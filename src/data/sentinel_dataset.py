"""Sentinel-2 time series Dataset for land take segmentation."""
import bisect
from pathlib import Path
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.config import (
    SENTINEL_DIR,
    MASK_DIR,
    SENTINEL_YEARS,
    ACQUISITIONS_PER_YEAR_SENTINEL,
    load_metadata
)
from src.data.file_helpers import find_file_by_prefix


class SentinelDataset(Dataset):
    DATASET_NAME = "sentinel"

    def __init__(
        self,
        ids,
        transform,
        frequency: str | int | None = None,
        prediction_horizon: int = 2,
        input_years: int | None = None,
        calibrate_mode: bool = False,
    ):
        """Sentinel-2 time series dataset for land take segmentation.

        Args:
            ids: list of REFIDs (tile identifiers).
            transform: spatial/spectral transforms (flips, rotations, normalization).
            frequency: bi-quarterly (default) or "annual".
            prediction_horizon (K): zero out the last K years so the model predicts
                K years ahead. E.g. K=2 hides 2023-2024 for a tile ending in 2024.
            input_years (N): keep start_year + latest N-1 years before cutoff.
                None = all available years.
            calibrate_mode: skip temporal masking/padding, used when computing
                normalization stats so zeros don't skew mean/std.
        """

        self.transform = transform
        self.frequency = frequency
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years
        self.calibrate_mode = calibrate_mode
        self.metadata = load_metadata()
        self.tile_years: dict[str, list[int]] = {}

        self.steps_per_year = 1 if self.frequency == "annual" else ACQUISITIONS_PER_YEAR_SENTINEL
        self.max_timesteps = len(SENTINEL_YEARS) * self.steps_per_year

        # Drop tiles with no metadata or whose cutoff/start year is out of range.
        # e.g. end_year=2020 with K=5 gives cutoff=2015, before our data starts.
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)
            
            if meta is None:
                dropped.append(fid)
                print(f"[Sentinel] Excluded {fid}: No metadata.")
                continue

            if self.calibrate_mode:
                filtered.append(fid)
                self.tile_years[fid] = SENTINEL_YEARS
                continue

            if meta.start_year < SENTINEL_YEARS[0]:
                dropped.append(fid)
                print(f"[Sentinel] Excluded {fid}: has annotation start year before {SENTINEL_YEARS[0]}.")
                continue

            cutoff_year = meta.end_year - prediction_horizon
            tile_years = [
                y for y in SENTINEL_YEARS
                if meta.start_year <= y <= meta.end_year
            ]
            if not tile_years or cutoff_year not in tile_years:
                print(f"[Sentinel] Excluded {fid}: Valid data window is empty or missing the {cutoff_year} cutoff year.")
                dropped.append(fid)
            else:
                filtered.append(fid)
                self.tile_years[fid] = tile_years

        if dropped:
            print(
                f"[SentinelDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s). "
                f"{len(filtered)} tiles remain."
            )
        self.ids = filtered

        self.img_paths: dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}
        for fid in self.ids:
            self.img_paths[fid] = find_file_by_prefix(SENTINEL_DIR, fid)
            self.mask_paths[fid] = find_file_by_prefix(MASK_DIR, fid)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        meta = self.metadata[fid]
        tile_years = self.tile_years[fid]

        # read arrays
        with rasterio.open(self.img_paths[fid]) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1)  # (H, W)sq

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Warn if Sentinel and mask have different spatial dimensions
        if img.shape[-2:] != mask.shape:
            print(f"[WARN] spatial mismatch for {fid}: Sentinel {img.shape[-2:]} vs mask {mask.shape}")

        C = 9
        num_quarters = ACQUISITIONS_PER_YEAR_SENTINEL
        num_bands, H, W = img.shape
        num_years = num_bands // (C * num_quarters) # e.g. 126 // (9*2) = 7

        # reshape (num_bands, H, W ) to (num_years, num_quarters, C, H, W)
        img = img.reshape(num_years, num_quarters, C, H, W)


        # Slice to the valid tile_years range
        start_clip = tile_years[0] - SENTINEL_YEARS[0]
        end_clip   = tile_years[-1] - SENTINEL_YEARS[0]
        img = img[start_clip : end_clip + 1]


        if self.frequency == "annual":
            img = img.mean(axis=1)  
        else:
            # merge quarters and years to timesteps -> (T, C, H, W)
            img = img.reshape(-1, C, H, W)

        if self.calibrate_mode:
            # Return raw tensor immediately so stats aren't corrupted by zeros from padding
        
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
            mask = (mask > 0).long()
        
            if self.transform is not None:
                img, mask = self.transform(img, mask)
            return img, mask, torch.zeros(1)

        current_T = img.shape[0]

        # Position encoding: 1-indexed so that 0 marks padding.
        # U-TAE ignores any timestep with position=0 in attention.

        if self.frequency == "annual":
            start_pos = start_clip + 1    # e.g. 2016 → 1, 2018 → 3
        else:
            start_pos = (start_clip * num_quarters) + 1      # e.g. 2016 → 2, 2018 → 6

        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # to torch tensors
        img = torch.from_numpy(img).float()     # (num_years, C, H, W) or (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()

        # Apply transforms before zero-padding so padding doesn't affect stats.
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # Zero out timesteps after cutoff (end_year - K) so the model predicts K years ahead.
        cutoff_year = meta.end_year - self.prediction_horizon
        n_visible_years = bisect.bisect_right(tile_years, cutoff_year)
        n_visible = n_visible_years * self.steps_per_year

        # Apply prediction horizon (K) masking
        img[n_visible:] = 0.0
        positions[n_visible:] = 0

        # Input years (N) masking: keep start_year + the N-1 latest years before cutoff.
        if self.input_years is not None:
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(tile_years[:n_visible_years]):
                # Always keep start_year; mask anything before the N-1 window
                if y != tile_years[0] and y <= window_limit:
                    t_start = i * self.steps_per_year
                    t_end = t_start + self.steps_per_year

                    img[t_start:t_end] = 0.0
                    positions[t_start:t_end] = 0

        # Pad or truncate to max_timesteps (zeros = masked by U-TAE)
        if current_T < self.max_timesteps:
            pad_len = self.max_timesteps - current_T
            img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > self.max_timesteps:
            img = img[:self.max_timesteps]
            positions = positions[:self.max_timesteps]

        return img, mask, positions
