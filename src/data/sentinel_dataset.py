from pathlib import Path
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.config import (
    SENTINEL_DIR,
    MASK_DIR,
    YEARS
)


def find_file_by_prefix(base_dir: Path, fid: str) -> Path:
    """Find the unique .tif file in base_dir whose name starts with fid."""
    candidates = sorted(list(base_dir.glob(f"{fid}*.tif")))
    if not candidates:
        raise FileNotFoundError(f"No file starting with {fid} in {base_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple files starting with {fid} in {base_dir}: {candidates}")
    return candidates[0]


class SentinelDataset(Dataset):
    DATASET_NAME = "sentinel"

    def __init__(
        self,
        ids,
        transform,
        slice_mode: str = None,
        frequency: str | int | None = None,
        end_years: dict[str, int] | None = None,
        start_years: dict[str, int] | None = None,
        max_timesteps: int | None = None,
        prediction_horizon: int = 2,
        input_years: int | None = None,
    ):
        """
        ids: list of REFIDs
        transform: normalization + augmentation
        slice_mode: None, "first_half", or a specific year to slice up to
        frequency: None (quarterly, default) or "annual"
        end_years: {refid: changeYear} — timesteps after changeYear are zeroed
        start_years: {refid: startYear} — used as per-tile reference year anchor
        max_timesteps: pad/truncate time axis to this length for batching
        prediction_horizon (K): model sees data up to changeYear-K
        input_years (N): reference year + latest N-1 years before cutoff;
            None means all available years
        """
        self.slice_mode = slice_mode
        self.transform = transform
        self.frequency = frequency
        self.end_years = end_years
        self.start_years = start_years
        self.max_timesteps = max_timesteps
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years

        # Drop tiles whose cutoff year falls outside YEARS
        if end_years is not None and prediction_horizon > 0:
            filtered, dropped = [], []
            for fid in ids:
                ey = end_years.get(fid)
                if ey is not None and (ey - prediction_horizon) not in YEARS:
                    dropped.append(fid)
                else:
                    filtered.append(fid)
            if dropped:
                print(
                    f"[SentinelDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s) "
                    f"whose cutoff year falls outside YEARS. {len(filtered)} tiles remain."
                )
            self.ids = filtered
        else:
            self.ids = list(ids)

        # Drop tiles with fewer than N years between reference year and cutoff
        if input_years is not None and end_years is not None:
            filtered, dropped = [], []
            for fid in self.ids:
                ey = end_years.get(fid)
                if ey is None:
                    filtered.append(fid)
                    continue
                cutoff_year = ey - prediction_horizon
                if cutoff_year not in YEARS:
                    filtered.append(fid)
                    continue
                tile_start = start_years.get(fid, YEARS[0]) if start_years else YEARS[0]
                anchor_year = max(tile_start, YEARS[0])
                available_years = YEARS.index(cutoff_year) - YEARS.index(anchor_year) + 1
                if available_years < input_years:
                    dropped.append(fid)
                else:
                    filtered.append(fid)
            if dropped:
                print(
                    f"[SentinelDataset] N={input_years}: excluded {len(dropped)} tile(s) "
                    f"with fewer than {input_years} available years. {len(filtered)} tiles remain."
                )
            self.ids = filtered

        # Pre-resolve paths
        self.img_paths: dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}
        for fid in self.ids:
            self.img_paths[fid] = find_file_by_prefix(SENTINEL_DIR, fid)
            self.mask_paths[fid] = find_file_by_prefix(MASK_DIR, fid)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]

        with rasterio.open(self.img_paths[fid]) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1)  # (H, W)

        if img.shape[-2:] != mask.shape:
            print(f"[WARN] spatial mismatch for {fid}: Sentinel {img.shape[-2:]} vs mask {mask.shape}")

        # Reshape to (num_years, num_quarters, C, H, W)
        C = 9
        num_bands, H, W = img.shape
        num_quarters = 2
        num_years = num_bands // (num_quarters * C)
        if num_bands != num_years * num_quarters * C:
            raise ValueError(f"Band count {num_bands} not divisible by {num_quarters * C} for {fid}")
        img = img.reshape(num_years, num_quarters, C, H, W)

        if self.slice_mode in YEARS:
            end_idx = YEARS.index(int(self.slice_mode))
            img = img[0: end_idx + 1]

        if self.frequency == "annual":
            img = img.mean(axis=1)  # (num_years, C, H, W)
        else:
            img = img.reshape(img.shape[0] * img.shape[1], C, H, W)

        if self.slice_mode == "first_half":
            img = img[: img.shape[0] // 2]

        # Per-tile positions: encoded relative to BASE_YEAR=2015 so position 0
        # is never assigned (kept free for U-TAE pad detection).
        # Quarterly: 2016Q2=2 ... 2025Q3=21 | Annual: 2016=1 ... 2025=10
        current_T = img.shape[0]
        BASE_YEAR = 2015
        tile_start_year = self.start_years.get(fid, YEARS[0]) if self.start_years else YEARS[0]
        tile_start_year = max(tile_start_year, YEARS[0])
        start_pos = (tile_start_year - BASE_YEAR) * (1 if self.frequency == "annual" else 2)
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # Mask timesteps after the cutoff (changeYear - K) to zero so U-TAE
        # ignores them via pad_value=0.0
        if self.end_years is not None:
            end_year = self.end_years.get(fid)
            if end_year is not None:
                cutoff_year = end_year - self.prediction_horizon
                assert cutoff_year in YEARS, f"cutoff_year {cutoff_year} not in YEARS for {fid}"
                steps_per_year = 1 if self.frequency == "annual" else 2
                cutoff_year_idx = YEARS.index(cutoff_year)
                n_valid = min((cutoff_year_idx + 1) * steps_per_year, current_T)
                img[n_valid:] = 0.0
                positions[n_valid:] = 0

                # Select reference year + latest N-1 years, zero the gap between them
                if self.input_years is not None:
                    tile_start = self.start_years.get(fid, YEARS[0]) if self.start_years else YEARS[0]
                    anchor_year_idx = YEARS.index(max(tile_start, YEARS[0]))
                    anchor_ts = anchor_year_idx * steps_per_year
                    if anchor_ts > 0:
                        img[:anchor_ts] = 0.0
                        positions[:anchor_ts] = 0
                    gap_start = (anchor_year_idx + 1) * steps_per_year
                    gap_end = (cutoff_year_idx - (self.input_years - 2)) * steps_per_year
                    if gap_start < gap_end:
                        img[gap_start:gap_end] = 0.0
                        positions[gap_start:gap_end] = 0

        # Pad or truncate to max_timesteps for batching
        if self.max_timesteps is not None:
            if current_T < self.max_timesteps:
                pad_len = self.max_timesteps - current_T
                img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
                positions = F.pad(positions, (0, pad_len))
            elif current_T > self.max_timesteps:
                img = img[:self.max_timesteps]
                positions = positions[:self.max_timesteps]

        return img, mask, positions
