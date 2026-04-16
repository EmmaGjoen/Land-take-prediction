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
        """
        - ids: list of REFIDs (filename stems without the long suffix)
        - transform: Transform to apply (flips, rotations, normalization)
        - frequency: two quarters a year as default, optional: annual
        - prediction_horizon (K): Number of years before final year in timeseries to cut off.
            With K=2, the model only sees data up to final year-2, forcing it to
            predict land take K years in advance.
        - input_years (N): reference year + latest N-1 years before cutoff;
            None means all available years except the ones masked by the prediction  horizon
        - calibrate mode: use when computing normalization stats to avoid zero padding affecting the mean and std
        
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

        # Keep tiles that have at least 1 visible Sentinel year before the cutoff.
        # A tile is usable if cutoff_year is within the Sentinel record AND >= startYear.
        # Tiles with fewer visible years than N are kept and zero-padded — this maximises
        # the dataset while keeping the temporal masking experiment intact.
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
                print(f"[Sentinel] xcluded {fid}: has annotation start year before {SENTINEL_YEARS[0]}.")
                continue

            cutoff_year = meta.end_year - prediction_horizon
            if cutoff_year in ALL_YEARS and cutoff_year >= meta.start_year:
                filtered.append(fid)
                self.tile_years[fid] = tile_years

        if dropped:
            print(
                f"[SentinelDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s) "
                f"with no visible years before cutoff. {len(filtered)} tiles remain."
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
            mask = src_m.read(1)  # (H, W)

        # Warn if Sentinel and mask have different spatial dimensions
        if img.shape[-2:] != mask.shape:
            print(f"[WARN] spatial mismatch for {fid}: Sentinel {img.shape[-2:]} vs mask {mask.shape}")

        C = 9
        num_quarters = ACQUISITIONS_PER_YEAR_SENTINEL
        num_bands, H, W = img.shape
        num_years = num_bands // (C * num_quarters) # e.g. 126 // (9*2) = 7

        # reshape (num_bands, H, W ) to (num_years, num_quarters, C, H, W)
        img = img.reshape(num_years, num_quarters, C, H, W)


        # Slice embedding to match the valid tile_years
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

        # Position encoding: encode each timestep's absolute temporal position.
        # Positions are 1-indexed so that 0 is always available to mark padding.
        # U-TAE masks out any timestep with position=0 from attention.

        if self.frequency == "annual":
            start_pos = start_clip + 1    # e.g. 2016 → 1, 2018 → 3
        else:
            start_pos = (start_clip * num_quarters) + 1      # e.g. 2016 → 2, 2018 → 6

        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # to torch tensors
        img = torch.from_numpy(img).float()     # (num_years, C, H, W) or (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()

        # Apply transforms (normalization + augmentation) before any zero-padding, 
        # so that padding zeros do not affect normalization statistics.
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # Zero out timesteps after cutoff year (end_year - K) so U-TAE ignores them.
        # The model only sees data up to the cutoff, forcing it to predict K years ahead.
        cutoff_year = meta.end_year - self.prediction_horizon
        cutoff_idx  = tile_years.index(cutoff_year)
        n_visible = (cutoff_idx + 1) * self.steps_per_year

        # Apply prediction horizon (K) masking
        img[n_visible_timesteps:]       = 0.0
        positions[n_visible_timesteps:] = 0

        # Apply input years (N) masking 
        if self.input_years is not None:
            # N total years: the start_year, plus the (N-1) latest years ending at cutoff
            window_limit = cutoff_year - (self.input_years - 1)
            
            # Iterate only through the years that survived the cutoff
            for i, y in enumerate(tile_years[:cutoff_idx + 1]):
                # If the year is NOT the start year of the tile AND falls before our N-1 window, mask it
                if y != tile_years[0] and y <= window_limit:
                    t_start = i * self.steps_per_year
                    t_end = t_start + self.steps_per_year

                    img[t_start:t_end] = 0.0
                    positions[t_start:t_end] = 0

        # Pad or truncate to max_timesteps
        # Padding is zeros so U-TAE treats them as masked timesteps
        if current_T < self.max_timesteps:
            pad_len = self.max_timesteps - current_T
            img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > self.max_timesteps:
            img = img[:self.max_timesteps]
            positions = positions[:self.max_timesteps]

        return img, mask, positions
