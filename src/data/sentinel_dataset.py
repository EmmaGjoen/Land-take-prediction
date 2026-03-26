from pathlib import Path
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.config import (
    SENTINEL_DIR,
    MASK_DIR,
    ALL_YEARS,
    MAX_TIMESTEPS,
    ACQUISITIONS_PER_YEAR,
    load_metadata
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
        frequency: str | int | None = None,
        prediction_horizon: int = 2,
        input_years: int | None = None,
        calibrate_mode: bool = False,
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        transform: Transform to apply (flips, rotations, normalization)
        frequency: two quarters a year as default, optional: annual
        prediction_horizon (K): Number of years before final year in timeseries to cut off.
            With K=2, the model only sees data up to final year-2, forcing it to
            predict land take K years in advance.
        input_years (N): reference year + latest N-1 years before cutoff;
            None means all available years except the ones masked by the prediction  horizon
        """

        self.transform = transform
        self.frequency = frequency
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years
        self.calibrate_mode = calibrate_mode
        self.metadata = load_metadata()

        # Drop tiles whose cutoff year falls outside the available Sentinel record.
        # Example: endYear=2022 with K=5 → cutoff=2017, which is before our data starts.
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)
            if meta is None:
                dropped.append(fid)
                print(f"No metadata is found for {fid}. Check your metadata CSV.")
                continue
            if self.calibrate_mode:
                filtered.append(fid)
                continue

            cutoff_year = meta.end_year - prediction_horizon
            if cutoff_year in ALL_YEARS:
                filtered.append(fid)
            else:
                dropped.append(fid)
        if dropped:
            print(
                f"[SentinelDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s), whose cutoff year falls outside available data. "
                f"{len(filtered)} tiles remain."
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
        meta = self.metadata[fid]

        # read arrays
        with rasterio.open(self.img_paths[fid]) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1)  # (H, W)

        # Warn if Sentinel and mask have different spatial dimensions
        if img.shape[-2:] != mask.shape:
            print(f"[WARN] spatial mismatch for {fid}: Sentinel {img.shape[-2:]} vs mask {mask.shape}")

        C = 9
        num_quarters = ACQUISITIONS_PER_YEAR
        num_bands, H, W = img.shape
        num_years = num_bands // (C * num_quarters) # e.g. 126 // (9*2) = 7
        T = num_years * num_quarters  # e.g. 7 * 2 = 14

        # reshape (num_bands, H, W ) to (num_years, num_quarters, C, H, W)
        img = img.reshape(num_years, num_quarters, C, H, W)

        file_years = list(range(ALL_YEARS[0], ALL_YEARS[0] + num_years))  # always [2016, 2017, ..., 2024]
        valid_years = [y for y in file_years if meta.start_year <= y <= meta.end_year]
        start_clip = valid_years[0] - ALL_YEARS[0]
        end_clip   = valid_years[-1] - ALL_YEARS[0]

        # Slice img along the year axis
        img = img[start_clip : end_clip + 1]   # shape: (num_valid_years, num_quarters, C, H, W)
        file_years = valid_years               # update to reflect clipped range


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
            start_pos = file_years[0] - ALL_YEARS[0] + 1    # e.g. 2016 → 1, 2018 → 3
        else:
            start_pos = (file_years[0] - ALL_YEARS[0] + 1) * 2      # e.g. 2016 → 2, 2018 → 6

        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # to torch tensors
        img = torch.from_numpy(img).float()     # (num_years, C, H, W) or (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()

        # Apply transforms (normalization + augmentation) before any zero-padding, 
        # so that padding zeros do not affect normalization statistics.
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # Zero out timesteps after cutoff year (endYear - K) so U-TAE ignores them.
        # The model only sees data up to the cutoff, forcing it to predict K years ahead.
        cutoff_year = meta.end_year - self.prediction_horizon
        steps_per_year = 1 if self.frequency == "annual" else num_quarters

        if cutoff_year not in file_years:
            n_visible_timesteps = current_T
            print(f"[WARN] {fid}: cutoff_year {cutoff_year} is not in file_years {file_years}")
            cutoff_idx = len(file_years) - 1 # Fallback
        else:
            cutoff_idx  = file_years.index(cutoff_year)
            n_visible_timesteps = min((cutoff_idx + 1) * steps_per_year, current_T)

        # Aplly prediction horizon (K) masking
        img[n_visible_timesteps:]       = 0.0
        positions[n_visible_timesteps:] = 0

        # Apply input years (N) masking 
        if self.input_years is not None:
            # N total years: the start_year, plus the (N-1) latest years ending at cutoff
            window_limit = cutoff_year - (self.input_years - 1)
            
            # Iterate only through the years that survived the cutoff
            for i, y in enumerate(file_years[:cutoff_idx + 1]):
                # If the year is NOT the start_year AND falls before our N-1 window, mask it
                if y != meta.start_year and y <= window_limit:
                    t_start = i * steps_per_year
                    t_end = t_start + steps_per_year

                    img[t_start:t_end] = 0.0
                    positions[t_start:t_end] = 0

        # Pad or truncate to max_timesteps
        # Padding is zeros so U-TAE treats them as masked timesteps
        if current_T < MAX_TIMESTEPS:
            pad_len = MAX_TIMESTEPS - current_T
            img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > MAX_TIMESTEPS:
            img = img[:MAX_TIMESTEPS]
            positions = positions[:MAX_TIMESTEPS]

        return img, mask, positions
