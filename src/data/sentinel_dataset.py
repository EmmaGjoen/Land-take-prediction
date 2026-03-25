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
    """
    Find the unique .tif file in base_dir whose name starts with fid.

    Example:
        fid  = "a-0-84261876000576_52-28383748670937"
        file = "a-0-84261876000576_52-28383748670937_RGBNIRRSWIRQ_Mosaic.tif"

    Raises FileNotFoundError if no match, RuntimeError if multiple matches.
    """
    candidates = sorted(
        list(base_dir.glob(f"{fid}*.tif"))
    )
    if not candidates:
        raise FileNotFoundError(f"No file starting with {fid} in {base_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple files starting with {fid} in {base_dir}: {candidates}")
    return candidates[0]


class SentinelDataset(Dataset):
    DATASET_NAME = "sentinel"
    """
    Loads Sentinel-2 time series data and reshapes it into (T, C, H, W)
    so it can be fed directly to temporal models (like U-TAE).

    Each REFID maps to a .tif file whose name starts with that REFID,
    found in SENTINEL_DIR and MASK_DIR respectively.
    """

    def __init__(
        self,
        ids,
        transform,
        frequency: str | int | None = None,
        prediction_horizon: int = 2,
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        transform: Transform to apply (flips, rotations, normalization)
        frequency: two quarters a year as default, optional: annual
        prediction_horizon: Number of years before final year in timeseries to cut off.
            With K=2, the model only sees data up to final year-2, forcing it to
            predict land take K years in advance.
        """

        self.transform = transform
        self.frequency = frequency
        self.prediction_horizon = prediction_horizon
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

        # Pre-resolve image and mask paths once for stability and speed
        self.img_paths: dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}

        for fid in self.ids:
            self.img_paths[fid] = find_file_by_prefix(SENTINEL_DIR, fid)
            self.mask_paths[fid] = find_file_by_prefix(MASK_DIR, fid)

    def __len__(self):
        # One sample per chip
        return len(self.ids)

    def __getitem__(self, idx):
        # Direct mapping: each idx corresponds to one 64×64 chip
        fid = self.ids[idx]
        meta = self.metadata[fid]
        img_path = self.img_paths[fid]
        mask_path = self.mask_paths[fid]

        # read arrays
        with rasterio.open(img_path) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1)  # (H, W)

        # Warn if Sentinel and mask have different spatial dimensions
        if img.shape[-2:] != mask.shape:
            print(f"[WARN] spatial mismatch for {fid}: Sentinel {img.shape[-2:]} vs mask {mask.shape}")

        C = 9
        num_quarters = ACQUISITIONS_PER_YEAR
        num_bands, H, W = img.shape
        num_years = num_bands // (C * num_quarters) # e.g. 126 // (9*2) = 7
        # num_years = meta.end_year - meta.start_year + 1 # e.g. 2018-2025 = 8 years
        T = num_years * num_quarters  # e.g. 7 * 2 = 14

        # if num_bands != T * C:
        #     # raise ValueError(
        #     #     f"Expected number of years to be {meta.end_year - meta.start_year +1}, got {num_years} for {fid} at {img_path}"
        #     # )
        #     print(f"[WARN] {fid}: expected {T * C} bands but .tif has {num_bands} — using file band count")
        #     num_years = num_bands // (C * num_quarters)
        
        # reshape (num_bands, H, W ) to (num_years, num_quarters, C, H, W)
        img = img.reshape(num_years, num_quarters, C, H, W)



        # file_years = list(range(meta.start_year, meta.start_year + num_years)) 
        file_years = list(range(ALL_YEARS[0], ALL_YEARS[0] + num_years))  # always [2016, 2017, ..., 2024]
        # valid_years = [y for y in file_years if y in ALL_YEARS]     # Clip to only years within ALL_YEARS
        valid_years = [y for y in file_years if meta.start_year <= y <= meta.end_year]
        # start_clip = valid_years[0] - meta.start_year
        # end_clip   = valid_years[-1] - meta.start_year

        start_clip = valid_years[0] - ALL_YEARS[0]
        end_clip   = valid_years[-1] - ALL_YEARS[0]

        # Slice img along the year axis
        img = img[start_clip : end_clip + 1]   # shape: (num_valid_years, num_quarters, C, H, W)
        file_years = valid_years               # update to reflect clipped range


        if self.frequency == "annual":
            # if T % num_quarters != 0:
            #     raise ValueError(
            #         f"Timesteps ({T}) not divisible by ({num_quarters}) quarters per year for {fid} at {img_path}"
            #     )
            # aggregate quarters -> (num_years, C, H, W)
            img = img.mean(axis=1)  
        else:
            # merge quarters and years to timesteps -> (T, C, H, W)
            img = img.reshape(-1, C, H, W)

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
        if cutoff_year not in file_years:
            n_visible_timesteps = current_T
            print(f"[WARN] {fid}: cutoff_year {cutoff_year} is not in file_years {file_years}")
        else:
            cutoff_idx  = file_years.index(cutoff_year)
            steps_per_year = 1 if self.frequency == "annual" else num_quarters
            n_visible_timesteps = min((cutoff_idx + 1) * steps_per_year, current_T)

        img[n_visible_timesteps:]       = 0.0
        positions[n_visible_timesteps:] = 0

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
