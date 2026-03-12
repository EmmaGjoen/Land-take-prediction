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
    """
    Find a file in base_dir whose name starts with fid and ends with .tif or .tiff.

    Example:
        fid = "a22-0323..._37-98..."
        file = "a22-0323..._37-98..._RGBNIRRSWIRQ_Mosaic.tif"

    This assumes there is exactly one such file per fid.
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
    Loads time series data and reshapes it into (T, C, H, W)
    so it can be fed directly to temporal models (like the FCEF baseline).

    Assumptions:
      - `ids` are REFIDs that match the *prefix* of the filenames in
        SENTINEL_DIR / MASK_DIR.
    """

    def __init__(
        self,
        ids,
        transform,
        slice_mode: str = None,
        frequency: str | int | None = None,
        end_years: dict[str, int] | None = None,
        max_timesteps: int | None = None,
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        slice_mode: None or "first_half", or a specific year (as int): 2019, 2020, 2021, 2022, 2023
        transform: Transform to apply (flips, rotations, normalization)
        frequency: two quarters a year as default, optional: annual
        end_years: mapping of refid -> endYear from annotations metadata.
            Sentinel timesteps after endYear are zeroed AFTER normalization,
            so U-TAE's pad_value=0.0 detection masks them correctly.
        max_timesteps: If set, pad or truncate the time axis to this length.
            Leave as None when all tiles share the same T.
        """
        self.ids = ids
        self.slice_mode = slice_mode
        self.transform = transform
        self.frequency = frequency
        self.end_years = end_years
        self.max_timesteps = max_timesteps

        # Pre-resolve image and mask paths once for stability and speed
        self.img_paths: dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}

        for fid in self.ids:
            img_path = find_file_by_prefix(SENTINEL_DIR, fid)
            mask_path = find_file_by_prefix(MASK_DIR, fid)

            self.img_paths[fid] = img_path
            self.mask_paths[fid] = mask_path

    def __len__(self):
        # One sample per chip
        return len(self.ids)

    def __getitem__(self, idx):
        # Direct mapping: each idx corresponds to one 64×64 chip
        fid = self.ids[idx]

        img_path = self.img_paths[fid]
        mask_path = self.mask_paths[fid]

        # read arrays
        with rasterio.open(img_path) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1)  # (H, W)

        # Warn if Sentinel and mask have different spatial dimensions (data quality check)
        if img.shape[-2:] != mask.shape:
            print(
                f"[WARN] spatial mismatch for {fid}: "
                f"Sentinel {img.shape[-2:]} vs mask {mask.shape}"
            )

        # reshape to (T, C, H, W)
        # (old data:) Expected layout: 126 = 7 years * 2 quarters * 9 bands
        C = 9
        num_bands, H, W = img.shape
        T = num_bands // C
        num_quarters = 2
        num_years = T // num_quarters
        
        if num_bands != num_years *num_quarters * C:
            raise ValueError(
                f"Expected bands to be divisible by {num_quarters * C}, got {num_bands} for {fid} at {img_path}"
            )
        img = img.reshape(num_years, num_quarters, C, H, W)

        if self.slice_mode in YEARS:
            end_idx = YEARS.index(int(self.slice_mode))
            img = img[0: end_idx +1]

        if self.frequency == "annual":
            if T % num_quarters != 0:
                raise ValueError(
                    f"Timesteps ({T}) not divisible by steps_per_year ({num_quarters}) for {fid} at {img_path}"
                )
            img = img.mean(axis=1)  # aggregate quarters -> (num_years, C, H, W)
        else:
            new_T = img.shape[0] * img.shape[1]
            img = img.reshape(new_T, C, H, W)

        # optionally take first half of the time series
        if self.slice_mode == "first_half":
            img = img[: img.shape[0] // 2]

        # Compute positions encoding absolute temporal location.
        # All Sentinel tiles start at 2018; we encode relative to base year 2016
        # so positions are non-zero (avoids collision with pad position=0).
        # Annual:    2018=2, 2019=3, ..., 2024=8
        # Quarterly: 2018Q1=4, 2018Q2=5, 2019Q1=6, ..., 2024Q2=17
        current_T = img.shape[0]
        BASE_YEAR = 2016
        START_YEAR = 2018
        if self.frequency == "annual":
            start_pos = START_YEAR - BASE_YEAR          # 2
        else:
            start_pos = (START_YEAR - BASE_YEAR) * 2   # 4
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # to torch tensors
        img = torch.from_numpy(img).float()     # (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()

        # Apply transforms (which handle padding/cropping via CenterCropTS)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # Apply endYear masking AFTER normalization so masked timesteps are
        # exactly 0.0 — the value U-TAE's pad_value detection compares against.
        if self.end_years is not None:
            end_year = self.end_years.get(fid)
            if end_year is not None and end_year in YEARS:
                if self.frequency == "annual":
                    n_valid = min(YEARS.index(end_year) + 1, current_T)
                else:
                    n_valid = min((YEARS.index(end_year) + 1) * 2, current_T)
                img[n_valid:] = 0.0
                positions[n_valid:] = 0

        # Pad or truncate to max_timesteps if set (needed when T varies per tile).
        # Padding is zeros so U-TAE treats them as masked timesteps automatically.
        if self.max_timesteps is not None:
            if current_T < self.max_timesteps:
                pad_len = self.max_timesteps - current_T
                img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
                positions = F.pad(positions, (0, pad_len))
            elif current_T > self.max_timesteps:
                img = img[:self.max_timesteps]
                positions = positions[:self.max_timesteps]

        return img, mask, positions
