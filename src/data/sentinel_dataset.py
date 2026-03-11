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
        max_timesteps: int = 20 #(10 years (2016-2025) * 2 quarters = 20)
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        slice_mode: None or "first_half", or a specific year (as int): 2019, 2020, 2021, 2022, 2023
        transform: Transform to apply (flips, rotations, normalization)
        frequency: two quarters a year as default, optional: annual
        max_timesteps: The global maximum sequence length to pad all chips to.
        """
        self.ids = ids
        self.slice_mode = slice_mode
        self.transform = transform
        self.frequency = frequency

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
            current_T = img.shape[0]
            img = img[: current_T // 2]

        # to torch tensors
        img = torch.from_numpy(img).float()     # (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()


        
        
        # Apply transforms (which handle padding/cropping via CenterCropTS)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # # Assuming you can extract the start_year from your filename or metadata
        # # For example, if fid = "2018_tile_45", start_year = 2018
        # start_year = int(fid.split("_")[0]) 
        
        # # Calculate offset from your absolute earliest year (e.g., 2016)
        # # 2 quarters per year = multiply by 2
        # offset = (start_year - 2016) * 2 
        
        # # Create a tensor of sequential positions: e.g., [4, 5, 6, 7...] for a 2018 start
        # positions = torch.arange(offset, offset + current_T, dtype=torch.long)

        positions = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])




        # Apply padding after transform(normalization) so the empty timesteps remain exactly 0.0
        if self.max_timesteps is not None:
            current_T = img.shape[0]
            if current_T < self.max_timesteps:
                pad_len = self.max_timesteps - current_T
                img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
                # Pad positions with 0s at the end
                positions = F.pad(positions, (0, pad_len))
            elif current_T > self.max_timesteps:
                img = img[:self.max_timesteps]
                positions = positions[:self.max_timesteps]

        return img, mask, positions
