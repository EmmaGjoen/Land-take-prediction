from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.config import (
    ALPHAEARTH_DIR
)

def find_file_by_prefix(base_dir: Path, fid: str) -> Path:
    """
    Find a file in base_dir whose name starts with fid and ends with .tif or .tiff.

    Example:
        fid = "a22-0323..._37-98..."
        file = "a22-0323..._37-98..._VEY_Mosaic.tif"

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


class AlphaEarthDataset(Dataset):
    DATASET_NAME = "alphaearth"
    """
    Loads .....

    Assumptions:
      - `ids` are REFIDs that match the *prefix* of the filenames in ALPHAEARTH_DIR.
    """

    def __init__(
        self,
        ids,
        transform,
        slice_mode: str = None,
        frequency: str = None
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        slice_mode: None or "first_half"
        transform: Transform to apply (flips, rotations, normalization)
        """
        self.ids = ids
        self.slice_mode = slice_mode
        self.transform = transform
        self.frequency = frequency

        # Pre-resolve image and mask paths once for stability and speed
        self.emb_paths: dict[str, Path] = {}

        for fid in self.ids:
            emb_path = find_file_by_prefix(ALPHAEARTH_DIR, fid)
            self.emb_paths[fid] = emb_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        emb_path = self.emb_paths[fid]

        # 1) read arrays
        with rasterio.open(emb_path) as src:
            emb = src.read()  # (bands, H, W)

        # 2) reshape to (T, C, H, W)
        # Expected layout: 448 = 7 years * 64 dim
        num_bands, H, W = emb.shape
        if num_bands != 448:
            raise ValueError(
                f"Expected 448 bands for AlphaEarth, got {num_bands} for {fid} at {emb_path}"
            )
        emb = emb.reshape(7, 64, H, W)

        # 4) to torch tensors
        emb = torch.from_numpy(emb).float()     # (T, C, H, W)

        # temporal allignment with sentinel
        if self.frequency != "annual":
            emb = emb.repeat_interleave(repeats=2, dim=0) # (14, 64, H, W)

        # 3) optionally take first half of the time series
        if self.slice_mode == "first_half":
            T = emb.shape[0]
            emb = emb[: T // 2]

        # 5) Apply transforms (which handle padding/cropping via CenterCropTS)
        if self.transform is not None:
            dummy_mask = torch.zeros((H, W), dtype=torch.long)
            emb = self.transform(emb, dummy_mask)

        return emb
