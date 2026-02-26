"""Dataset for GeoTessera embeddings snapped to mask grid.

Each file contains 128-band embeddings for a single year. Multiple years are
stacked along the temporal dimension to form a (T, C, H, W) tensor that
aligns temporally with the Sentinel time series.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.config import TESSERA_DIR


class TesseraDataset(Dataset):
    """Load GeoTessera embeddings as a temporal stack (T, C, H, W).

    Files are expected at ``TESSERA_DIR/{refid}_tessera_{year}_snapped.tif``
    with 128 bands each. Years are stacked in chronological order.

    The temporal axis is aligned with SentinelDataset by repeating each
    yearly embedding twice (matching 2 quarters per year in Sentinel).

    Args:
        ids: Reference IDs matching the mask filename prefix.
        transform: ComposeTS-style transform applied to (emb, dummy_mask).
        slice_mode: ``None`` keeps all timesteps; ``"first_half"`` keeps the
            first half (matching the Sentinel first-half convention).
        frequency: ``None`` repeats yearly embeddings ×2 to match Sentinel's
            bi-quarterly cadence; ``"annual"`` keeps one timestep per year.
        years: Years to load. Defaults to 2018-2024 (7 years) to match Sentinel.
    """

    DATASET_NAME = "tessera"

    # Sentinel has 126 bands = 7 years × 2 quarters × 9 spectral bands.
    from src.config import YEARS as YEARS_DEFAULT
    BANDS_PER_YEAR = 128

    def __init__(
        self,
        ids: list[str],
        transform,
        slice_mode: Optional[str] = None,
        frequency: Optional[str] = None,
        years: Optional[list[int]] = None,
    ):
        self.ids = ids
        self.transform = transform
        self.slice_mode = slice_mode
        self.frequency = frequency
        self.years = years or self.YEARS_DEFAULT

        self.emb_paths: dict[str, list[Path]] = {}
        self.valid_ids = []
        self.excluded_ids = {}
        for fid in self.ids:
            paths = []
            missing_years = []
            for year in self.years:
                p = TESSERA_DIR / f"{fid}_tessera_{year}_snapped.tif"
                if not p.exists():
                    missing_years.append(year)
                else:
                    paths.append(p)
            if not missing_years:
                self.emb_paths[fid] = paths
                self.valid_ids.append(fid)
            else:
                self.excluded_ids[fid] = missing_years
        if self.excluded_ids:
            print(f"[TesseraDataset] Excluded {len(self.excluded_ids)} tiles missing required years.")
            for fid, years in self.excluded_ids.items():
                print(f"  Excluded {fid}: missing years {years}")
        self.ids = self.valid_ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        fid = self.ids[idx]
        paths = self.emb_paths[fid]

        yearly_arrays = []
        for path in paths:
            with rasterio.open(path) as src:
                arr = src.read()  # (128, H, W)
            if arr.shape[0] != self.BANDS_PER_YEAR:
                raise ValueError(
                    f"Expected {self.BANDS_PER_YEAR} bands, got {arr.shape[0]} "
                    f"for {fid} at {path}"
                )
            yearly_arrays.append(arr)

        H, W = yearly_arrays[0].shape[1], yearly_arrays[0].shape[2]

        # Stack: (num_years, 128, H, W)
        emb = np.stack(yearly_arrays, axis=0)
        emb = torch.from_numpy(emb).float()  # (T_years, C, H, W)

        # Temporal alignment with Sentinel: 2 quarters per year
        if self.frequency != "annual":
            emb = emb.repeat_interleave(repeats=2, dim=0)  # (T_years*2, C, H, W)

        if self.slice_mode == "first_half":
            T = emb.shape[0]
            emb = emb[: T // 2]

        dummy_mask = torch.zeros((H, W), dtype=torch.long)
        if self.transform is not None:
            emb, dummy_mask = self.transform(emb, dummy_mask)

        return emb, dummy_mask
