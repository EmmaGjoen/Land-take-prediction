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
        years: Years to load. Defaults to 2017-2024 (8 years).
    """

    DATASET_NAME = "tessera"

    YEARS_DEFAULT = list(range(2017, 2025))  # 2017-2024
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
        for fid in self.ids:
            paths = []
            for year in self.years:
                p = TESSERA_DIR / f"{fid}_tessera_{year}_snapped.tif"
                if not p.exists():
                    raise FileNotFoundError(
                        f"Missing Tessera file for {fid} year={year}: {p}"
                    )
                paths.append(p)
            self.emb_paths[fid] = paths

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

        if self.transform is not None:
            dummy_mask = torch.zeros((H, W), dtype=torch.long)
            emb = self.transform(emb, dummy_mask)

        return emb
