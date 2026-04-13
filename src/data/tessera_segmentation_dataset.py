from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from file_helpers import find_file_by_prefix
from src.config import TESSERA_YEARS, MASK_DIR, TESSERA_DIR, load_metadata

class TesseraSegmentationDataset(Dataset):
    """Load GeoTessera yearly embeddings paired with land-take segmentation masks postitions encoding for the embedding timeseries.

    The loaded years are stacked chronologically to form a ``(T, 128, H, W)``
    tensor, padded with zeros to ``len(TESSERA_YEARS)`` so that all samples in a batch
    have the same temporal length.

    Args:
        ids: list of REFIDs
        transform: transforms to apply (flips, rotations)
        prediction_horizon (K): Number of years before final year in timeseries to cut off.
            With K=2, the model only sees data up to final year-2, forcing it to
            predict land take K years in advance.
        input_years (N): reference year + latest N-1 years before cutoff;
            None means all years up to cutoff

    **Tile filtering** at construction time (logged):

        * Tiles with no metadata or whose cutoff is out of range.
        * Tiles missing any any TESSERA file for years within [start_year, end_year]
        * Tiles with start year before the available TESSERA_YEARS
    """

    DATASET_NAME = "tessera"
    _BANDS_PER_YEAR = 128
    _EXPECTED_BANDS = len(TESSERA_YEARS) * _BANDS_PER_YEAR  #1024

    def __init__(
        self,
        ids: list[str],
        transform,
        prediction_horizon: int = 2,
        input_years: int | None = None,
    ):
        self.transform = transform
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years
        self.max_timesteps = len(TESSERA_YEARS)
        self.metadata = load_metadata()
        self.tile_years: dict[str, list[int]] = {}

        # Drop tiles with no metadata or whose cutoff or start year is out of range 
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)

            if meta is None:
                dropped.append(fid)
                print(f"[TesseraDataset] Excluded {fid}: No metadata.")
                continue

            if meta.start_year < TESSERA_YEARS[0]:
                dropped.append(fid)
                print(f"[TesseraDataset] Excluded {fid}: has annotation start year before {TESSERA_YEARS[0]}.")
                continue

            cutoff_year = meta.end_year - prediction_horizon
            tile_years = [y for y in TESSERA_YEARS if meta.start_year <= y <= meta.end_year]

            if not tile_years or cutoff_year not in tile_years:
                dropped.append(fid)
                print(f"[TesseraDataset] Excluded {fid}: Valid data window is empty or missing the {cutoff_year} cutoff year.")
            else:
                filtered.append(fid)

        if dropped:
            print(
                f"[TesseraDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s). "
                f"{len(filtered)} remain."
            )
        self.ids = filtered


        # ADDITIONAL TESSERA FILTERING: Exclude tiles with missing years/files
        self.emb_paths: dict[str, list[Path]] = {}
        self.mask_paths: dict[str, Path] = {}
        valid_ids: list[str] = []
        excluded_tessera: dict[str, list[int]] = {}

        for fid in self.ids:
            # Tessera needs to resolve multiple specific files based on tile_years
            missing, paths = [], []
            for year in tile_years:
                p = TESSERA_DIR / f"{fid}_tessera_{year}_snapped.tif"
                if p.exists():
                    paths.append(p)
                else:
                    missing.append(year)
            if missing:
                excluded_tessera[fid] = missing
                del self.tile_years[fid]
                continue

            self.emb_paths[fid] = paths
            self.mask_paths[fid] = find_file_by_prefix(MASK_DIR, fid)
            valid_ids.append(fid)


        if excluded_tessera:
            print(
                f"[TesseraDataset] Excluded {len(excluded_tessera)} tile(s) "
                f"missing required TESSERA files:"
            )
            for fid, my in excluded_tessera.items():
                print(f"  {fid}: missing years {my}")

        self.ids = valid_ids


    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        fid = self.ids[idx]
        meta = self.metadata[fid]
        tile_years = self.tile_years[fid] 

        # Load TESSERA embeddings (Naturally sliced by the tile_years list)
        yearly = []
        for path in self.emb_paths[fid]:
            with rasterio.open(path) as src:
                arr = src.read()  # (128, H, W)
            if arr.shape[0] != self._BANDS_PER_YEAR:
                raise ValueError(
                    f"{fid}: expected {self._BANDS_PER_YEAR} bands, "
                    f"got {arr.shape[0]} at {path}"
                )
            yearly.append(arr)

        emb = np.stack(yearly, axis=0)         # (T, 128, H, W)
        current_T = emb.shape[0]

        # Load segmentation mask
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1)           # (H, W)

        # To torch tensors
        emb = torch.from_numpy(emb).float()   
        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()

        # Annual temporal positions
        start_pos = tile_years[0] - TESSERA_YEARS[0] + 1
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # Apply transform 
        if self.transform is not None:
            emb, mask = self.transform(emb, mask)

        # Temporal masking
        cutoff_year = meta.end_year - self.prediction_horizon
        cutoff_idx = tile_years.index(cutoff_year)
        n_visible = cutoff_idx + 1

        emb[n_visible:] = 0.0
        positions[n_visible:] = 0

        # input_years (N) windowing: keep start_year(tile_years[0]) + latest (N-1) years before cutoff
        if self.input_years is not None:
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(tile_years[:cutoff_idx + 1]):
                if y != tile_years[0] and y <= window_limit:
                    emb[i] = 0.0
                    positions[i] = 0

        # Pad to max_timesteps for consistent batching
        if current_T < self.max_timesteps:
            pad_len = self.max_timesteps - current_T
            emb = F.pad(emb, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > self.max_timesteps:
            emb = emb[:self.max_timesteps]
            positions = positions[:self.max_timesteps]

        return emb, mask, positions