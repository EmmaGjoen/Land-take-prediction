"""Standalone dataset for GeoTessera embeddings with land-take segmentation masks.

Provides (img, mask, positions) triples that mirror the SentinelDataset interface
so that U-TAE can be trained on TESSERA embeddings alone, without any Sentinel data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import ALL_YEARS, MASK_DIR, TESSERA_DIR, load_metadata


class TesseraSegmentationDataset(Dataset):
    """Load GeoTessera yearly embeddings paired with land-take segmentation masks.

    Each tile has one 128-band GeoTIFF per year located at::

        TESSERA_DIR/{refid}_tessera_{year}_snapped.tif

    Only years that fall within ``[meta.start_year, meta.end_year]`` are loaded
    per tile; other requested years are ignored.  This mirrors how SentinelDataset
    clips the Sentinel time series to each tile's VHR window.

    The loaded years are stacked chronologically to form a ``(T, 128, H, W)``
    tensor, padded with zeros to ``len(years)`` so that all samples in a batch
    have the same temporal length.

    **Temporal positions** follow the same convention as SentinelDataset (annual
    mode): ``position = year - ALL_YEARS[0] + 1``, giving the first year in
    ``ALL_YEARS`` → 1.  Position 0 is reserved for U-TAE's pad-value masking;
    zeroed/padded timesteps always receive position 0.

    **Temporal masking** (applied after the transform):

    * Timesteps after ``endYear − K`` are zeroed (prediction horizon).
    * When ``input_years=N``: ``startYear`` is always visible; years between
      ``startYear+1`` and ``cutoff − (N−2)`` are zeroed.

    **Tile filtering** at construction time (logged to stdout):

    * Tiles with no metadata or whose cutoff ``(endYear − K)`` is out of range.
    * Tiles missing any TESSERA file for years within ``[startYear, endYear]``.
    * Tiles missing a mask file in ``MASK_DIR``.

    Args:
        ids: Reference IDs (tile name prefixes, e.g. ``"R101C117"``).
        transform: ``ComposeTS``-style callable applied jointly to
            ``(img, mask)``, e.g. crop → augmentation → normalisation.
        years: Ordered list of years to consider.  Only the subset that falls
            within each tile's ``[startYear, endYear]`` window is loaded.
        prediction_horizon: ``K`` — the model sees data strictly before
            ``endYear − K``.
        input_years: ``N`` — keep ``startYear`` plus the latest ``N−1`` years
            before the cutoff; zero the intervening gap.  ``None`` keeps all.
    """

    DATASET_NAME = "tessera"
    BANDS_PER_YEAR = 128

    def __init__(
        self,
        ids: list[str],
        transform,
        years: Optional[list[int]] = None,
        prediction_horizon: int = 2,
        input_years: Optional[int] = None,
    ):
        self.transform = transform
        self.years = list(years) if years is not None else list(ALL_YEARS)
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years
        self.max_timesteps = len(self.years)  # pad all samples to this length

        self.metadata = load_metadata()

        # ------------------------------------------------------------------ #
        # Step 1: drop tiles with no metadata or whose cutoff is out of range #
        # ------------------------------------------------------------------ #
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)
            if meta is None:
                dropped.append(fid)
                print(f"[TesseraSegmentationDataset] No metadata for {fid}, skipping.")
                continue
            cutoff_year = meta.end_year - prediction_horizon
            # valid_years for this tile: years within [start, end] and in self.years
            tile_years = [y for y in self.years if meta.start_year <= y <= meta.end_year]
            if not tile_years or cutoff_year not in tile_years:
                dropped.append(fid)
            else:
                filtered.append(fid)
        if dropped:
            print(
                f"[TesseraSegmentationDataset] K={prediction_horizon}: excluded "
                f"{len(dropped)} tile(s) whose cutoff year is outside available data. "
                f"{len(filtered)} remain."
            )
        ids = filtered

        # ------------------------------------------------------------------ #
        # Step 2: resolve file paths; exclude tiles with missing files        #
        # ------------------------------------------------------------------ #
        self.emb_paths: dict[str, list[Path]] = {}
        self.mask_paths: dict[str, Path] = {}
        self.tile_years: dict[str, list[int]] = {}
        valid_ids: list[str] = []
        excluded_tessera: dict[str, list[int]] = {}

        for fid in ids:
            meta = self.metadata[fid]
            tile_years = [y for y in self.years if meta.start_year <= y <= meta.end_year]

            missing, paths = [], []
            for year in tile_years:
                p = TESSERA_DIR / f"{fid}_tessera_{year}_snapped.tif"
                if p.exists():
                    paths.append(p)
                else:
                    missing.append(year)
            if missing:
                excluded_tessera[fid] = missing
                continue

            mask_candidates = sorted(MASK_DIR.glob(f"{fid}*.tif"))
            if not mask_candidates:
                print(f"[TesseraSegmentationDataset] Excluded {fid}: no mask file found.")
                continue

            self.emb_paths[fid] = paths
            self.mask_paths[fid] = mask_candidates[0]
            self.tile_years[fid] = tile_years
            valid_ids.append(fid)

        if excluded_tessera:
            print(
                f"[TesseraSegmentationDataset] Excluded {len(excluded_tessera)} tile(s) "
                f"missing required TESSERA files."
            )
            for fid, my in excluded_tessera.items():
                print(f"  {fid}: missing years {my}")

        self.ids = valid_ids

    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        fid = self.ids[idx]
        meta = self.metadata[fid]
        tile_years = self.tile_years[fid]   # years actually loaded for this tile

        # ---- Load TESSERA embeddings -------------------------------------- #
        yearly = []
        for path in self.emb_paths[fid]:
            with rasterio.open(path) as src:
                arr = src.read()  # (128, H, W)
            if arr.shape[0] != self.BANDS_PER_YEAR:
                raise ValueError(
                    f"{fid}: expected {self.BANDS_PER_YEAR} bands, "
                    f"got {arr.shape[0]} at {path}"
                )
            yearly.append(arr)

        img = np.stack(yearly, axis=0)         # (T, 128, H, W)
        img = torch.from_numpy(img).float()
        current_T = img.shape[0]

        # ---- Load segmentation mask --------------------------------------- #
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask_arr = src_m.read(1)           # (H, W)
        mask = torch.from_numpy(mask_arr).long()
        mask = (mask > 0).long()

        # ---- Annual temporal positions ------------------------------------ #
        # Encoding matches SentinelDataset annual mode:
        #   position = (year - ALL_YEARS[0]) + 1  →  first year in ALL_YEARS gets 1
        # Position 0 is reserved for pad-value masking by U-TAE.
        start_pos = tile_years[0] - ALL_YEARS[0] + 1
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # ---- Apply transform (crop / augmentation / normalisation) -------- #
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # ---- Temporal masking --------------------------------------------- #
        cutoff_year = meta.end_year - self.prediction_horizon
        cutoff_idx = tile_years.index(cutoff_year)   # guaranteed present (filtered in __init__)
        n_visible = cutoff_idx + 1                    # steps_per_year = 1 (annual)

        # Zero all timesteps after the prediction-horizon cutoff
        img[n_visible:] = 0.0
        positions[n_visible:] = 0

        # input_years (N) windowing: keep startYear + latest (N-1) years before cutoff
        if self.input_years is not None:
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(tile_years[:cutoff_idx + 1]):
                if y != meta.start_year and y <= window_limit:
                    img[i] = 0.0
                    positions[i] = 0

        # ---- Pad to max_timesteps for consistent batching ----------------- #
        if current_T < self.max_timesteps:
            pad_len = self.max_timesteps - current_T
            img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > self.max_timesteps:
            img = img[:self.max_timesteps]
            positions = positions[:self.max_timesteps]

        return img, mask, positions
