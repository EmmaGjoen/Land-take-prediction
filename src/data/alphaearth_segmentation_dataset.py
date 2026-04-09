"""Standalone dataset for AlphaEarth embeddings with land-take segmentation masks.

Provides (img, mask, positions) triples that mirror the SentinelDataset interface
so that U-TAE can be trained on AlphaEarth embeddings alone, without Sentinel data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import ALPHAEARTH_YEARS, ALPHAEARTH_DIR, MASK_DIR, load_metadata

# AlphaEarth embeddings span exactly these years (fixed by the GEE export).
_BANDS_PER_YEAR = 64
_EXPECTED_BANDS = len(ALPHAEARTH_YEARS) * _BANDS_PER_YEAR  # 448

def find_file_by_prefix(base_dir: Path, fid: str) -> Path:
    """
    Find the unique .tif file in base_dir whose name starts with fid.
    """
    candidates = sorted(base_dir.glob(f"{fid}*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No file starting with {fid!r} in {base_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple files starting with {fid!r} in {base_dir}: {candidates}")
    return candidates[0]


class AlphaEarthSegmentationDataset(Dataset):
    """Load AlphaEarth annual embeddings paired with land-take segmentation masks.

    Each tile has a single 448-band GeoTIFF (7 years × 64 dims) located at::

        ALPHAEARTH_DIR/{refid}_VEY_Mosaic.tif

    The array is reshaped to ``(7, 64, H, W)`` and then clipped to the years
    that fall within ``[meta.start_year, meta.end_year]``.  Tiles whose VHR
    window starts before 2018 (AlphaEarth's first year) will have their valid
    window floored at 2018; those with ``endYear > 2024`` are clipped to 2024
    (with K≥1 this is always sufficient since the cutoff ≤ endYear − 1 ≤ 2024).

    The result is padded with zeros to ``len(ALPHAEARTH_YEARS)`` timesteps so
    that all samples in a batch have the same temporal length.

    **Temporal positions** follow the same annual convention as SentinelDataset:
    ``position = year − ALL_YEARS[0] + 1``.  Position 0 is reserved for
    U-TAE's pad-value masking.

    **Temporal masking** (applied after the transform):

    * Timesteps after ``endYear − K`` are zeroed (prediction horizon).
    * When ``input_years=N``: ``startYear`` is always visible; years between
      ``startYear+1`` and ``cutoff − (N−2)`` are zeroed.

    **Tile filtering** at construction time (logged to stdout):

    * Tiles with no metadata or whose cutoff ``(endYear − K)`` is out of range.
    * Tiles missing an AlphaEarth file in ``ALPHAEARTH_DIR``.
    * Tiles missing a mask file in ``MASK_DIR``.

    Args:
        ids: Reference IDs (tile name prefixes).
        transform: ``ComposeTS``-style callable applied jointly to
            ``(img, mask)``, e.g. crop → augmentation → normalisation.
        prediction_horizon: ``K`` — the model sees data strictly before
            ``endYear − K``.
        input_years: ``N`` — keep ``startYear`` plus the latest ``N−1`` years
            before the cutoff; zero the intervening gap.  ``None`` keeps all.
    """

    DATASET_NAME = "alphaearth"
    YEARS = ALPHAEARTH_YEARS

    def __init__(
        self,
        ids: list[str],
        transform,
        prediction_horizon: int = 2,
        input_years: Optional[int] = None,
    ):
        self.transform = transform
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years
        self.max_timesteps = len(ALPHAEARTH_YEARS)
        self.metadata = load_metadata()
        self.tile_years: dict[str, list[int]] = {}

        # ------------------------------------------------------------------ #
        # Drop tiles with no metadata or whose cutoff or start year is out of range #
        # ------------------------------------------------------------------ #
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)
            
            if meta is None:
                dropped.append(fid)
                print(f"[AlphaEarth] No metadata for {fid}, skipping.")
                continue
            
            if meta.start_year < ALPHAEARTH_YEARS[0]:
                dropped.append(fid)
                print(f"[AlphaEarth] file {fid} has annotation start year before {ALPHAEARTH_YEARS[0]}, skipping.")
                continue

            cutoff_year = meta.end_year - prediction_horizon
            tile_years = [
                y for y in ALPHAEARTH_YEARS
                if meta.start_year <= y <= meta.end_year
            ]

            if not tile_years or cutoff_year not in tile_years:
                print(f"[AlphaEarth] Excluded {fid}: Valid data window is empty or missing the {cutoff_year} cutoff year.")
                dropped.append(fid)
            else:
                filtered.append(fid)
                self.tile_years[fid] = tile_years
        
        if dropped:
            print(
                f"[AlphaEarthDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s)"
                f"{len(filtered)} remain."
            )
        self.ids = filtered

        self.emb_paths:  dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}
        
        for fid in self.ids:
            self.emb_paths[fid]  = find_file_by_prefix(ALPHAEARTH_DIR, fid)
            self.mask_paths[fid] = find_file_by_prefix(MASK_DIR, fid)


    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        fid = self.ids[idx]
        meta = self.metadata[fid]
        tile_years = self.tile_years[fid]

        # ---- Load AlphaEarth embeddings ----------------------------------- #
        with rasterio.open(self.emb_paths[fid]) as src:
            arr = src.read()  # (448, H, W)

        num_bands, H, W = arr.shape
        if num_bands != _EXPECTED_BANDS:
            raise ValueError(
                f"{fid}: expected {_EXPECTED_BANDS} bands, got {num_bands}"
            )

        # Reshape to (7, 64, H, W) and slice to the tile's valid year window
        full_img = arr.reshape(len(ALPHAEARTH_YEARS), _BANDS_PER_YEAR, H, W)
        start_idx = ALPHAEARTH_YEARS.index(tile_years[0])
        end_idx   = ALPHAEARTH_YEARS.index(tile_years[-1])
        img = full_img[start_idx : end_idx + 1]    # (T, 64, H, W)
        img = torch.from_numpy(img).float()
        current_T = img.shape[0]

        # ---- Load segmentation mask --------------------------------------- #
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask_arr = src_m.read(1)               # (H, W)
        mask = torch.from_numpy(mask_arr).long()
        mask = (mask > 0).long()

        # ---- Annual temporal positions ------------------------------------ #
        # position = year − ALL_YEARS[0] + 1; position 0 reserved for padding
        start_pos = tile_years[0] - ALL_YEARS[0] + 1
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # ---- Apply transform (crop / augmentation / normalisation) -------- #
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # ---- Temporal masking --------------------------------------------- #
        cutoff_year = meta.end_year - self.prediction_horizon
        cutoff_idx  = tile_years.index(cutoff_year)
        n_visible   = cutoff_idx + 1

        img[n_visible:] = 0.0
        positions[n_visible:] = 0

        if self.input_years is not None:
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(tile_years[:cutoff_idx + 1]):
                if y != meta.start_year and y <= window_limit:
                    img[i] = 0.0
                    positions[i] = 0

        # ---- Pad to max_timesteps for consistent batching ----------------- #
        if current_T < self.max_timesteps:
            pad_len = self.max_timesteps - current_T
            img       = F.pad(img,       (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > self.max_timesteps:
            img       = img[:self.max_timesteps]
            positions = positions[:self.max_timesteps]

        return img, mask, positions
