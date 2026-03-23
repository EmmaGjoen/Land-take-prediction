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
    TileMetadata,
    load_metadata,
)

BANDS_PER_TIMESTEP = 9   # spectral channels per acquisition
ACQUISITIONS_PER_YEAR = 2  # Sentinel: two per year (bi-annual)
BASE_YEAR = 2015           # position 0 is reserved → year 2016 = pos 1 (annual) or pos 2 (quarterly)


def find_file_by_prefix(base_dir: Path, fid: str) -> Path:
    candidates = sorted(base_dir.glob(f"{fid}*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No file starting with {fid!r} in {base_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple files starting with {fid!r} in {base_dir}: {candidates}")
    return candidates[0]


class SentinelDataset(Dataset):
    DATASET_NAME = "sentinel"

    def __init__(
        self,
        ids: list[str],
        transform,
        # metadata: dict[str, TileMetadata] | None = None,
        frequency: str | int | None = None,
        max_timesteps: int | None = None,
        prediction_horizon: int = 2,
        input_years: int | None = None,
    ):
        """
        Parameters
        ----------
        ids               : list of REFIDs to use
        transform         : normalization / augmentation callable
        metadata          : {refid: TileMetadata}; loaded from CSV if None
        frequency         : None → bi-annual (2 acquisitions/yr), "annual" → mean-pooled
        max_timesteps     : pad / truncate time axis to this length for batching
        prediction_horizon: K — model sees data up to (endYear - K)
        input_years       : N — use reference year + latest N-1 years before cutoff;
                            None means all available years
        """
        
        metadata = load_metadata()

        self.metadata   = metadata
        self.transform  = transform
        self.frequency  = frequency
        self.max_timesteps    = max_timesteps
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years

        # ── Filter 1: cutoff year (endYear - K) must exist in ALL_YEARS ──────
        filtered, dropped = [], []
        for fid in ids:
            meta = metadata.get(fid)
            if meta is None:
                dropped.append((fid, "no metadata"))
                continue
            cutoff = meta.end_year - prediction_horizon
            if cutoff not in ALL_YEARS:
                dropped.append((fid, f"cutoff {cutoff} outside ALL_YEARS"))
            else:
                filtered.append(fid)
        if dropped:
            print(
                f"[SentinelDataset] Dropped {len(dropped)} tile(s) at init "
                f"(K={prediction_horizon}): {[d[1] for d in dropped[:5]]} ..."
            )
        ids = filtered

        # ── Filter 2: enough years between startYear and cutoff for N ────────
        if input_years is not None:
            filtered, dropped = [], []
            for fid in ids:
                meta = metadata[fid]
                cutoff     = meta.end_year - prediction_horizon
                tile_start = max(meta.start_year, ALL_YEARS[0])
                available  = ALL_YEARS.index(cutoff) - ALL_YEARS.index(tile_start) + 1
                if available < input_years:
                    dropped.append(fid)
                else:
                    filtered.append(fid)
            if dropped:
                print(
                    f"[SentinelDataset] N={input_years}: excluded {len(dropped)} tile(s) "
                    f"with fewer than {input_years} available years. {len(filtered)} remain."
                )
            ids = filtered

        self.ids = ids

        # ── Pre-resolve paths ─────────────────────────────────────────────────
        self.img_paths:  dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}
        for fid in self.ids:
            self.img_paths[fid]  = find_file_by_prefix(SENTINEL_DIR, fid)
            self.mask_paths[fid] = find_file_by_prefix(MASK_DIR, fid)

    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        fid  = self.ids[idx]
        meta = self.metadata[fid]

        steps_per_year = 1 if self.frequency == "annual" else ACQUISITIONS_PER_YEAR

        # ── Load raw raster ───────────────────────────────────────────────────
        with rasterio.open(self.img_paths[fid]) as src:
            img = src.read()   # (num_bands, H, W)
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1)  # (H, W)

        if img.shape[-2:] != mask.shape:
            print(f"[WARN] spatial mismatch for {fid}: "
                  f"Sentinel {img.shape[-2:]} vs mask {mask.shape}")

        # ── Reshape to (num_years, acq_per_year, C, H, W) ────────────────────
        C = BANDS_PER_TIMESTEP
        num_bands, H, W = img.shape
        total_bands_per_year = C * ACQUISITIONS_PER_YEAR
        if num_bands % total_bands_per_year != 0:
            raise ValueError(
                f"{fid}: band count {num_bands} not divisible by {total_bands_per_year}"
            )
        num_years_in_file = num_bands // total_bands_per_year
        img = img.reshape(num_years_in_file, ACQUISITIONS_PER_YEAR, C, H, W)

        # ── Derive calendar years present in this file ────────────────────────
        # The file always starts at meta.start_year — confirmed by GEE export.
        file_years = list(range(meta.start_year, meta.start_year + num_years_in_file))

        # ── Clip to [startYear, endYear] from metadata ────────────────────────
        valid_mask_years = [
            y for y in file_years
            if meta.start_year <= y <= meta.end_year
        ]
        # Safety: only keep years that are in ALL_YEARS
        valid_mask_years = [y for y in valid_mask_years if y in ALL_YEARS]

        start_idx = file_years.index(valid_mask_years[0])
        end_idx   = file_years.index(valid_mask_years[-1])
        img = img[start_idx : end_idx + 1]           # (T_valid, 2, C, H, W)
        current_years = valid_mask_years

        # ── Apply frequency pooling ───────────────────────────────────────────
        if self.frequency == "annual":
            img = img.mean(axis=1)                   # (T, C, H, W)
        else:
            img = img.reshape(img.shape[0] * ACQUISITIONS_PER_YEAR, C, H, W)  # (T*2, C, H, W)

        current_T = img.shape[0]

        # ── Build positional indices ──────────────────────────────────────────
        # Position 0 is reserved (U-TAE pad detection).
        # annual:     year Y → pos (Y - BASE_YEAR)         e.g. 2016 → 1
        # bi-annual:  year Y, acq q → pos (Y-BASE_YEAR)*2 + q   e.g. 2016Q1→2, 2016Q2→3
        if self.frequency == "annual":
            positions = torch.tensor(
                [y - BASE_YEAR for y in current_years], dtype=torch.long
            )
        else:
            positions = torch.tensor(
                [(y - BASE_YEAR) * 2 + q
                 for y in current_years
                 for q in range(ACQUISITIONS_PER_YEAR)],
                dtype=torch.long,
            )

        img  = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # ── Zero-mask timesteps after cutoff (endYear - K) ───────────────────
        cutoff_year = meta.end_year - self.prediction_horizon
        assert cutoff_year in ALL_YEARS, \
            f"cutoff_year {cutoff_year} not in ALL_YEARS for {fid}"

        if cutoff_year in current_years:
            cutoff_idx_in_current = current_years.index(cutoff_year)
            n_valid = min((cutoff_idx_in_current + 1) * steps_per_year, current_T)
        else:
            # cutoff is beyond the file's range — keep everything
            n_valid = current_T

        img[n_valid:]       = 0.0
        positions[n_valid:] = 0

        # ── Zero-mask the gap when input_years=N ─────────────────────────────
        if self.input_years is not None and cutoff_year in current_years:
            cutoff_idx  = current_years.index(cutoff_year)
            anchor_year = max(meta.start_year, ALL_YEARS[0])
            anchor_idx  = current_years.index(anchor_year) if anchor_year in current_years else 0

            anchor_ts  = anchor_idx * steps_per_year
            # zero everything before the anchor
            if anchor_ts > 0:
                img[:anchor_ts]       = 0.0
                positions[:anchor_ts] = 0

            # zero the gap between anchor+1 and the last N-1 years before cutoff
            gap_start = (anchor_idx + 1) * steps_per_year
            gap_end   = (cutoff_idx - (self.input_years - 2)) * steps_per_year
            if gap_start < gap_end:
                img[gap_start:gap_end]       = 0.0
                positions[gap_start:gap_end] = 0

        # ── Pad / truncate to max_timesteps for batching ──────────────────────
        if self.max_timesteps is not None:
            if current_T < self.max_timesteps:
                pad_len  = self.max_timesteps - current_T
                img      = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
                positions = F.pad(positions, (0, pad_len))
            elif current_T > self.max_timesteps:
                img       = img[:self.max_timesteps]
                positions = positions[:self.max_timesteps]

        return img, mask, positions