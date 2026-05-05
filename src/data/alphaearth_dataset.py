import bisect
from pathlib import Path

import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.data.file_helpers import find_file_by_prefix
from src.config import ALL_YEARS, ALPHAEARTH_YEARS, ALPHAEARTH_DIR, MASK_DIR, load_metadata

_BANDS_PER_YEAR = 64
_EXPECTED_BANDS = len(ALPHAEARTH_YEARS) * _BANDS_PER_YEAR  # 8 years x 64 = 512

class AlphaEarthDataset(Dataset):
    """AlphaEarth annual embeddings paired with land-take segmentation masks.

    Args:
        ids: list of REFIDs.
        transform: spatial transforms (flips, rotations).
        prediction_horizon (K): zero out the last K years so the model
            predicts K years ahead.
        input_years (N): keep start_year + latest N-1 years before cutoff.
            None = all years up to cutoff.

    Tiles are filtered at init (logged) if they lack metadata, have
    start_year before ALPHAEARTH_YEARS, or miss an embedding/mask file.
    """
    DATASET_NAME = "alphaearth"

    @staticmethod
    def get_ref_ids(alphaearth_dir: Path) -> list[str]:
        """Return sorted unique REFIDs found in alphaearth_dir."""
        files = sorted(alphaearth_dir.glob("*_VEY_Mosaic.tif"))
        return sorted({f.stem.removesuffix("_VEY_Mosaic") for f in files})

    def __init__(
        self,
        ids: list[str],
        transform,
        prediction_horizon: int = 2,
        input_years: int | None = None
    ):
        self.transform = transform
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years
        self.max_timesteps = len(ALPHAEARTH_YEARS)
        self.metadata = load_metadata()
        self.tile_years: dict[str, list[int]] = {}

        # Drop tiles with no metadata or whose cutoff or start year is out of range
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)

            if meta is None:
                dropped.append(fid)
                print(f"[AlphaEarth] Excluded {fid}: No metadata.")
                continue

            if meta.start_year < ALPHAEARTH_YEARS[0]:
                dropped.append(fid)
                print(f"[AlphaEarth] Excluded {fid}: has annotation start year before {ALPHAEARTH_YEARS[0]}.")
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
                f"[AlphaEarthDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s). "
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

        with rasterio.open(self.emb_paths[fid]) as src:
            emb = src.read()  # (num_bands, H, W)
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1)  # (H, W)

        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        C = _BANDS_PER_YEAR
        num_bands, H, W = emb.shape
        num_years = num_bands // C

        if num_bands != _EXPECTED_BANDS:
            raise ValueError(f"{fid}: expected {_EXPECTED_BANDS} bands, got {num_bands}")

        emb = emb.reshape(num_years, C, H, W)

        # Slice to the valid tile_years range
        start_clip = tile_years[0] - ALPHAEARTH_YEARS[0]
        end_clip   = tile_years[-1] - ALPHAEARTH_YEARS[0]

        emb = emb[start_clip : end_clip + 1]   # shape: (num_valid_years, C, H, W)
        current_T = emb.shape[0]

        # To torch tensors
        emb  = torch.from_numpy(emb).float()
        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()

        # Position encoding: 1-indexed, shared origin ALL_YEARS[0]=2016.
        # Position 0 = padding (ignored by U-TAE attention).
        start_pos = tile_years[0] - ALL_YEARS[0] + 1
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # Apply transforms before zero padding
        if self.transform is not None:
            emb, mask = self.transform(emb, mask)

        # Temporal masking
        cutoff_year = meta.end_year - self.prediction_horizon
        n_visible = bisect.bisect_right(tile_years, cutoff_year)

        emb[n_visible:] = 0.0
        positions[n_visible:] = 0

        if self.input_years is not None:
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(tile_years[:n_visible]):
                if y != tile_years[0] and y <= window_limit:
                    emb[i] = 0.0
                    positions[i] = 0

        # Pad to max_timesteps for consistent tensor size
        if current_T < self.max_timesteps:
            pad_len = self.max_timesteps - current_T
            emb = F.pad(emb, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > self.max_timesteps:
            emb = emb[:self.max_timesteps]
            positions = positions[:self.max_timesteps]

        return emb, mask, positions
