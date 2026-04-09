from pathlib import Path

import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import ALPHAEARTH_YEARS, ALPHAEARTH_DIR, MASK_DIR, load_metadata

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


class AlphaEarthDataset(Dataset):
    """Loads AlphaEarth annual embeddings paired with land-take segmentation masks and postitions encoding for the embedding timeseries.

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
        * Tiles missing an AlphaEarth file in ALPHAEARTH_DIR.
        * Tiles with start year before the available ALPHAEARTH_YEARS
    """
    DATASET_NAME = "alphaearth"

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

        C = _BANDS_PER_YEAR
        num_bands, H, W = emb.shape
        num_years = num_bands // C

        if num_bands != _EXPECTED_BANDS:
            raise ValueError(f"{fid}: expected {_EXPECTED_BANDS} bands, got {num_bands}")

        emb = emb.reshape(num_years, C, H, W)

        # Slice embedding to match the valid tile_years
        start_clip = tile_years[0] - ALPHAEARTH_YEARS[0]
        end_clip   = tile_years[-1] - ALPHAEARTH_YEARS[0]

        emb = emb[start_clip : end_clip + 1]   # shape: (num_valid_years, C, H, W)
        current_T = emb.shape[0]

        # Position encoding
        # 1-indexed absolute temporal position. 0 is reserved for padding.
        start_pos = start_clip + 1
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # To torch tensors
        emb  = torch.from_numpy(emb).float()
        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()
    
        # Apply transforms before zero padding
        if self.transform is not None:
            emb, mask = self.transform(emb, mask)

        # Temporal masking 
        cutoff_year = meta.end_year - self.prediction_horizon
        cutoff_idx  = tile_years.index(cutoff_year)
        n_visible   = cutoff_idx + 1

        emb[n_visible:] = 0.0
        positions[n_visible:] = 0

        if self.input_years is not None:
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(tile_years[:cutoff_idx + 1]):
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
