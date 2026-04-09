from pathlib import Path
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import (
    ALPHAEARTH_DIR,
    MASK_DIR,
    ALPHAEARTH_YEARS,
    load_metadata,
)


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
    DATASET_NAME = "alphaearth"

    def __init__(
        self,
        ids,
        transform,
        prediction_horizon: int = 2,
        input_years: int | None = None,
    ):
        """
        - ids: list of REFIDs
        - transform: transforms to apply (flips, rotations)
        - prediction_horizon (K): Number of years before final year in timeseries to cut off.
            With K=2, the model only sees data up to final year-2, forcing it to
            predict land take K years in advance.
        - input_years (N): reference year + latest N-1 years before cutoff;
            None means all years up to cutoff
        """
        self.transform          = transform
        self.prediction_horizon = prediction_horizon
        self.input_years        = input_years
        self.metadata           = load_metadata()

        self.years_range = ALPHAEARTH_YEARS
        self.max_timesteps = len(ALPHAEARTH_YEARS)

        # Drop tiles whose cutoff year falls outside the available AlphaEarth record
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)
            if meta is None:
                dropped.append(fid)
                print(f"No metadata found for {fid}. Check your metadata CSV.")
                continue
            cutoff_year = meta.end_year - prediction_horizon
            if cutoff_year in self.years_range:
                filtered.append(fid)
            else:
                dropped.append(fid)

        if dropped:
            print(
                f"[AlphaEarthDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s), whose cutoff year falls outside available data. "
                f"{len(filtered)} tiles remain."
            )

        self.ids = filtered

        # Pre-resolve paths
        self.emb_paths:  dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}
        for fid in self.ids:
            self.emb_paths[fid]  = find_file_by_prefix(ALPHAEARTH_DIR, fid)
            self.mask_paths[fid] = find_file_by_prefix(MASK_DIR, fid)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid  = self.ids[idx]
        meta = self.metadata[fid]

        # Load embedding and mask
        with rasterio.open(self.emb_paths[fid]) as src:
            emb = src.read() # (num_bands, H, W)
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1) # (H, W)

        C = 64
        num_bands, H, W = emb.shape
        num_years = num_bands // C # e.g. 448 // 64 = 7

        if num_bands % C != 0:
            raise ValueError(
                f"Expected bands divisible by {C}, got {num_bands} for {fid}"
            )

        # Reshape to (num_years, C, H, W)
        emb = emb.reshape(num_years, C, H, W)

        file_years  = list(range(self.years_range[0], self.years_range[0] + num_years))
        valid_years = [y for y in file_years if meta.start_year <= y <= meta.end_year]
        start_clip  = valid_years[0] - self.years_range[0]
        end_clip    = valid_years[-1] - self.years_range[0]

        # Slice img along the year axis
        emb = emb[start_clip : end_clip + 1]   # shape: (num_valid_years, C, H, W)
        file_years = valid_years               # update to reflect clipped range

        current_T = emb.shape[0]
    
        # Position encoding: encode each timestep's absolute temporal position.
        # Positions are 1-indexed so that 0 is always available to mark padding.
        # U-TAE masks out any timestep with position=0 from attention.
        start_pos = file_years[0] - self.years_range[0] + 1
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # to torch tensors
        emb  = torch.from_numpy(emb).float()
        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()

        # Apply transforms before any zero-padding
        if self.transform is not None:
            emb, mask = self.transform(emb, mask)

        # Zero out timesteps after cutoff year (end_year - K)
        cutoff_year = meta.end_year - self.prediction_horizon

        if cutoff_year not in file_years:
            n_visible_timesteps = current_T
            print(f"[WARN] {fid}: cutoff_year {cutoff_year} not in file_years {file_years}")
            cutoff_idx = len(file_years) - 1
        else:
            cutoff_idx = file_years.index(cutoff_year)
            n_visible_timesteps = min((cutoff_idx + 1), current_T)

        emb[n_visible_timesteps:]       = 0.0
        positions[n_visible_timesteps:] = 0

        # Zero out gap years when input_years=N is set
        if self.input_years is not None:
            # Keep start_year (reference) + the (N-1) latest years ending at cutoff
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(file_years[:cutoff_idx + 1]):
                # If the year is NOT the start_year AND falls before our N-1 window, mask it
                if y != meta.start_year and y <= window_limit:
                    emb[i:i+1]       = 0.0
                    positions[i:i+1] = 0

        # Pad or truncate to MAX_TIMESTEPS for batching
        if current_T < MAX_TIMESTEPS:
            pad_len   = MAX_TIMESTEPS - current_T
            emb       = F.pad(emb, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > MAX_TIMESTEPS:
            emb       = emb[:MAX_TIMESTEPS]
            positions = positions[:MAX_TIMESTEPS]

        return emb, mask, positions
    