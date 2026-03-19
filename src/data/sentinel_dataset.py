from pathlib import Path
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.config import (
    SENTINEL_DIR,
    MASK_DIR,
    YEARS
)


def find_file_by_prefix(base_dir: Path, fid: str) -> Path:
    """
    Find a file in base_dir whose name starts with fid and ends with .tif or .tiff.

    Example:
        fid = "a22-0323..._37-98..."
        file = "a22-0323..._37-98..._RGBNIRRSWIRQ_Mosaic.tif"

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


class SentinelDataset(Dataset):
    DATASET_NAME = "sentinel"
    """
    Loads time series data and reshapes it into (T, C, H, W)
    so it can be fed directly to temporal models (like the FCEF baseline).

    Assumptions:
      - `ids` are REFIDs that match the *prefix* of the filenames in
        SENTINEL_DIR / MASK_DIR.
    """

    def __init__(
        self,
        ids,
        transform,
        slice_mode: str = None,
        frequency: str | int | None = None,
        end_years: dict[str, int] | None = None,
        start_years: dict[str, int] | None = None,
        max_timesteps: int | None = None,
        prediction_horizon: int = 2,
        input_years: int | None = None,
    ):
        """
        ids: list of REFIDs (filename stems without the long suffix)
        slice_mode: None or "first_half", or a specific year (as int): 2019, 2020, 2021, 2022, 2023
        transform: Transform to apply (flips, rotations, normalization)
        frequency: two quarters a year as default, optional: annual
        end_years: mapping of refid -> endYear from annotations metadata.
            Sentinel timesteps after endYear are zeroed after normalization,
            so U-TAE's pad_value=0.0 detection masks them correctly.
        start_years: mapping of refid -> startYear from annotations metadata.
            Used as the per-tile anchor year for input_years selection.
            startYear values before YEARS[0] (2018) are clamped to YEARS[0]
            since Sentinel data does not extend before 2018.
        max_timesteps: If set, pad or truncate the time axis to this length.
            Leave as None when all tiles share the same T.
        prediction_horizon: Number of years before endYear to cut off (K).
            With K=2, the model only sees data up to endYear-2, forcing it to
            predict land take K years in advance (per tile, using each tile's endYear).
        input_years: If set, selects N years per tile: the per-tile anchor year
            (startYear, clamped to 2018) plus the latest N-1 years up to the
            cutoff (endYear - K). Timesteps before the anchor and in the gap
            between anchor and latest window are zeroed. With N=1 only the
            anchor year is visible; with N >= total available years all years
            from anchor to cutoff are visible.
        """
        self.slice_mode = slice_mode
        self.transform = transform
        self.frequency = frequency
        self.end_years = end_years
        self.start_years = start_years
        self.max_timesteps = max_timesteps
        self.prediction_horizon = prediction_horizon
        self.input_years = input_years

        # Drop tiles whose cutoff year falls outside the Sentinel record (YEARS).
        # This happens when prediction_horizon is large relative to a tile's endYear,
        # e.g. endYear=2022 with K=5 → cutoff=2017 < YEARS[0].
        if end_years is not None and prediction_horizon > 0:
            filtered = []
            dropped = []
            for fid in ids:
                ey = end_years.get(fid)
                if ey is not None and (ey - prediction_horizon) not in YEARS:
                    dropped.append(fid)
                else:
                    filtered.append(fid)
            if dropped:
                print(
                    f"[SentinelDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s) "
                    f"whose cutoff year falls outside YEARS (e.g. endYear too small). "
                    f"{len(filtered)} tiles remain."
                )
            self.ids = filtered
        else:
            self.ids = list(ids)

        # Drop tiles that don't have enough years for the requested sequence length N.
        if input_years is not None and end_years is not None:
            filtered = []
            dropped = []
            for fid in self.ids:
                ey = end_years.get(fid)
                if ey is None:
                    filtered.append(fid)
                    continue
                cutoff_year = ey - prediction_horizon
                if cutoff_year not in YEARS:
                    filtered.append(fid)
                    continue
                tile_start = start_years.get(fid, YEARS[0]) if start_years else YEARS[0]
                anchor_year = max(tile_start, YEARS[0])
                cutoff_year_idx = YEARS.index(cutoff_year)
                anchor_year_idx = YEARS.index(anchor_year)
                available_years = cutoff_year_idx - anchor_year_idx + 1
                if available_years < input_years:
                    dropped.append(fid)
                else:
                    filtered.append(fid)
            if dropped:
                print(
                    f"[SentinelDataset] N={input_years}: excluded {len(dropped)} tile(s) "
                    f"with fewer than {input_years} available years between reference and cutoff. "
                    f"{len(filtered)} tiles remain."
                )
            self.ids = filtered

        # Pre-resolve image and mask paths once for stability and speed
        self.img_paths: dict[str, Path] = {}
        self.mask_paths: dict[str, Path] = {}

        for fid in self.ids:
            img_path = find_file_by_prefix(SENTINEL_DIR, fid)
            mask_path = find_file_by_prefix(MASK_DIR, fid)

            self.img_paths[fid] = img_path
            self.mask_paths[fid] = mask_path

    def __len__(self):
        # One sample per chip
        return len(self.ids)

    def __getitem__(self, idx):
        # Direct mapping: each idx corresponds to one 64×64 chip
        fid = self.ids[idx]

        img_path = self.img_paths[fid]
        mask_path = self.mask_paths[fid]

        # read arrays
        with rasterio.open(img_path) as src:
            img = src.read()  # (bands, H, W)
        with rasterio.open(mask_path) as src_m:
            mask = src_m.read(1)  # (H, W)

        # Warn if Sentinel and mask have different spatial dimensions (data quality check)
        if img.shape[-2:] != mask.shape:
            print(
                f"[WARN] spatial mismatch for {fid}: "
                f"Sentinel {img.shape[-2:]} vs mask {mask.shape}"
            )

        # reshape to (T, C, H, W)
        # (old data:) Expected layout: 126 = 7 years * 2 quarters * 9 bands
        C = 9
        num_bands, H, W = img.shape
        T = num_bands // C
        num_quarters = 2
        num_years = T // num_quarters
        
        if num_bands != num_years *num_quarters * C:
            raise ValueError(
                f"Expected bands to be divisible by {num_quarters * C}, got {num_bands} for {fid} at {img_path}"
            )
        img = img.reshape(num_years, num_quarters, C, H, W)

        if self.slice_mode in YEARS:
            end_idx = YEARS.index(int(self.slice_mode))
            img = img[0: end_idx +1]

        if self.frequency == "annual":
            if T % num_quarters != 0:
                raise ValueError(
                    f"Timesteps ({T}) not divisible by steps_per_year ({num_quarters}) for {fid} at {img_path}"
                )
            img = img.mean(axis=1)  # aggregate quarters -> (num_years, C, H, W)
        else:
            new_T = img.shape[0] * img.shape[1]
            img = img.reshape(new_T, C, H, W)

        # optionally take first half of the time series
        if self.slice_mode == "first_half":
            img = img[: img.shape[0] // 2]

        # Compute positions encoding absolute temporal location.
        # All Sentinel tiles start at 2018; we encode relative to base year 2016
        # so positions are non-zero (avoids collision with pad position=0).
        # Annual:    2018=2, 2019=3, ..., 2024=8
        # Quarterly: 2018Q1=4, 2018Q2=5, 2019Q1=6, ..., 2024Q2=17
        current_T = img.shape[0]
        BASE_YEAR = 2016
        START_YEAR = 2018
        if self.frequency == "annual":
            start_pos = START_YEAR - BASE_YEAR          # 2
        else:
            start_pos = (START_YEAR - BASE_YEAR) * 2   # 4
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        # to torch tensors
        img = torch.from_numpy(img).float()     # (T, C, H, W)
        mask = torch.from_numpy(mask).long()    # (H, W)
        mask = (mask > 0).long()

        # Apply transforms (which handle padding/cropping via CenterCropTS)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # Apply prediction_horizon masking AFTER normalization so masked timesteps are
        # exactly 0.0 — the value U-TAE's pad_value detection compares against.
        # With prediction_horizon=K, the model only sees data up to (endYear - K),
        # forcing it to predict land take K years in advance (per tile).
        if self.end_years is not None:
            end_year = self.end_years.get(fid)
            if end_year is not None:
                cutoff_year = end_year - self.prediction_horizon
                assert cutoff_year in YEARS, (
                    f"cutoff_year {cutoff_year} not in YEARS for {fid} — "
                    f"should have been filtered out in __init__"
                )
                if self.frequency == "annual":
                    n_valid = min(YEARS.index(cutoff_year) + 1, current_T)
                else:
                    n_valid = min((YEARS.index(cutoff_year) + 1) * 2, current_T)
                img[n_valid:] = 0.0
                positions[n_valid:] = 0

                # input_years masking: keep per-tile anchor year (startYear,
                # clamped to YEARS[0]) + latest N-1 years up to the cutoff.
                # Years before the anchor and the gap between anchor and the
                # latest window are zeroed. This lets us vary temporal context
                # length (N) independently of prediction horizon (K), while
                # using as much of each tile's available history as possible.
                if self.input_years is not None:
                    steps_per_year = 1 if self.frequency == "annual" else 2
                    cutoff_year_idx = YEARS.index(cutoff_year)

                    # Per-tile anchor: startYear clamped to first Sentinel year
                    tile_start = self.start_years.get(fid, YEARS[0]) if self.start_years else YEARS[0]
                    anchor_year = max(tile_start, YEARS[0])
                    anchor_year_idx = YEARS.index(anchor_year)

                    # Zero timesteps before the anchor
                    anchor_ts = anchor_year_idx * steps_per_year
                    if anchor_ts > 0:
                        img[:anchor_ts] = 0.0
                        positions[:anchor_ts] = 0

                    # First timestep of the latest (N-1)-year window
                    latest_window_start = (cutoff_year_idx - (self.input_years - 2)) * steps_per_year
                    # Gap = everything after anchor and before the latest window
                    gap_start = (anchor_year_idx + 1) * steps_per_year
                    gap_end = latest_window_start
                    if gap_start < gap_end:
                        img[gap_start:gap_end] = 0.0
                        positions[gap_start:gap_end] = 0

        # Pad or truncate to max_timesteps if set (needed when T varies per tile).
        # Padding is zeros so U-TAE treats them as masked timesteps automatically.
        if self.max_timesteps is not None:
            if current_T < self.max_timesteps:
                pad_len = self.max_timesteps - current_T
                img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_len))
                positions = F.pad(positions, (0, pad_len))
            elif current_T > self.max_timesteps:
                img = img[:self.max_timesteps]
                positions = positions[:self.max_timesteps]

        return img, mask, positions
