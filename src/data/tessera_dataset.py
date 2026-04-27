import bisect
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.file_helpers import find_file_by_prefix
from src.config import (
    ALL_YEARS, TESSERA_YEARS, MASK_DIR,
    tessera_tif_path, load_metadata,
)


class TesseraDataset(Dataset):
    """Load GeoTessera yearly embeddings paired with land-take segmentation masks.

    The loaded years are stacked chronologically to form a (T, 128, H, W) tensor,
    then padded with zeros to len(TESSERA_YEARS) so all samples in a batch share
    the same temporal length.

    Args:
        ids: list of REFIDs.
        transform: spatial transforms applied jointly to (emb, mask).
        prediction_horizon (K): zero timesteps from end_year-K onwards, forcing
            the model to predict land take K years ahead.
        input_years (N): keep start_year plus the latest N-1 years before the
            cutoff; None keeps all visible years.

    Tile filtering at construction time (logged to stdout):
        - Tiles with no metadata or whose start_year predates TESSERA_YEARS.
        - Tiles with no visible years before the cutoff.
        - Tiles missing any required TESSERA file.
        - Tiles missing a mask file.
    """

    DATASET_NAME = "tessera"
    BANDS_PER_YEAR = 128

    @staticmethod
    def get_ref_ids(tessera_dir: Path) -> list[str]:
        """Return sorted unique REFIDs found in tessera_dir.

        Filenames follow the convention {refid}_tessera_{year}_snapped.tif.
        """
        files = sorted(tessera_dir.glob("*_tessera_*_snapped.tif"))
        return sorted({f.name.split("_tessera_")[0] for f in files})

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

        # Step 1: drop tiles with no metadata, out-of-range start year,
        # or no visible years before the cutoff.
        filtered, dropped = [], []
        for fid in ids:
            meta = self.metadata.get(fid)
            if meta is None:
                dropped.append(fid)
                print(f"[TesseraDataset] Excluded {fid}: no metadata.")
                continue

            if meta.start_year < TESSERA_YEARS[0]:
                dropped.append(fid)
                print(
                    f"[TesseraDataset] Excluded {fid}: "
                    f"start_year {meta.start_year} predates TESSERA_YEARS."
                )
                continue

            cutoff_year = meta.end_year - prediction_horizon
            tile_years = [y for y in TESSERA_YEARS if meta.start_year <= y <= meta.end_year]

            if not tile_years or cutoff_year not in tile_years:
                dropped.append(fid)
                print(
                    f"[TesseraDataset] Excluded {fid}: "
                    f"cutoff year {cutoff_year} not in valid window."
                )
                continue

            filtered.append(fid)
            self.tile_years[fid] = tile_years

        if dropped:
            print(
                f"[TesseraDataset] K={prediction_horizon}: excluded {len(dropped)} tile(s). "
                f"{len(filtered)} remain."
            )

        # Step 2: resolve file paths; exclude tiles with missing TESSERA or mask files.
        self.emb_paths: dict[str, list[Path]] = {}
        self.mask_paths: dict[str, Path] = {}
        valid_ids: list[str] = []
        excluded_tessera: dict[str, list[int]] = {}

        for fid in filtered:
            tile_years = self.tile_years[fid]
            missing, paths = [], []
            for year in tile_years:
                p = tessera_tif_path(fid, year)
                if p.exists():
                    paths.append(p)
                else:
                    missing.append(year)

            if missing:
                excluded_tessera[fid] = missing
                continue

            try:
                mask_path = find_file_by_prefix(MASK_DIR, fid)
            except FileNotFoundError:
                print(f"[TesseraDataset] Excluded {fid}: no mask file found.")
                continue

            self.emb_paths[fid] = paths
            self.mask_paths[fid] = mask_path
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

        # Load TESSERA embeddings stacked chronologically.
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

        emb = np.stack(yearly, axis=0)  # (T, 128, H, W)
        current_T = emb.shape[0]

        # Load segmentation mask.
        with rasterio.open(self.mask_paths[fid]) as src_m:
            mask = src_m.read(1)  # (H, W)

        emb = torch.from_numpy(emb).float()
        mask = torch.from_numpy(mask).long()
        mask = (mask > 0).long()

        # Annual temporal positions; position 0 reserved for padding.
        # ALL_YEARS[0] is the shared origin across all modalities (year 2016 -> pos 1).
        start_pos = tile_years[0] - ALL_YEARS[0] + 1
        positions = torch.arange(start_pos, start_pos + current_T, dtype=torch.long)

        if self.transform is not None:
            emb, mask = self.transform(emb, mask)

        # Temporal masking: zero out timesteps after the cutoff.
        cutoff_year = meta.end_year - self.prediction_horizon
        n_visible = bisect.bisect_right(tile_years, cutoff_year)

        emb[n_visible:] = 0.0
        positions[n_visible:] = 0

        # input_years (N) windowing: keep start_year + latest N-1 years before cutoff.
        if self.input_years is not None:
            window_limit = cutoff_year - (self.input_years - 1)
            for i, y in enumerate(tile_years[:n_visible]):
                if y != tile_years[0] and y <= window_limit:
                    emb[i] = 0.0
                    positions[i] = 0

        # Pad to max_timesteps for consistent batching.
        if current_T < self.max_timesteps:
            pad_len = self.max_timesteps - current_T
            emb = F.pad(emb, (0, 0, 0, 0, 0, 0, 0, pad_len))
            positions = F.pad(positions, (0, pad_len))
        elif current_T > self.max_timesteps:
            emb = emb[:self.max_timesteps]
            positions = positions[:self.max_timesteps]

        return emb, mask, positions
