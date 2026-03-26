from pathlib import Path
import csv
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT       = ROOT / "data" / "raw"
SENTINEL_DIR    = DATA_ROOT / "Sentinel_v2"
MASK_DIR        = DATA_ROOT / "Land_take_masks_coarse"
VHR_DIR         = DATA_ROOT / "vhr"
PLANETSCOPE_DIR = DATA_ROOT / "PlanetScope"
ALPHAEARTH_DIR  = DATA_ROOT / "AlphaEarth"
METADATA_PATH   = DATA_ROOT / "annotations_metadata_final.csv"
TESSERA_DIR = ROOT / "data" / "processed" / "tessera" / "snapped_to_mask_grid"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PATCH_SIZE = 256
BATCH_SIZE = 8
LR         = 1e-3
EPOCHS     = 10

for d in [REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class TileMetadata:
    refid:      str
    start_year: int   # year of first VHR image → first valid Sentinel year
    end_year:   int   # year of second VHR image → last valid Sentinel year


def load_metadata(skip_na: bool = True) -> dict[str, TileMetadata]:
    """
    Return {refid: TileMetadata} for all tiles.

    Parameters
    ----------
    skip_na : if True, rows where startYear or endYear is 'NA' are silently dropped.
    """
    meta: dict[str, TileMetadata] = {}
    with open(METADATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            sy, ey = row["startYear"], row["endYear"]
            if skip_na and (sy == "NA" or ey == "NA"):
                continue
            meta[row["REFID"]] = TileMetadata(
                refid=row["REFID"],
                start_year=int(sy),
                end_year=int(ey),
            )
    return meta

def load_all_years() -> list[int]:
    """Derive the full year range from the metadata file itself."""
    meta = load_metadata()
    min_year = min(m.start_year for m in meta.values())
    max_year = max(m.end_year   for m in meta.values())
    return list(range(min_year, max_year + 1))

ALL_YEARS: list[int] = load_all_years()
ACQUISITIONS_PER_YEAR = 2
MAX_TIMESTEPS = len(ALL_YEARS) * ACQUISITIONS_PER_YEAR
