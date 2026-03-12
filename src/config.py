from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

# Data folders on IDUN
DATA_ROOT = ROOT / "data" / "raw"

SENTINEL_DIR    = DATA_ROOT / "Sentinel"
MASK_DIR        = DATA_ROOT / "Land_take_masks_coarse"
VHR_DIR         = DATA_ROOT / "vhr"
PLANETSCOPE_DIR = DATA_ROOT / "PlanetScope"
ALPHAEARTH_DIR  = DATA_ROOT / "AlphaEarth"

METADATA_PATH = DATA_ROOT / "annotations_metadata_final.csv"


def load_end_years() -> dict[str, int]:
    """Return {refid: endYear} for all tiles with valid metadata.

    Tiles with NA endYear (two known bad entries) are silently skipped.
    """
    end_years: dict[str, int] = {}
    with open(METADATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["endYear"] != "NA":
                end_years[row["REFID"]] = int(row["endYear"])
    return end_years

# Tessera embeddings (snapped to mask grid)
TESSERA_DIR = ROOT / "data" / "processed" / "tessera" / "snapped_to_mask_grid"

# Output / reports
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Training defaults
PATCH_SIZE = 256
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10

# Create folders if missing 
for d in [REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Chronological mapping of years present in the Sentinel time-series (ordered)
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
