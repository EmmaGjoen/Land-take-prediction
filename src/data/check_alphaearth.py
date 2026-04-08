import sys
from pathlib import Path

# Add workspace root to path so src module can be found
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import rasterio
from src.config import load_metadata, ALPHAEARTH_DIR 
from src.data.ae import find_file_by_prefix

C = 64
meta = load_metadata()
mismatches = 0

for fid, m in meta.items():
    try:
        path = find_file_by_prefix(ALPHAEARTH_DIR, fid)
    except FileNotFoundError:
        print(f"[MISSING] No AlphaEarth file found for {fid}")
        continue

    with rasterio.open(path) as src:
        num_bands = src.count   # src.count gives band count, not src.shape
        H, W = src.height, src.width

    expected_years = m.end_year - m.start_year + 1
    expected_bands = expected_years * C

    if num_bands != expected_bands:
        actual_years = num_bands // C
        print(
            f"[MISMATCH] {fid}: metadata says {expected_years} years "
            f"({m.start_year}–{m.end_year}) = {expected_bands} bands, "
            f"but .tif has {num_bands} bands = {actual_years} years"
        )
        mismatches += 1

print(f"\nDone. {mismatches} mismatches out of {len(meta)} tiles.")