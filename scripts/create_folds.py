"""Generate geographic 5-fold cross-validation assignments for all tiles.

Coordinates are extracted directly from each tile's REFID and K-means
clustering on (lon, lat) produces five spatially compact folds, mirroring
the approach used in the PASTIS benchmark (Garnot & Landrieu, ICCV 2021).
Spatial separation avoids the performance overestimation (up to 28%) that
random splits can introduce due to spatial autocorrelation.

Run once before training:

    python scripts/create_folds.py

The assignments are written to ``src/data/geographic_folds.csv`` and should
be committed to the repository so that all experiments use identical splits.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_metadata
from src.data.splits import (
    FOLDS_PATH,
    create_geographic_folds,
    get_fold_splits,
    parse_coords_from_refid,
    save_folds,
)

N_FOLDS = 5
RANDOM_STATE = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate geographic folds.")
    parser.add_argument(
        "--min-start-year", 
        type=int, 
        default=None,
        help="Filter out tiles with a start_year strictly before this value."
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=FOLDS_PATH,
        help="Path to save the generated folds CSV. Defaults to FOLDS_PATH."
    )
    args = parser.parse_args()

    metadata = load_metadata()
    refids = []
    dropped_count = 0
    for fid, meta in metadata.items():
        if args.min_start_year is not None and meta.start_year < args.min_start_year:
            dropped_count += 1
            continue
        refids.append(fid)
        
    refids = sorted(refids)
    print(f"Loaded {len(refids)} tiles from metadata.")
    if args.min_start_year is not None:
        print(f"Excluded {dropped_count} tiles with start_year < {args.min_start_year}.")

    print(f"\nCreating {N_FOLDS} geographic folds via K-means (random_state={RANDOM_STATE})...")
    fold_assignments = create_geographic_folds(refids, n_folds=N_FOLDS, random_state=RANDOM_STATE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_folds(fold_assignments, args.output)
    print(f"Saved fold assignments → {args.output}")

    # Per-fold statistics
    fold_refids: dict[int, list[str]] = defaultdict(list)
    for refid, fold in fold_assignments.items():
        fold_refids[fold].append(refid)

    print(f"\n{'─' * 62}")
    print(f"{'Fold':<6} {'Tiles':<8} {'Lon range':<24} {'Lat range'}")
    print(f"{'─' * 62}")
    for fold_id in range(N_FOLDS):
        members = fold_refids[fold_id]
        coords = [parse_coords_from_refid(r) for r in members]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        print(
            f"  {fold_id:<4} {len(members):<8}"
            f"{min(lons):+7.1f} → {max(lons):+6.1f}    "
            f"{min(lats):5.1f} → {max(lats):.1f}"
        )
    print(f"{'─' * 62}")

    # CV split summary
    print(f"\n5-fold CV splits  (test | val | train):")
    for test_fold in range(N_FOLDS):
        train_ids, val_ids, test_ids = get_fold_splits(fold_assignments, test_fold, N_FOLDS)
        print(
            f"  fold={test_fold}:  "
            f"test={len(test_ids):>3}  val={len(val_ids):>3}  train={len(train_ids):>3}"
        )

    print(
        f"\nCommit {FOLDS_PATH.name} to the repository to lock in these splits."
    )


if __name__ == "__main__":
    main()
