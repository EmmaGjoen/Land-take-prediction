"""Train/val/test split utilities.

Two strategies:

* ``get_splits``: legacy random 70/15/15 split.
* ``create_geographic_folds`` + ``get_fold_splits``: geographic 5-fold CV
  via K-means on (lon, lat). Follows the PASTIS benchmark approach
  (Garnot & Landrieu, ICCV 2021) to avoid spatial autocorrelation.

Typical workflow::

    python scripts/create_folds.py                         # run once
    folds = load_folds()
    train_ids, val_ids, test_ids = get_fold_splits(folds, test_fold=0)
"""

import csv
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

# Pre-computed fold assignments CSV, committed for reproducibility.
FOLDS_PATH = Path(__file__).parent / "geographic_folds_2017.csv"


def get_splits(
    ref_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[list[str], list[str], list[str]]:
    """Split reference IDs into train/val/test with a fixed random seed."""
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
    
    # First split: separate train from (val + test)
    train_ref_ids, val_test_ref_ids = train_test_split(
        ref_ids,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
    )
    
    # Second split: separate val from test
    # Adjust the test_size to be relative to the remaining data
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_ref_ids, test_ref_ids = train_test_split(
        val_test_ref_ids,
        test_size=relative_test_size,
        random_state=random_state,
    )
    
    return train_ref_ids, val_ref_ids, test_ref_ids


# ── Geographic k-fold cross-validation ──────────────────────────────────────


def parse_coords_from_refid(refid: str) -> tuple[float, float]:
    """Extract (longitude, latitude) encoded in a tile refid.

    Refids follow the convention ``a{lon}_{lat}`` where the decimal point in
    each coordinate is represented as a dash, for example::

        a-7-20196668794104_53-29507776919935  →  lon=-7.20, lat=53.30
        a12-46613619061323_57-24443634830578  →  lon=+12.47, lat=57.24

    Args:
        refid: Tile reference ID from annotations_metadata_final.csv.

    Returns:
        ``(longitude, latitude)`` as floats.
    """
    lon_raw, lat_raw = refid.split("_", 1)
    lon_str = lon_raw.removeprefix("a")

    def _parse(s: str) -> float:
        negative = s.startswith("-")
        if negative:
            s = s[1:]
        return (-1 if negative else 1) * float(s.replace("-", ".", 1))

    return _parse(lon_str), _parse(lat_raw)


def create_geographic_folds(
    refids: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, int]:
    """Cluster tiles into geographic folds using K-means on coordinates.

    Produces ``n_folds`` spatially compact groups, following the PASTIS
    benchmark approach.

    Args:
        refids: Tile reference IDs whose coordinates are encoded in the ID.
        n_folds: Number of geographic folds (default 5).
        random_state: Seed for K-means initialisation (default 42).

    Returns:
        Mapping of ``refid → fold_index`` (0-indexed).
    """
    import numpy as np
    from sklearn.cluster import KMeans

    coords = np.array([parse_coords_from_refid(r) for r in refids])
    kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords)
    return {refid: int(label) for refid, label in zip(refids, labels)}


def save_folds(fold_assignments: dict[str, int], path: Path = FOLDS_PATH) -> None:
    """Save fold assignments to a CSV file.

    Args:
        fold_assignments: Mapping returned by :func:`create_geographic_folds`.
        path: Destination CSV (default: ``src/data/geographic_folds.csv``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["refid", "fold"])
        for refid, fold in sorted(fold_assignments.items()):
            writer.writerow([refid, fold])


def load_folds(path: Path = FOLDS_PATH) -> dict[str, int]:
    """Load fold assignments saved by :func:`save_folds`.

    Args:
        path: CSV file written by :func:`save_folds`.

    Returns:
        Mapping of ``refid → fold_index``.

    Raises:
        FileNotFoundError: If the folds file does not exist.
            Generate it by running ``python scripts/create_folds.py``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fold assignments not found at {path}.\n"
            "Generate them by running:  python scripts/create_folds.py"
        )
    with open(path, newline="") as f:
        return {row["refid"]: int(row["fold"]) for row in csv.DictReader(f)}


def get_fold_splits(
    fold_assignments: dict[str, int],
    test_fold: int,
    n_folds: int = 5,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train_ids, val_ids, test_ids) for one round of k-fold CV.

    The test set is ``test_fold``; validation is the adjacent fold
    ``(test_fold + 1) % n_folds``; training is all remaining folds.

    Args:
        fold_assignments: Mapping of refid → fold index.
        test_fold: Fold index to use as test set (0-indexed).
        n_folds: Total number of folds (default 5).

    Returns:
        Three lists of refids: ``(train_ids, val_ids, test_ids)``.
    """
    val_fold = (test_fold + 1) % n_folds
    train_ids = [r for r, f in fold_assignments.items() if f not in (test_fold, val_fold)]
    val_ids   = [r for r, f in fold_assignments.items() if f == val_fold]
    test_ids  = [r for r, f in fold_assignments.items() if f == test_fold]
    return train_ids, val_ids, test_ids
