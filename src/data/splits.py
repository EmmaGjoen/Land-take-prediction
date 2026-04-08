"""
Shared train/val/test split utilities for fair model comparison.

Two splitting strategies are provided:

* **Random split** (``get_splits``): legacy 70/15/15 random split kept for
  backward compatibility.
* **Geographic 5-fold CV** (``create_geographic_folds``, ``get_fold_splits``):
  tiles are clustered into five spatially compact groups via K-means on
  (lon, lat) coordinates encoded in each refid.  This mirrors the approach
  used in the PASTIS benchmark (Garnot & Landrieu, ICCV 2021) and avoids
  spatial autocorrelation inflating test metrics.

Typical workflow::

    # Once, before training:
    python scripts/create_folds.py   # → src/data/geographic_folds.csv

    # At training time (fold 0 = test, fold 1 = val, folds 2-4 = train):
    from src.data.splits import load_folds, get_fold_splits
    folds = load_folds()
    train_ids, val_ids, test_ids = get_fold_splits(folds, test_fold=0)
"""

import csv
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

# Path where geographic fold assignments are persisted.
# Small CSV (~10 KB) committed to the repository for full reproducibility.
FOLDS_PATH = Path(__file__).parent / "geographic_folds.csv"


def get_splits(
    ref_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[list[str], list[str], list[str]]:
    """
    Split reference IDs into train/val/test sets with fixed random seed.
    
    This function ensures reproducible splits across different notebooks and models,
    enabling fair baseline comparison under identical data conditions.
    
    Args:
        ref_ids: List of reference IDs (e.g., tile identifiers)
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_ref_ids, val_ref_ids, test_ref_ids)
    
    Raises:
        ValueError: If ratios don't sum to 1.0
    
    Example:
        >>> ref_ids = ["tile_001", "tile_002", ..., "tile_034"]
        >>> train_ids, val_ids, test_ids = get_splits(ref_ids)
        >>> len(train_ids), len(val_ids), len(test_ids)
        (23, 5, 5)  # For 34 total tiles with 70/15/15 split
    """
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


def get_ref_ids_from_directory(
    directory: Path,
    pattern: str = "*_RGBNIRRSWIRQ_Mosaic.tif",
    exclude_suffix: str = "_RGBNIRRSWIRQ_Mosaic",
) -> list[str]:
    """
    Extract reference IDs from filenames in a directory.
    
    Args:
        directory: Directory containing data files
        pattern: Glob pattern to match files (default: Sentinel pattern)
        exclude_suffix: Suffix to remove from filenames to get ref_id
    
    Returns:
        Sorted list of reference IDs
    
    Example:
        >>> from src.config import SENTINEL_DIR
        >>> ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
        >>> ref_ids[:3]
        ['R101C117', 'R101C118', 'R101C119']
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    ref_ids = [f.stem.replace(exclude_suffix, "") for f in files]
    return ref_ids


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
    """Cluster tiles into spatially separated folds using K-means on coordinates.

    Produces ``n_folds`` geographically compact groups, mirroring the approach
    used in the PASTIS benchmark.  A fixed ``random_state`` ensures that the
    same fold structure is produced every time.

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
    """Persist fold assignments to a CSV file for reproducible reuse.

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
