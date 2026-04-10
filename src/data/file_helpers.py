from pathlib import Path

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

def get_ref_ids_from_tessera_dir(tessera_dir: Path) -> list[str]:
    """Return sorted unique REFIDs found in TESSERA_DIR.

    Filenames follow the convention ``{refid}_tessera_{year}_snapped.tif``;
    the REFID is everything before the first ``_tessera_`` token.
    """
    files = sorted(tessera_dir.glob("*_tessera_*_snapped.tif"))
    ref_ids = sorted({f.name.split("_tessera_")[0] for f in files})
    return ref_ids