"""Filename and reference ID helpers shared across datasets."""
from pathlib import Path

def find_file_by_prefix(base_dir: Path, fid: str) -> Path:
    """Find the unique .tif file in base_dir whose name starts with fid."""
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
    """Extract sorted reference IDs from filenames matching a glob pattern."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    ref_ids = [f.stem.replace(exclude_suffix, "") for f in files]
    return ref_ids
