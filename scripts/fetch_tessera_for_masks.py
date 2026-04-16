"""
Fetch GeoTessera embeddings for all tiles in the metadata and snap to mask grid.

Tiles are discovered via load_metadata() + MASK_DIR, mirroring the training
datasets, so fetch coverage always matches what the model expects.

When a mask's bounding box overlaps multiple GeoTessera tiles, all tiles are
downloaded and merged with rasterio.merge before snapping to the mask grid.

Usage:
    python scripts/fetch_tessera_for_masks.py
    python scripts/fetch_tessera_for_masks.py --year 2023
    python scripts/fetch_tessera_for_masks.py --year 2017-2024
    python scripts/fetch_tessera_for_masks.py --force   # reprocess existing
"""
from __future__ import annotations

import argparse
import logging
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
from geotessera import GeoTessera

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MASK_DIR, load_metadata

_WGS84 = CRS.from_epsg(4326)
_thread_local = threading.local()

# Per-GeoTessera-tile locks: prevents two workers from downloading the same
# 0.1° grid tile simultaneously, avoiding races on the shared .npy cache.
_tile_locks: dict[tuple, threading.Lock] = {}
_tile_locks_mutex = threading.Lock()


def _get_gt(embeddings_dir: Path | None = None) -> GeoTessera:
    """Return a thread-local GeoTessera instance (created once per thread).

    embeddings_dir sets where intermediate .npy tile cache is stored.
    Defaults to cwd if not specified (GeoTessera default).
    """
    if not hasattr(_thread_local, "gt"):
        kwargs = {"embeddings_dir": str(embeddings_dir)} if embeddings_dir is not None else {}
        _thread_local.gt = GeoTessera(**kwargs, verify_hashes=False)
    return _thread_local.gt


def _geotessera_tile_keys(bbox: tuple[float, float, float, float], year: int) -> list[tuple]:
    """Compute which GeoTessera 0.1° grid cells a bbox overlaps, keyed by (lon, lat, year).

    Used to acquire per-tile locks before downloading so parallel workers
    never race on the same underlying GeoTessera file.
    """
    import math
    min_lon, min_lat, max_lon, max_lat = bbox
    keys = []
    lon = math.floor(min_lon / 0.1) * 0.1
    while lon <= max_lon + 1e-9:
        lat = math.floor(min_lat / 0.1) * 0.1
        while lat <= max_lat + 1e-9:
            keys.append((round(lon, 1), round(lat, 1), year))
            lat = round(lat + 0.1, 1)
        lon = round(lon + 0.1, 1)
    return keys


def _acquire_tile_locks(keys: list[tuple]) -> list[threading.Lock]:
    """Acquire locks for the given GeoTessera tile keys in sorted order (avoids deadlock)."""
    with _tile_locks_mutex:
        for k in keys:
            if k not in _tile_locks:
                _tile_locks[k] = threading.Lock()
        locks = [_tile_locks[k] for k in sorted(keys)]
    for lock in locks:
        lock.acquire()
    return locks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def refid_from_mask_path(mask_path: Path) -> str:
    name = mask_path.name
    if not name.endswith("_mask.tif"):
        raise ValueError(f"Expected *_mask.tif, got: {name}")
    return name.removesuffix("_mask.tif")


def bbox_from_mask(mask_path: Path) -> tuple[float, float, float, float]:
    """Return bounding box in WGS84 degrees (min_lon, min_lat, max_lon, max_lat).

    Reprojects from the mask's native CRS if it is not already WGS84, so that
    GeoTessera's tile registry (which indexes in 0.1° WGS84 tiles) receives
    correct coordinates regardless of the mask projection.
    """
    with rasterio.open(mask_path) as src:
        b = src.bounds
        crs = src.crs
    if crs != _WGS84:
        left, bottom, right, top = transform_bounds(crs, _WGS84, b.left, b.bottom, b.right, b.top)
    else:
        left, bottom, right, top = b.left, b.bottom, b.right, b.top
    return (left, bottom, right, top)


def snap_tessera_to_mask_grid(tessera_tifs: list[Path], mask_path: Path, out_path: Path) -> None:
    """Reproject and snap one or more Tessera tile GeoTIFFs to the mask grid.

    When multiple tiles are provided they are merged spatially with
    rasterio.merge before reprojection, so masks that cross a tile
    boundary receive a spatially complete embedding.
    """
    with rasterio.open(mask_path) as msrc:
        dst_crs = msrc.crs
        dst_transform = msrc.transform
        dst_height = msrc.height
        dst_width = msrc.width

    datasets = [rasterio.open(p) for p in tessera_tifs]
    # MemoryFiles for any reprojected intermediates. Kept open until after merge.
    _memfiles: list[MemoryFile] = []
    try:
        if len(datasets) == 1:
            src_data = datasets[0].read()
            src_transform = datasets[0].transform
            src_crs = datasets[0].crs
            src_count = datasets[0].count
            src_dtype = datasets[0].dtypes[0]
        else:
            # Tessera tiles may span a UTM zone boundary and arrive in different
            # projected CRS (e.g. EPSG:32629 and EPSG:32630). rio_merge requires
            # identical CRS, so reproject all tiles to WGS84 first when mixed.
            unique_crs = {ds.crs for ds in datasets}
            if len(unique_crs) > 1:
                merge_inputs = []
                for ds in datasets:
                    t, w, h = calculate_default_transform(
                        ds.crs, _WGS84, ds.width, ds.height, *ds.bounds
                    )
                    profile = ds.profile.copy()
                    profile.update(driver="GTiff", crs=_WGS84, transform=t, width=w, height=h)
                    mf = MemoryFile()
                    _memfiles.append(mf)
                    with mf.open(**profile) as mem_ds:
                        for i in range(1, ds.count + 1):
                            reproject(
                                source=rasterio.band(ds, i),
                                destination=rasterio.band(mem_ds, i),
                                resampling=Resampling.nearest,
                            )
                    merge_inputs.append(mf.open())
            else:
                merge_inputs = datasets
            src_data, src_transform = rio_merge(merge_inputs)
            src_crs = merge_inputs[0].crs
            src_count = src_data.shape[0]
            src_dtype = src_data.dtype.name
    finally:
        for ds in datasets:
            ds.close()
        for mf in _memfiles:
            mf.close()

    # Write merged/single source into a MemoryFile so rasterio reproject
    # receives a proper dataset with embedded CRS rather than a bare numpy array.
    # Passing numpy arrays to reproject can trigger spurious CRS mismatch errors
    # when tiles have been closed after merging.
    src_profile = {
        "driver": "GTiff",
        "crs": src_crs,
        "transform": src_transform,
        "height": src_data.shape[1],
        "width": src_data.shape[2],
        "count": src_count,
        "dtype": src_dtype,
    }
    dst_profile = {
        **src_profile,
        "crs": dst_crs,
        "transform": dst_transform,
        "height": dst_height,
        "width": dst_width,
        "compress": "lzw",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as mem_src:
            mem_src.write(src_data)
        with memfile.open() as mem_src:
            with rasterio.open(out_path, "w", **dst_profile) as dst:
                for band_idx in range(1, src_count + 1):
                    reproject(
                        source=rasterio.band(mem_src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        resampling=Resampling.nearest,
                    )


def fetch_one_mask(mask_path: Path, year: int, out_dir: Path, gt: GeoTessera | None = None, skip_existing: bool = True, refid: str | None = None) -> str:
    """Fetch Tessera embeddings for a single mask and snap to its grid.

    Returns: 'exists', 'ok', 'skipped' (no Tessera coverage), or 'error' (exception).
    """
    if refid is None:
        refid = refid_from_mask_path(mask_path)
    if gt is None:
        gt = _get_gt(embeddings_dir=out_dir / "raw_downloads")
    snapped_path = out_dir / "snapped_to_mask_grid" / f"{refid}_tessera_{year}_snapped.tif"

    if skip_existing and snapped_path.exists():
        logging.info(f"[EXISTS] {refid} -> already processed")
        return "exists"

    try:
        bbox = bbox_from_mask(mask_path)

        tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
        if len(tiles) == 0:
            logging.warning(f"[SKIP] No Tessera coverage for {refid} year={year}")
            return "skipped"

        # Acquire per-GeoTessera-tile locks before downloading to prevent
        # parallel workers from racing on the same 0.1° grid cell (causes hash mismatch).
        tile_keys = _geotessera_tile_keys(bbox, year)
        locks = _acquire_tile_locks(tile_keys)
        try:
            raw_dir = out_dir / "raw_downloads" / refid
            files = gt.export_embedding_geotiffs(
                tiles_to_fetch=tiles, output_dir=str(raw_dir), bands=None, compress="lzw"
            )
        finally:
            for lock in locks:
                lock.release()

        if len(files) == 0:
            logging.error(f"[ERROR] Export returned no files for {refid} year={year} (registry has coverage — likely a download failure)")
            return "error"

        raw_tifs = [Path(f) for f in files]
        if len(raw_tifs) > 1:
            logging.info(f"[MERGE] {refid} year={year}: merging {len(raw_tifs)} tiles")
        snap_tessera_to_mask_grid(raw_tifs, mask_path, snapped_path)

        logging.info(f"[OK] {refid} -> {snapped_path}")
        return "ok"
    
    except Exception as e:
        logging.error(f"[ERROR] {refid}: {e}")
        return "error"


def parse_year_arg(year_str: str) -> list[int]:
    """Parse year argument. Supports single year (2024) or range (2018-2024)."""
    if "-" in year_str:
        start, end = year_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(year_str)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GeoTessera embeddings for HABLOSS masks")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tessera"), help="Output directory")
    parser.add_argument("--year", type=str, default="2017-2024", help="Year or year range (e.g., 2024 or 2017-2024)")
    parser.add_argument("--force", action="store_true", help="Reprocess existing files")
    parser.add_argument("--timeout", type=int, default=30, help="Socket timeout in seconds for tile downloads (default: 30)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel download threads (default: 4)")
    parser.add_argument("--retry-file", type=Path, default=None, help="File of 'refid_year' lines to retry (e.g. skipped_masks.txt). Only these tile-year pairs are processed.")
    args = parser.parse_args()

    socket.setdefaulttimeout(args.timeout)
    logging.info(f"Socket timeout: {args.timeout}s  |  workers: {args.workers}")

    out_dir: Path = args.out_dir
    years: list[int] = parse_year_arg(args.year)
    skip_existing: bool = not args.force

    # Discover tiles via metadata + MASK_DIR, mirroring how the training datasets work.
    metadata = load_metadata()
    mask_pairs: list[tuple[str, Path]] = []
    for refid in sorted(metadata.keys()):
        candidates = sorted(MASK_DIR.glob(f"{refid}*.tif"))
        if not candidates:
            logging.warning(f"[SKIP] No mask file found for {refid} in {MASK_DIR}")
            continue
        mask_pairs.append((refid, candidates[0]))

    if not mask_pairs:
        raise RuntimeError(f"No mask files found in {MASK_DIR.resolve()}")

    processed: list[str] = []
    existing: list[str] = []
    skipped: list[str] = []
    errored: list[str] = []

    # Build the full list of (refid, mask_path, year) tasks.
    # --retry-file restricts processing to only the listed refid+year pairs.
    if args.retry_file is not None:
        if not args.retry_file.exists():
            raise FileNotFoundError(f"--retry-file not found: {args.retry_file}")
        retry_pairs: set[tuple[str, int]] = set()
        for line in args.retry_file.read_text().strip().splitlines():
            refid, year_str = line.rsplit("_", 1)
            retry_pairs.add((refid, int(year_str)))
        mask_lookup = {refid: mp for refid, mp in mask_pairs}
        tasks = [
            (refid, mask_lookup[refid], year)
            for refid, year in sorted(retry_pairs)
            if refid in mask_lookup
        ]
        logging.info(f"--retry-file: retrying {len(tasks)} tile-year pairs from {args.retry_file}")
    else:
        logging.info(f"Processing {len(mask_pairs)} tiles for years {years[0]}-{years[-1]}...")
        tasks = [
            (refid, mp, year)
            for year in years
            for refid, mp in mask_pairs
        ]

    def _fetch(task: tuple) -> tuple[str, str]:
        refid, mp, year = task
        result = fetch_one_mask(mp, year=year, out_dir=out_dir, skip_existing=skip_existing, refid=refid)
        return f"{refid}_{year}", result

    logging.info(f"Starting {len(tasks)} fetch tasks with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, t): t for t in tasks}
        task_years = sorted({t[2] for t in tasks})
        year_counts: dict[int, dict] = {
            y: {"ok": 0, "exists": 0, "skipped": 0, "error": 0} for y in task_years
        }
        for future in as_completed(futures):
            key, result = future.result()
            year = int(key.split("_")[-1])
            year_counts[year][result if result in year_counts[year] else "error"] += 1
            if result == "ok":
                processed.append(key)
            elif result == "exists":
                existing.append(key)
            elif result == "skipped":
                skipped.append(key)
            else:
                errored.append(key)

    for year in task_years:
        c = year_counts[year]
        logging.info(
            f"Year {year}: {c['ok']} new, {c['exists']} existing, "
            f"{c['skipped']} skipped, {c['error']} errors"
        )

    # Summary
    logging.info(f"\n{'='*50}")
    logging.info(f"TOTAL SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"New:       {len(processed)} mask-year combinations")
    logging.info(f"Existing:  {len(existing)} mask-year combinations (already processed)")
    logging.info(f"Skipped:   {len(skipped)} mask-year combinations (no Tessera coverage)")
    logging.info(f"Errors:    {len(errored)} mask-year combinations (exception during fetch/snap)")
    total_possible = len(tasks)
    logging.info(f"Total:     {len(processed) + len(existing)}/{total_possible} mask-year combinations with embeddings")

    # Write current run's skipped masks (no Tessera coverage). This reflects only the
    # current run — re-fetching a previously skipped mask will remove it from the file.
    skipped_file = out_dir / "skipped_masks.txt"
    if skipped:
        skipped_file.parent.mkdir(parents=True, exist_ok=True)
        skipped_file.write_text("\n".join(sorted(skipped)))
        logging.info(f"Skipped masks ({len(skipped)}) saved to: {skipped_file}")
    elif skipped_file.exists():
        skipped_file.unlink()
        logging.info("No skipped masks this run; removed stale skipped_masks.txt")

    # Write errored masks so they can be retried with --retry-file.
    errors_file = out_dir / "errors_masks.txt"
    if errored:
        errors_file.parent.mkdir(parents=True, exist_ok=True)
        errors_file.write_text("\n".join(sorted(errored)))
        logging.info(f"Errored masks ({len(errored)}) saved to: {errors_file}")
        logging.info(f"Retry with: python scripts/fetch_tessera_for_masks.py --retry-file {errors_file}")
    elif errors_file.exists():
        errors_file.unlink()
        logging.info("No errors this run; removed stale errors_masks.txt")


if __name__ == "__main__":
    main()
