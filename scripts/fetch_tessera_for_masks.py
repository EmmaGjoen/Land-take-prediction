"""
Fetch GeoTessera embeddings for HABLOSS mask tiles and snap to mask grid. 2024 default year. 

Usage:
    python scripts/fetch_tessera_for_masks.py
    python scripts/fetch_tessera_for_masks.py --year 2023
    python scripts/fetch_tessera_for_masks.py --masks-dir data/raw/masks --out-dir data/processed/tessera
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from geotessera import GeoTessera

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
    with rasterio.open(mask_path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)  # min_lon, min_lat, max_lon, max_lat (EPSG:4326 for HABLOSS)


def snap_tessera_to_mask_grid(tessera_tif: Path, mask_path: Path, out_path: Path) -> None:
    with rasterio.open(mask_path) as msrc:
        dst_crs = msrc.crs
        dst_transform = msrc.transform
        dst_height = msrc.height
        dst_width = msrc.width

    with rasterio.open(tessera_tif) as tsrc:
        src_crs = tsrc.crs
        src_transform = tsrc.transform
        src_count = tsrc.count
        src_dtype = tsrc.dtypes[0]

        profile = tsrc.profile.copy()
        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=dst_transform,
            height=dst_height,
            width=dst_width,
            count=src_count,
            dtype=src_dtype,
            compress="lzw",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for band in range(1, src_count + 1):
                src_data = tsrc.read(band)
                dst_data = np.zeros((dst_height, dst_width), dtype=src_data.dtype)

                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
                dst.write(dst_data, band)


def fetch_one_mask(mask_path: Path, year: int, out_dir: Path, gt: GeoTessera, skip_existing: bool = True) -> str:
    """Fetch Tessera embeddings for a single mask and snap to its grid.
    
    Returns: 'exists', 'ok', or 'skipped'
    """
    refid = refid_from_mask_path(mask_path)
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

        raw_dir = out_dir / "raw_downloads" / refid
        files = gt.export_embedding_geotiffs(
            tiles_to_fetch=tiles, output_dir=str(raw_dir), bands=None, compress="lzw"
        )
        if len(files) == 0:
            logging.warning(f"[SKIP] Export failed for {refid}")
            return "skipped"

        raw_tif = Path(files[0])
        snap_tessera_to_mask_grid(raw_tif, mask_path, snapped_path)

        logging.info(f"[OK] {refid} -> {snapped_path}")
        return "ok"
    
    except Exception as e:
        logging.error(f"[ERROR] {refid}: {e}")
        return "skipped"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GeoTessera embeddings for HABLOSS masks")
    parser.add_argument("--masks-dir", type=Path, default=Path("data/raw/masks"), help="Directory containing mask files")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tessera"), help="Output directory")
    parser.add_argument("--year", type=int, default=2024, help="Year for Tessera embeddings")
    parser.add_argument("--force", action="store_true", help="Reprocess existing files")
    args = parser.parse_args()

    masks_dir: Path = args.masks_dir
    out_dir: Path = args.out_dir
    year: int = args.year
    skip_existing: bool = not args.force

    mask_paths = sorted(masks_dir.glob("*_mask.tif"))
    if not mask_paths:
        raise RuntimeError(f"No masks found in {masks_dir.resolve()}")

    logging.info(f"Processing {len(mask_paths)} masks for year {year}...")
    
    gt = GeoTessera()
    processed: list[str] = []
    existing: list[str] = []
    skipped: list[str] = []

    for mp in mask_paths:
        result = fetch_one_mask(mp, year=year, out_dir=out_dir, gt=gt, skip_existing=skip_existing)
        if result == "ok":
            processed.append(mp.name)
        elif result == "exists":
            existing.append(mp.name)
        else:
            skipped.append(mp.name)

    # Summary
    logging.info("=" * 50)
    logging.info(f"New:       {len(processed)} masks")
    logging.info(f"Existing:  {len(existing)} masks (already processed)")
    logging.info(f"Skipped:   {len(skipped)} masks (no Tessera coverage)")
    logging.info(f"Total:     {len(processed) + len(existing)}/{len(mask_paths)} masks with embeddings")
    
    # Load previously skipped masks and merge with current skipped
    skipped_file = out_dir / "skipped_masks.txt"
    all_skipped = set(skipped)
    if skipped_file.exists():
        content = skipped_file.read_text().strip()
        if content:
            all_skipped.update(content.split("\n"))
    
    if all_skipped:
        skipped_file.parent.mkdir(parents=True, exist_ok=True)
        skipped_file.write_text("\n".join(sorted(all_skipped)))
        logging.info(f"All skipped masks ({len(all_skipped)}) saved to: {skipped_file}")


if __name__ == "__main__":
    main()
