"""
Generate a summary of which masks have GeoTessera embeddings for which years.
Scans files on disk to create an accurate coverage report.

Usage:
    python scripts/generate_tessera_summary.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_filename(filename: str) -> tuple[str, int | None]:
    """Extract mask ID and year from filename like 'maskid_tessera_2024_snapped.tif'."""
    parts = filename.replace("_snapped.tif", "").split("_tessera_")
    if len(parts) == 2:
        mask_id = parts[0]
        try:
            year = int(parts[1])
        except ValueError:
            year = None
        return mask_id, year
    return filename, None


def scan_coverage(tessera_dir: Path, masks_dir: Path, years: list[int]) -> dict:
    """Scan files to determine actual coverage."""
    
    # Get all mask IDs from masks directory
    all_masks = set()
    for mask_file in masks_dir.glob("*_mask.tif"):
        mask_id = mask_file.name.removesuffix("_mask.tif")
        all_masks.add(mask_id)
    
    logging.info(f"Found {len(all_masks)} masks in {masks_dir}")
    
    # Scan tessera files
    tessera_files = list(tessera_dir.glob("*_tessera_*_snapped.tif"))
    logging.info(f"Found {len(tessera_files)} tessera files in {tessera_dir}")
    
    # Build coverage map
    coverage = defaultdict(set)  # mask_id -> set of years
    
    for tif_path in tessera_files:
        mask_id, year = parse_filename(tif_path.name)
        if mask_id and year:
            # Verify file is not empty
            if tif_path.stat().st_size > 0:
                coverage[mask_id].add(year)
    
    return {
        "all_masks": all_masks,
        "coverage": dict(coverage),
        "years": years,
    }


def generate_markdown_summary(data: dict, out_path: Path) -> None:
    """Generate a markdown summary file."""
    all_masks = data["all_masks"]
    coverage = data["coverage"]
    years = data["years"]
    
    # Calculate stats
    masks_with_any_coverage = set(coverage.keys())
    masks_without_coverage = all_masks - masks_with_any_coverage
    
    # Coverage by year
    coverage_by_year = {y: 0 for y in years}
    for mask_id, mask_years in coverage.items():
        for year in mask_years:
            if year in coverage_by_year:
                coverage_by_year[year] += 1
    
    # Full coverage (all years)
    full_coverage = [m for m, yrs in coverage.items() if set(years).issubset(yrs)]

    # Missing coverage by year
    missing_by_year: dict[int, list[str]] = {}
    for year in years:
        masks_with_year = {m for m, yrs in coverage.items() if year in yrs}
        missing_by_year[year] = sorted(all_masks - masks_with_year)
    
    report = f"""# GeoTessera Coverage Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

| Metric | Count |
|--------|-------|
| Total masks | {len(all_masks)} |
| Masks with any coverage | {len(masks_with_any_coverage)} |
| Masks with full coverage ({years[0]}-{years[-1]}) | {len(full_coverage)} |
| Masks without any coverage | {len(masks_without_coverage)} |
| Years analyzed | {years[0]}-{years[-1]} ({len(years)} years) |
| Total embedding files | {sum(len(yrs) for yrs in coverage.values())} |

## Coverage by Year

| Year | Masks with embeddings | Percentage |
|------|----------------------|------------|
"""
    
    for year in years:
        count = coverage_by_year.get(year, 0)
        pct = (count / len(all_masks) * 100) if all_masks else 0
        report += f"| {year} | {count} / {len(all_masks)} | {pct:.1f}% |\n"

    report += """
## Missing Embeddings by Year

| Year | Missing masks | Percentage |
|------|---------------|------------|
"""

    for year in years:
        missing = missing_by_year.get(year, [])
        missing_count = len(missing)
        pct_missing = (missing_count / len(all_masks) * 100) if all_masks else 0
        report += f"| {year} | {missing_count} / {len(all_masks)} | {pct_missing:.1f}% |\n"
    
    # Detailed coverage table
    report += """
## Detailed Coverage Matrix

| Mask ID | """ + " | ".join(str(y) for y in years) + """ | Total |
|---------|""" + "|".join(["---"] * len(years)) + """|-------|
"""
    
    for mask_id in sorted(all_masks):
        mask_years = coverage.get(mask_id, set())
        row = f"| {mask_id[:40]}{'...' if len(mask_id) > 40 else ''} |"
        for year in years:
            row += " ✓ |" if year in mask_years else " - |"
        row += f" {len(mask_years)}/{len(years)} |"
        report += row + "\n"
    
    # List masks without coverage
    if masks_without_coverage:
        report += f"""
## Masks Without Any Coverage ({len(masks_without_coverage)})

"""
        for mask_id in sorted(masks_without_coverage):
            report += f"- {mask_id}\n"

    # List missing embeddings by year
    for year in years:
        missing = missing_by_year.get(year, [])
        if missing:
            report += f"""
## Missing Embeddings for {year} ({len(missing)})

"""
            for mask_id in missing:
                report += f"- {mask_id}\n"
    
    # List masks with full coverage
    if full_coverage:
        report += f"""
## Masks With Full Coverage ({len(full_coverage)})

"""
        for mask_id in sorted(full_coverage):
            report += f"- {mask_id}\n"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logging.info(f"Summary saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GeoTessera coverage summary")
    parser.add_argument(
        "--tessera-dir",
        type=Path,
        default=Path("data/processed/tessera/snapped_to_mask_grid"),
        help="Directory containing tessera GeoTIFFs"
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("data/raw/masks"),
        help="Directory containing mask files"
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/processed/tessera/COVERAGE_SUMMARY.md"),
        help="Output markdown file"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2018-2024",
        help="Year range"
    )
    args = parser.parse_args()
    
    # Parse years
    start, end = args.years.split("-")
    years = list(range(int(start), int(end) + 1))
    
    # Scan coverage
    data = scan_coverage(args.tessera_dir, args.masks_dir, years)
    
    # Generate report
    generate_markdown_summary(data, args.out_file)
    
    # Print quick summary to console
    logging.info("=" * 50)
    logging.info("QUICK SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Masks with coverage: {len(data['coverage'])} / {len(data['all_masks'])}")
    logging.info(f"Total embedding files: {sum(len(yrs) for yrs in data['coverage'].values())}")


if __name__ == "__main__":
    main()
