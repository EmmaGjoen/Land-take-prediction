"""
Exploratory Data Analysis for GeoTessera embeddings.

Usage:
    python scripts/eda_tessera.py
    python scripts/eda_tessera.py --tessera-dir data/processed/tessera/snapped_to_mask_grid
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import matplotlib.pyplot as plt
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def analyze_single_file(tif_path: Path) -> dict:
    """Extract statistics from a single GeoTIFF file."""
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()  # Shape: (bands, height, width)

            # Handle nodata
            nodata = src.nodata
            if nodata is not None:
                mask = data != nodata
                valid_data = data[mask]
            else:
                valid_data = data.flatten()

            stats = {
                "path": str(tif_path),
                "name": tif_path.stem,
                "bands": src.count,
                "height": src.height,
                "width": src.width,
                "crs": str(src.crs),
                "dtype": str(src.dtypes[0]),
                "bounds": src.bounds,
                "resolution": src.res,
                "nodata": nodata,
                "min": float(np.min(valid_data)) if len(valid_data) > 0 else None,
                "max": float(np.max(valid_data)) if len(valid_data) > 0 else None,
                "mean": float(np.mean(valid_data)) if len(valid_data) > 0 else None,
                "std": float(np.std(valid_data)) if len(valid_data) > 0 else None,
                "nan_count": int(np.isnan(data).sum()),
                "nodata_percentage": float((data == nodata).sum() / data.size * 100) if nodata else 0,
            }

            # Per-band statistics
            band_stats = []
            for b in range(src.count):
                band_data = data[b].flatten()
                if nodata is not None:
                    band_data = band_data[band_data != nodata]
                if len(band_data) > 0:
                    band_stats.append({
                        "band": b + 1,
                        "min": float(np.min(band_data)),
                        "max": float(np.max(band_data)),
                        "mean": float(np.mean(band_data)),
                        "std": float(np.std(band_data)),
                    })
            stats["band_stats"] = band_stats

        return stats
    except RasterioIOError as exc:
        logging.warning("Skipping unreadable file %s (%s)", tif_path, exc)
        return {"path": str(tif_path), "name": tif_path.stem, "error": str(exc)}


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


def generate_coverage_table(file_stats: list[dict], years: list[int]) -> dict:
    """Generate a coverage table showing which masks have data for which years."""
    coverage = defaultdict(dict)
    
    for stat in file_stats:
        mask_id, year = parse_filename(stat["name"])
        if year:
            coverage[mask_id][year] = True
    
    return dict(coverage)


def plot_band_distributions(file_stats: list[dict], out_dir: Path, sample_size: int = 5) -> None:
    """Plot distribution of values across bands for sample files."""
    sample = file_stats[:sample_size]
    
    fig, axes = plt.subplots(len(sample), 1, figsize=(12, 3 * len(sample)))
    if len(sample) == 1:
        axes = [axes]
    
    for ax, stat in zip(axes, sample):
        if stat["band_stats"]:
            bands = [b["band"] for b in stat["band_stats"]]
            means = [b["mean"] for b in stat["band_stats"]]
            stds = [b["std"] for b in stat["band_stats"]]
            
            ax.errorbar(bands, means, yerr=stds, fmt='o-', capsize=3, alpha=0.7)
            ax.set_xlabel("Band")
            ax.set_ylabel("Value")
            ax.set_title(f"{stat['name'][:50]}...")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / "band_distributions.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved band distribution plot: {out_path}")


def plot_coverage_heatmap(coverage: dict, years: list[int], out_dir: Path) -> None:
    """Plot heatmap of coverage across masks and years."""
    masks = sorted(coverage.keys())
    
    if len(masks) == 0:
        logging.warning("No coverage data to plot")
        return
    
    # Create matrix
    matrix = np.zeros((len(masks), len(years)))
    for i, mask in enumerate(masks):
        for j, year in enumerate(years):
            if coverage.get(mask, {}).get(year, False):
                matrix[i, j] = 1
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(masks) * 0.3)))
    im = ax.imshow(matrix, aspect='auto', cmap='Greens', interpolation='nearest')
    
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mask (index)")
    ax.set_title(f"GeoTessera Coverage: {len(masks)} masks × {len(years)} years")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No data', 'Has data'])
    
    plt.tight_layout()
    out_path = out_dir / "coverage_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved coverage heatmap: {out_path}")


def generate_markdown_report(
    file_stats: list[dict],
    coverage: dict,
    years: list[int],
    out_dir: Path,
    masks_dir: Path,
) -> Path:
    """Generate a markdown summary report."""
    report_path = out_dir / "tessera_eda_report.md"
    
    total_masks = len(coverage)
    total_years = len(years)
    total_files = len(file_stats)
    
    # Calculate coverage stats
    coverage_by_year = {y: 0 for y in years}
    for mask_data in coverage.values():
        for year in mask_data:
            if year in coverage_by_year:
                coverage_by_year[year] += 1
    
    # Get sample stats
    if file_stats:
        sample = file_stats[0]
        bands = sample["bands"]
        dtype = sample["dtype"]
        resolution = sample["resolution"]
    else:
        bands = dtype = resolution = "N/A"
    
    report = f"""# GeoTessera EDA Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total masks | {total_masks} |
| Years analyzed | {years[0]}-{years[-1]} ({total_years} years) |
| Total embedding files | {total_files} |
| Embedding bands | {bands} |
| Data type | {dtype} |
| Resolution | {resolution} |

## Coverage by Year

| Year | Masks with coverage | Percentage |
|------|---------------------|------------|
"""
    
    for year in years:
        count = coverage_by_year.get(year, 0)
        pct = (count / total_masks * 100) if total_masks > 0 else 0
        report += f"| {year} | {count} | {pct:.1f}% |\n"
    
    report += f"""
## Data Statistics

Based on {min(len(file_stats), 10)} sample files:

| Statistic | Value |
|-----------|-------|
"""
    
    if file_stats:
        all_mins = [s["min"] for s in file_stats if s["min"] is not None]
        all_maxs = [s["max"] for s in file_stats if s["max"] is not None]
        all_means = [s["mean"] for s in file_stats if s["mean"] is not None]
        
        if all_mins:
            report += f"| Min value | {min(all_mins):.4f} |\n"
            report += f"| Max value | {max(all_maxs):.4f} |\n"
            report += f"| Mean value (avg) | {np.mean(all_means):.4f} |\n"
    
    report += """
## Files

### Coverage Heatmap
![Coverage Heatmap](coverage_heatmap.png)

### Band Distributions
![Band Distributions](band_distributions.png)

## Masks Without Full Coverage

Masks missing data for some years:

| Mask ID | Missing Years |
|---------|---------------|
"""
    
    for mask_id, mask_years in sorted(coverage.items()):
        missing = [y for y in years if y not in mask_years]
        if missing:
            report += f"| {mask_id[:40]}... | {', '.join(map(str, missing))} |\n"
    
    report_path.write_text(report)
    logging.info(f"Saved markdown report: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for GeoTessera embeddings")
    parser.add_argument(
        "--tessera-dir", 
        type=Path, 
        default=Path("data/processed/tessera/snapped_to_mask_grid"),
        help="Directory containing snapped Tessera GeoTIFFs"
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("data/raw/masks"),
        help="Directory containing original masks"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/tessera/eda"),
        help="Output directory for EDA results"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2018-2024",
        help="Year range to analyze"
    )
    args = parser.parse_args()
    
    # Parse years
    start, end = args.years.split("-")
    years = list(range(int(start), int(end) + 1))
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tessera files
    tif_files = sorted(args.tessera_dir.glob("*_snapped.tif"))
    logging.info(f"Found {len(tif_files)} Tessera GeoTIFF files")
    
    if not tif_files:
        logging.error(f"No files found in {args.tessera_dir}")
        return
    
    # Analyze files
    logging.info("Analyzing files...")
    file_stats = []
    error_files = []
    for i, tif_path in enumerate(tif_files):
        if i % 50 == 0:
            logging.info(f"Processing {i+1}/{len(tif_files)}...")
        stats = analyze_single_file(tif_path)
        if "error" in stats:
            error_files.append(stats)
            continue
        file_stats.append(stats)
    
    # Generate coverage table
    coverage = generate_coverage_table(file_stats, years)
    logging.info(f"Found {len(coverage)} unique masks with coverage data")
    if error_files:
        logging.warning(f"Skipped {len(error_files)} unreadable files during EDA")
    
    # Generate plots
    logging.info("Generating plots...")
    plot_band_distributions(file_stats, args.out_dir)
    plot_coverage_heatmap(coverage, years, args.out_dir)
    
    # Generate markdown report
    report_path = generate_markdown_report(
        file_stats, coverage, years, args.out_dir, args.masks_dir
    )
    
    logging.info(f"\nEDA complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()
