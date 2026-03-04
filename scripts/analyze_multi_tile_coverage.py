"""
Analyse the spatial coverage impact of the multi-tile GeoTessera merge fix.

For each mask, computes:
  - How many GeoTessera 0.1-degree tiles it overlaps
  - For multi-tile masks: the maximum fraction of the bbox a single tile covers
    (upper bound on what the old fetch logic could supply)

Outputs (all under data/processed/tessera/multi_tile_analysis/):
  results.csv: one row per mask with geometry and split label
  summary.md: markdown + LaTeX table ready for a research paper
  figures/tile_distribution.png: tile-count histogram and coverage boxplot

Usage:
    python scripts/analyze_multi_tile_coverage.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

MASK_DIR    = ROOT / "data" / "raw" / "masks"
TESSERA_DIR = ROOT / "data" / "processed" / "tessera" / "snapped_to_mask_grid"
SENTINEL_DIR = ROOT / "data" / "raw" / "Sentinel"

TESSERA_YEARS = [2018, 2019, 2020]
TILE_SIZE_DEG = 0.1
OUT_DIR = ROOT / "data" / "processed" / "tessera" / "multi_tile_analysis"


# ── split helpers (inlined from src/data/splits.py to avoid torch dependency) ─

def get_ref_ids_from_directory(directory: Path, pattern: str = "*_RGBNIRRSWIRQ_Mosaic.tif") -> list[str]:
    files = sorted(Path(directory).glob(pattern))
    return [f.stem.replace("_RGBNIRRSWIRQ_Mosaic", "") for f in files]


def get_splits(ref_ids: list[str], random_state: int = 42):
    train_ids, val_test_ids = train_test_split(ref_ids, test_size=0.30, random_state=random_state)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.50, random_state=random_state)
    return train_ids, val_ids, test_ids


# ── geometry helpers (logic mirrors figure.py) ───────────────────────────────

def mask_bounds_wgs84(mask_path: Path) -> tuple[float, float, float, float]:
    """Return mask bounds reprojected to WGS84 degrees."""
    with rasterio.open(mask_path) as src:
        b = src.bounds
        crs = src.crs
    wgs84 = CRS.from_epsg(4326)
    if crs != wgs84:
        return transform_bounds(crs, wgs84, b.left, b.bottom, b.right, b.top)
    return b.left, b.bottom, b.right, b.top


def tiles_for_bbox(left: float, bottom: float, right: float, top: float) -> list[tuple[float, float]]:
    """Return (tile_left, tile_bottom) corners for every 0.1-degree cell intersecting the bbox."""
    col_min = math.floor(left / TILE_SIZE_DEG)
    col_max = math.floor(right / TILE_SIZE_DEG)
    row_min = math.floor(bottom / TILE_SIZE_DEG)
    row_max = math.floor(top / TILE_SIZE_DEG)
    return [
        (c * TILE_SIZE_DEG, r * TILE_SIZE_DEG)
        for c in range(col_min, col_max + 1)
        for r in range(row_min, row_max + 1)
    ]


def tile_coverage_fractions(
    left: float, bottom: float, right: float, top: float,
    tiles: list[tuple[float, float]],
) -> list[float]:
    """For each tile, return the fraction of the mask bbox area it covers.

    Fractions sum to 1.0 for non-degenerate masks (tiles partition the bbox).
    """
    mask_area = (right - left) * (top - bottom)
    fractions = []
    for tx, ty in tiles:
        ix_l = max(left, tx)
        ix_b = max(bottom, ty)
        ix_r = min(right, tx + TILE_SIZE_DEG)
        ix_t = min(top, ty + TILE_SIZE_DEG)
        if ix_r > ix_l and ix_t > ix_b:
            fractions.append((ix_r - ix_l) * (ix_t - ix_b) / mask_area)
        else:
            fractions.append(0.0)
    return fractions


# ── data helpers ─────────────────────────────────────────────────────────────

def has_full_embedding(refid: str) -> bool:
    """True if all three training-year snapped files exist for this mask."""
    return all(
        (TESSERA_DIR / f"{refid}_tessera_{year}_snapped.tif").exists()
        for year in TESSERA_YEARS
    )


def build_split_map() -> dict[str, str]:
    """Map refid -> 'train' | 'val' | 'test'. Returns empty dict if Sentinel unavailable."""
    if not SENTINEL_DIR.exists():
        print(f"[warn] SENTINEL_DIR not found ({SENTINEL_DIR}); split labels will be 'unknown'")
        return {}
    all_ids = get_ref_ids_from_directory(SENTINEL_DIR)
    if not all_ids:
        print(f"[warn] No Sentinel files found in {SENTINEL_DIR}; split labels will be 'unknown'")
        return {}
    train_ids, val_ids, test_ids = get_splits(all_ids)
    split_map = {id_: "train" for id_ in train_ids}
    split_map.update({id_: "val" for id_ in val_ids})
    split_map.update({id_: "test" for id_ in test_ids})
    return split_map


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse_masks() -> list[dict]:
    mask_paths = sorted(MASK_DIR.glob("*_mask.tif"))
    if not mask_paths:
        raise RuntimeError(f"No masks found in {MASK_DIR.resolve()}")

    split_map = build_split_map()
    records = []

    for mp in mask_paths:
        refid = mp.name.removesuffix("_mask.tif")
        left, bottom, right, top = mask_bounds_wgs84(mp)
        tiles = tiles_for_bbox(left, bottom, right, top)
        n_tiles = len(tiles)

        fractions = tile_coverage_fractions(left, bottom, right, top, tiles)
        max_tile_frac = max(fractions) if fractions else 0.0

        records.append({
            "refid": refid,
            "n_tiles": n_tiles,
            "snapped_exists": has_full_embedding(refid),
            "max_tile_fraction": round(max_tile_frac, 4),
            "missing_fraction": round(1.0 - max_tile_frac, 4),
            "split": split_map.get(refid, "unknown"),
        })

    return records


# ── outputs ──────────────────────────────────────────────────────────────────

def write_csv(records: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return path


def write_summary(records: list[dict]) -> Path:
    total = len(records)
    single = [r for r in records if r["n_tiles"] == 1]
    multi  = [r for r in records if r["n_tiles"] > 1]
    multi_with_emb = [r for r in multi if r["snapped_exists"]]

    lines: list[str] = []

    lines.append("# Multi-Tile GeoTessera Coverage Impact\n\n")
    lines.append(f"Training years analysed: {TESSERA_YEARS[0]}–{TESSERA_YEARS[-1]}\n\n")

    lines.append("## Overall summary\n\n")
    lines.append("| Category | Count | % of masks |\n")
    lines.append("|---|---|---|\n")
    lines.append(f"| Total masks | {total} | 100% |\n")
    lines.append(f"| Single-tile (always complete) | {len(single)} | {len(single)/total*100:.1f}% |\n")
    lines.append(f"| Multi-tile (previously incomplete) | {len(multi)} | {len(multi)/total*100:.1f}% |\n")
    lines.append(f"| — with snapped embeddings present | {len(multi_with_emb)} | {len(multi_with_emb)/total*100:.1f}% |\n")
    lines.append("\n")

    if multi:
        missing = [r["missing_fraction"] for r in multi]
        lines.append("## Spatial coverage loss under old logic (multi-tile masks)\n\n")
        lines.append(
            "`missing_fraction` = portion of each mask's bbox that was **never** covered even "
            "in the best case, where one tile covered the maximum possible area.\n\n"
        )
        lines.append(f"| Statistic | Value |\n|---|---|\n")
        lines.append(f"| Mean missing fraction | {np.mean(missing)*100:.1f}% |\n")
        lines.append(f"| Median missing fraction | {np.median(missing)*100:.1f}% |\n")
        lines.append(f"| Min missing fraction | {np.min(missing)*100:.1f}% |\n")
        lines.append(f"| Max missing fraction | {np.max(missing)*100:.1f}% |\n")
        lines.append("\n")

    lines.append("## Breakdown by split\n\n")
    lines.append("| Split | Single-tile | Multi-tile | Multi-tile with embedding | Total |\n")
    lines.append("|---|---|---|---|---|\n")
    for sp in ["train", "val", "test"]:
        sp_recs = [r for r in records if r["split"] == sp]
        sp_single      = sum(1 for r in sp_recs if r["n_tiles"] == 1)
        sp_multi       = sum(1 for r in sp_recs if r["n_tiles"] > 1)
        sp_multi_emb   = sum(1 for r in sp_recs if r["n_tiles"] > 1 and r["snapped_exists"])
        lines.append(f"| {sp} | {sp_single} | {sp_multi} | {sp_multi_emb} | {len(sp_recs)} |\n")
    lines.append("\n")

    lines.append("## LaTeX table\n\n```latex\n")
    lines.append("\\begin{table}[h]\n\\centering\n")
    lines.append(
        "\\caption{GeoTessera tile overlap per HABLOSS mask. "
        "Under the previous fetch logic, masks overlapping multiple tiles received "
        "embeddings covering only a single tile's extent. "
        "The updated logic merges all tiles before snapping.}\n"
    )
    lines.append("\\label{tab:tessera-tile-coverage}\n")
    lines.append("\\begin{tabular}{lrr}\n\\toprule\n")
    lines.append("Category & Count & \\% of masks \\\\\n\\midrule\n")
    lines.append(f"Total masks & {total} & 100\\% \\\\\n")
    lines.append(f"Single-tile (always complete) & {len(single)} & {len(single)/total*100:.1f}\\% \\\\\n")
    lines.append(f"Multi-tile (previously incomplete) & {len(multi)} & {len(multi)/total*100:.1f}\\% \\\\\n")
    if multi:
        missing = [r["missing_fraction"] for r in multi]
        lines.append(
            f"\\quad Mean missing fraction (old logic) & "
            f"\\multicolumn{{2}}{{r}}{{{np.mean(missing)*100:.1f}\\%}} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n```\n")

    path = OUT_DIR / "summary.md"
    path.write_text("".join(lines))
    return path


def make_figure(records: list[dict]) -> Path:
    multi = [r for r in records if r["n_tiles"] > 1]
    max_n = max(r["n_tiles"] for r in records)

    tile_counts = [sum(1 for r in records if r["n_tiles"] == n) for n in range(1, max_n + 1)]
    bar_colors = ["#4878CF" if n == 1 else "#D65F5F" for n in range(1, max_n + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): tile count distribution
    ax = axes[0]
    bars = ax.bar(range(1, max_n + 1), tile_counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, tile_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    str(count), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(1, max_n + 1))
    ax.set_xlabel("Number of GeoTessera tiles overlapping mask bbox")
    ax.set_ylabel("Number of masks")
    ax.set_title("(a)  Tile overlap distribution")
    ax.set_ylim(0, max(tile_counts) * 1.2)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4878CF", label="Single-tile (always correct)"),
        Patch(facecolor="#D65F5F", label="Multi-tile (previously incomplete)"),
    ], fontsize=8, loc="upper right")

    # Panel (b): best-case single-tile coverage fraction for multi-tile masks
    ax = axes[1]
    if multi:
        max_fracs = [r["max_tile_fraction"] for r in multi]
        bp = ax.boxplot(max_fracs, vert=True, patch_artist=True, widths=0.4,
                        boxprops=dict(facecolor="#D65F5F", alpha=0.6),
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))
        ax.axhline(1.0, color="#4878CF", linestyle="--", linewidth=1.5,
                   label="New logic: 100% coverage")
        ax.set_xticks([1])
        ax.set_xticklabels([f"Multi-tile masks\n(n={len(multi)})"])
        ax.set_ylabel("Best-case single-tile coverage (old logic)")
        ax.set_title("(b)  Spatial completeness under old vs. new logic")
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No multi-tile masks", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_title("(b)  Coverage fraction (multi-tile)")

    fig.suptitle(
        f"GeoTessera multi-tile coverage impact  |  {len(records)} masks  |  years {TESSERA_YEARS[0]}–{TESSERA_YEARS[-1]}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "tile_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Analysing mask–tile overlap...\n")
    records = analyse_masks()

    csv_path     = write_csv(records)
    summary_path = write_summary(records)
    fig_path     = make_figure(records)

    print(f"Saved CSV     → {csv_path}")
    print(f"Saved summary → {summary_path}")
    print(f"Saved figure  → {fig_path}")

    total  = len(records)
    single = [r for r in records if r["n_tiles"] == 1]
    multi  = [r for r in records if r["n_tiles"] > 1]

    print(f"\nTotal masks:  {total}")
    print(f"Single-tile:  {len(single)} ({len(single)/total*100:.1f}%)")
    print(f"Multi-tile:   {len(multi)} ({len(multi)/total*100:.1f}%)")
    if multi:
        fracs = [r["missing_fraction"] for r in multi]
        print(f"  Mean missing fraction (old logic): {np.mean(fracs)*100:.1f}%")
        print(f"  Max missing fraction  (old logic): {np.max(fracs)*100:.1f}%")


if __name__ == "__main__":
    main()
