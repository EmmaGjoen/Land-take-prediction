"""
Generate a figure showing how a HABLOSS land-take mask aligns with the
GeoTessera 0.1° embedding grid.

The figure has three panels:
  (a) The binary change mask with its geographic extent.
  (b) The GeoTessera tile grid surrounding the mask, with the mask bbox
      overlaid to show single-tile vs. multi-tile overlap.
  (c) The snapped (reprojected) Tessera embedding (band-1 shown) next to
      the mask, demonstrating pixel-level alignment after snapping.

Usage:
    python figure.py                          # auto-pick first available mask
    python figure.py --refid <REFID>          # specific mask
    python figure.py --refid <REFID> --year 2024
    python figure.py --out figures/alignment.pdf
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from geotessera import GeoTessera


# ── paths ────────────────────────────────────────────────────────────────
MASK_DIR = Path("data/raw/masks")
SNAPPED_DIR = Path("data/processed/tessera/snapped_to_mask_grid")

# GeoTessera tiles live on a 0.1° global grid
TILE_SIZE_DEG = 0.1


# ── helpers ──────────────────────────────────────────────────────────────

def mask_bounds(mask_path: Path):
    """Return (left, bottom, right, top) in WGS84 degrees, plus the native CRS."""
    with rasterio.open(mask_path) as src:
        b = src.bounds
        crs = src.crs
    wgs84 = CRS.from_epsg(4326)
    if crs != wgs84:
        left, bottom, right, top = transform_bounds(crs, wgs84, b.left, b.bottom, b.right, b.top)
    else:
        left, bottom, right, top = b.left, b.bottom, b.right, b.top
    return left, bottom, right, top, crs


def tiles_for_bbox(left, bottom, right, top, tile_size=TILE_SIZE_DEG):
    """Return a list of (tile_left, tile_bottom) corners for every 0.1°
    grid cell that intersects the given bounding box."""
    col_min = math.floor(left / tile_size)
    col_max = math.floor(right / tile_size)
    row_min = math.floor(bottom / tile_size)
    row_max = math.floor(top / tile_size)
    tiles = []
    for c in range(col_min, col_max + 1):
        for r in range(row_min, row_max + 1):
            tiles.append((c * tile_size, r * tile_size))
    return tiles


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(refid: str, year: int, out_path: Path | None = None):
    mask_path = MASK_DIR / f"{refid}_mask.tif"
    snapped_path = SNAPPED_DIR / f"{refid}_tessera_{year}_snapped.tif"

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        mask_transform = src.transform
        mask_h, mask_w = src.height, src.width
        native_bounds = src.bounds
        crs = src.crs

    wgs84 = CRS.from_epsg(4326)
    if crs != wgs84:
        left, bottom, right, top = transform_bounds(crs, wgs84, *native_bounds)
    else:
        left, bottom, right, top = native_bounds

    grid_tiles = tiles_for_bbox(left, bottom, right, top)
    n_tiles = len(grid_tiles)

    has_snapped = snapped_path.exists()
    if has_snapped:
        with rasterio.open(snapped_path) as src:
            emb_band1 = src.read(1)
            emb_bands = src.count
    else:
        emb_band1 = None
        emb_bands = "?"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    ax = axes[0]
    ax.imshow(mask_data, cmap="Greys_r", interpolation="nearest",
              extent=[left, right, bottom, top])
    ax.set_title("(a)  Land-take mask", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.ticklabel_format(useOffset=False)
    res_x = abs(mask_transform.a)
    ax.text(0.02, 0.02,
            f"{mask_h}×{mask_w} px\n~{res_x:.4f}°/px",
            transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax = axes[1]
    # Pad the view to show surrounding grid context
    pad = TILE_SIZE_DEG * 1.5
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(bottom - pad, top + pad)

    for (tx, ty) in tiles_for_bbox(left - pad, bottom - pad,
                                    right + pad, top + pad):
        rect = mpatches.Rectangle(
            (tx, ty), TILE_SIZE_DEG, TILE_SIZE_DEG,
            linewidth=0.6, edgecolor="#888888", facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    colors_overlap = plt.cm.Set2(np.linspace(0, 1, max(n_tiles, 2)))
    for i, (tx, ty) in enumerate(grid_tiles):
        rect = mpatches.Rectangle(
            (tx, ty), TILE_SIZE_DEG, TILE_SIZE_DEG,
            linewidth=1.4, edgecolor="black",
            facecolor=colors_overlap[i % len(colors_overlap)],
            alpha=0.35, zorder=2,
        )
        ax.add_patch(rect)
        cx = tx + TILE_SIZE_DEG / 2
        cy = ty + TILE_SIZE_DEG / 2
        ax.text(cx, cy, f"tile {i+1}", fontsize=7, ha="center", va="center",
                fontweight="bold", zorder=3)

    mask_rect = mpatches.Rectangle(
        (left, bottom), right - left, top - bottom,
        linewidth=2.2, edgecolor="#d62728", facecolor="none",
        zorder=4, label="Mask extent",
    )
    ax.add_patch(mask_rect)

    ax.set_title(f"(b)  GeoTessera 0.1° grid\n({n_tiles} tile{'s' if n_tiles != 1 else ''} overlap)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False)

    ax = axes[2]
    if has_snapped:
        ax.imshow(emb_band1, cmap="viridis", interpolation="nearest",
                  extent=[left, right, bottom, top])
        xs = np.linspace(left, right, mask_w)
        ys = np.linspace(top, bottom, mask_h)
        xx, yy = np.meshgrid(xs, ys)
        ax.contour(xx, yy, mask_data, levels=[0.5], colors=["red"], linewidths=1.0)
        ax.set_title(f"(c)  Snapped embedding (band 1 / {emb_bands})\n"
                     f"red contour = mask boundary",
                     fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Snapped file\nnot available",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title("(c)  Snapped embedding", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.ticklabel_format(useOffset=False)

    fig.suptitle(
        f"Spatial alignment: HABLOSS mask → GeoTessera embedding\n"
        f"Ref ID: {refid}   |   Year: {year}   |   "
        f"Overlapping tiles: {n_tiles}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)
    return n_tiles


# ── find interesting examples ────────────────────────────────────────────

def scan_for_multi_tile_masks(year: int = 2024, max_scan: int | None = None):
    """Scan all masks and report how many GeoTessera tiles each overlaps."""
    mask_paths = sorted(MASK_DIR.glob("*_mask.tif"))
    if max_scan:
        mask_paths = mask_paths[:max_scan]

    results = []
    for mp in mask_paths:
        refid = mp.name.removesuffix("_mask.tif")
        left, bottom, right, top, _ = mask_bounds(mp)
        tiles = tiles_for_bbox(left, bottom, right, top)
        results.append((refid, len(tiles), (left, bottom, right, top)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Figure: mask-to-GeoTessera tile alignment"
    )
    parser.add_argument("--refid", type=str, default=None,
                        help="Reference ID of the mask to visualise")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (e.g. figures/alignment.pdf)")
    parser.add_argument("--scan", action="store_true",
                        help="Scan all masks and report tile overlap counts")
    args = parser.parse_args()

    if args.scan:
        print("Scanning all masks for GeoTessera tile overlap...\n")
        results = scan_for_multi_tile_masks(year=args.year)
        print(f"{'REFID':<60s}  {'TILES':>5s}   BBOX")
        print("-" * 110)
        for refid, n, bbox in results:
            bbox_str = f"({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})"
            marker = " ← MULTI" if n > 1 else ""
            print(f"{refid:<60s}  {n:>5d}   {bbox_str}{marker}")

        multi = [r for r in results if r[1] > 1]
        print(f"\n{len(multi)} / {len(results)} masks overlap >1 tile")

        if multi and args.out:
            out_dir = Path(args.out)
            for refid, n, _ in multi[:5]:
                make_figure(refid, args.year,
                            out_path=out_dir / f"{refid}_alignment.png")
        return

    if args.refid is None:
        results = scan_for_multi_tile_masks(year=args.year)
        multi = [r for r in results if r[1] > 1]
        if multi:
            chosen = multi[0][0]
            print(f"Auto-selected multi-tile mask: {chosen} "
                  f"({multi[0][1]} tiles)")
        else:
            chosen = results[0][0]
            print(f"No multi-tile masks found; using: {chosen}")
        args.refid = chosen

    out = Path(args.out) if args.out else None
    n = make_figure(args.refid, args.year, out_path=out)
    print(f"Done — mask overlaps {n} GeoTessera tile(s)")


if __name__ == "__main__":
    main()
