"""
Verify that Tessera embeddings align with masks by plotting them side by side.

Usage:
    python scripts/verify_tessera_alignment.py
    python scripts/verify_tessera_alignment.py --n 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import rasterio
import numpy as np


def verify_alignment(mask_path: Path, tessera_path: Path) -> dict:
    """Compare metadata between mask and tessera file."""
    with rasterio.open(mask_path) as msrc:
        mask_meta = {
            "crs": msrc.crs,
            "bounds": msrc.bounds,
            "shape": (msrc.height, msrc.width),
            "transform": msrc.transform,
        }
        mask_data = msrc.read(1)

    with rasterio.open(tessera_path) as tsrc:
        tessera_meta = {
            "crs": tsrc.crs,
            "bounds": tsrc.bounds,
            "shape": (tsrc.height, tsrc.width),
            "transform": tsrc.transform,
            "bands": tsrc.count,
        }
        tessera_data = tsrc.read(1)  # First band for visualization

    matches = {
        "crs": mask_meta["crs"] == tessera_meta["crs"],
        "shape": mask_meta["shape"] == tessera_meta["shape"],
        "bounds": mask_meta["bounds"] == tessera_meta["bounds"],
        "transform": mask_meta["transform"] == tessera_meta["transform"],
    }

    return {
        "mask_meta": mask_meta,
        "tessera_meta": tessera_meta,
        "matches": matches,
        "mask_data": mask_data,
        "tessera_data": tessera_data,
    }


def plot_comparison(mask_path: Path, tessera_path: Path, save_path: Path | None = None) -> None:
    """Plot mask and tessera side by side."""
    result = verify_alignment(mask_path, tessera_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mask
    axes[0].imshow(result["mask_data"], cmap="viridis")
    axes[0].set_title(f"Mask\n{mask_path.name}")
    axes[0].axis("off")
    
    # Tessera (first band)
    axes[1].imshow(result["tessera_data"], cmap="viridis")
    axes[1].set_title(f"Tessera (band 1)\n{tessera_path.name}")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(result["mask_data"], cmap="Reds", alpha=0.5)
    axes[2].imshow(result["tessera_data"], cmap="Blues", alpha=0.5)
    axes[2].set_title("Overlay (red=mask, blue=tessera)")
    axes[2].axis("off")
    
    # Add metadata comparison as text
    matches = result["matches"]
    match_text = "\n".join([f"{k}: {'✓' if v else '✗'}" for k, v in matches.items()])
    fig.text(0.02, 0.02, f"Alignment check:\n{match_text}", fontsize=10, family="monospace")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Tessera alignment with masks")
    parser.add_argument("--masks-dir", type=Path, default=Path("data/raw/masks"))
    parser.add_argument("--tessera-dir", type=Path, default=Path("data/processed/tessera/snapped_to_mask_grid"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tessera/verification"))
    parser.add_argument("--n", type=int, default=3, help="Number of samples to check")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    mask_paths = sorted(args.masks_dir.glob("*_mask.tif"))[:args.n]
    
    all_match = True
    for mask_path in mask_paths:
        refid = mask_path.name.removesuffix("_mask.tif")
        tessera_path = args.tessera_dir / f"{refid}_tessera_{args.year}_snapped.tif"
        
        if not tessera_path.exists():
            print(f"[SKIP] No tessera file for {refid}")
            continue
        
        result = verify_alignment(mask_path, tessera_path)
        matches = result["matches"]
        
        print(f"\n{refid}:")
        print(f"  CRS match:       {matches['crs']}")
        print(f"  Shape match:     {matches['shape']} ({result['mask_meta']['shape']})")
        print(f"  Bounds match:    {matches['bounds']}")
        print(f"  Transform match: {matches['transform']}")
        
        if not all(matches.values()):
            all_match = False
        
        # Save plot
        plot_comparison(
            mask_path, 
            tessera_path, 
            save_path=args.out_dir / f"{refid}_verification.png"
        )
    
    print("\n" + "=" * 50)
    if all_match:
        print("✓ All checked files are properly aligned!")
    else:
        print("✗ Some files have alignment issues - check the output above")


if __name__ == "__main__":
    main()