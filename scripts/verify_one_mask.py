#!/usr/bin/env python3
"""
Wrapper to run verify_tessera_alignment on a single mask selected by 1-based index.

This safely loads the existing `verify_tessera_alignment.py` module from the
`scripts/` directory and calls its functions. It forces a non-interactive
matplotlib backend so it is safe to run on a headless cluster.

Usage:
    python scripts/verify_one_mask_by_index.py --index 5 --year 2024
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys


def load_verify_module(module_path: Path):
    """Load the verify_tessera_alignment.py module from a path and return it."""
    # Ensure matplotlib uses a non-interactive backend before the module imports it
    os.environ.setdefault("MPLBACKEND", "Agg")

    spec = importlib.util.spec_from_file_location("verify_tessera_alignment", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verification for a single mask by index")
    parser.add_argument("--masks-dir", type=Path, default=Path("data/raw/masks"))
    parser.add_argument("--tessera-dir", type=Path, default=Path("data/processed/tessera/snapped_to_mask_grid"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tessera/verification"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--index", type=int, required=True, help="1-based index into sorted mask list")
    args = parser.parse_args()

    mask_paths = sorted(args.masks_dir.glob("*_mask.tif"))
    if not mask_paths:
        print("No mask files found in", args.masks_dir, file=sys.stderr)
        sys.exit(2)

    idx = args.index - 1
    if idx < 0 or idx >= len(mask_paths):
        print(f"Index {args.index} out of range (1..{len(mask_paths)})", file=sys.stderr)
        sys.exit(3)

    mask_path = mask_paths[idx]
    refid = mask_path.name.removesuffix("_mask.tif")
    tessera_path = args.tessera_dir / f"{refid}_tessera_{args.year}_snapped.tif"

    if not tessera_path.exists():
        print(f"[SKIP] No tessera file for {refid}")
        return

    # Load the existing verify module from the scripts directory
    repo_scripts = Path(__file__).parent
    verify_mod_path = repo_scripts / "verify_tessera_alignment.py"
    verify = load_verify_module(verify_mod_path)

    # Call verify + plot functions from the loaded module
    try:
        # run verification and save a plot (we don't need the returned dict here)
        verify.verify_alignment(mask_path, tessera_path)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        verify.plot_comparison(mask_path, tessera_path, save_path=args.out_dir / f"{refid}_verification.png")
    except Exception as exc:
        print(f"Error while verifying {refid}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
