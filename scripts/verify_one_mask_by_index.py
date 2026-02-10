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
import csv
import fcntl
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


_CSV_HEADER = ["refid", "status", "crs", "shape", "bounds", "transform", "mask_shape", "tessera_shape", "tessera_bands"]


def _append_result_row(results_file: Path, row: dict) -> None:
    """Append a single result row to a shared CSV file (file-lock safe for SLURM)."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_file.exists() or results_file.stat().st_size == 0

    with open(results_file, "a", newline="") as fh:
        # Use an exclusive lock so parallel SLURM tasks don't interleave writes
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verification for a single mask by index")
    parser.add_argument("--masks-dir", type=Path, default=Path("data/raw/masks"))
    parser.add_argument("--tessera-dir", type=Path, default=Path("data/processed/tessera/snapped_to_mask_grid"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tessera/verification"))
    parser.add_argument("--results-file", type=Path,
                        default=Path("data/processed/tessera/verification/results.csv"),
                        help="Shared CSV file where each task appends its result row")
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
        _append_result_row(args.results_file, {"refid": refid, "status": "SKIP"})
        return

    # Load the existing verify module from the scripts directory
    repo_scripts = Path(__file__).parent
    verify_mod_path = repo_scripts / "verify_tessera_alignment.py"
    verify = load_verify_module(verify_mod_path)

    # Call verify + plot functions from the loaded module
    try:
        result = verify.verify_alignment(mask_path, tessera_path)
        matches = result["matches"]

        # Print textual verification results (captured by SLURM log files)
        print(f"\n{refid}:")
        print(f"  CRS match:       {matches['crs']}")
        print(f"  Shape match:     {matches['shape']} ({result['mask_meta']['shape']})")
        print(f"  Bounds match:    {matches['bounds']}")
        print(f"  Transform match: {matches['transform']}")

        args.out_dir.mkdir(parents=True, exist_ok=True)

        # Reuse the result so we don't read the files a second time
        verify.plot_comparison(
            mask_path,
            tessera_path,
            save_path=args.out_dir / f"{refid}_verification.png",
            result=result,
        )

        status = "OK" if all(matches.values()) else "FAIL"

        # Persist structured result to shared CSV
        _append_result_row(args.results_file, {
            "refid": refid,
            "status": status,
            "crs": matches["crs"],
            "shape": matches["shape"],
            "bounds": matches["bounds"],
            "transform": matches["transform"],
            "mask_shape": result["mask_meta"]["shape"],
            "tessera_shape": result["tessera_meta"]["shape"],
            "tessera_bands": result["tessera_meta"]["bands"],
        })

        if status == "FAIL":
            print(f"[FAIL] Alignment mismatch for {refid}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"[OK] {refid} aligned correctly")
    except Exception as exc:
        print(f"Error while verifying {refid}: {exc}", file=sys.stderr)
        _append_result_row(args.results_file, {"refid": refid, "status": f"ERROR: {exc}"})
        raise


if __name__ == "__main__":
    main()
