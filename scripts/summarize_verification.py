#!/usr/bin/env python3
"""
Summarize verification results from the shared CSV produced by the SLURM array job.

Usage:
    python scripts/summarize_verification.py
    python scripts/summarize_verification.py --results-file path/to/results.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tessera verification results")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("data/processed/tessera/verification/results.csv"),
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        return

    with open(args.results_file, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print("Results file is empty.")
        return

    status_counts = Counter(r["status"] for r in rows)

    print("=" * 60)
    print("TESSERA VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total entries:  {len(rows)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:>12s}:  {count}")
    print()

    # List failures
    failures = [r for r in rows if r["status"] == "FAIL"]
    if failures:
        print(f"FAILURES ({len(failures)}):")
        print("-" * 60)
        for r in failures:
            mismatches = [
                k for k in ("crs", "shape", "bounds", "transform")
                if r.get(k) == "False"
            ]
            print(f"  {r['refid']}  — mismatches: {', '.join(mismatches)}")
            print(f"      mask shape: {r.get('mask_shape', '?')}  "
                  f"tessera shape: {r.get('tessera_shape', '?')}  "
                  f"bands: {r.get('tessera_bands', '?')}")
        print()

    # List errors
    errors = [r for r in rows if r["status"].startswith("ERROR")]
    if errors:
        print(f"ERRORS ({len(errors)}):")
        print("-" * 60)
        for r in errors:
            print(f"  {r['refid']}  — {r['status']}")
        print()

    # List skips
    skips = [r for r in rows if r["status"] == "SKIP"]
    if skips:
        print(f"SKIPPED ({len(skips)}):")
        print("-" * 60)
        for r in skips:
            print(f"  {r['refid']}")
        print()

    if not failures and not errors:
        print("All verified masks are properly aligned!")
    print("=" * 60)


if __name__ == "__main__":
    main()
