"""
Aggregate 5-fold CV results from WandB.

Supports two experiment types via --vary:

  K experiment (prediction horizon):   vary K, fix N
    python scripts/aggregate_cv_results.py --vary K --k_values 1 2 3 --input_years 4

  N experiment (input years):          vary N, fix K
    python scripts/aggregate_cv_results.py --vary N --n_values 1 2 3 4 --k 2

Prints a table of mean ± std for each metric across folds, one row per varied value.
Add --detail for per-fold breakdown.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import wandb


WANDB_PROJECT = "data_variasjon_utae"
WANDB_ENTITY = "nina_prosjektoppgave"

TEST_METRICS = ["test_f1", "test_precision", "test_recall", "test_iou", "test_accuracy"]
METRIC_LABELS = {
    "test_f1":        "F1",
    "test_precision": "Precision",
    "test_recall":    "Recall",
    "test_iou":       "IoU",
    "test_accuracy":  "Accuracy",
}
N_FOLDS = 5


def get_group_name(dataset: str, k: int, input_years) -> str:
    n_label = str(input_years) if input_years is not None else "all"
    return f"UTAE_{dataset}_K{k}_N{n_label}"


def fetch_runs_for_group(api: wandb.Api, group: str) -> list:
    """Return all runs belonging to a WandB group, sorted by name."""
    runs = api.runs(
        path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": group},
    )
    return list(runs)


def extract_test_metrics(run) -> dict | None:
    """
    Pull the final logged value for each test metric from a run's history.
    Returns None if no test metrics were logged (run may still be running or crashed).
    """
    summary = run.summary._json_dict
    result = {}
    for metric in TEST_METRICS:
        val = summary.get(metric)
        if val is None:
            return None  # incomplete run
        result[metric] = float(val)
    return result


def aggregate_group(runs: list) -> dict:
    """
    Given runs for a single group (one per fold), compute mean and std
    for each test metric.

    Returns:
        {
          "n_runs": int,
          "fold_metrics": [{"fold": int, "test_f1": float, ...}, ...],
          "mean": {"test_f1": float, ...},
          "std":  {"test_f1": float, ...},
        }
    """
    fold_metrics = []
    missing_folds = []

    for run in runs:
        metrics = extract_test_metrics(run)
        if metrics is None:
            missing_folds.append(run.name)
            continue
        # Try to parse fold index from run name (e.g. "UTAE_sentinel_K2_N4_fold3")
        fold_idx = None
        for part in run.name.split("_"):
            if part.startswith("fold"):
                try:
                    fold_idx = int(part[4:])
                except ValueError:
                    pass
        fold_metrics.append({"fold": fold_idx, "run_name": run.name, **metrics})

    values = defaultdict(list)
    for fm in fold_metrics:
        for metric in TEST_METRICS:
            values[metric].append(fm[metric])

    mean = {m: float(np.mean(v)) for m, v in values.items()}
    std  = {m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for m, v in values.items()}

    return {
        "n_runs": len(fold_metrics),
        "missing": missing_folds,
        "fold_metrics": fold_metrics,
        "mean": mean,
        "std": std,
    }


def print_table(results: dict, row_label: str):
    """
    Print a formatted comparison table.
    results: {row_value: aggregate_group_output}
    row_label: "K" or "N"
    """
    col_w = 18

    header_metrics = [METRIC_LABELS[m] for m in TEST_METRICS]
    header = f"{row_label:>4}  {'Folds':>5}  " + "  ".join(f"{h:^{col_w}}" for h in header_metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for val in sorted(results.keys()):
        r = results[val]
        mean = r["mean"]
        std  = r["std"]
        cells = []
        for m in TEST_METRICS:
            cell = f"{mean[m]:.4f} ± {std[m]:.4f}"
            cells.append(f"{cell:^{col_w}}")
        folds_str = f"{r['n_runs']}/{N_FOLDS}"
        display_val = str(val) if val is not None else "all"
        print(f"{row_label}={display_val:<3}  {folds_str:>5}  " + "  ".join(cells))
        if r["missing"]:
            print(f"         ^ missing runs: {r['missing']}")

    print("=" * len(header))
    print()


def print_per_fold_detail(label: str, val, agg: dict):
    """Print per-fold breakdown for a single row (K or N value)."""
    display_val = str(val) if val is not None else "all"
    print(f"\n--- Per-fold detail for {label}={display_val} ---")
    header = f"  {'Run':<40}" + "".join(f"  {METRIC_LABELS[m]:>10}" for m in TEST_METRICS)
    print(header)
    for fm in sorted(agg["fold_metrics"], key=lambda x: x.get("fold") or 0):
        row = f"  {fm['run_name']:<40}" + "".join(f"  {fm[m]:>10.4f}" for m in TEST_METRICS)
        print(row)
    mean = agg["mean"]
    std  = agg["std"]
    print("  " + "-" * (len(header) - 2))
    mean_row = f"  {'mean':<40}" + "".join(f"  {mean[m]:>10.4f}" for m in TEST_METRICS)
    std_row  = f"  {'std':<40}" + "".join(f"  {std[m]:>10.4f}" for m in TEST_METRICS)
    print(mean_row)
    print(std_row)


def empty_result():
    return {
        "n_runs": 0, "missing": [], "fold_metrics": [],
        "mean": {m: float("nan") for m in TEST_METRICS},
        "std":  {m: float("nan") for m in TEST_METRICS},
    }


def fetch_and_aggregate(api, dataset, k, input_years, detail, row_label, row_val):
    group = get_group_name(dataset, k, input_years)
    print(f"  Fetching group: {group} ...", end=" ", flush=True)
    runs = fetch_runs_for_group(api, group)
    print(f"{len(runs)} run(s) found")

    if not runs:
        print(f"  WARNING: No runs found for '{group}'. Check jobs completed and naming matches.")
        return empty_result()

    agg = aggregate_group(runs)
    if detail:
        print_per_fold_detail(row_label, row_val, agg)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Aggregate 5-fold CV WandB results")
    parser.add_argument("--vary", choices=["K", "N"], default="K",
                        help="Which dimension to vary across table rows (default: K)")
    parser.add_argument("--dataset", default="sentinel",
                        help="Dataset name in run group (default: sentinel)")
    # K-experiment args
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 3],
                        help="[--vary K] Prediction horizons to compare (default: 1 2 3)")
    parser.add_argument("--input_years", type=int, default=4,
                        help="[--vary K] Fixed N input years (default: 4; 0 = all)")
    # N-experiment args
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="[--vary N] Input year counts to compare (e.g. 1 2 3 4; 0 = all)")
    parser.add_argument("--k", type=int, default=2,
                        help="[--vary N] Fixed prediction horizon K (default: 2)")
    # Shared
    parser.add_argument("--detail", action="store_true",
                        help="Also print per-fold breakdown for each row")
    args = parser.parse_args()

    print(f"\nConnecting to WandB project: {WANDB_ENTITY}/{WANDB_PROJECT}")
    api = wandb.Api()

    results = {}

    if args.vary == "K":
        input_years = None if args.input_years == 0 else args.input_years
        for k in args.k_values:
            results[k] = fetch_and_aggregate(api, args.dataset, k, input_years,
                                             args.detail, "K", k)
        print(f"\nSummary: {args.dataset.upper()}, N={input_years or 'all'} input years, varying K")
        print_table(results, row_label="K")

    else:  # vary N
        n_values = args.n_values if args.n_values is not None else [1, 2, 3, 4]
        for n_raw in n_values:
            input_years = None if n_raw == 0 else n_raw
            results[input_years] = fetch_and_aggregate(api, args.dataset, args.k, input_years,
                                                       args.detail, "N", input_years)
        print(f"\nSummary: {args.dataset.upper()}, K={args.k}, varying N")
        print_table(results, row_label="N")

    print("Note: For land-take detection (rare positive class), prioritize F1, Precision,")
    print("      and Recall. IoU is also informative. Accuracy is misleading on imbalanced data.\n")


if __name__ == "__main__":
    main()
