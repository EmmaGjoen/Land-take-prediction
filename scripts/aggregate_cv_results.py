"""
Aggregate 5-fold CV results from WandB.

Primary metric: **pooled IoU**, confusion matrices are summed across all folds
before computing metrics, so every tile contributes equally regardless of fold
size.  This matches the evaluation protocol of the U-TAE / PASTIS benchmark
(Garnot & Landrieu, ICCV 2021; github.com/VSainteuf/utae-paps).

Secondary metric: macro mean ± std across fold scores (diagnostic only).

Supports two experiment types via --vary:

  K experiment (prediction horizon):
    python scripts/aggregate_cv_results.py --vary K --k_values 1 2 3 4 5 --dataset sentinel

  N experiment (input years):
    python scripts/aggregate_cv_results.py --vary N --n_values 2 3 4 0 --k 2 --dataset sentinel

  Modality comparison (fixed K=2, N=all, one row per dataset):
    python scripts/aggregate_cv_results.py --modality --datasets sentinel tessera alphaearth

Add --detail for per-fold breakdown.

Note: pooled metrics require runs to have logged test_tp/fp/tn/fn (available
after the confusion-matrix logging update).  Older runs fall back to macro
mean ± std with a warning.
"""

import argparse
import os
import sys
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import wandb


# ── stdout tee ────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


@contextmanager
def tee_stdout(filepath):
    tee = _Tee(filepath)
    sys.stdout = tee
    try:
        yield filepath
    finally:
        sys.stdout = tee._stdout
        tee.close()


# ── constants ─────────────────────────────────────────────────────────────────

WANDB_PROJECT = "data_variasjon_utae"
WANDB_ENTITY  = "nina_prosjektoppgave"

SCALAR_METRICS = ["test_f1", "test_precision", "test_recall", "test_iou", "test_accuracy"]
CM_KEYS        = ["test_tp", "test_fp", "test_tn", "test_fn"]
METRIC_LABELS  = {
    "test_f1":        "F1",
    "test_precision": "Precision",
    "test_recall":    "Recall",
    "test_iou":       "IoU",
    "test_accuracy":  "Accuracy",
}
N_FOLDS = 5


# ── WandB helpers ─────────────────────────────────────────────────────────────

def get_group_name(dataset: str, k: int, input_years, tag: str = "") -> str:
    n_label = str(input_years) if input_years is not None else "all"
    base = f"UTAE_{dataset}_K{k}_N{n_label}"
    return f"{base}_{tag}" if tag else base


def fetch_runs_for_group(api: wandb.Api, group: str) -> list:
    runs = api.runs(
        path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": group},
    )
    return list(runs)


def extract_run_data(run) -> dict | None:
    """
    Pull scalar metrics and, if available, raw confusion matrix totals.
    Returns None if any scalar metric is missing (run incomplete/crashed).
    """
    summary = run.summary._json_dict

    result = {}
    for metric in SCALAR_METRICS:
        val = summary.get(metric)
        if val is None:
            return None
        result[metric] = float(val)

    # Confusion matrix totals (optional and only present in runs after the
    # pooled-aggregation update)
    for key in CM_KEYS:
        val = summary.get(key)
        result[key] = int(val) if val is not None else None

    return result


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_group(runs: list) -> dict:
    """
    Compute both pooled and macro-averaged metrics for a group of fold runs.

    Returns
    -------
    {
      "n_runs":       int,
      "missing":      [run_name, ...],
      "fold_data":    [{fold, run_name, ...metrics, ...cm_keys}, ...],
      "has_cm":       bool,           # True if all folds logged CM values
      "pooled":       {metric: float} | None,
      "mean":         {metric: float},
      "std":          {metric: float},
    }
    """
    fold_data = []
    missing = []

    for run in runs:
        data = extract_run_data(run)
        if data is None:
            missing.append(run.name)
            continue
        fold_idx = None
        for part in run.name.split("_"):
            if part.startswith("fold"):
                try:
                    fold_idx = int(part[4:])
                except ValueError:
                    pass
        fold_data.append({"fold": fold_idx, "run_name": run.name, **data})

    # ── pooled (primary) ──────────────────────────────────────────────────
    has_cm = all(fd["test_tp"] is not None for fd in fold_data) and len(fold_data) > 0
    pooled = None
    if has_cm:
        tp = sum(fd["test_tp"] for fd in fold_data)
        fp = sum(fd["test_fp"] for fd in fold_data)
        tn = sum(fd["test_tn"] for fd in fold_data)
        fn = sum(fd["test_fn"] for fd in fold_data)
        eps = 1e-8
        pooled = {
            "test_iou":       tp / (tp + fp + fn + eps),
            "test_f1":        2*tp / (2*tp + fp + fn + eps),
            "test_precision": tp / (tp + fp + eps),
            "test_recall":    tp / (tp + fn + eps),
            "test_accuracy":  (tp + tn) / (tp + tn + fp + fn + eps),
        }

    # ── macro mean ± std (secondary) ─────────────────────────────────────
    values = defaultdict(list)
    for fd in fold_data:
        for m in SCALAR_METRICS:
            values[m].append(fd[m])

    mean = {m: float(np.mean(v)) for m, v in values.items()}
    std  = {m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            for m, v in values.items()}

    return {
        "n_runs":    len(fold_data),
        "missing":   missing,
        "fold_data": fold_data,
        "has_cm":    has_cm,
        "pooled":    pooled,
        "mean":      mean,
        "std":       std,
    }


# ── printing ──────────────────────────────────────────────────────────────────

def _header_row(row_label: str, col_w: int) -> str:
    header_metrics = [METRIC_LABELS[m] for m in SCALAR_METRICS]
    return (
        f"{row_label:>10}  {'Folds':>5}  "
        + "  ".join(f"{h:^{col_w}}" for h in header_metrics)
    )


def print_pooled_table(results: dict, row_label: str):
    col_w = 10
    header = _header_row(row_label, col_w)
    sep = "=" * len(header)
    print(f"\nPooled metrics")
    print(sep)
    print(header)
    print(sep)
    for val in sorted(results.keys(), key=lambda x: (x is None, x)):
        r = results[val]
        display_val = str(val) if val is not None else "all"
        folds_str = f"{r['n_runs']}/{N_FOLDS}"
        if r["pooled"] is not None:
            cells = [f"{r['pooled'][m]:^{col_w}.4f}" for m in SCALAR_METRICS]
        else:
            cells = [f"{'n/a':^{col_w}}"] * len(SCALAR_METRICS)
            print(f"  [{display_val}] WARNING: no confusion matrix data — re-run with updated training script")
        print(f"{row_label}={display_val:<8}  {folds_str:>5}  " + "  ".join(cells))
        if r["missing"]:
            print(f"           ^ missing runs: {r['missing']}")
    print(sep)


def print_macro_table(results: dict, row_label: str):
    col_w = 18
    header = _header_row(row_label, col_w)
    sep = "=" * len(header)
    print(f"\nMacro mean ± std (secondary / diagnostic — unequal fold sizes affect reliability)")
    print(sep)
    print(header)
    print(sep)
    for val in sorted(results.keys(), key=lambda x: (x is None, x)):
        r = results[val]
        display_val = str(val) if val is not None else "all"
        folds_str = f"{r['n_runs']}/{N_FOLDS}"
        cells = [
            f"{r['mean'][m]:.4f}±{r['std'][m]:.4f}".center(col_w)
            for m in SCALAR_METRICS
        ]
        print(f"{row_label}={display_val:<8}  {folds_str:>5}  " + "  ".join(cells))
        if r["missing"]:
            print(f"           ^ missing runs: {r['missing']}")
    print(sep)


def print_per_fold_detail(label: str, val, agg: dict):
    display_val = str(val) if val is not None else "all"
    print(f"\n--- Per-fold detail: {label}={display_val} ---")
    header = f"  {'Run':<45}" + "".join(f"  {METRIC_LABELS[m]:>10}" for m in SCALAR_METRICS)
    print(header)
    for fd in sorted(agg["fold_data"], key=lambda x: x.get("fold") or 0):
        row = f"  {fd['run_name']:<45}" + "".join(f"  {fd[m]:>10.4f}" for m in SCALAR_METRICS)
        print(row)
    print("  " + "-" * (len(header) - 2))
    mean_row = f"  {'macro mean':<45}" + "".join(f"  {agg['mean'][m]:>10.4f}" for m in SCALAR_METRICS)
    std_row  = f"  {'macro std':<45}"  + "".join(f"  {agg['std'][m]:>10.4f}"  for m in SCALAR_METRICS)
    print(mean_row)
    print(std_row)
    if agg["pooled"]:
        pool_row = f"  {'pooled':<45}" + "".join(f"  {agg['pooled'][m]:>10.4f}" for m in SCALAR_METRICS)
        print(pool_row)


def empty_result():
    nan = {m: float("nan") for m in SCALAR_METRICS}
    return {"n_runs": 0, "missing": [], "fold_data": [], "has_cm": False,
            "pooled": None, "mean": nan, "std": nan}


# ── fetch + aggregate ─────────────────────────────────────────────────────────

def fetch_and_aggregate(api, dataset, k, input_years, detail, row_label, row_val, tag=""):
    group = get_group_name(dataset, k, input_years, tag)
    print(f"  Fetching group: {group} ...", end=" ", flush=True)
    runs = fetch_runs_for_group(api, group)
    print(f"{len(runs)} run(s) found")

    if not runs:
        print(f"  WARNING: No runs found for '{group}'.")
        return empty_result()

    agg = aggregate_group(runs)
    if detail:
        print_per_fold_detail(row_label, row_val, agg)
    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate 5-fold CV WandB results")
    parser.add_argument("--vary", choices=["K", "N"], default="K",
                        help="Which dimension to vary (default: K)")
    parser.add_argument("--dataset", default="sentinel",
                        help="Dataset name in WandB group (default: sentinel)")
    parser.add_argument("--tag", default="",
                        help="Experiment tag appended to group name (e.g. 'slicing' or 'modality')")
    # K-experiment
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--input_years", type=int, default=0,
                        help="Fixed N input years (0 = all, default: 0)")
    # N-experiment
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="Input year counts to compare (0 = all)")
    parser.add_argument("--k", type=int, default=2,
                        help="Fixed prediction horizon K (default: 2)")
    # Modality comparison
    parser.add_argument("--modality", action="store_true",
                        help="Compare modalities: one row per dataset (K=2, N=all)")
    parser.add_argument("--datasets", nargs="+", default=["sentinel", "tessera", "alphaearth"],
                        help="Datasets to compare in --modality mode")
    # Shared
    parser.add_argument("--detail", action="store_true",
                        help="Print per-fold breakdown for each row")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        if args.modality:
            tag_str = f"_{args.tag}" if args.tag else ""
            args.output = f"results/cv_modality{tag_str}.txt"
        elif args.vary == "K":
            n_label = "all" if args.input_years == 0 else args.input_years
            tag_str = f"_{args.tag}" if args.tag else ""
            args.output = f"results/cv_K_{args.dataset}_N{n_label}{tag_str}.txt"
        else:
            tag_str = f"_{args.tag}" if args.tag else ""
            args.output = f"results/cv_N_{args.dataset}_K{args.k}{tag_str}.txt"

    with tee_stdout(args.output) as out_path:
        print(f"\nConnecting to WandB: {WANDB_ENTITY}/{WANDB_PROJECT}")
        api = wandb.Api()
        results = {}

        if args.modality:
            for ds in args.datasets:
                results[ds] = fetch_and_aggregate(
                    api, ds, k=2, input_years=None,
                    detail=args.detail, row_label="dataset", row_val=ds,
                    tag=args.tag,
                )
            print(f"\nModality comparison: K=2, N=all")
            print_pooled_table(results, row_label="dataset")
            print_macro_table(results, row_label="dataset")

        elif args.vary == "K":
            input_years = None if args.input_years == 0 else args.input_years
            for k in args.k_values:
                results[k] = fetch_and_aggregate(
                    api, args.dataset, k, input_years,
                    args.detail, "K", k, tag=args.tag,
                )
            print(f"\nSlicing experiment: {args.dataset.upper()}, N={input_years or 'all'}, varying K")
            print_pooled_table(results, row_label="K")
            print_macro_table(results, row_label="K")

        else:  # vary N
            n_values = args.n_values if args.n_values is not None else [2, 3, 4, 0]
            for n_raw in n_values:
                input_years = None if n_raw == 0 else n_raw
                results[input_years] = fetch_and_aggregate(
                    api, args.dataset, args.k, input_years,
                    args.detail, "N", input_years, tag=args.tag,
                )
            print(f"\nSlicing experiment: {args.dataset.upper()}, K={args.k}, varying N")
            print_pooled_table(results, row_label="N")
            print_macro_table(results, row_label="N")

        print("\nNote: for land-take detection (rare positive class), IoU and F1 are")
        print("      the most informative metrics. Accuracy is misleading on imbalanced data.\n")
        print("Reference: Garnot & Landrieu (ICCV 2021) — pooled confusion matrices")
        print("           across folds: github.com/VSainteuf/utae-paps\n")

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
