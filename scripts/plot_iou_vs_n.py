"""Plot per-fold test IoU vs number of input years N for the N-slicing experiment.

Fetches per-fold test_iou from WandB (group: UTAE_sentinel_K2_N{n}_slicing_v2)
and saves a line plot to reports/figures/iou_vs_n.pdf and .png.

Usage:
    python scripts/plot_iou_vs_n.py
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".venv/lib/python3.11/site-packages"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "nina_prosjektoppgave"
PROJECT = "data_variasjon_utae"
K = 2
N_VALUES = [1, 2, 3, 4, "all"]
TAG = "slicing_v2"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

OUT_DIR = root / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_data() -> dict[int, dict]:
    """Return {fold: {n: test_iou}} from WandB."""
    api = wandb.Api()
    data: dict[int, dict] = {}
    for n in N_VALUES:
        group = f"UTAE_sentinel_K{K}_N{n}_{TAG}"
        runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})
        for r in runs:
            fold = r.config.get("cv_fold")
            iou = r.summary.get("test_iou")
            if fold is not None and iou is not None:
                data.setdefault(fold, {})[n] = iou
    return data


def plot(data: dict[int, dict]) -> None:
    # Use numeric x-axis: N=all maps to 5 with a custom tick label
    x_numeric = [1, 2, 3, 4, 5]
    x_labels = ["1", "2", "3", "4", "all"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for fold, color, marker in zip(sorted(data.keys()), COLORS, MARKERS):
        ious = [data[fold].get(n, float("nan")) for n in N_VALUES]
        ax.plot(x_numeric, ious, color=color, marker=marker,
                linewidth=1.4, markersize=5, label=f"Fold {fold}")

    mean_ious = [np.nanmean([data[f].get(n, float("nan")) for f in data]) for n in N_VALUES]
    ax.plot(x_numeric, mean_ious, color="black", linewidth=2.2,
            linestyle="--", marker="o", markersize=5, label="Mean", zorder=5)

    ax.set_xlabel("Input years $N$", fontsize=11)
    ax.set_ylabel("Test IoU", fontsize=11)
    ax.set_xticks(x_numeric)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "iou_vs_n.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "iou_vs_n.png", dpi=200, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / 'iou_vs_n.pdf'} and .png")


if __name__ == "__main__":
    data = fetch_data()
    for fold in sorted(data):
        row = "  ".join(f"N={n}: {data[fold].get(n, float('nan')):.4f}" for n in N_VALUES)
        print(f"Fold {fold}:  {row}")
    plot(data)
