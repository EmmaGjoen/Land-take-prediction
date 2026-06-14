"""Qualitative prediction maps for N=1,3,all on one easy and one hard tile.

Grid: 6 rows (3 N values x 2 tiles) x 4 columns (context composite, GT, prob, pred).
The input column shows a mean RGB composite over all timesteps visible to the model,
so it changes with N (N=1 shows only the start year. N=all shows the full history).
Saves to reports/figures/qualitative_n_slicing.pdf and .png.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".venv/lib/python3.11/site-packages"))
sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SENTINEL_DIR, MASK_DIR
from src.data.sentinel_dataset import SentinelDataset
from src.data.splits import load_folds, get_fold_splits
from src.data.file_helpers import get_ref_ids_from_directory
from src.data.transform import ComposeTS, NormalizeBy, CenterCropTS, Normalize, compute_normalization_stats
from src.models.external.utae import UTAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TILES = {
    "dense": ("a4-63523450914143_51-75228829204871", 0),   # fold 0, 53% land take
    "sparse": ("a24-34041727116894_57-09974939549575", 1), # fold 1, 7% fragmented
}
K = 2
# N=None means "all available years" in SentinelDataset
N_VALUES = [1, 3, None]
N_LABELS = ["1", "3", "all"]
CHIP_SIZE = 64

OUT_DIR = root / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_norm_stats(fold: int) -> tuple[list, list]:
    all_ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
    all_ref_ids = [fid for fid in all_ref_ids if list(MASK_DIR.glob(f"{fid}*.tif"))]
    fold_assignments = load_folds()
    fold_assignments = {r: f for r, f in fold_assignments.items() if r in set(all_ref_ids)}
    train_ids, _, _ = get_fold_splits(fold_assignments, fold)
    ds = SentinelDataset(train_ids, transform=ComposeTS([NormalizeBy(10000.0)]),
                         calibrate_mode=True)
    return compute_normalization_stats(ds, num_samples=500)


def load_model(n: int | None, fold: int, input_dim: int) -> torch.nn.Module:
    n_label = n if n is not None else "all"
    ckpt = root / "checkpoints" / f"utae_sentinel_K{K}_N{n_label}_fold{fold}" / "best_model.pth"
    model = UTAE(input_dim=input_dim, out_conv=[32, 2], pad_value=0.0)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def get_sample(refid: str, n: int | None, mean: list, std: list):
    transform = ComposeTS([CenterCropTS(CHIP_SIZE), NormalizeBy(10000.0), Normalize(mean, std)])
    ds = SentinelDataset([refid], transform=transform, prediction_horizon=K, input_years=n)
    assert len(ds) == 1, f"Tile {refid} was excluded for K={K}, N={n}"
    return ds[0]


def frame_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a single (C, H, W) frame to a display-ready RGB with percentile stretch."""
    rgb = np.stack([frame[2], frame[1], frame[0]], axis=-1)
    for i in range(3):
        lo, hi = np.percentile(rgb[..., i], 2), np.percentile(rgb[..., i], 98)
        rgb[..., i] = np.clip((rgb[..., i] - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def get_frames(img: torch.Tensor, positions: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (start_frame_rgb, last_frame_rgb) for the visible timesteps."""
    visible = (positions > 0).nonzero(as_tuple=True)[0]
    start_rgb = frame_rgb(img[visible[0]].numpy())
    last_rgb  = frame_rgb(img[visible[-1]].numpy())
    return start_rgb, last_rgb


def count_visible(positions: torch.Tensor) -> int:
    # Each year has 2 timesteps (bi-quarterly); divide to get years
    return int((positions > 0).sum().item()) // 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Computing normalization stats...")
norm_stats: dict[int, tuple] = {}
for _, (_, fold) in TILES.items():
    if fold not in norm_stats:
        print(f"  fold {fold}...")
        norm_stats[fold] = get_norm_stats(fold)

results = {}

for tile_label, (refid, fold) in TILES.items():
    mean, std = norm_stats[fold]
    sample_img, _, _ = get_sample(refid, N_VALUES[0], mean, std)
    input_dim = sample_img.shape[1]

    for n, n_label in zip(N_VALUES, N_LABELS):
        print(f"  {tile_label}  N={n_label}...")
        img, mask, positions = get_sample(refid, n, mean, std)
        n_yrs = count_visible(positions)
        model = load_model(n, fold, input_dim)

        with torch.no_grad():
            logits = model(img.unsqueeze(0), batch_positions=positions.unsqueeze(0))
        prob = F.softmax(logits, dim=1)[0, 1].numpy()
        pred = (prob > 0.5).astype(np.uint8)

        start_rgb, last_rgb = get_frames(img, positions)
        results[(tile_label, n_label)] = (start_rgb, last_rgb, mask.numpy(), prob, pred, n_yrs)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

col_titles = ["First input image", "Last input image", "Ground truth", "Pred. probability", "Prediction"]
tile_display = {"dense": "Dense land take sample", "sparse": "Sparse land take sample"}
n_rows = len(N_VALUES)

for tile_label in list(TILES.keys()):
    fig, axes = plt.subplots(n_rows, 5, figsize=(9.0, n_rows * 1.8))
    plt.subplots_adjust(hspace=0.08, wspace=0.04)

    for row, n_label in enumerate(N_LABELS):
        start_rgb, last_rgb, gt, prob, pred, n_yrs = results[(tile_label, n_label)]

        axes[row, 0].imshow(start_rgb)
        axes[row, 1].imshow(last_rgb)
        axes[row, 2].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(prob, cmap="viridis", vmin=0, vmax=1)
        axes[row, 4].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"$N={n_label}$", fontsize=8, labelpad=4)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, pad=4)

    fig.suptitle(tile_display[tile_label], fontsize=10, y=1.02)

    cax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="P(land take)")

    plt.savefig(OUT_DIR / f"qualitative_n_slicing_{tile_label}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT_DIR / f"qualitative_n_slicing_{tile_label}.png", bbox_inches="tight", dpi=150)
    print(f"Saved to {OUT_DIR / f'qualitative_n_slicing_{tile_label}.pdf'}")
    plt.close()
