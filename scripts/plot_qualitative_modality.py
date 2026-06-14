"""Qualitative modality comparison: AlphaEarth, TESSERA, Sentinel-2.

Grid: 2 rows (dense, sparse tiles) x 4 columns (GT, AlphaEarth prob, TESSERA prob, Sentinel-2 prob).
K=2, N=all. Saves to reports/figures/qualitative_modality.pdf and .png.
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
from src.data.alphaearth_dataset import AlphaEarthDataset
from src.data.tessera_dataset import TesseraDataset
from src.data.splits import load_folds, get_fold_splits
from src.data.file_helpers import get_ref_ids_from_directory
from src.data.transform import ComposeTS, NormalizeBy, CenterCropTS, Normalize, compute_normalization_stats
from src.models.external.utae import UTAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TILES = {
    "dense": ("a-2-52025858362194_53-72050616794933", 2),  # fold 2, 47% land-take (start 2018)
    "sparse": ("a24-34041727116894_57-09974939549575", 1), # fold 1, 7% fragmented (start 2018)
}
K = 2
CHIP_SIZE = 64

OUT_DIR = root / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = ["alphaearth", "tessera", "sentinel"]
MOD_LABELS = {"alphaearth": "AlphaEarth", "tessera": "TESSERA", "sentinel": "Sentinel-2"}


def get_sentinel_norm_stats(fold: int) -> tuple[list, list]:
    all_ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
    all_ref_ids = [fid for fid in all_ref_ids if list(MASK_DIR.glob(f"{fid}*.tif"))]
    fold_assignments = load_folds()
    fold_assignments = {r: f for r, f in fold_assignments.items() if r in set(all_ref_ids)}
    train_ids, _, _ = get_fold_splits(fold_assignments, fold)
    ds = SentinelDataset(train_ids, transform=ComposeTS([NormalizeBy(10000.0)]),
                         calibrate_mode=True)
    return compute_normalization_stats(ds, num_samples=500)


def get_sample(refid: str, modality: str, sentinel_mean: list, sentinel_std: list):
    if modality == "sentinel":
        transform = ComposeTS([CenterCropTS(CHIP_SIZE), NormalizeBy(10000.0),
                               Normalize(sentinel_mean, sentinel_std)])
        ds = SentinelDataset([refid], transform=transform, prediction_horizon=K, input_years=None)
    elif modality == "alphaearth":
        ds = AlphaEarthDataset([refid], transform=ComposeTS([CenterCropTS(CHIP_SIZE)]),
                               prediction_horizon=K, input_years=None)
    elif modality == "tessera":
        ds = TesseraDataset([refid], transform=ComposeTS([CenterCropTS(CHIP_SIZE)]),
                            prediction_horizon=K, input_years=None)
    assert len(ds) == 1, f"Tile {refid} was excluded for modality={modality}"
    return ds[0]


def load_model(modality: str, fold: int, input_dim: int) -> torch.nn.Module:
    ckpt = root / "checkpoints" / f"utae_{modality}_K{K}_Nall_fold{fold}" / "best_model.pth"
    model = UTAE(input_dim=input_dim, out_conv=[32, 2], pad_value=0.0)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Computing Sentinel normalization stats...")
sentinel_norm: dict[int, tuple] = {}
for _, (_, fold) in TILES.items():
    if fold not in sentinel_norm:
        print(f"  fold {fold}...")
        sentinel_norm[fold] = get_sentinel_norm_stats(fold)

results = {}

for tile_label, (refid, fold) in TILES.items():
    mean, std = sentinel_norm[fold]
    for modality in MODALITIES:
        print(f"  {tile_label}  {modality}...")
        img, mask, positions = get_sample(refid, modality, mean, std)
        input_dim = img.shape[1]
        model = load_model(modality, fold, input_dim)

        with torch.no_grad():
            logits = model(img.unsqueeze(0), batch_positions=positions.unsqueeze(0))
        prob = F.softmax(logits, dim=1)[0, 1].numpy()

        results[(tile_label, modality)] = (mask.numpy(), prob)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

tile_labels = list(TILES.keys())
tile_display = {"dense": "Dense land take sample", "sparse": "Sparse land take sample"}
col_titles = ["Ground truth"] + [MOD_LABELS[m] for m in MODALITIES]
n_rows = len(tile_labels)

fig, axes = plt.subplots(n_rows, 4, figsize=(7.5, n_rows * 2.0))
plt.subplots_adjust(hspace=0.08, wspace=0.04)

for row, tile_label in enumerate(tile_labels):
    gt = results[(tile_label, MODALITIES[0])][0]
    axes[row, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_ylabel(tile_display[tile_label], fontsize=8, labelpad=4)

    for col, modality in enumerate(MODALITIES, start=1):
        _, prob = results[(tile_label, modality)]
        axes[row, col].imshow(prob, cmap="viridis", vmin=0, vmax=1)

    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=9, pad=4)

cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, cax=cax, label="P(land take)")

plt.savefig(OUT_DIR / "qualitative_modality.pdf", bbox_inches="tight", dpi=150)
plt.savefig(OUT_DIR / "qualitative_modality.png", bbox_inches="tight", dpi=150)
print(f"Saved to {OUT_DIR / 'qualitative_modality.pdf'}")
