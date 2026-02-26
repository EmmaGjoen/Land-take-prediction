"""
FCEF + GeoTessera Early Fusion Training Script for Land-Take Prediction

Fuses Sentinel time series with GeoTessera embeddings via channel concatenation.
"""

import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb

root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.config import SENTINEL_DIR, TESSERA_DIR
from src.data.sentinel_dataset import SentinelDataset
from src.data.tessera_dataset import TesseraDataset
from src.data.wrap_datasets import FusedSentinelTesseraDataset
from src.data.splits import get_splits, get_ref_ids_from_directory
from src.data.transform import (
    compute_normalization_stats,
    ComposeTS,
    NormalizeBy,
    CenterCropTS,
    Normalize,
    RandomFlipTS,
    RandomRotate90TS,
)
from src.models.external.torchrs_fc_cd import FCEF
from src.utils.visualization import log_masks
from src.utils.metrics import compute_confusion_binary, compute_metrics_from_confusion


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Random seed
    "random_seed": 42,

    # Data splits
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # Model
    "architecture": "FCEF",
    "num_classes": 2,

    # Data
    "temporal_mode": "first_half",  # keep first half of the temporal axis
    "img_frequency": "annual",
    "chip_size": 64,
    # Only first 3 years are used in 'first_half' mode
    "tessera_years": [2018, 2019, 2020],

    # Training
    "epochs": 50,
    "learning_rate": 1e-3,
    "batch_size": 4,
    "augment_train": True,

    # Normalization
    "normalization": "scale_10000_plus_standardize",
    "num_samples_for_stats": 2000,

    # DataLoader
    "num_workers": 4,

    # WandB
    "wandb_project": "data_variasjon_fcef",
    "wandb_entity": "nina_prosjektoppgave",
}


# ============================================================================
# SETUP
# ============================================================================

def set_random_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")


def get_device() -> torch.device:
    """Get device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _is_valid_tif(path: Path, min_size_bytes: int = 1024) -> bool:
    """Check that a .tif file exists, is large enough, and can be opened."""
    if not path.exists():
        return False
    if path.stat().st_size < min_size_bytes:
        return False
    try:
        import rasterio
        with rasterio.open(path) as src:
            _ = src.count  # quick sanity read
        return True
    except Exception:
        return False


def filter_ids_by_tessera_availability(
    ref_ids: list[str],
    years: list[int],
    split_name: str = "",
) -> list[str]:
    """Keep only ref_ids that have all required Tessera embedding files.

    Checks that each file exists, has a reasonable size, and is a valid
    GeoTIFF (can be opened by rasterio). Logs every skipped tile.
    """
    valid, skipped_missing, skipped_corrupt = [], [], []
    for fid in ref_ids:
        missing_years = []
        corrupt_years = []
        for y in years:
            p = TESSERA_DIR / f"{fid}_tessera_{y}_snapped.tif"
            if not p.exists():
                missing_years.append(y)
            elif not _is_valid_tif(p):
                corrupt_years.append(y)
        if missing_years:
            skipped_missing.append((fid, missing_years))
        elif corrupt_years:
            skipped_corrupt.append((fid, corrupt_years))
        else:
            valid.append(fid)

    if skipped_missing:
        tag = f" [{split_name}]" if split_name else ""
        print(f"\n⚠ Skipped {len(skipped_missing)} tile(s){tag} — missing Tessera embeddings:")
        for fid, yrs in skipped_missing:
            print(f"  • {fid}  (missing years: {yrs})")
    if skipped_corrupt:
        tag = f" [{split_name}]" if split_name else ""
        print(f"\n⚠ Skipped {len(skipped_corrupt)} tile(s){tag} — CORRUPT Tessera files:")
        for fid, yrs in skipped_corrupt:
            paths = [TESSERA_DIR / f"{fid}_tessera_{y}_snapped.tif" for y in yrs]
            sizes = [p.stat().st_size if p.exists() else 0 for p in paths]
            print(f"  • {fid}  (corrupt years: {yrs}, sizes: {sizes} bytes)")
        print("  → Re-run fetch_tessera_for_masks.py with --force to regenerate them.")
    return valid


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_random_seeds(CONFIG["random_seed"])
    device = get_device()

    # ------------------------------------------------------------------
    # Data splits
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATA SPLITS")
    print("=" * 80)
    all_ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
    print(f"Total reference IDs found: {len(all_ref_ids)}")

    train_ref_ids, val_ref_ids, test_ref_ids = get_splits(
        all_ref_ids,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"],
        random_state=CONFIG["random_seed"],
    )

    print(f"Train tiles: {len(train_ref_ids)} (~{100 * len(train_ref_ids) / len(all_ref_ids):.0f}%)")
    print(f"Val tiles:   {len(val_ref_ids)} (~{100 * len(val_ref_ids) / len(all_ref_ids):.0f}%)")
    print(f"Test tiles:  {len(test_ref_ids)} (~{100 * len(test_ref_ids) / len(all_ref_ids):.0f}%)")
    print(f"Using shared splits (random_state={CONFIG['random_seed']})")

    # Filter to tiles that have all required Tessera embeddings
    tessera_years = CONFIG["tessera_years"]
    train_ref_ids = filter_ids_by_tessera_availability(train_ref_ids, tessera_years, "train")
    val_ref_ids   = filter_ids_by_tessera_availability(val_ref_ids,   tessera_years, "val")
    test_ref_ids  = filter_ids_by_tessera_availability(test_ref_ids,  tessera_years, "test")

    total_after = len(train_ref_ids) + len(val_ref_ids) + len(test_ref_ids)
    print(f"\nAfter Tessera filtering: {total_after}/{len(all_ref_ids)} tiles remain")
    print(f"  Train: {len(train_ref_ids)}  Val: {len(val_ref_ids)}  Test: {len(test_ref_ids)}")

    if total_after == 0:
        raise RuntimeError("No tiles remain after filtering — check TESSERA_DIR and file names.")

    # ------------------------------------------------------------------
    # Normalization statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("NORMALIZATION")
    print("=" * 80)

    # Sentinel stats
    temp_transform_sen = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
        NormalizeBy(10000.0),
    ])
    temp_ds_sen = SentinelDataset(
        train_ref_ids,
        slice_mode=CONFIG["temporal_mode"],
        frequency=CONFIG["img_frequency"],
        transform=temp_transform_sen,
    )
    print("Estimating Sentinel per-channel mean/std from training data...")
    mean_sen, std_sen = compute_normalization_stats(
        temp_ds_sen, num_samples=CONFIG["num_samples_for_stats"]
    )
    print(f"Sentinel stats: {len(mean_sen)} channels")
    print(f"  mean (first 5): {[f'{m:.4f}' for m in mean_sen[:5]]}")
    print(f"  std  (first 5): {[f'{s:.4f}' for s in std_sen[:5]]}")

    # Tessera stats
    temp_transform_tess = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
    ])
    # no slice_mode for Tessera. The year list already selects the
    # temporal subset that matches Sentinel's first-half slice.
    temp_ds_tess = TesseraDataset(
        train_ref_ids,
        slice_mode=None,
        frequency=CONFIG["img_frequency"],
        transform=temp_transform_tess,
        years=CONFIG["tessera_years"],
    )
    print("Estimating Tessera per-channel mean/std from training data...")
    mean_tess, std_tess = compute_normalization_stats(
        temp_ds_tess, num_samples=CONFIG["num_samples_for_stats"]
    )
    print(f"Tessera stats: {len(mean_tess)} channels")
    print(f"  mean (first 5): {[f'{m:.4f}' for m in mean_tess[:5]]}")
    print(f"  std  (first 5): {[f'{s:.4f}' for s in std_tess[:5]]}")

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    if CONFIG["augment_train"]:
        train_transform_sen = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),
            RandomFlipTS(p_horizontal=0.5, p_vertical=0.5),
            RandomRotate90TS(),
            NormalizeBy(10000.0),
            Normalize(mean_sen, std_sen),
        ])
        train_transform_tess = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),
            RandomFlipTS(p_horizontal=0.5, p_vertical=0.5),
            RandomRotate90TS(),
            Normalize(mean_tess, std_tess),
        ])
    else:
        train_transform_sen = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),
            NormalizeBy(10000.0),
            Normalize(mean_sen, std_sen),
        ])
        train_transform_tess = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),
            Normalize(mean_tess, std_tess),
        ])

    val_transform_sen = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
        NormalizeBy(10000.0),
        Normalize(mean_sen, std_sen),
    ])
    val_transform_tess = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
        Normalize(mean_tess, std_tess),
    ])

    test_transform_sen = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
        NormalizeBy(10000.0),
        Normalize(mean_sen, std_sen),
    ])
    test_transform_tess = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
        Normalize(mean_tess, std_tess),
    ])

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATASETS")
    print("=" * 80)

    train_ds_sen = SentinelDataset(train_ref_ids, slice_mode=CONFIG["temporal_mode"], frequency=CONFIG["img_frequency"], transform=train_transform_sen)
    val_ds_sen   = SentinelDataset(val_ref_ids,   slice_mode=CONFIG["temporal_mode"], frequency=CONFIG["img_frequency"], transform=val_transform_sen)
    test_ds_sen  = SentinelDataset(test_ref_ids,  slice_mode=CONFIG["temporal_mode"], frequency=CONFIG["img_frequency"], transform=test_transform_sen)

    tess_kwargs = dict(years=CONFIG["tessera_years"], frequency=CONFIG["img_frequency"])
    # No slice_mode for Tessera: year list already defines the temporal range.
    train_ds_tess = TesseraDataset(train_ref_ids, slice_mode=None, transform=train_transform_tess, **tess_kwargs)
    val_ds_tess   = TesseraDataset(val_ref_ids,   slice_mode=None, transform=val_transform_tess,   **tess_kwargs)
    test_ds_tess  = TesseraDataset(test_ref_ids,  slice_mode=None, transform=test_transform_tess,  **tess_kwargs)

    train_ds = FusedSentinelTesseraDataset(train_ds_sen, train_ds_tess)
    val_ds   = FusedSentinelTesseraDataset(val_ds_sen,   val_ds_tess)
    test_ds  = FusedSentinelTesseraDataset(test_ds_sen,  test_ds_tess)

    print(f"Datasets created for {CONFIG['chip_size']}×{CONFIG['chip_size']} chips")
    print(f"Train chips: {len(train_ds)} (from {len(train_ref_ids)} REFIDs). with flips + rotations")
    print(f"Val chips:   {len(val_ds)} (from {len(val_ref_ids)} REFIDs). no augmentation")
    print(f"Test chips:  {len(test_ds)} (from {len(test_ref_ids)} REFIDs). no augmentation")
    print(f"Augmentation enabled: {CONFIG['augment_train']}")

    # Temporal alignment sanity check: verify Sentinel and Tessera have same T
    print("\n--- Temporal alignment check ---")
    sample_sen, _ = train_ds_sen[0]
    sample_tess, _ = train_ds_tess[0]
    T_sen, C_sen = sample_sen.shape[0], sample_sen.shape[1]
    T_tess, C_tess = sample_tess.shape[0], sample_tess.shape[1]
    print(f"  Sentinel:  T={T_sen}, C={C_sen}  (shape {tuple(sample_sen.shape)})")
    print(f"  Tessera:   T={T_tess}, C={C_tess}  (shape {tuple(sample_tess.shape)})")
    print(f"  Tessera years: {CONFIG['tessera_years']}")
    if T_sen != T_tess:
        raise RuntimeError(
            f"Temporal mismatch! Sentinel has {T_sen} timesteps but Tessera has "
            f"{T_tess}. Ensure tessera_years covers the same number of years as "
            f"the Sentinel data ({T_sen} annual steps)."
        )
    print(f"  ✓ Temporal axes match: T={T_sen}")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def worker_init_fn(worker_id):
        worker_seed = CONFIG["random_seed"] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(CONFIG["random_seed"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    print(f"Dataloaders created with reproducible shuffling (seed={CONFIG['random_seed']})")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL")
    print("=" * 80)
    sample_x, _ = next(iter(train_loader))
    _, T, C, H, W = sample_x.shape

    model = FCEF(channels=C, t=T, num_classes=CONFIG["num_classes"]).to(device)

    print("FCEF model created")
    print(f"  Channels (Sentinel + Tessera): {C}")
    print(f"  Timesteps: {T}")
    print(f"  Classes: {CONFIG['num_classes']}")
    print(f"  Input shape: (B, {T}, {C}, {H}, {W})")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("WANDB INITIALIZATION")
    print("=" * 80)
    run = wandb.init(
        entity=CONFIG["wandb_entity"],
        project=CONFIG["wandb_project"],
        name=f"FCEF_{train_ds.DATASET_NAME}_{CONFIG['img_frequency']}_chip{CONFIG['chip_size']}_t{T}",
        config={
            "learning_rate": CONFIG["learning_rate"],
            "architecture": CONFIG["architecture"],
            "dataset": train_ds.DATASET_NAME,
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "chip_size": CONFIG["chip_size"],
            "augment_train": CONFIG["augment_train"],
            "augmentation": "flips_rotations" if CONFIG["augment_train"] else "none",
            "temporal_mode": CONFIG["temporal_mode"],
            "num_timesteps": T,
            "num_channels": C,
            "tessera_years": CONFIG["tessera_years"],
            "train_chips": len(train_ds),
            "val_chips": len(val_ds),
            "test_chips": len(test_ds),
            "tiles_total": len(all_ref_ids),
            "tiles_after_tessera_filter": total_after,
            "tiles_skipped_tessera": len(all_ref_ids) - total_after,
            "normalization": CONFIG["normalization"],
            "random_seed": CONFIG["random_seed"],
            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],
            "test_ratio": CONFIG["test_ratio"],
        },
    )
    print("✓ WandB initialized")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        for x, mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}"):
            x = x.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        sum_tp = sum_fp = sum_tn = sum_fn = 0
        with torch.no_grad():
            for x, mask in val_loader:
                x = x.to(device)
                mask = mask.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, mask)
                val_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                tp, fp, tn, fn = compute_confusion_binary(pred, mask, positive_class=1)
                sum_tp += tp
                sum_fp += fp
                sum_tn += tn
                sum_fn += fn

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

        run.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "IoU": val_metrics["iou"],
            "F1": val_metrics["f1"],
            "Precision": val_metrics["precision"],
            "Recall": val_metrics["recall"],
            "Accuracy": val_metrics["accuracy"],
        })

        print(
            f"Epoch {epoch + 1}/{CONFIG['epochs']}: "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} | "
            f"IoU={val_metrics['iou']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"Prec={val_metrics['precision']:.4f} "
            f"Rec={val_metrics['recall']:.4f} "
            f"Acc={val_metrics['accuracy']:.4f}"
        )

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    model.eval()
    test_loss = 0.0
    sum_tp = sum_fp = sum_tn = sum_fn = 0

    with torch.no_grad():
        for x, mask in test_loader:
            x = x.to(device)
            mask = mask.to(device)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, mask)
            test_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            tp, fp, tn, fn = compute_confusion_binary(pred, mask, positive_class=1)
            sum_tp += tp
            sum_fp += fp
            sum_tn += tn
            sum_fn += fn

    avg_test_loss = test_loss / len(test_loader)
    test_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

    print("Test Set Results:")
    print(f"  Loss:      {avg_test_loss:.4f}")
    print(f"  IoU:       {test_metrics['iou']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")

    run.log({
        "test_loss": avg_test_loss,
        "test_iou": test_metrics["iou"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_accuracy": test_metrics["accuracy"],
    })

    print("\nLogging test set masks...")
    log_masks(model, test_loader, device, step=CONFIG["epochs"], name_prefix="test", max_batches=10)

    run.finish()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("Final Validation Metrics:")
    print(f"  Loss: {avg_val_loss:.4f}")
    print(f"  IoU:  {val_metrics['iou']:.4f}")
    print(f"  F1:   {val_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
