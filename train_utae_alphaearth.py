"""U-TAE trained on AlphaEarth embeddings only (no Sentinel imagery).

This script trains the U-TAE temporal attention model using 64-band-per-year
AlphaEarth embeddings as the sole input modality. It is the counterpart to
``train_utae_tessera.py`` and is intended for a direct modality-comparison
experiment in the master's thesis.

Usage::

    python train_utae_alphaearth.py [--prediction_horizon K] [--input_years N]

The ``--prediction_horizon`` (K) and ``--input_years`` (N) arguments mirror
those in ``train_utae_tessera.py`` so that identical temporal settings can be
applied across both modalities.
"""

import json
import os
import subprocess

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.config import ALPHAEARTH_DIR, MASK_DIR, ALPHAEARTH_YEARS
from src.data.alphaearth_dataset import AlphaEarthDataset
from src.data.splits import get_splits, load_folds, get_fold_splits
from src.utils.training import set_random_seeds, get_device
from src.data.transform import (
    ComposeTS,
    RandomCropTS,
    CenterCropTS,
    RandomFlipTS,
    RandomRotate90TS,
)
from src.models.external.utae import UTAE
from src.utils.visualization import log_masks
from src.utils.metrics import compute_confusion_binary, compute_metrics_from_confusion
from src.utils.focal_loss import FocalLoss


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Reproducibility
    "random_seed": 42,

    # Data splits
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # Model
    "architecture": "u-tae-alphaearth",
    "num_classes": 2,

    # Data
    "chip_size": 64,
    "prediction_horizon": 2,    # K: zero timesteps from (end_year - K) onwards
    "input_years": None,        # N: keep start_year + latest N-1 years; None = all

    # Loss
    "focal_gamma": 2.0,         # Focal loss focusing parameter (Lin et al., 2017)

    # Training
    "epochs": 75,
    "learning_rate": 1e-3,
    "lr_patience": 7,           # Epochs without val_loss improvement before LR halves
    "lr_factor": 0.5,
    "early_stopping_patience": 15,  # Epochs without val IoU improvement before stopping
    "batch_size": 4,
    "augment_train": True,      # Random flips + 90 degree rotations

    "normalization": None,

    # DataLoader
    "num_workers": 4,

    # WandB
    "wandb_project": "data_variasjon_utae",
    "wandb_entity": "nina_prosjektoppgave",
}


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train U-TAE on AlphaEarth embeddings (no Sentinel data)."
    )
    parser.add_argument(
        "--prediction_horizon", type=int, default=None,
        help="Override CONFIG prediction_horizon (K)",
    )
    parser.add_argument(
        "--input_years", type=int, default=None,
        help="Override CONFIG input_years (N): number of years to show before cutoff",
    )
    parser.add_argument(
        "--fold", type=int, default=None, choices=range(5), metavar="0-4",
        help=(
            "Geographic fold to use as test set (0-4).  "
            "Requires src/data/geographic_folds.csv; run scripts/create_folds.py first.  "
            "Val = (fold+1) %% 5; train = remaining folds.  "
            "If omitted, falls back to the legacy random 70/15/15 split."
        ),
    )
    parser.add_argument(
        "--folds-file", type=Path, default=None,
        help="Path to folds CSV (default: src/data/geographic_folds.csv).",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional tag appended to WandB group name (e.g. 'slicing' or 'modality').",
    )
    args = parser.parse_args()

    if args.prediction_horizon is not None:
        CONFIG["prediction_horizon"] = args.prediction_horizon
        print(f"prediction_horizon overridden via CLI: K={CONFIG['prediction_horizon']}")
    if args.input_years is not None:
        CONFIG["input_years"] = args.input_years
        print(f"input_years overridden via CLI: N={CONFIG['input_years']}")
    CONFIG["fold"] = args.fold

    torch.use_deterministic_algorithms(True, warn_only=True)
    set_random_seeds(CONFIG["random_seed"])
    device = get_device()

    # ------------------------------------------------------------------ #
    # CHECKPOINT SETUP                                                     #
    # ------------------------------------------------------------------ #
    n_label = CONFIG["input_years"] if CONFIG["input_years"] is not None else "all"
    fold_label = f"_fold{args.fold}" if args.fold is not None else ""
    run_name = f"utae_alphaearth_K{CONFIG['prediction_horizon']}_N{n_label}{fold_label}"
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_hash = "unknown"
    print(f"Git commit: {git_hash}")

    config_snapshot = {**CONFIG, "git_commit": git_hash}
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, default=str)
    )
    print(f"Config saved to: {checkpoint_dir / 'config.json'}")

    # ------------------------------------------------------------------ #
    # DATA SPLITS                                                          #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("DATA SPLITS")
    print("=" * 80)

    all_ref_ids = AlphaEarthDataset.get_ref_ids(ALPHAEARTH_DIR)
    print(f"Unique REFIDs found in ALPHAEARTH_DIR: {len(all_ref_ids)}")

    all_ref_ids = [fid for fid in all_ref_ids if list(MASK_DIR.glob(f"{fid}*.tif"))]
    print(f"After filtering to tiles with masks: {len(all_ref_ids)}")

    if CONFIG["fold"] is not None:
        # Geographic 5-fold CV: load pre-computed fold assignments and filter
        # to tiles available on disk so all modalities see the same tile pool.
        fold_assignments = load_folds(path=args.folds_file) if args.folds_file else load_folds()
        fold_assignments = {r: f for r, f in fold_assignments.items() if r in set(all_ref_ids)}
        train_ref_ids, val_ref_ids, test_ref_ids = get_fold_splits(fold_assignments, CONFIG["fold"])
        print(f"✓ Geographic 5-fold CV  (test_fold={CONFIG['fold']}, val_fold={(CONFIG['fold']+1)%5})")
    else:
        train_ref_ids, val_ref_ids, test_ref_ids = get_splits(
            all_ref_ids,
            train_ratio=CONFIG["train_ratio"],
            val_ratio=CONFIG["val_ratio"],
            test_ratio=CONFIG["test_ratio"],
            random_state=CONFIG["random_seed"],
        )
        print(f"✓ Legacy random split (random_state={CONFIG['random_seed']})")

    print(f"Train tiles: {len(train_ref_ids)} (~{100 * len(train_ref_ids) / len(all_ref_ids):.0f}%)")
    print(f"Val tiles:   {len(val_ref_ids)} (~{100 * len(val_ref_ids) / len(all_ref_ids):.0f}%)")
    print(f"Test tiles:  {len(test_ref_ids)} (~{100 * len(test_ref_ids) / len(all_ref_ids):.0f}%)")
    print(f"  AlphaEarth years: {ALPHAEARTH_YEARS[0]}-{ALPHAEARTH_YEARS[-1]}")

    # ------------------------------------------------------------------ #
    # DATASETS                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("DATASETS")
    print("=" * 80)

    if CONFIG["augment_train"]:
        train_transform = ComposeTS([
            RandomCropTS(CONFIG["chip_size"]),
            RandomFlipTS(p_horizontal=0.5, p_vertical=0.5),
            RandomRotate90TS(),
        ])
    else:
        train_transform = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),
        ])

    eval_transform = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
    ])

    shared_ds_kwargs = dict(
        prediction_horizon=CONFIG["prediction_horizon"],
        input_years=CONFIG["input_years"],
    )

    train_ds = AlphaEarthDataset(
        train_ref_ids, transform=train_transform, **shared_ds_kwargs
    )
    val_ds = AlphaEarthDataset(
        val_ref_ids, transform=eval_transform, **shared_ds_kwargs
    )
    test_ds = AlphaEarthDataset(
        test_ref_ids, transform=eval_transform, **shared_ds_kwargs
    )

    print(f"✓ Datasets created for {CONFIG['chip_size']}x{CONFIG['chip_size']} chips")
    print(f"  Train: {len(train_ref_ids)} input → {len(train_ds)} usable ({len(train_ref_ids) - len(train_ds)} excluded) | augmentation: {'flips + rotations' if CONFIG['augment_train'] else 'none'}")
    print(f"  Val:   {len(val_ref_ids)} input → {len(val_ds)} usable ({len(val_ref_ids) - len(val_ds)} excluded)")
    print(f"  Test:  {len(test_ref_ids)} input → {len(test_ds)} usable ({len(test_ref_ids) - len(test_ds)} excluded)")

    # ------------------------------------------------------------------ #
    # DATALOADERS                                                          #
    # ------------------------------------------------------------------ #
    def worker_init_fn(worker_id: int) -> None:
        seed = CONFIG["random_seed"] + worker_id
        np.random.seed(seed)
        random.seed(seed)

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
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(CONFIG["random_seed"]),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(CONFIG["random_seed"]),
    )

    print(f"✓ Dataloaders created (reproducible shuffle seed={CONFIG['random_seed']})")

    # ------------------------------------------------------------------ #
    # MODEL                                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("MODEL")
    print("=" * 80)

    sample_x, _, _ = next(iter(train_loader))
    _, T, C, H, W = sample_x.shape

    model = UTAE(
        input_dim=C,
        out_conv=[32, CONFIG["num_classes"]],
        pad_value=0.0,
    )

    print(f"✓ U-TAE model created")
    print(f"  Input modality: AlphaEarth embeddings (no Sentinel)")
    print(f"  Channels (C):   {C}  (64 per AlphaEarth year)")
    print(f"  Timesteps (T):  {T}  (annual)")
    print(f"  Classes:        {CONFIG['num_classes']}")
    print(f"  Input shape:    (B, {T}, {C}, {H}, {W})")

    criterion = FocalLoss(gamma=CONFIG["focal_gamma"])
    print(f"  Loss:           FocalLoss(gamma={CONFIG['focal_gamma']})")

    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["lr_factor"],
        patience=CONFIG["lr_patience"],
    )

    model = model.to(device)
    criterion = criterion.to(device)

    # Warm-up pass in FP32 to initialise U-TAE's dynamic shapes before
    # any AMP context, preventing shape-mismatch bugs on the first forward.
    print("Initialising U-TAE dynamic shapes (FP32 warm-up pass)...")
    model.eval()
    with torch.no_grad():
        dummy_x = torch.zeros(1, T, C, H, W, device=device)
        dummy_pos = torch.zeros(1, T, dtype=torch.long, device=device)
        _ = model(dummy_x, batch_positions=dummy_pos)
    print("✓ Warm-up complete")

    # ------------------------------------------------------------------ #
    # WANDB                                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("WANDB INITIALISATION")
    print("=" * 80)

    n_label = CONFIG["input_years"] if CONFIG["input_years"] is not None else "all"
    fold_label = f"_fold{CONFIG['fold']}" if CONFIG["fold"] is not None else ""
    tag_suffix = f"_{args.tag}" if args.tag else ""
    group_name = f"UTAE_{train_ds.DATASET_NAME}_K{CONFIG['prediction_horizon']}_N{n_label}{tag_suffix}"
    run = wandb.init(
        entity=CONFIG["wandb_entity"],
        project=CONFIG["wandb_project"],
        name=f"{group_name}{fold_label}",
        group=group_name,
        config={
            "architecture": CONFIG["architecture"],
            "dataset": train_ds.DATASET_NAME,
            "input_modality": "alphaearth_only",
            "experiment_tag": args.tag or None,
            "alphaearth_channels": C,
            "num_timesteps": T,
            "prediction_horizon_K": CONFIG["prediction_horizon"],
            "input_years_N": CONFIG["input_years"],
            "cv_fold": CONFIG["fold"],
            "split_strategy": "geographic_5fold_cv" if CONFIG["fold"] is not None else "random_70_15_15",
            "epochs": CONFIG["epochs"],
            "learning_rate": CONFIG["learning_rate"],
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_patience": CONFIG["lr_patience"],
            "lr_factor": CONFIG["lr_factor"],
            "batch_size": CONFIG["batch_size"],
            "chip_size": CONFIG["chip_size"],
            "augment_train": CONFIG["augment_train"],
            "augmentation": "flips_rotations" if CONFIG["augment_train"] else "none",
            "normalization": CONFIG["normalization"],
            "loss": "focal_loss",
            "focal_gamma": CONFIG["focal_gamma"],
            "train_tiles_input": len(train_ref_ids),
            "train_tiles_used": len(train_ds),
            "val_tiles_input": len(val_ref_ids),
            "val_tiles_used": len(val_ds),
            "test_tiles_input": len(test_ref_ids),
            "test_tiles_used": len(test_ds),
            "early_stopping_patience": CONFIG["early_stopping_patience"],
            "random_seed": CONFIG["random_seed"],
            "git_commit": git_hash,
        },
    )
    print("✓ WandB initialised")

    # ------------------------------------------------------------------ #
    # TRAINING LOOP                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_iou = 0.0
    epochs_without_improvement = 0
    for epoch in range(CONFIG["epochs"]):
        # ---- Train ---- #
        model.train()
        total_loss = 0.0
        for x, mask, positions in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}"):
            x = x.to(device)
            mask = mask.to(device)
            positions = positions.to(device)

            optimizer.zero_grad()
            logits = model(x, batch_positions=positions)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validate ---- #
        model.eval()
        val_loss = 0.0
        sum_tp = sum_fp = sum_tn = sum_fn = 0
        with torch.no_grad():
            for x, mask, positions in val_loader:
                x = x.to(device)
                mask = mask.to(device)
                positions = positions.to(device)

                logits = model(x, batch_positions=positions)
                loss = criterion(logits, mask)
                val_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                tp, fp, tn, fn = compute_confusion_binary(pred, mask, positive_class=1)
                sum_tp += tp; sum_fp += fp; sum_tn += tn; sum_fn += fn

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
        else:
            epochs_without_improvement += 1

        run.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "IoU": val_metrics["iou"],
            "F1": val_metrics["f1"],
            "Precision": val_metrics["precision"],
            "Recall": val_metrics["recall"],
            "Accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
        })

        print(
            f"Epoch {epoch + 1}/{CONFIG['epochs']}: "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f} | "
            f"IoU={val_metrics['iou']:.4f}  "
            f"F1={val_metrics['f1']:.4f}  "
            f"Prec={val_metrics['precision']:.4f}  "
            f"Rec={val_metrics['recall']:.4f}  "
            f"Acc={val_metrics['accuracy']:.4f}"
        )

        if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
            print(
                f"\nEarly stopping: no val IoU improvement for "
                f"{CONFIG['early_stopping_patience']} epochs (stopped at epoch {epoch + 1})."
            )
            break

    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    print(f"\nBest model (val_IoU={best_val_iou:.4f}) -> {checkpoint_dir / 'best_model.pth'}")
    print(f"Final model -> {checkpoint_dir / 'final_model.pth'}")

    # ------------------------------------------------------------------ #
    # TEST EVALUATION                                                      #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    # Always evaluate with the best checkpoint, not the final epoch
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pth", map_location=device))
    model.eval()
    test_loss = 0.0
    sum_tp = sum_fp = sum_tn = sum_fn = 0

    with torch.no_grad():
        for x, mask, positions in test_loader:
            x = x.to(device)
            mask = mask.to(device)
            positions = positions.to(device)

            logits = model(x, batch_positions=positions)
            loss = criterion(logits, mask)
            test_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            tp, fp, tn, fn = compute_confusion_binary(pred, mask, positive_class=1)
            sum_tp += tp; sum_fp += fp; sum_tn += tn; sum_fn += fn

    avg_test_loss = test_loss / len(test_loader)
    test_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

    print(f"Test results:")
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

    print("\nLogging test set masks to WandB...")
    log_masks(model, test_loader, device, step=CONFIG["epochs"], name_prefix="test", max_batches=10)

    run.finish()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final validation metrics:")
    print(f"  Loss: {avg_val_loss:.4f}  IoU: {val_metrics['iou']:.4f}  F1: {val_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
