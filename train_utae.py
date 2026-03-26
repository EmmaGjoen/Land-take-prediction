import os

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

from src.config import SENTINEL_DIR, MASK_DIR
import rasterio
from src.data.sentinel_dataset import SentinelDataset
from src.data.splits import get_splits, get_ref_ids_from_directory
from src.data.transform import (
    compute_normalization_stats,
    ComposeTS,
    NormalizeBy,
    RandomCropTS,
    CenterCropTS,
    Normalize,
    RandomFlipTS,
    RandomRotate90TS
)
from src.models.external.utae import UTAE
from src.utils.visualization import log_masks
from src.utils.metrics import compute_confusion_binary, compute_metrics_from_confusion
from src.utils.focal_loss import FocalLoss


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
    "architecture": "u-tae",
    "num_classes": 2,
    
    # Data
    "img_frequency": None,
    "chip_size": 64,
    "prediction_horizon": 2,        # K: zero timesteps from (endYear - K) onwards per tile
    "input_years": None,            # N: only show the last N years before the cutoff; None = all available
    "focal_gamma": 2.0,             # focusing parameter for focal loss (Lin et al., 2017)

    # Training
    "epochs": 75,
    "learning_rate": 1e-3,
    "lr_patience": 7,               # epochs with no val_loss improvement before LR halves
    "lr_factor": 0.5,               # multiply LR by this when patience runs out
    "batch_size": 4,
    "augment_train": True,  # Enable spatial augmentation (flips, rotations)
    
    # Normalization
    "normalization": "scale_10000_plus_standardize",
    "num_samples_for_stats": 2000,
    
    # DataLoader
    "num_workers": 4,
    
    # WandB
    "wandb_project": "data_variasjon_utae",
    "wandb_entity": "nina_prosjektoppgave",
}


# ============================================================================
# SETUP
# ============================================================================

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")



def get_device():
    """Get device for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_horizon", type=int, default=None,
                        help="Override CONFIG prediction_horizon (K)")
    parser.add_argument("--input_years", type=int, default=None,
                        help="Override CONFIG input_years (N): number of years before cutoff to show")
    args = parser.parse_args()
    if args.prediction_horizon is not None:
        CONFIG["prediction_horizon"] = args.prediction_horizon
        print(f"prediction_horizon overridden via CLI: K={CONFIG['prediction_horizon']}")
    if args.input_years is not None:
        CONFIG["input_years"] = args.input_years
        print(f"input_years overridden via CLI: N={CONFIG['input_years']}")

    # Set random seeds
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_random_seeds(CONFIG["random_seed"])
    
    # Get device
    device = get_device()
    
    # Get data splits
    print("\n" + "="*80)
    print("DATA SPLITS")
    print("="*80)
    all_ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
    print(f"Total reference IDs found in Sentinel dir: {len(all_ref_ids)}")
    # Keep only tiles that also have a mask — new coarse masks don't cover all old Sentinel tiles
    all_ref_ids = [fid for fid in all_ref_ids if list(MASK_DIR.glob(f"{fid}*.tif"))]
    print(f"After filtering to tiles with masks: {len(all_ref_ids)}")
    
    train_ref_ids, val_ref_ids, test_ref_ids = get_splits(
        all_ref_ids,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"],
        random_state=CONFIG["random_seed"],
    )
    
    print(f"Train tiles: {len(train_ref_ids)} (~{100*len(train_ref_ids)/len(all_ref_ids):.0f}%)")
    print(f"Val tiles: {len(val_ref_ids)} (~{100*len(val_ref_ids)/len(all_ref_ids):.0f}%)")
    print(f"Test tiles: {len(test_ref_ids)} (~{100*len(test_ref_ids)/len(all_ref_ids):.0f}%)")
    print(f"✓ Using SHARED splits with U-Net baseline (random_state={CONFIG['random_seed']})")

    
    # Compute normalization stats
    print("\n" + "="*80)
    print("NORMALIZATION")
    print("="*80)
    temp_train_transform = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),
        NormalizeBy(10000.0),
    ])
    
    temp_train_ds = SentinelDataset(
        train_ref_ids,
        frequency=CONFIG["img_frequency"],
        transform=temp_train_transform,
    )
    
    print("Estimating per-channel mean and std from training data...")
    mean, std = compute_normalization_stats(temp_train_ds, num_samples=CONFIG["num_samples_for_stats"])
    print(f"✓ Computed normalization stats: {len(mean)} channels")
    print(f"  Mean (first 5): {[f'{m:.4f}' for m in mean[:5]]}")
    print(f"  Std (first 5): {[f'{s:.4f}' for s in std[:5]]}")
    
    # Create datasets
    print("\n" + "="*80)
    print("DATASETS")
    print("="*80)
    
    # Training transform with spatial augmentation (flips + rotations)
    # Always center-crop first to handle variable input sizes
    if CONFIG["augment_train"]:
        train_transform = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
            RandomFlipTS(p_horizontal=0.5, p_vertical=0.5),
            RandomRotate90TS(),
            NormalizeBy(10000.0),
            Normalize(mean, std),
        ])
    else:
        train_transform = ComposeTS([
            CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
            NormalizeBy(10000.0),
            Normalize(mean, std),
        ])
    
    # Val/test transforms: no augmentation, only normalization (but still crop)
    val_transform = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
        NormalizeBy(10000.0),
        Normalize(mean, std),
    ])
    
    test_transform = ComposeTS([
        CenterCropTS(CONFIG["chip_size"]),  # Pad/crop to 64×64
        NormalizeBy(10000.0),
        Normalize(mean, std),
    ])
    
    train_ds = SentinelDataset(
        train_ref_ids,
        frequency=CONFIG["img_frequency"],
        transform=train_transform,
        prediction_horizon=CONFIG["prediction_horizon"],
        input_years=CONFIG["input_years"],
    )

    val_ds = SentinelDataset(
        val_ref_ids,
        frequency=CONFIG["img_frequency"],
        transform=val_transform,
        prediction_horizon=CONFIG["prediction_horizon"],
        input_years=CONFIG["input_years"],
    )
    test_ds = SentinelDataset(
        test_ref_ids,
        frequency=CONFIG["img_frequency"],
        transform=test_transform,
        prediction_horizon=CONFIG["prediction_horizon"],
        input_years=CONFIG["input_years"],
    )
    
    print(f"✓ Datasets created for pre-cropped {CONFIG['chip_size']}×{CONFIG['chip_size']} chips")
    print(f"Train chips: {len(train_ds)} (from {len(train_ref_ids)} REFIDs) - with flips + rotations")
    print(f"Val chips: {len(val_ds)} (from {len(val_ref_ids)} REFIDs) - no augmentation")
    print(f"Test chips: {len(test_ds)} (from {len(test_ref_ids)} REFIDs) - no augmentation")
    print(f"Augmentation enabled: {CONFIG['augment_train']}")
    
    # Create dataloaders
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
        generator=torch.Generator().manual_seed(CONFIG["random_seed"])
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(CONFIG["random_seed"])
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(CONFIG["random_seed"])
    )
    
    print(f"✓ Dataloaders created with reproducible shuffling (seed={CONFIG['random_seed']})")
    
    # Build model
    print("\n" + "="*80)
    print("MODEL")
    print("="*80)
    sample_x, _, _ = next(iter(train_loader))
    _, T, C, H, W = sample_x.shape
    
    model = UTAE(
        input_dim=C,
        out_conv=[32, CONFIG["num_classes"]],
        pad_value=0.0,             
    )
    
    print(f"✓ U-TAE model created")
    print(f"  Channels: {C}")
    print(f"  Timesteps: {T}")
    print(f"  Classes: {CONFIG['num_classes']}")
    print(f"  Input shape: (B, {T}, {C}, {H}, {W})")
    
    # Loss, optimizer
    criterion = FocalLoss(gamma=CONFIG["focal_gamma"]).to(device)
    print(f"Using focal loss with gamma={CONFIG['focal_gamma']}")
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["lr_factor"],
        patience=CONFIG["lr_patience"],
    )
    
    # Initialize WandB
    print("\n" + "="*80)
    model = model.to(device)
    
    # --- WARM-UP PASS ---
    print("Initializing U-TAE dynamic shapes to prevent AMP bug.")
    model.eval() # Set to eval to avoid affecting BatchNorm
    with torch.no_grad():
        # Create a single FP32 dummy batch matching your chip dimensions
        dummy_x = torch.zeros(1, T, C, H, W, device=device)
        dummy_pos = torch.zeros(1, T, dtype=torch.long, device=device)
        
        # Run it through the model OUTSIDE of the autocast context
        _ = model(dummy_x, batch_positions=dummy_pos)
    print("✓ U-TAE shapes initialized safely in FP32")
    # -----------------------------


    print("WANDB INITIALIZATION")
    print("="*80)
    n_label = CONFIG["input_years"] if CONFIG["input_years"] is not None else "all"
    run = wandb.init(
        entity=CONFIG["wandb_entity"],
        project=CONFIG["wandb_project"],
        name=f"UTAE_{train_ds.DATASET_NAME}_K{CONFIG['prediction_horizon']}_N{n_label}_freq:{CONFIG['img_frequency']}",
        config={
            "architecture": CONFIG["architecture"],
            "dataset": train_ds.DATASET_NAME,
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "chip_size": CONFIG["chip_size"],
            "augment_train": CONFIG["augment_train"],
            "augmentation": "flips_rotations" if CONFIG["augment_train"] else "none",
            "num_timesteps": T,
            "train_chips": len(train_ds),
            "val_chips": len(val_ds),
            "test_chips": len(test_ds),
            "normalization": CONFIG["normalization"],
            "random_seed": CONFIG["random_seed"],
            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],
            "test_ratio": CONFIG["test_ratio"],
            "prediction_horizon": CONFIG["prediction_horizon"],
            "loss": "weighted_cross_entropy",
            "prediction_horizon_K": CONFIG["prediction_horizon"],
            "input_years_N": CONFIG["input_years"],
            "num_timesteps": T,
            "epochs": CONFIG["epochs"],
            "learning_rate": CONFIG["learning_rate"],
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_patience": CONFIG["lr_patience"],
            "lr_factor": CONFIG["lr_factor"],
            "batch_size": CONFIG["batch_size"],
            "chip_size": CONFIG["chip_size"],
            "augmentation": "flips_rotations" if CONFIG["augment_train"] else "none",
            "normalization": CONFIG["normalization"],
            "loss": "focal_loss",
            "focal_gamma": CONFIG["focal_gamma"],
            "train_chips": len(train_ds),
            "val_chips": len(val_ds),
            "test_chips": len(test_ds),
            "random_seed": CONFIG["random_seed"],
        },
    )
    print("✓ WandB initialized")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        total_loss = 0.0
        for x, mask, positions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
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

        # Validation
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
                sum_tp += tp
                sum_fp += fp
                sum_tn += tn
                sum_fn += fn

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

        # Step LR scheduler — halves LR if val_loss hasn't improved for lr_patience epochs
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to WandB
        run.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "IoU": val_metrics['iou'],
            "F1": val_metrics['f1'],
            "Precision": val_metrics['precision'],
            "Recall": val_metrics['recall'],
            "Accuracy": val_metrics['accuracy'],
            "learning_rate": current_lr,
        })

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']}: "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} | "
            f"IoU={val_metrics['iou']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"Prec={val_metrics['precision']:.4f} "
            f"Rec={val_metrics['recall']:.4f} "
            f"Acc={val_metrics['accuracy']:.4f}"
        )

    
    # Test set evaluation
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
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
            sum_tp += tp
            sum_fp += fp
            sum_tn += tn
            sum_fn += fn

    avg_test_loss = test_loss / len(test_loader)
    test_metrics = compute_metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)
    
    print(f"Test Set Results:")
    print(f"  Loss: {avg_test_loss:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Log test metrics to WandB
    run.log({
        "test_loss": avg_test_loss,
        "test_iou": test_metrics['iou'],
        "test_f1": test_metrics['f1'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_accuracy": test_metrics['accuracy'],
    })
    
    # Log combined masks from multiple test batches
    print("\nLogging test set masks...")
    log_masks(model, test_loader, device, step=CONFIG["epochs"], name_prefix="test", max_batches=10)
    
    # Finish WandB
    run.finish()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final Validation Metrics:")
    print(f"  Loss: {avg_val_loss:.4f}")
    print(f"  IoU: {val_metrics['iou']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()