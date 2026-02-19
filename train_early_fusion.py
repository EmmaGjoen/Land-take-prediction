"""
FCEF Early Fusion Training Script for Land-Take Prediction

Fair comparison setup with U-Net baseline: shared splits, normalization, patch size, random seeds
"""

import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb
import traceback

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.config import SENTINEL_DIR
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
    "temporal_mode": "first_half",  # 7 timesteps
    "chip_size": 64, 
    
    # Training
    "epochs": 50,
    "learning_rate": 1e-3,
    "batch_size": 4,
    "augment_train": True,  # Enable spatial augmentation (flips, rotations)
    
    # Normalization
    "normalization": "scale_10000_plus_standardize",
    "num_samples_for_stats": 2000,
    
    # DataLoader
    "num_workers": 4,
    
    # WandB
    "wandb_project": "Baseline",
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
    # Set random seeds
    set_random_seeds(CONFIG["random_seed"])
    
    # Get device
    device = get_device()
    
    # Get data splits
    print("\n" + "="*80)
    print("DATA SPLITS")
    print("="*80)
    all_ref_ids = get_ref_ids_from_directory(SENTINEL_DIR)
    print(f"Total reference IDs found: {len(all_ref_ids)}")
    
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
        slice_mode=CONFIG["temporal_mode"],
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
        slice_mode=CONFIG["temporal_mode"],
        transform=train_transform,
    )
    val_ds = SentinelDataset(
        val_ref_ids,
        slice_mode=CONFIG["temporal_mode"],
        transform=val_transform,
    )
    test_ds = SentinelDataset(
        test_ref_ids,
        slice_mode=CONFIG["temporal_mode"],
        transform=test_transform,
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
        batch_size=1,  # Use batch_size=1 for stable validation on small datasets
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Use batch_size=1 for stable test evaluation
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    
    print(f"✓ Dataloaders created with reproducible shuffling (seed={CONFIG['random_seed']})")
    
    # Build model
    print("\n" + "="*80)
    print("MODEL")
    print("="*80)
    sample_x, _ = next(iter(train_loader))
    _, T, C, H, W = sample_x.shape
    
    model = FCEF(channels=C, t=T, num_classes=CONFIG["num_classes"]).to(device)
    
    print(f"✓ FCEF model created")
    print(f"  Channels: {C}")
    print(f"  Timesteps: {T}")
    print(f"  Classes: {CONFIG['num_classes']}")
    print(f"  Input shape: (B, {T}, {C}, {H}, {W})")
    
    # Loss, optimizer, and scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize WandB
    print("\n" + "="*80)
    print("WANDB INITIALIZATION")
    print("="*80)
    run = wandb.init(
        entity=CONFIG["wandb_entity"],
        project=CONFIG["wandb_project"],
        name=f"FCEF_{train_ds.DATASET_NAME}_chip{CONFIG['chip_size']}_t{T}",
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
            "train_chips": len(train_ds),
            "val_chips": len(val_ds),
            "test_chips": len(test_ds),
            "normalization": CONFIG["normalization"],
            "random_seed": CONFIG["random_seed"],
            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],
            "test_ratio": CONFIG["test_ratio"],
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
        for x, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
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

        # Log to WandB
        run.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "IoU": val_metrics['iou'],
            "F1": val_metrics['f1'],
            "Precision": val_metrics['precision'],
            "Recall": val_metrics['recall'],
            "Accuracy": val_metrics['accuracy']
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
