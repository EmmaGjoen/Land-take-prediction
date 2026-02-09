import traceback

import numpy as np
import torch
import torch.nn.functional as F
import wandb


def upscale_mask(mask, scale: int = 4):
    """
    Upscale a 2D numpy mask (H, W) with values 0 or 255
    to a larger size using nearest-neighbor interpolation.

    Args:
        mask: 2D numpy array (H, W)
        scale: Upscaling factor (default 4)

    Returns:
        Upscaled mask (H*scale, W*scale)
    """
    t = torch.from_numpy(mask)[None, None].float()  # (1,1,H,W)
    t_up = F.interpolate(t, scale_factor=scale, mode="nearest")
    return t_up[0, 0].byte().numpy()


def log_masks(model, loader, device, step, name_prefix="val", max_batches=10):
    """
    Log ground-truth and predicted segmentation masks to WandB as combined side-by-side images.
    Iterates over multiple batches and creates a single image per sample with GT on left, prediction on right.
    Visualizes masks as black (0) and white (255).

    Args:
        model: The model to evaluate
        loader: DataLoader to sample from
        device: Device for inference
        step: WandB step (typically epoch number)
        name_prefix: Prefix for WandB keys (e.g., "val", "test")
        max_batches: Maximum number of batches to process (default 10)
    """
    try:
        model.eval()
        combined_images = []

        with torch.no_grad():
            loader_iter = iter(loader)
            for b_idx in range(max_batches):
                try:
                    imgs, masks = next(loader_iter)
                except StopIteration:
                    print(f"[INFO] log_masks ({name_prefix}): reached end of loader at batch {b_idx}")
                    break
                except RuntimeError as e:
                    print(f"[WARN] log_masks ({name_prefix}) batch {b_idx} failed: {e}")
                    continue

                if imgs.shape[0] == 0:
                    continue

                B = imgs.shape[0]
                x = imgs.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu()  # (B, H, W)
                masks = masks.cpu()

                # Convert to uint8 and scale to 0/255 for visibility
                masks_vis = (masks * 255).byte().numpy()  # (B, H, W)
                preds_vis = (preds * 255).byte().numpy()  # (B, H, W)

                for i in range(B):
                    if len(masks_vis[i].shape) != 2 or len(preds_vis[i].shape) != 2:
                        continue

                    # Combine GT (left) and prediction (right) side-by-side
                    combined = np.concatenate([masks_vis[i], preds_vis[i]], axis=1)  # (64, 128)
                    # Upscale for better visualization
                    upscaled = upscale_mask(combined, scale=4)  # (256, 512)
                    combined_images.append(wandb.Image(upscaled, caption=f"{name_prefix}_GT_left_PRED_right_b{b_idx}_i{i}"))

        # Log combined GT+prediction images
        if len(combined_images) > 0:
            wandb.log({f"{name_prefix}_combined_masks": combined_images}, step=step)
            print(f"[INFO] Logged {len(combined_images)} combined mask images for {name_prefix}")
        else:
            print(f"[WARN] log_masks ({name_prefix}): no valid samples to log")

    except Exception as e:
        print(f"[ERROR] log_masks ({name_prefix}) failed: {e}")
        traceback.print_exc()
