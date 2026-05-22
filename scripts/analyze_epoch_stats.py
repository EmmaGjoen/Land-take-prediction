"""Print early-stopping and best-epoch statistics from WandB for all experiments."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".venv/lib/python3.11/site-packages"))
sys.path.insert(0, str(root))

import numpy as np
import wandb

api = wandb.Api()
ENTITY = "nina_prosjektoppgave"
PROJECT = "data_variasjon_utae"
MAX_EPOCHS = 150

experiments = {
    "K-slicing (N=3, slicing_v3)": [f"UTAE_sentinel_K{k}_N3_slicing_v3" for k in [1, 2, 3, 4, 5]],
    "N-slicing (K=2, slicing_v2)": [
        "UTAE_sentinel_K2_N1_slicing_v2",
        "UTAE_sentinel_K2_N2_slicing_v2",
        "UTAE_sentinel_K2_N3_slicing_v2",
        "UTAE_sentinel_K2_N4_slicing_v2",
        "UTAE_sentinel_K2_Nall_slicing_v2",
    ],
    "Modality (K=2, N=all, modality_v2)": [
        "UTAE_sentinel_K2_Nall_modality_v2",
        "UTAE_alphaearth_K2_Nall_modality_v2",
        "UTAE_tessera_K2_Nall_modality_v2",
    ],
}

for exp_name, groups in experiments.items():
    print(f"\n{'='*68}")
    print(f"  {exp_name}")
    print(f"{'='*68}")
    all_best_epochs, all_total_epochs = [], []
    early_stopped = total_runs = 0

    for group in groups:
        runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})
        for run in runs:
            if run.state != "finished":
                continue
            total_runs += 1
            total_ep = run.summary.get("epoch")
            if total_ep is None:
                continue
            hist = run.history(pandas=False)
            iou_vals = [
                (row.get("epoch"), row.get("IoU"))
                for row in hist
                if row.get("IoU") is not None and row.get("epoch") is not None
            ]
            if not iou_vals:
                continue
            best_ep = max(iou_vals, key=lambda x: x[1])[0]
            all_total_epochs.append(total_ep)
            all_best_epochs.append(best_ep)
            if total_ep < MAX_EPOCHS - 1:
                early_stopped += 1
            print(f"  {group:47s}  total={total_ep:3d}  best@{best_ep:3d}")

    if all_total_epochs:
        print(f"\n  Runs: {total_runs}  |  Early stopped (<{MAX_EPOCHS}): {early_stopped}/{total_runs}")
        print(f"  Total epochs:   min={min(all_total_epochs):3d}  median={int(np.median(all_total_epochs)):3d}  max={max(all_total_epochs):3d}")
        print(f"  Best val epoch: min={min(all_best_epochs):3d}  median={int(np.median(all_best_epochs)):3d}  max={max(all_best_epochs):3d}")
