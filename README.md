# Predicting Land Take Using Satellite Image Time Series: Prediction Horizons, Temporal Sequence Length and Data Representations

**Emma Gjøen and Cecilia Møller Blom**  
Master's Thesis in Computer Science, NTNU, June 2026

We compare three input representations for binary land take segmentation with U-TAE: raw Sentinel-2 time series, TESSERA embeddings, and AlphaEarth embeddings. Evaluation uses geographic 5-fold cross-validation with pooled confusion-matrix aggregation following the PASTIS protocol.

## Data

All data lives on IDUN and is gitignored. Expected layout:

```
data/
  raw/
    Sentinel_v2/                  Sentinel-2 mosaics (9 bands, bi-annual, 2016-2024)
    Land_take_masks_coarse/       Binary change masks (~261 tiles)
    AlphaEarth_v2/                AlphaEarth embeddings (64 dims/year, 2017-2024)
    annotations_metadata_final.csv
  processed/
    tessera/
      snapped_to_mask_grid/       TESSERA embeddings aligned to mask grid (128 dims/year, 2017-2024)
```

## Repository structure

```
src/
  config.py                       Paths, shared constants, year ranges
  data/
    sentinel_dataset.py           Sentinel-2 time series dataset
    tessera_dataset.py            TESSERA embedding dataset
    alphaearth_dataset.py         AlphaEarth embedding dataset
    splits.py                     Geographic 5-fold CV and legacy random split
    transform.py                  Crop, flip, normalisation
    file_helpers.py               Shared filename / refid utilities
    geographic_folds_2017.csv     Pre-computed fold assignments, 232 tiles (committed)
  models/external/
    utae.py                       U-TAE model
    backbones/                    LTAE and ConvLSTM backbones
  utils/
    training.py                   Seed and device helpers
    focal_loss.py                 Focal loss
    metrics.py                    Binary segmentation metrics
    visualization.py              WandB mask logging

scripts/
  create_folds.py                 Generate geographic 5-fold CV assignments (run once)
  fetch_tessera_for_masks.py      Download and snap TESSERA embeddings to mask grid
  aggregate_cv_results.py         Pool confusion matrices across folds from WandB
  analyze_multi_tile_coverage.py  Check TESSERA coverage across tiles
  analyze_epoch_stats.py          Print early-stopping and best-epoch statistics from WandB
  generate_tessera_summary.py     Per-tile embedding coverage summary
  eda_tessera.py                  Exploratory analysis of TESSERA embeddings
  plot_iou_vs_k.py                Per-fold test IoU vs prediction horizon K
  plot_iou_vs_n.py                Per-fold test IoU vs input years N
  plot_qualitative_k.py           Qualitative visualisations for K-slicing experiment
  plot_qualitative_n.py           Qualitative visualisations for N-slicing experiment
  plot_qualitative_modality.py    Qualitative comparison across modalities

train_utae.py                     Train U-TAE on Sentinel-2
train_utae_tessera.py             Train U-TAE on TESSERA embeddings
train_utae_alphaearth.py          Train U-TAE on AlphaEarth embeddings

submit_modality_v3.sh             Submit modality experiment across all three modalities

slurm_utae.sh                     SLURM array job, Sentinel-2 experiment (folds 0-4)
slurm_utae_tessera.sh             SLURM array job, TESSERA experiment
slurm_utae_alphaearth.sh          SLURM array job, AlphaEarth experiment
slurm_aggregate.sh                Aggregate K-slicing or N-slicing results from WandB
slurm_aggregate_modality_v3.sh    Aggregate modality experiment results (modality_v3 tag)
slurm_fetch_tessera.sh            Fetch TESSERA embeddings
slurm_generate_tessera_summary.sh Summarise TESSERA coverage
slurm_eda_tessera.sh              Run TESSERA EDA
slurm_plot_qualitative.sh         Run all qualitative plot scripts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
```

On IDUN, load the Python module first:

```bash
module load Python/3.11.3-GCCcore-12.3.0
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add a WandB key:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

## Before training

Generate the geographic fold assignments once. The output is committed so all
experiments share identical splits.

```bash
python scripts/create_folds.py
```

Fetch TESSERA embeddings if not already present:

```bash
sbatch slurm_fetch_tessera.sh
```

## Training

Each training script accepts `--prediction_horizon K`, `--input_years N`, and
`--fold 0-4`. The SLURM scripts run all five folds as an array job.

```bash
sbatch --export=K=2,INPUT_YEARS=4 slurm_utae.sh
sbatch --export=K=2,INPUT_YEARS=4 slurm_utae_tessera.sh
sbatch --export=K=2,INPUT_YEARS=4 slurm_utae_alphaearth.sh
```

Single fold:

```bash
sbatch --export=K=2,INPUT_YEARS=4 --array=0-0 slurm_utae_tessera.sh
```

## Aggregating results

After all folds finish, pool the per-fold confusion matrices logged to WandB:

```bash
python scripts/aggregate_cv_results.py --modality --datasets sentinel tessera alphaearth
python scripts/aggregate_cv_results.py --vary K --k_values 1 2 3 4 5 --dataset sentinel
```

Pooled IoU is the primary metric. Macro mean and standard deviation across
folds are reported as a diagnostic.

## Monitoring jobs

```bash
squeue -u $USER
scancel <job_id>
tail -f logs/utae_tessera/fold0_<job_id>.out
```
