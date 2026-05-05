# Land-take Prediction

Master's thesis at NTNU in collaboration with NINA. The project compares three
input modalities for binary land-take segmentation with U-TAE: raw Sentinel-2
time series, GeoTessera embeddings, and AlphaEarth embeddings. Evaluation uses
geographic 5-fold cross-validation with pooled confusion-matrix aggregation
following the PASTIS protocol.

## Data

All data lives on IDUN and is gitignored. Expected layout:

```
data/
  raw/
    Sentinel_v2/                  Sentinel-2 mosaics (9 bands, bi-annual, 2016-2024)
    Land_take_masks_coarse/       Binary change masks (~261 tiles)
    AlphaEarth/                   AlphaEarth embeddings (64 dims/year, 2017-2024)
    annotations_metadata_final.csv
  processed/
    tessera/
      snapped_to_mask_grid/       GeoTessera embeddings aligned to mask grid (128 dims/year, 2017-2024)
```

## Repository structure

```
src/
  config.py                       Paths, shared constants, year ranges
  data/
    sentinel_dataset.py           Sentinel-2 time series dataset
    tessera_dataset.py            GeoTessera embedding dataset
    alphaearth_dataset.py         AlphaEarth embedding dataset
    splits.py                     Geographic 5-fold CV and legacy random split
    transform.py                  Crop, flip, normalisation
    file_helpers.py               Shared filename / refid utilities
    geographic_folds.csv          Pre-computed fold assignments (committed)
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
  fetch_tessera_for_masks.py      Download and snap GeoTessera embeddings to mask grid
  aggregate_cv_results.py         Pool confusion matrices across folds from WandB
  analyze_multi_tile_coverage.py  Check tessera coverage across tiles
  generate_tessera_summary.py     Per-tile embedding coverage summary
  verify_one_mask_by_index.py     Verify spatial alignment for a single tile (SLURM array)
  summarize_verification.py       Aggregate alignment verification results
  eda_tessera.py                  Exploratory analysis of tessera embeddings

train_utae.py                     Train U-TAE on Sentinel-2
train_utae_tessera.py             Train U-TAE on GeoTessera embeddings
train_utae_alphaearth.py          Train U-TAE on AlphaEarth embeddings

slurm_utae.sh                     SLURM array job, Sentinel experiment (folds 0-4)
slurm_utae_tessera.sh             SLURM array job, GeoTessera experiment
slurm_utae_alphaearth.sh          SLURM array job, AlphaEarth experiment
slurm_aggregate.sh                Run aggregate_cv_results.py
slurm_fetch_tessera.sh            Fetch GeoTessera embeddings
slurm_generate_tessera_summary.sh Summarise tessera coverage
slurm_verify_tessera_alignment_array.sh  Verify tessera alignment (array)
slurm_eda_tessera.sh              Run tessera EDA
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

Fetch GeoTessera embeddings if not already present:

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
