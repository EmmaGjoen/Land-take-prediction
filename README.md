# Land-take Prediction

Master's thesis at NTNU in collaboration with NINA. The project compares three input modalities for binary land-take segmentation using U-TAE: raw Sentinel-2 imagery, GeoTessera embeddings, and AlphaEarth embeddings.

## Data

All data lives on IDUN and is gitignored. The required layout is:

```
data/
  raw/
    Sentinel_v2/                  Sentinel-2 mosaics (9 bands, bi-annual, 2016-2024)
    Land_take_masks_coarse/       Binary change masks (~261 tiles)
    AlphaEarth/                   AlphaEarth embeddings (64 dims/year, 2018-2024)
    annotations_metadata_final.csv
  processed/
    tessera/
      snapped_to_mask_grid/       GeoTessera embeddings aligned to mask grid (2017-2024)
```

## Project Structure

```
scripts/
  create_folds.py                 Generate geographic 5-fold CV assignments (run once)
  fetch_tessera_for_masks.py      Download and snap GeoTessera embeddings to mask grid
  analyze_multi_tile_coverage.py  Check tessera coverage across tiles
  generate_tessera_summary.py     Print per-tile embedding coverage summary
  verify_one_mask_by_index.py     Verify spatial alignment for a single tile (used by SLURM array)
  summarize_verification.py       Aggregate verification results
  eda_tessera.py                  Exploratory analysis of tessera embeddings

src/
  config.py                       All paths and shared constants
  data/
    sentinel_dataset.py           Sentinel-2 time series dataset
    tessera_segmentation_dataset.py  GeoTessera dataset
    alphaearth_segmentation_dataset.py  AlphaEarth dataset
    splits.py                     Geographic 5-fold CV and legacy random split utilities
    transform.py                  Shared transforms (crop, flip, normalize)
    geographic_folds.csv          Pre-computed fold assignments (committed)
  models/
    external/
      utae.py                     U-TAE model
      backbones/                  LTAE and ConvLSTM backbones
  utils/
    training.py                   Shared set_random_seeds and get_device
    focal_loss.py                 Focal loss
    metrics.py                    Binary segmentation metrics
    visualization.py              WandB mask logging

train_utae.py                     Train U-TAE on Sentinel-2
train_utae_tessera.py             Train U-TAE on GeoTessera embeddings
train_utae_alphaearth.py          Train U-TAE on AlphaEarth embeddings

slurm_utae.sh                     SLURM array job for Sentinel experiment (folds 0-4)
slurm_utae_tessera.sh             SLURM array job for GeoTessera experiment
slurm_utae_alphaearth.sh          SLURM array job for AlphaEarth experiment
slurm_fetch_tessera.sh            Fetch GeoTessera embeddings
slurm_generate_tessera_summary.sh Summarize tessera coverage
slurm_verify_tessera_alignment_array.sh  Verify tessera alignment (array job)
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

Add your WandB key:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

## Before Training

Run once to generate the geographic fold assignments:

```bash
python scripts/create_folds.py
```

This writes `src/data/geographic_folds.csv` and prints per-fold tile counts and coordinate ranges. Commit the file so all experiments use identical splits.

Fetch GeoTessera embeddings (if not already done):

```bash
sbatch slurm_fetch_tessera.sh
```

## Training

Each training script supports `--prediction_horizon K`, `--input_years N`, and `--fold 0-4`. The SLURM scripts run all five folds as an array job.

```bash
# Run all five folds for each modality
sbatch --export=K=2,INPUT_YEARS=4 slurm_utae.sh
sbatch --export=K=2,INPUT_YEARS=4 slurm_utae_tessera.sh
sbatch --export=K=2,INPUT_YEARS=4 slurm_utae_alphaearth.sh
```

To run a single fold:

```bash
sbatch --export=K=2,INPUT_YEARS=4 --array=0-0 slurm_utae_tessera.sh
```

## Monitoring Jobs

```bash
squeue -u $USER
scancel <job_id>
tail -f logs/utae_tessera/fold0_<job_id>.out
```
