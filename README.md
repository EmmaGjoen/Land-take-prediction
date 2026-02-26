# Land-take Prediction

Master's thesis project at NTNU in collaboration with NINA.

## Goal

Explore how different input representations (e.g., satellite embeddings, raw imagery) affect land-take prediction models.

## Project Structure

```
├── scripts/                            # Utility scripts
│   ├── fetch_tessera_for_masks.py          # Download & snap GeoTessera embeddings
│   ├── eda_tessera.py                      # EDA for Tessera embeddings
│   ├── generate_tessera_summary.py         # Generate coverage summary
│   ├── verify_tessera_alignment.py         # Verify spatial alignment
│   ├── verify_one_mask_by_index.py         # Verify a single mask
│   └── summarize_verification.py           # Summarize verification results
├── src/
│   ├── config.py                       # Paths and training defaults
│   ├── data/                           # Dataset classes and transforms
│   │   ├── sentinel_dataset.py             # Sentinel-2 time series loader
│   │   ├── tessera_dataset.py              # GeoTessera embedding loader
│   │   ├── alphaearth_dataset.py           # AlphaEarth embedding loader
│   │   ├── habloss_dataset.py              # HABLOSS VHR loader
│   │   ├── wrap_datasets.py                # Fused dataset wrappers
│   │   ├── splits.py                       # Train/val/test splits
│   │   └── transform.py                    # Shared transforms
│   ├── models/
│   │   └── external/
│   │       └── torchrs_fc_cd.py            # FCEF / FC-Siam models
│   └── utils/
│       ├── metrics.py                      # Binary segmentation metrics
│       └── visualization.py                # WandB mask logging
├── data/
│   ├── raw/                            # Raw input data (from HABLOSS)
│   │   ├── Sentinel/                       # Sentinel-2 mosaics (126 bands)
│   │   ├── masks/                          # Binary change masks
│   │   ├── AlphaEarth/                     # AlphaEarth embeddings (448 bands)
│   │   ├── PlanetScope/                    # PlanetScope mosaics
│   │   └── vhr/                            # Google VHR imagery
│   ├── processed/
│   │   └── tessera/
│   │       ├── snapped_to_mask_grid/       # Final: aligned to mask grid (used by training)
│   │       ├── raw_downloads/              # Intermediate: raw API downloads
│   │       ├── raw_tiles/                  # Intermediate: per-mask tiles
│   │       ├── tiffs/                      # Intermediate: grid tiffs
│   │       └── embeddings/                 # Intermediate: yearly embeddings
│   └── tessera/                        # Tessera EDA outputs & coverage plots
├── train_early_fusion.py               # FCEF with Sentinel only
├── train_fcef_tessera.py               # FCEF with Sentinel + GeoTessera
├── train_fcef_wrapped_data.py          # FCEF with Sentinel + AlphaEarth
├── slurm_*.sh                          # SLURM job scripts for IDUN
├── IDUN_GUIDE.md                       # Guide for running on IDUN
└── requirements.txt
```

> **Note:** `data/` is gitignored. The full raw dataset (Sentinel, masks, AlphaEarth, PlanetScope, VHR) lives on IDUN.
> Only `Sentinel/` and `masks/` are required for Tessera training.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

## Usage

### Fetch GeoTessera embeddings

```bash
python scripts/fetch_tessera_for_masks.py --year 2018-2024
```

### Train models

```bash
# FCEF with Sentinel only
python train_early_fusion.py

# FCEF with Sentinel + GeoTessera
python train_fcef_tessera.py

# FCEF with Sentinel + AlphaEarth
python train_fcef_wrapped_data.py
```

For running on IDUN, see [IDUN_GUIDE.md](IDUN_GUIDE.md).
