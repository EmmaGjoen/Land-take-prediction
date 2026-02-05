# Land-take Prediction

Master's thesis project at NTNU in collaboration with NINA.

## Goal

Explore how different input representations (e.g., satellite embeddings, raw imagery) affect land-take prediction models.

## Project Structure

```
├── scripts/                # Utility scripts
│   └── fetch_tessera_for_masks.py  # Download & align GeoTessera embeddings
├── src/
│   ├── config.py           # Configuration
│   ├── data/               # Data loading and transforms
│   ├── models/             # Model architectures
│   └── training/           # Training utilities
├── data/
│   ├── raw/                # Raw input data (masks, etc.)
│   └── processed/          # Processed data (tessera embeddings, etc.)
├── train_early_fusion.py   # Main training script
├── slurm_*.sh              # SLURM job scripts for IDUN
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

## Usage

### Fetch GeoTessera embeddings

```bash
python scripts/fetch_tessera_for_masks.py --year 2024
```

### Train model

```bash
python train_early_fusion.py
```

For running on IDUN, see [IDUN_GUIDE.md](IDUN_GUIDE.md).
