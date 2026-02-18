#!/bin/bash
#SBATCH --job-name=eda_tessera
#SBATCH --output=logs/eda_tessera_%j.out
#SBATCH --error=logs/eda_tessera_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=CPUQ

echo "=========================================="
echo "Job started: $(date)"
echo "Running on host: $(hostname)"
echo "=========================================="

module purge
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"

mkdir -p logs

python scripts/eda_tessera.py \
    --tessera-dir data/processed/tessera/snapped_to_mask_grid \
    --masks-dir data/raw/masks \
    --out-dir data/processed/tessera/eda \
    --years 2018-2024

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
