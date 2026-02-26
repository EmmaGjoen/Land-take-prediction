#!/bin/bash

#SBATCH --job-name=fetch_tessera
#SBATCH --account=share-ie-idi
#SBATCH --output=logs/fetch_tessera_%j.out
#SBATCH --error=logs/fetch_tessera_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=========================================="
echo "Job started: $(date)"
echo "Running on host: $(hostname)"
echo "=========================================="

# Load Python module
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source .venv/bin/activate

# Create logs directory if needed
mkdir -p logs

# Run the script (processes years 2018-2024 by default)
# Add --force to re-download corrupt/truncated files
python scripts/fetch_tessera_for_masks.py "$@"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
