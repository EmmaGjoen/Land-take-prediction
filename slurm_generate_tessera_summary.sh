#!/bin/bash
#SBATCH --job-name=tessera_summary
#SBATCH --output=logs/tessera_summary_%j.out
#SBATCH --error=logs/tessera_summary_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=GPUQ

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
mkdir -p "${SUBMIT_DIR}/verification"

OUT_FILE="${SUBMIT_DIR}/COVERAGE_SUMMARY.md"

echo "Working dir: $(pwd)"
echo "Writing summary to: $OUT_FILE"

python scripts/generate_tessera_summary.py \
  --tessera-dir data/processed/tessera/snapped_to_mask_grid \
  --masks-dir data/raw/masks \
  --out-file "$OUT_FILE" \
  --years 2018-2024

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
