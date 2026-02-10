#!/bin/bash
#SBATCH --job-name=verify_tessera
#SBATCH --output=logs/verify_%A_%a.out
#SBATCH --error=logs/verify_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=GPUQ
#SBATCH --array=1-100%20   # replace 100 with number of masks; %20 limits concurrent jobs

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
mkdir -p verification

python scripts/verify_one_mask_by_index.py --index ${SLURM_ARRAY_TASK_ID} --year 2024

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
