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
mkdir -p data/processed/tessera/verification

# Clear stale results CSV on the first array task only
RESULTS_FILE="data/processed/tessera/verification/results.csv"
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MIN}" ]; then
    rm -f "$RESULTS_FILE"
    echo "Cleared previous results file"
fi

python scripts/verify_one_mask_by_index.py \
    --index ${SLURM_ARRAY_TASK_ID} \
    --year 2024 \
    --results-file "$RESULTS_FILE"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
echo ""
echo "After all array tasks complete, run:"
echo "  python scripts/summarize_verification.py"
