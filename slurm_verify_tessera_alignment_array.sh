#!/bin/bash
#SBATCH --job-name=verify_tessera
#SBATCH --account=share-ie-idi
#SBATCH --output=logs/verify_%A_%a.out
#SBATCH --error=logs/verify_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=CPUQ
#SBATCH --array=1-55%20

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

# Year to verify: pass as first argument to sbatch, e.g.:
#   sbatch slurm_verify_tessera_alignment_array.sh 2021
# Defaults to 2018.
YEAR="${1:-2018}"

# Results file is year-specific so multiple years can be verified independently
RESULTS_FILE="data/processed/tessera/verification/results_${YEAR}.csv"

# Clear stale results CSV on the first array task only
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MIN}" ]; then
    rm -f "$RESULTS_FILE"
    echo "Cleared previous results file for year ${YEAR}"
fi

python scripts/verify_one_mask_by_index.py \
    --index ${SLURM_ARRAY_TASK_ID} \
    --year ${YEAR} \
    --results-file "$RESULTS_FILE"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
echo ""
echo "After all array tasks complete, run:"
echo "  python scripts/summarize_verification.py --results-file data/processed/tessera/verification/results_${YEAR}.csv"
echo ""
echo "To verify a different year:"
echo "  sbatch slurm_verify_tessera_alignment_array.sh 2024"
