#!/bin/bash
#
# Aggregate 5-fold CV results from WandB after U-TAE training completes.
# Runs on CPU. Submit after training array jobs finish.
#
# K experiment (vary prediction horizon, fix N):
#   sbatch --export=VARY=K,K_VALUES="1 2 3",INPUT_YEARS=4 slurm_aggregate.sh
#
# N experiment (vary input years, fix K):
#   sbatch --export=VARY=N,N_VALUES="1 2 3 4",K=2 slurm_aggregate.sh
#
#SBATCH --job-name=cv_aggregate
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

echo "=========================================="
echo "CV Results Aggregation"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node(s):   $SLURM_NODELIST"
echo "=========================================="
echo ""

module purge
module load Python/3.11.3-GCCcore-12.3.0

WORKDIR=${SLURM_SUBMIT_DIR}
cd "$WORKDIR"

source .venv/bin/activate

export $(grep -v '^#' /cluster/home/$USER/Land-take-prediction/.env | xargs)

mkdir -p logs

# Shared parameters
VARY=${VARY:-K}
DATASET=${DATASET:-sentinel}

echo "Experiment type:  --vary $VARY"
echo "Dataset:          $DATASET"

CMD="python scripts/aggregate_cv_results.py --vary $VARY --dataset $DATASET --detail"

if [ "$VARY" = "K" ]; then
    K_VALUES=${K_VALUES:-"1 2 3"}
    INPUT_YEARS=${INPUT_YEARS:-}
    echo "K values:         $K_VALUES"
    echo "Input years N:    ${INPUT_YEARS:-all}"
    CMD="$CMD --k_values $K_VALUES"
    if [ -n "$INPUT_YEARS" ]; then
        CMD="$CMD --input_years $INPUT_YEARS"
    else
        CMD="$CMD --input_years 0"
    fi
else
    N_VALUES=${N_VALUES:-"1 2 3 4"}
    K=${K:-2}
    echo "N values:         $N_VALUES"
    echo "K (fixed):        $K"
    CMD="$CMD --n_values $N_VALUES --k $K"
fi

echo ""
echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "=========================================="
echo "Aggregation complete."
echo "=========================================="
