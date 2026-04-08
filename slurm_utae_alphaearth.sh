#!/bin/bash
#
# 5-fold geographic CV for U-TAE + AlphaEarth.
# Submits one job per fold (array tasks 0–4).
#
# Usage:
#   sbatch --export=K=2,INPUT_YEARS=4 slurm_utae_alphaearth.sh
#   sbatch --export=K=1              slurm_utae_alphaearth.sh   # N=all
#
# The SLURM_ARRAY_TASK_ID is used as the --fold argument (0–4).
#
#SBATCH --job-name=utae_alphaearth
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=logs/utae_alphaearth/fold%a_%A.out
#SBATCH --error=logs/utae_alphaearth/fold%a_%A.err

echo "=========================================="
echo "Starting U-TAE + AlphaEarth training job"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array task:    $SLURM_ARRAY_TASK_ID  (= fold index)"
echo "Node(s):       $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "=========================================="
echo ""

module purge
module load Python/3.11.3-GCCcore-12.3.0

WORKDIR=${SLURM_SUBMIT_DIR}
cd "$WORKDIR"

source .venv/bin/activate

export $(grep -v '^#' /cluster/home/$USER/Land-take-prediction/.env | xargs)

mkdir -p logs/utae_alphaearth

K=${K:-2}
INPUT_YEARS=${INPUT_YEARS:-}
FOLD=${SLURM_ARRAY_TASK_ID}

echo "Prediction horizon K=${K}"
echo "Input years N=${INPUT_YEARS:-all}"
echo "Fold: ${FOLD}"
echo ""

echo "GPU status:"
nvidia-smi || echo "nvidia-smi not available"
echo ""

CMD="python train_utae_alphaearth.py --prediction_horizon $K --fold $FOLD"
if [ -n "$INPUT_YEARS" ]; then
    CMD="$CMD --input_years $INPUT_YEARS"
fi

echo "Starting: $CMD"
$CMD

echo ""
echo "=========================================="
echo "Job finished: fold ${FOLD}"
echo "=========================================="
