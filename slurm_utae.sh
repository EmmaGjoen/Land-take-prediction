#!/bin/bash

#SBATCH --job-name=utae_K${K:-2}_N${INPUT_YEARS:-all}
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/utae/utae_%j.out
#SBATCH --error=logs/utae/utae_%j.err

echo "=========================================="
echo "Starting U-TAE training job"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node(s):       $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPUs:          $SLURM_GPUS"
echo "=========================================="
echo ""


module purge
module load Python/3.11.3-GCCcore-12.3.0

WORKDIR=${SLURM_SUBMIT_DIR}
cd "$WORKDIR"

# Activate project venv
source .venv/bin/activate

# Get .env variables (like wandb api key)
export $(grep -v '^#' /cluster/home/$USER/Land-take-prediction/.env | xargs)

# Install/update packages to ensure compatibility
echo "Installing/updating packages..."
pip install --upgrade torch==2.1.0 torchvision==0.16.0 --quiet
echo "Package installation complete"
echo ""

echo "Running from directory: $WORKDIR"
echo ""

echo "GPU status:"
nvidia-smi || echo "nvidia-smi not available"
echo ""

mkdir -p logs/utae

# Accept prediction horizon K and input window N from environment
K=${K:-2}
INPUT_YEARS=${INPUT_YEARS:-}

echo "Prediction horizon K=${K}"
echo "Input years N=${INPUT_YEARS:-all}"

# Build python command — only pass --input_years if INPUT_YEARS is set
CMD="python train_utae.py --prediction_horizon $K"
if [ -n "$INPUT_YEARS" ]; then
    CMD="$CMD --input_years $INPUT_YEARS"
fi

echo "Starting: $CMD"
$CMD

echo ""
echo "=========================================="
echo "Job finished"
echo "=========================================="
