#!/bin/bash
#SBATCH --job-name=cv_modality_v3
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/aggregate_modality_v3_%j.out
#SBATCH --error=logs/aggregate_modality_v3_%j.err

module purge
module load Python/3.11.3-GCCcore-12.3.0

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export $(grep -v '^#' /cluster/home/$USER/Land-take-prediction/.env | xargs)

python scripts/aggregate_cv_results.py \
    --modality \
    --input_years 3 \
    --tag modality_v3 \
    --detail \
    --output results/cv_modality_modality_v3.txt
