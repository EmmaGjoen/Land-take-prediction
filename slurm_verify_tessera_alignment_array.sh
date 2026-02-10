#!/bin/bash
#SBATCH --job-name=verify_tessera
#SBATCH --output=logs/verify_%A_%a.out
#SBATCH --error=logs/verify_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=GPUQ
#SBATCH --array=1-100%20   # replace 100 with number of masks; %20 limits concurrent jobs

### Adjust modules/conda env to your site configuration ###
module load Anaconda3
source activate tessera

cd /home/youruser/path/to/Land-take-prediction
mkdir -p logs
mkdir -p verification

# SLURM_ARRAY_TASK_ID is 1-based and maps to --index in the wrapper
python scripts/verify_one_mask_by_index.py --index ${SLURM_ARRAY_TASK_ID} --year 2024
