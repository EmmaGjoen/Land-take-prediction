#!/bin/bash
#SBATCH --job-name=plot_qualitative
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/plot_qualitative.out
#SBATCH --error=logs/plot_qualitative.err

module purge
module load Python/3.11.3-GCCcore-12.3.0

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export $(grep -v '^#' /cluster/home/$USER/Land-take-prediction/.env | xargs)

python scripts/plot_qualitative_k.py
python scripts/plot_qualitative_n.py
python scripts/plot_qualitative_modality.py
