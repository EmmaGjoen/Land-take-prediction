#!/bin/bash
#SBATCH --job-name=tessera_summary
#SBATCH --output=logs/tessera_summary_%j.out
#SBATCH --error=logs/tessera_summary_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=standard

### Adjust the lines below to match your IDUN environment ###
module load Anaconda3
source activate tessera

cd /home/youruser/path/to/Land-take-prediction
mkdir -p logs

python scripts/generate_tessera_summary.py \
  --tessera-dir data/processed/tessera/snapped_to_mask_grid \
  --masks-dir data/raw/masks \
  --out-file COVERAGE_SUMMARY.md \
  --years 2017-2024
