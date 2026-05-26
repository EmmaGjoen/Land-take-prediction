#!/bin/bash
# Submit modality_v3 training: K=2, N=3, geographic_folds_2017.csv for all three modalities.
#
# NOTE: This overwrites checkpoints/utae_sentinel_K2_N3_fold{0-4}, which were
#       previously trained on the full folds (slicing_v2). The K-slicing qualitative
#       PDFs in reports/figures/ are already saved, so this is safe.
#
# WandB groups created:
#   UTAE_sentinel_K2_N3_modality_v3
#   UTAE_alphaearth_K2_N3_modality_v3
#   UTAE_tessera_K2_N3_modality_v3
#
# Usage:
#   bash submit_modality_v3.sh

set -e
FOLDS="src/data/geographic_folds_2017.csv"

echo "Submitting Sentinel-2  K=2 N=3 modality_v3 ..."
sbatch --export=K=2,INPUT_YEARS=3,TAG=modality_v3,FOLDS_FILE="${FOLDS}" slurm_utae.sh

echo "Submitting AlphaEarth  K=2 N=3 modality_v3 ..."
sbatch --export=K=2,INPUT_YEARS=3,TAG=modality_v3,FOLDS_FILE="${FOLDS}" slurm_utae_alphaearth.sh

echo "Submitting GeoTessera  K=2 N=3 modality_v3 ..."
sbatch --export=K=2,INPUT_YEARS=3,TAG=modality_v3,FOLDS_FILE="${FOLDS}" slurm_utae_tessera.sh

echo "Done. 3 x 5-fold array jobs submitted."
echo "Monitor with: squeue -u \$USER"
