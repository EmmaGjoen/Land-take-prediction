# Running Training Scripts on IDUN

This guide explains how to run the training scripts on the IDUN GPU cluster.

## Prerequisites

- SSH access to `idun.hpc.ntnu.no`
- Project cloned to `/cluster/home/$USER/Land-take-prediction/`
- WandB API key in `.env` file

## First-Time Setup on IDUN

### 1. Clone the project

```bash
ssh username@idun.hpc.ntnu.no
cd /cluster/home/$USER
git clone https://github.com/EmmaGjoen/Land-take-prediction.git
cd Land-take-prediction
```

### 2. Create a virtual environment

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure WandB

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

### 4. Data paths

`src/config.py` resolves all paths relative to the project root. After
transferring data, the expected layout is:

```
Land-take-prediction/
├── data/
│   ├── raw/
│   │   ├── Sentinel_v2/
│   │   ├── Land_take_masks_coarse/
│   │   ├── AlphaEarth_v2/
│   │   └── annotations_metadata_final.csv
│   └── processed/
│       └── tessera/
│           └── snapped_to_mask_grid/
└── ...
```

## Transferring Code and Data to IDUN

Code travels via git. Data (gitignored) must be transferred separately.

**Code:** push locally and pull on IDUN.

```bash
# Local machine
git push

# IDUN
cd /cluster/home/$USER/Land-take-prediction
git pull
```

**Data:** rsync from local machine.

```bash
rsync -avz --progress data/raw/Sentinel_v2/ username@idun.hpc.ntnu.no:/cluster/home/$USER/Land-take-prediction/data/raw/Sentinel_v2/
rsync -avz --progress data/raw/Land_take_masks_coarse/ username@idun.hpc.ntnu.no:/cluster/home/$USER/Land-take-prediction/data/raw/Land_take_masks_coarse/
rsync -avz --progress data/raw/AlphaEarth_v2/ username@idun.hpc.ntnu.no:/cluster/home/$USER/Land-take-prediction/data/raw/AlphaEarth_v2/
rsync -avz --progress data/raw/annotations_metadata_final.csv username@idun.hpc.ntnu.no:/cluster/home/$USER/Land-take-prediction/data/raw/
```

TESSERA embeddings are fetched directly on IDUN via the `slurm_fetch_tessera.sh` job.

## Submitting Training Jobs

Optional: adjust `--job-name` and `--time` at the top of any SLURM script before submitting.

### Sentinel-2

```bash
sbatch --export=K=2,INPUT_YEARS=3 slurm_utae.sh
```

### TESSERA embeddings

```bash
sbatch --export=K=2,INPUT_YEARS=3 slurm_utae_tessera.sh
```

### AlphaEarth embeddings

```bash
sbatch --export=K=2,INPUT_YEARS=3 slurm_utae_alphaearth.sh
```

### Modality comparison (all three modalities, K=2, N=3)

```bash
bash submit_modality_v3.sh
```

## Monitoring Jobs

```bash
# View all running jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# Follow output log in real-time
tail -f logs/utae_sentinel/fold0_<job_id>.out
tail -f logs/utae_tessera/fold0_<job_id>.out
tail -f logs/utae_alphaearth/<job_id>_fold0.out

# View error log
tail -f logs/utae_sentinel/fold0_<job_id>.err
```
