# Running Training Scripts on IDUN

This guide explains how to run the training scripts on the IDUN GPU cluster.

---

## Prerequisites

- SSH access to `idun.hpc.ntnu.no`
- Project already cloned to `/cluster/home/$USER/Land-take-prediction/` (see Setup)
- WandB API key in `.env` file (see Configure WandB)

---

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

`src/config.py` resolves all paths relative to the project root — no edits needed.
After transferring data (see below), the layout should be:

```
Land-take-prediction/
├── data/
│   └── raw/
│       ├── Sentinel/
│       ├── Land_take_masks_coarse/
│       └── annotations_metadata_final.csv
└── ...
```

---

## Transferring Code and Data to IDUN

**Code** travels via git. **Data** (gitignored) must be rsynced separately.

Run this script from your local machine to do both in one step:

```bash
bash sync_to_idun.sh
```

The script:
1. Pushes your local commits to GitHub
2. rsyncs `data/raw/Sentinel/`, `data/raw/Land_take_masks_coarse/`, and
   `data/raw/annotations_metadata_final.csv` to IDUN

After it finishes, SSH to IDUN and pull the code:

```bash
ssh username@idun.hpc.ntnu.no
cd /cluster/home/$USER/Land-take-prediction
git pull
```

---

## Submitting Training Jobs

Optional: adjust `--job-name` and `--time` at the top of any SLURM script before submitting.

### U-TAE

```bash
sbatch slurm_utae.sh
```

### FCEF + Tessera embeddings

```bash
sbatch slurm_fcef_tessera.sh
```

### FCEF (Sentinel only)

```bash
sbatch slurm_fcef.sh
```

### FCEF + AlphaEarth

```bash
sbatch slurm_fcef_alpha.sh
```

### U-Net

```bash
sbatch slurm_unet.sh
```

---

## Monitoring Jobs

```bash
# View all your running jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# Follow output log in real-time
tail -f logs/utae/utae_<job_id>.out
tail -f logs/fcef/fcef_tessera_<job_id>.out

# View error log
tail -f logs/utae/utae_<job_id>.err
```
