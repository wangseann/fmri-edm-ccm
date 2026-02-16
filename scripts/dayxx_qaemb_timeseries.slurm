#!/bin/bash
#SBATCH --job-name=qaemb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --mem=192G
#SBATCH --output=/work/PaoU/seann/fmri-edm-ccm/scripts/qaemb_%j.out
#SBATCH --error=/work/PaoU/seann/fmri-edm-ccm/scripts/qaemb_%j.err

set -euo pipefail

# Path to repo
PROJECT_ROOT=/work/PaoU/seann/fmri-edm-ccm

# Activate environment
source /work/PaoU/seann/miniforge3/etc/profile.d/conda.sh
conda activate qaemb-saion


# Per-job scratch + persistent caches.
export TMPDIR="/scratch/${USER}/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/.mpl"
export QAEMB_CACHE_ROOT="/scratch/${USER}/qaemb_cache"
export XDG_CACHE_HOME="${QAEMB_CACHE_ROOT}/xdg"
export HF_HOME="${QAEMB_CACHE_ROOT}/hf"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$PROJECT_ROOT/logs"

# CPU-side tokenization/inference helpers.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=true

# Runtime parameters (override via sbatch --export or environment)
export CONFIG=${CONFIG:-configs/demo.yaml}
export SUBJECT=${SUBJECT:-UTS01}
export STORY=${STORY:-wheretheressmoke}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export PLOT_PREVIEW=${PLOT_PREVIEW:-false}
export QAEMB_PIPELINE_BATCH_SIZE=${QAEMB_PIPELINE_BATCH_SIZE:-16}

cd "$PROJECT_ROOT"
bash "$PROJECT_ROOT/scripts/dayxx_qaemb_timeseries.sh"
