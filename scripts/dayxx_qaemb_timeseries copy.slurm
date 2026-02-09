#!/bin/bash
#SBATCH --job-name=qaemb
#SBATCH --partition=test-gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/flash/PaoU/seann/fmri-edm-ccm/scripts/qaemb_%j.out
#SBATCH --error=/flash/PaoU/seann/fmri-edm-ccm/scripts/qaemb_%j.err

set -euo pipefail

# Path to repo
PROJECT_ROOT=/flash/PaoU/seann/fmri-edm-ccm

# Activate environment
module purge
source "$HOME/venvs/fmri-edm-ccm-saion/bin/activate"

# Per-job cache dirs (avoid shared cache contention)
export TMPDIR="/scratch/${USER}/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/.cache"
export MPLCONFIGDIR="$TMPDIR/.mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PROJECT_ROOT/logs"

# Runtime parameters (override via sbatch --export or environment)
export CONFIG=${CONFIG:-configs/demo.yaml}
export SUBJECT=${SUBJECT:-UTS01}
export STORY=${STORY:-wheretheressmoke}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export PLOT_PREVIEW=${PLOT_PREVIEW:-true}

cd "$PROJECT_ROOT"
bash "$PROJECT_ROOT/scripts/dayxx_qaemb_timeseries.sh"
