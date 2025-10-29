#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=1-00:00:00
#SBATCH --job-name=day18_story_cache
#SBATCH --output=day18_story_cache_%A-%a.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=se-wang@oist.jp
#SBATCH --array=0

set -euo pipefail

PROJECT_ROOT=/flash/PaoU/seann/fmri-edm-ccm

module purge
module load python/3.11
source "$PROJECT_ROOT/.venv/bin/activate"

export TMPDIR="/scratch/$USER/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/.cache"
export MPLCONFIGDIR="$TMPDIR/.mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

SUBJECT_LIST=(${SUBJECT_LIST:-UTS01})
INDEX=${SLURM_ARRAY_TASK_ID}
SUBJECT=${SUBJECT_LIST[$INDEX]}

export SUBJECTS=${SUBJECT}
export STORIES=${STORIES:-}
export STORY_LIST_PATH=${STORY_LIST_PATH:-misc/story_list.txt}
export CACHE_ROOT=${CACHE_ROOT:-}
export SEMANTIC_COMPONENTS=${SEMANTIC_COMPONENTS:-}
export DRY_RUN=${DRY_RUN:-false}
export CONFIG=${CONFIG:-configs/demo.yaml}

bash "${PROJECT_ROOT}/scripts/run_story_cache_batch.sh"
