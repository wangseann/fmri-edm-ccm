#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=4-00:00:00
#SBATCH --job-name=day26_lasso_gaussian
#SBATCH --output=day26_lasso_gaussian_%A-%a.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=se-wang@oist.jp
#SBATCH --array=0-11

set -euo pipefail

PROJECT_ROOT=/flash/PaoU/seann/fmri-edm-ccm

module purge
module load python/3.11
source "$PROJECT_ROOT/.venv/bin/activate"

# Per-job temp/cache to avoid cross-run cache races
export TMPDIR="/scratch/$USER/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/.cache"
export MPLCONFIGDIR="$TMPDIR/.mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

SUBJECT=UTS01
STORY=wheretheressmoke
TARGET_LIST=(
  cat_abstract
  cat_temporal
  cat_professional
  cat_visual
  cat_violent
  cat_tactile
  cat_communal
  cat_mental
  cat_numeric
  cat_emotional
  cat_social
  cat_locational
)

INDEX=${SLURM_ARRAY_TASK_ID}
TARGET=${TARGET_LIST[$INDEX]}
TARGET_SAFE=$(echo "${TARGET}" | tr -c '[:alnum:]_-' '_')

export SUBJECT STORY TARGET

export WINDOW_START=${WINDOW_START:-0.0}
export WINDOW_STOP=${WINDOW_STOP:-1.25}
export WINDOW_STEP=${WINDOW_STEP:-0.25}
export METHODS="${METHODS:-gaussian}"  # moving_average gaussian
export WINDOWS="5.0 10.0 15.0 20.0"
export MAX_PREDICTORS=${MAX_PREDICTORS:-90}
export TAU_GRID="${TAU_GRID:-1 2}"    # simple tau grid for lag building

export USE_CONCAT=${USE_CONCAT:-false} # true or false
export CONCAT_MANIFEST=${CONCAT_MANIFEST:-}
export CONCAT_STORY_LABEL=${CONCAT_STORY_LABEL:-ten_stories}
export CONCAT_FEATURES_ROOT="${CONCAT_FEATURES_ROOT:-${PROJECT_ROOT}/features_day26_concat_test}"
export CONCAT_OUTPUT_SUBDIR=${CONCAT_OUTPUT_SUBDIR:-ten_stories/day24_subject_concat}
export CONCAT_STORY_ORDER="adollshouse adventuresinsayingyes breakingupintheageofgoogle cocoonoflove comingofageondeathrow eyespy firetestforlove gangstersandcookies wheretheressmoke wildwomenanddancingqueens"
export CONCAT_FORCE=${CONCAT_FORCE:-false} # true or false

export APPLY_HUTH_PREPROC=${APPLY_HUTH_PREPROC:-true}
export PREPROC_WINDOW=${PREPROC_WINDOW:-120.0}
export PREPROC_TRIM=${PREPROC_TRIM:-10.0}
export PREPROC_POLYORDER=${PREPROC_POLYORDER:-2}
export PREPROC_ZSCORE=${PREPROC_ZSCORE:-true}

FIGS_STORY_DIR=${STORY}
if [[ "${USE_CONCAT}" == "true" && -n "${CONCAT_STORY_LABEL}" ]]; then
    FIGS_STORY_DIR=${CONCAT_STORY_LABEL}
fi

# Use separate dirs so Ridge/Lasso outputs don't collide
export FEATURES_EVAL_BASE="features_day26_lasso_eval_90_detrend_gaussian/${FIGS_STORY_DIR}/${TARGET_SAFE}"
export FIGS_BASE="figs_new/${SUBJECT}/${FIGS_STORY_DIR}/day26_lasso_cli_90_detrend_gaussian/${TARGET_SAFE}"

bash "${PROJECT_ROOT}/scripts/run_day26_smoothing_lasso.sh"