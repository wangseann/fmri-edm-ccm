#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --job-name=str1_sub2_guassian_CAE
#SBATCH --output=str1_sub2_CAE_detrend_gaussian_%A-%a.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=se-wang@oist.jp
#SBATCH --array=0-60
# 0-59 corresponds to 12 categories * 5 window sizes (default: includes 0.0 baseline). Adjust if you change lists below.
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

SUBJECT=UTS03
STORY=wheretheressmoke #myfirstdaywiththeyankees #life #howtodraw #souls #odetostepfather #legacy #avatar #alternateithicatom
BOLD_RUN=${BOLD_RUN:-run-3}
#TARGET_LIST=(cat_abstract cat_temporal cat_professional cat_visual cat_violent cat_tactile cat_communal cat_mental cat_numeric cat_emotional cat_social cat_locational)
TARGET_LIST=(cat_social cat_emotional)
WINDOW_OPTIONS=(${WINDOW_OPTIONS:-0.0 10.0 20.0 30.0 40.0})

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_TARGETS=${#TARGET_LIST[@]}
NUM_WINDOWS=${#WINDOW_OPTIONS[@]}
TOTAL=$((NUM_TARGETS * NUM_WINDOWS))
if (( TASK_ID >= TOTAL )); then
    echo "Error: TASK_ID ${TASK_ID} out of range for ${TOTAL} target-window combinations." >&2
    exit 1
fi
TARGET_INDEX=$((TASK_ID / NUM_WINDOWS))
WINDOW_INDEX=$((TASK_ID % NUM_WINDOWS))
TARGET=${TARGET_LIST[$TARGET_INDEX]}
WINDOWS="${WINDOW_OPTIONS[$WINDOW_INDEX]}"
TARGET_SAFE=$(echo "${TARGET}" | tr -c '[:alnum:]_-' '_')

export SUBJECT STORY TARGET BOLD_RUN
export WINDOWS
export CCM_SAMPLES=${CCM_SAMPLES:-10}
export WINDOW_START=${WINDOW_START:-0.0}
export WINDOW_STOP=${WINDOW_STOP:-1.25}
export WINDOW_STEP=${WINDOW_STEP:-0.25}
export METHODS="${METHODS:-gaussian}" #moving_average gaussian
export MAX_PREDICTORS=${MAX_PREDICTORS:-90}
export USE_CONCAT=${USE_CONCAT:-false} #true or false
export CONCAT_MANIFEST=${CONCAT_MANIFEST:-}
export CONCAT_STORY_LABEL=${CONCAT_STORY_LABEL:-ten_stories}
export CONCAT_FEATURES_ROOT="${CONCAT_FEATURES_ROOT:-${PROJECT_ROOT}/features_day26_concat_test}"
export CONCAT_OUTPUT_SUBDIR=${CONCAT_OUTPUT_SUBDIR:-ten_stories/day24_subject_concat}
export CONCAT_STORY_ORDER="adollshouse adventuresinsayingyes breakingupintheageofgoogle cocoonoflove comingofageondeathrow eyespy firetestforlove gangstersandcookies wheretheressmoke wildwomenanddancingqueens"    #${CONCAT_STORY_ORDER:-}
export CONCAT_FORCE=${CONCAT_FORCE:-false} #true or false
export USE_CAE=${USE_CAE:-true} #true or false
export APPLY_HUTH_PREPROC=${APPLY_HUTH_PREPROC:-true}
export PREPROC_WINDOW=${PREPROC_WINDOW:-120.0}
export PREPROC_TRIM=${PREPROC_TRIM:-10.0}
export PREPROC_POLYORDER=${PREPROC_POLYORDER:-2}
export PREPROC_ZSCORE=${PREPROC_ZSCORE:-true}
SELECTION_TAG="rho"
if [[ "${USE_CAE}" == "true" ]]; then
    SELECTION_TAG="CAE"
fi
export SELECTION_TAG
FIGS_STORY_DIR=${STORY}
if [[ "${USE_CONCAT}" == "true" && -n "${CONCAT_STORY_LABEL}" ]]; then
    FIGS_STORY_DIR=${CONCAT_STORY_LABEL}
fi
FIGS_STORY_DIR_RUN=${FIGS_STORY_DIR}
if [[ -n "${BOLD_RUN}" ]]; then
    FIGS_STORY_DIR_RUN="${FIGS_STORY_DIR}/${BOLD_RUN}"
fi
export FEATURES_EVAL_BASE="features_new_day26_eval_cli_90_detrend_gaussian${SELECTION_TAG}/${FIGS_STORY_DIR_RUN}/${TARGET_SAFE}"
export FIGS_BASE="figs_new/${SUBJECT}/${FIGS_STORY_DIR_RUN}/day26_smoothing_cli_MDE90step_detrend_gaussian_${SELECTION_TAG}/${TARGET_SAFE}"

bash "${PROJECT_ROOT}/scripts/run_day26_smoothing.sh"
