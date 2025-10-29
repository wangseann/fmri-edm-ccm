#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=2-00:00:00
#SBATCH --job-name=day26_smoothing_90
#SBATCH --output=day26_smoothing_%A-%a.out
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
TARGET_LIST=(cat_abstract cat_temporal cat_professional cat_visual cat_violent cat_tactile cat_communal cat_mental cat_numeric cat_emotional cat_social cat_locational)

INDEX=${SLURM_ARRAY_TASK_ID}
TARGET=${TARGET_LIST[$INDEX]}
TARGET_SAFE=$(echo "${TARGET}" | tr -c '[:alnum:]_-' '_')

export SUBJECT STORY TARGET
export CCM_SAMPLES=${CCM_SAMPLES:-10}
export WINDOW_START=${WINDOW_START:-0.0}
export WINDOW_STOP=${WINDOW_STOP:-1.25}
export WINDOW_STEP=${WINDOW_STEP:-0.25}
export METHODS="${METHODS:-gaussian moving_average}"
export WINDOWS="0.0 0.25 1.0"
export MAX_PREDICTORS=${MAX_PREDICTORS:-90}
export USE_CONCAT=${USE_CONCAT:-false}
export CONCAT_MANIFEST=${CONCAT_MANIFEST:-}
export CONCAT_STORY_LABEL=${CONCAT_STORY_LABEL:-all_stories}
export CONCAT_FEATURES_ROOT="${CONCAT_FEATURES_ROOT:-${PROJECT_ROOT}/features_day26_concat_test}"
export CONCAT_OUTPUT_SUBDIR=${CONCAT_OUTPUT_SUBDIR:-all_stories/day24_subject_concat}
export CONCAT_STORY_ORDER=${CONCAT_STORY_ORDER:-}
export CONCAT_FORCE=${CONCAT_FORCE:-false}
FIGS_STORY_DIR=${STORY}
if [[ "${USE_CONCAT}" == "true" && -n "${CONCAT_STORY_LABEL}" ]]; then
    FIGS_STORY_DIR=${CONCAT_STORY_LABEL}
fi
export FEATURES_EVAL_BASE="features_day26_eval_cli_90_CAE/${FIGS_STORY_DIR}/${TARGET_SAFE}"
export FIGS_BASE="figs/${SUBJECT}/${FIGS_STORY_DIR}/day26_smoothing_cli_MDE90step_CAE/${TARGET_SAFE}"

bash "${PROJECT_ROOT}/scripts/run_day26_smoothing.sh"
