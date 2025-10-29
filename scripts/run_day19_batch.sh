#!/bin/bash

set -euo pipefail

CONFIG=${CONFIG:-configs/demo.yaml}
SUBJECTS_RAW=${SUBJECTS:-UTS01}
STORIES_RAW=${STORIES:-}
STORY_LIST_PATH=${STORY_LIST_PATH:-misc/story_list.txt}
FEATURES_ROOT=${FEATURES_ROOT:-features_day26_concat_test}
SECONDS_BIN_WIDTH=${SECONDS_BIN_WIDTH:-}
TEMPORAL_WEIGHTING=${TEMPORAL_WEIGHTING:-}
SMOOTHING_SECONDS=${SMOOTHING_SECONDS:-0.0}
SMOOTHING_METHOD=${SMOOTHING_METHOD:-moving_average}
GAUSSIAN_SIGMA=${GAUSSIAN_SIGMA:-}
PROTOTYPE_WEIGHT_POWER=${PROTOTYPE_WEIGHT_POWER:-}
DRY_RUN=${DRY_RUN:-false}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

declare -a SUBJECT_ARRAY=()
declare -a STORY_ARRAY=()

read -r -a SUBJECT_ARRAY <<< "${SUBJECTS_RAW}"

if [[ -n "${STORIES_RAW}" ]]; then
    read -r -a STORY_ARRAY <<< "${STORIES_RAW}"
elif [[ -f "${PROJECT_ROOT}/${STORY_LIST_PATH}" ]]; then
    mapfile -t STORY_ARRAY < "${PROJECT_ROOT}/${STORY_LIST_PATH}"
fi

CMD=("python" "${SCRIPT_DIR}/run_day19_batch.py" --config "${CONFIG}" --features-root "${FEATURES_ROOT}")

for SUBJECT in "${SUBJECT_ARRAY[@]}"; do
    [[ -n "${SUBJECT}" ]] && CMD+=(--subjects "${SUBJECT}")
done

if (( ${#STORY_ARRAY[@]} > 0 )); then
    CMD+=(--stories)
    CMD+=("${STORY_ARRAY[@]}")
fi

if [[ -n "${SECONDS_BIN_WIDTH}" ]]; then
    CMD+=(--seconds-bin-width "${SECONDS_BIN_WIDTH}")
fi

if [[ -n "${TEMPORAL_WEIGHTING}" ]]; then
    CMD+=(--temporal-weighting "${TEMPORAL_WEIGHTING}")
fi

CMD+=(--smoothing-seconds "${SMOOTHING_SECONDS}")
CMD+=(--smoothing-method "${SMOOTHING_METHOD}")

if [[ -n "${GAUSSIAN_SIGMA}" ]]; then
    CMD+=(--gaussian-sigma "${GAUSSIAN_SIGMA}")
fi

if [[ -n "${PROTOTYPE_WEIGHT_POWER}" ]]; then
    CMD+=(--prototype-weight-power "${PROTOTYPE_WEIGHT_POWER}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    CMD+=(--dry-run)
fi

cd "${PROJECT_ROOT}"

printf 'Running Day19 batch smoothing with command:\n  %s\n' "${CMD[*]}"
"${CMD[@]}"
