#!/bin/bash

set -euo pipefail

CONFIG=${CONFIG:-configs/demo.yaml}
SUBJECTS_RAW=${SUBJECTS:-UTS01}
STORIES_RAW=${STORIES:-}
STORY_LIST_PATH=${STORY_LIST_PATH:-misc/story_list.txt}
CACHE_ROOT=${CACHE_ROOT:-}
SEMANTIC_COMPONENTS=${SEMANTIC_COMPONENTS:-}
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

CMD=("python" "${SCRIPT_DIR}/run_story_cache_batch.py" --config "${CONFIG}")

for SUBJECT in "${SUBJECT_ARRAY[@]}"; do
    [[ -n "${SUBJECT}" ]] && CMD+=(--subjects "${SUBJECT}")
done

if (( ${#STORY_ARRAY[@]} > 0 )); then
    CMD+=(--stories)
    CMD+=("${STORY_ARRAY[@]}")
fi

if [[ -n "${CACHE_ROOT}" ]]; then
    CMD+=(--cache-root "${CACHE_ROOT}")
fi

if [[ -n "${SEMANTIC_COMPONENTS}" ]]; then
    CMD+=(--semantic-components "${SEMANTIC_COMPONENTS}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    CMD+=(--dry-run)
fi

cd "${PROJECT_ROOT}"

printf 'Running Day18 story-cache batch with command:\n  %s\n' "${CMD[*]}"
"${CMD[@]}"
