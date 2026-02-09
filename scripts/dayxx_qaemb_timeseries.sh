#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG=${CONFIG:-configs/demo.yaml}
SUBJECT=${SUBJECT:-UTS01}
STORY=${STORY:-wheretheressmoke}
LOG_LEVEL=${LOG_LEVEL:-INFO}
PLOT_PREVIEW=${PLOT_PREVIEW:-false}

CMD=("python" "${SCRIPT_DIR}/dayxx_qaemb_timeseries.py"
    --config "${CONFIG}"
    --subject "${SUBJECT}"
    --story "${STORY}"
    --log-level "${LOG_LEVEL}"
)

if [[ "${PLOT_PREVIEW}" == "true" ]]; then
    CMD+=(--plot-preview)
fi

cd "${PROJECT_ROOT}"
printf 'Running QA-Emb time series:\n  %s\n' "${CMD[*]}"
"${CMD[@]}"
