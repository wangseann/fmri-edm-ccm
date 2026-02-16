#!/bin/bash

set -euo pipefail

CONFIG=${CONFIG:-configs/demo.yaml}
SUBJECT=${SUBJECT:-UTS01}
STORY=${STORY:-wheretheressmoke}
BOLD_RUN=${BOLD_RUN:-5}
TARGET=${TARGET:-cat_abstract}
WINDOW_START=${WINDOW_START:-0.0}
WINDOW_STOP=${WINDOW_STOP:-1.25}
WINDOW_STEP=${WINDOW_STEP:-0.25}
CCM_SAMPLES=${CCM_SAMPLES:-${SAMPLE_STEPS:-10}}
FEATURES_EVAL_BASE=${FEATURES_EVAL_BASE:-features_day26_concat_test}
FIGS_BASE=${FIGS_BASE:-}
METHODS=${METHODS:-gaussian moving_average}
WINDOWS=${WINDOWS:-}
USE_CONCAT=${USE_CONCAT:-true}
CONCAT_MANIFEST=${CONCAT_MANIFEST:-}
CONCAT_STORY_LABEL=${CONCAT_STORY_LABEL:-all_stories}
CONCAT_FEATURES_ROOT=${CONCAT_FEATURES_ROOT:-}
CONCAT_OUTPUT_SUBDIR=${CONCAT_OUTPUT_SUBDIR:-}
CONCAT_STORY_ORDER=${CONCAT_STORY_ORDER:-}
CONCAT_FORCE=${CONCAT_FORCE:-false}
USE_CAE=${USE_CAE:-false}
APPLY_HUTH_PREPROC=${APPLY_HUTH_PREPROC:-true}
PREPROC_WINDOW=${PREPROC_WINDOW:-120.0}
PREPROC_TRIM=${PREPROC_TRIM:-20.0}
PREPROC_POLYORDER=${PREPROC_POLYORDER:-2}
PREPROC_ZSCORE=${PREPROC_ZSCORE:-true}
FMRI_SPACE=${FMRI_SPACE:-roi}
NO_LAG_PREDICTORS=${NO_LAG_PREDICTORS:-false}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CMD=("python" "${SCRIPT_DIR}/day26_smoothing_mde.py"
    --config "${CONFIG}"
    --subject "${SUBJECT}"
    --story "${STORY}"
    --target "${TARGET}"
    --window-start "${WINDOW_START}"
    --window-stop "${WINDOW_STOP}"
    --window-step "${WINDOW_STEP}"
    --ccm-samples "${CCM_SAMPLES}"
    --features-eval-base "${FEATURES_EVAL_BASE}"
)

if [[ -n "${BOLD_RUN}" ]]; then
    CMD+=(--bold-run "${BOLD_RUN}")
fi

if [[ -n "${FIGS_BASE}" ]]; then
    CMD+=(--figs-base "${FIGS_BASE}")
fi

if [[ "${USE_CAE}" == "true" ]]; then
    CMD+=(--use-cae)
fi

if [[ -n "${METHODS}" ]]; then
    CMD+=(--methods ${METHODS})
fi

if [[ -n "${WINDOWS}" ]]; then
    CMD+=(--windows ${WINDOWS})
fi

if [[ -n "${LIB_SIZES:-}" ]]; then
    CMD+=(--lib-sizes ${LIB_SIZES})
fi

if [[ -n "${TAU_GRID:-}" ]]; then
    CMD+=(--tau-grid ${TAU_GRID})
fi

MAX_PREDICTORS=${MAX_PREDICTORS:-${E_CAP:-}}

if [[ -n "${MAX_PREDICTORS}" ]]; then
    CMD+=(--max-predictors "${MAX_PREDICTORS}")
fi

CMD+=(--fmri-space "${FMRI_SPACE}")
if [[ "${NO_LAG_PREDICTORS}" == "true" ]]; then
    CMD+=(--no-lag-predictors)
fi

if [[ "${SKIP_EXTRA_PLOTS:-true}" == "true" ]]; then
    CMD+=(--skip-extra-plots)
fi

if [[ "${USE_CONCAT}" == "true" ]]; then
    CMD+=(--use-concat)
    if [[ -n "${CONCAT_MANIFEST}" ]]; then
        CMD+=(--concat-manifest "${CONCAT_MANIFEST}")
    fi
    if [[ -n "${CONCAT_STORY_LABEL}" ]]; then
        CMD+=(--concat-story-label "${CONCAT_STORY_LABEL}")
    fi
    if [[ -n "${CONCAT_FEATURES_ROOT}" ]]; then
        CMD+=(--concat-features-root "${CONCAT_FEATURES_ROOT}")
    fi
    if [[ -n "${CONCAT_OUTPUT_SUBDIR}" ]]; then
        CMD+=(--concat-output-subdir "${CONCAT_OUTPUT_SUBDIR}")
    fi
    if [[ -n "${CONCAT_STORY_ORDER}" ]]; then
        CMD+=(--concat-story-order ${CONCAT_STORY_ORDER})
    fi
    if [[ "${CONCAT_FORCE}" == "true" ]]; then
        CMD+=(--concat-force)
    fi
fi

if [[ "${APPLY_HUTH_PREPROC}" == "true" ]]; then
    CMD+=(--huth-preproc)
else
    CMD+=(--no-huth-preproc)
fi
CMD+=(--preproc-window "${PREPROC_WINDOW}")
CMD+=(--preproc-trim "${PREPROC_TRIM}")
CMD+=(--preproc-polyorder "${PREPROC_POLYORDER}")
if [[ "${PREPROC_ZSCORE}" == "false" ]]; then
    CMD+=(--no-preproc-zscore)
else
    CMD+=(--preproc-zscore)
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    CMD+=(--dry-run)
fi

cd "${PROJECT_ROOT}"

printf 'Running Day26 smoothing sweep with command:\n  %s\n' "${CMD[*]}"
"${CMD[@]}"
