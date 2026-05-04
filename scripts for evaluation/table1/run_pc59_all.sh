#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../server/common.sh"

RUN_NAME="table1_pc59"
SEMSEG_EVAL_CMD="${SEMSEG_EVAL_CMD:-}"
INFER_CMD="${INFER_CMD:-}"

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  activate_venv
fi
run_from_project_root

LOG_DIR="${OUT_ROOT}/${RUN_NAME}/logs"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
ensure_dir "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -n "${INFER_CMD}" ]]; then
  eval "${INFER_CMD}"
fi

if [[ -z "${SEMSEG_EVAL_CMD}" ]]; then
  echo "SEMSEG_EVAL_CMD is required to compute mIoU for PC-59." >&2
  exit 2
fi

eval "${SEMSEG_EVAL_CMD}"
