#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

activate_venv
run_from_project_root
ensure_var CHECKPOINT_PATH

export PYTHON_BIN
export DATA_ROOT
export OUT_ROOT
export BPE_PATH
export CHECKPOINT_PATH
export NUM_GPUS
export NUM_WORKERS
export VAL_BATCH_SIZE
export SAM3_RESOLUTION
export SACO_GOLD_RESOLUTION="${SACO_GOLD_RESOLUTION:-896}"
export GATHER_PRED_VIA_FILESYS
export SACO_GOLD_ANN="${SACO_GOLD_ANN:-${DATA_ROOT}/saco_gold/annotations}"
export SACO_GOLD_METACLIP_IMG="${SACO_GOLD_METACLIP_IMG:-${DATA_ROOT}/saco_gold/metaclip_images}"
export SACO_GOLD_SA1B_IMG="${SACO_GOLD_SA1B_IMG:-${DATA_ROOT}/saco_gold/sa1b_images}"
export SKIP_VENV=1

bash scripts/table1/run_saco_gold_all.sh
