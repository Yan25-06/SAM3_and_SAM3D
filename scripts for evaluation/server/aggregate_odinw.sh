#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

RUN_NAME="${1:-table3_odinw_text_only}"

activate_venv
run_from_project_root

python scripts/extract_odinw_results.py --res_dir "${OUT_ROOT}/${RUN_NAME}/logs"
