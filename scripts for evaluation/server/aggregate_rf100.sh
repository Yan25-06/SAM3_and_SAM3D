#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

RUN_NAME="${1:-table2_rf100_ft100}"

activate_venv
run_from_project_root

python scripts/extract_roboflow_vl100_results.py --path "${OUT_ROOT}/${RUN_NAME}"
