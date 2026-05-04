#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

activate_venv
RF100_VARIANT="${1:-full}"
download_rf100_variant "${RF100_VARIANT}"
