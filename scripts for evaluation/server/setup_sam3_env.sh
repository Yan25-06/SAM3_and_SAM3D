#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ensure_dir "${DATA_ROOT}"
ensure_dir "${OUT_ROOT}"
ensure_dir "${CACHE_ROOT}"
ensure_dir "${HF_HOME}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Cannot find ${PYTHON_BIN}. Set PYTHON_BIN in sam3_server.env." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

activate_venv
run_from_project_root

python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.10.0 torchvision --index-url "${PYTORCH_INDEX_URL}"
python -m pip install -e ".[train,dev]"
python -m pip install "huggingface_hub[cli]" rf100vl

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("gpu_name:", torch.cuda.get_device_name(0))
PY
