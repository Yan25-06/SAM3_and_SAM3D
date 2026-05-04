#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

activate_venv
run_from_project_root
ensure_dir "${HF_HOME}"
ensure_var HF_MODEL_REPO

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi
export HF_HOME

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_MODEL_REPO"]
cache_dir = os.environ["HF_HOME"]

path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
print(f"Downloaded or reused cache for {repo_id}: {path}")
PY
