#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

activate_venv
ensure_dir "${ODINW_ROOT}"

python - <<'PY'
import os
import tempfile
import urllib.request
import zipfile

root = "https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35"
datasets = [
    "AerialMaritimeDrone",
    "Aquarium",
    "CottontailRabbits",
    "EgoHands",
    "NorthAmericaMushrooms",
    "Packages",
    "PascalVOC",
    "Raccoon",
    "ShellfishOpenImages",
    "VehiclesOpenImages",
    "pistols",
    "pothole",
    "thermalDogsAndPeople",
]
out_root = os.environ["ODINW_ROOT"]

for name in datasets:
    target_dir = os.path.join(out_root, name)
    if os.path.isdir(target_dir):
        print(f"[skip] {name} already exists")
        continue

    url = f"{root}/{name}.zip"
    print(f"[download] {url}")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, f"{name}.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_root)
    print(f"[done] {name}")
PY
