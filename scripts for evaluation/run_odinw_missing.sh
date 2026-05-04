#!/usr/bin/env bash

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

CONFIG_ARG="${CONFIG_ARG:-configs/odinw13/odinw_text_only.yaml}"
CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/sam3/train/configs/odinw13/odinw_text_only.yaml}"
ODINW_ROOT="${ODINW_ROOT:-$REPO_ROOT/data/odinw13}"
OUT_DIR="${OUT_DIR:-/home/svo/Downloads/sam3/outputs/table2_odinw_zero_shot}"
BPE_PATH="${BPE_PATH:-$REPO_ROOT/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints/sam3.pt}"
NUM_GPUS="${NUM_GPUS:-1}"
USE_CLUSTER="${USE_CLUSTER:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESOLUTIONS="${RESOLUTIONS:-1008}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASETS=(
  "AerialMaritimeDrone_large"
  "Aquarium"
  "CottontailRabbits"
  "EgoHands_generic"
  "NorthAmericaMushrooms"
  "Packages"
  "PascalVOC"
  "Raccoon"
  "ShellfishOpenImages"
  "VehiclesOpenImages"
  "pistols"
  "pothole"
  "thermalDogsAndPeople"
)

SKIPPED=()
COMPLETED=()
FAILED=()

print_header() {
  echo "Repo root : $REPO_ROOT"
  echo "Config arg: $CONFIG_ARG"
  echo "Config fs : $CONFIG_FILE"
  echo "ODINW_ROOT: $ODINW_ROOT"
  echo "OUT_DIR   : $OUT_DIR"
  echo "BPE_PATH  : $BPE_PATH"
  echo "CKPT_PATH : $CKPT_PATH"
  echo "Batch size: $BATCH_SIZE"
  echo "Resolutions: $RESOLUTIONS"
  echo
}

has_valid_val_json() {
  local json_path="$1"
  if [[ ! -s "$json_path" ]]; then
    return 1
  fi

  "$PYTHON_BIN" - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit(1)

last = json.loads(lines[-1])
if not isinstance(last, dict) or not last:
    raise SystemExit(1)

if not any(k.endswith("coco_eval_bbox_AP") for k in last):
    raise SystemExit(1)
PY
}

backup_invalid_json_if_needed() {
  local json_path="$1"
  if [[ ! -e "$json_path" ]]; then
    return 0
  fi

  if has_valid_val_json "$json_path"; then
    return 0
  fi

  local backup_path="${json_path}.invalid.$(date +%Y%m%d_%H%M%S).bak"
  mv "$json_path" "$backup_path"
  echo "Backed up invalid val_stats: $backup_path"
}

run_dataset() {
  local dataset_name="$1"
  local task_index="$2"
  local log_dir="$OUT_DIR/logs/$dataset_name"
  local json_path="$log_dir/val_stats.json"

  if has_valid_val_json "$json_path"; then
    echo "[SKIP] $dataset_name already has a valid val_stats.json"
    SKIPPED+=("$dataset_name")
    return 0
  fi

  mkdir -p "$log_dir"
  backup_invalid_json_if_needed "$json_path"

  local resolution
  local status=1

  for resolution in $RESOLUTIONS; do
    echo "[RUN ] $dataset_name | task_index=$task_index | resolution=$resolution | batch_size=$BATCH_SIZE"

    if env PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
      "$PYTHON_BIN" sam3/train/train.py \
        -c "$CONFIG_ARG" \
        --use-cluster "$USE_CLUSTER" \
        --num-gpus "$NUM_GPUS" \
        "paths.odinw_data_root=$ODINW_ROOT" \
        "paths.experiment_log_dir=$OUT_DIR" \
        "paths.bpe_path=$BPE_PATH" \
        "submitit.job_array.task_index=$task_index" \
        "scratch.enable_segmentation=false" \
        "scratch.val_batch_size=$BATCH_SIZE" \
        "scratch.resolution=$resolution" \
        "+trainer.model.checkpoint_path=$CKPT_PATH" \
        "+trainer.model.load_from_HF=false"; then
      if has_valid_val_json "$json_path"; then
        echo "[ OK ] $dataset_name finished successfully"
        COMPLETED+=("$dataset_name")
        return 0
      fi

      echo "[WARN] $dataset_name finished but val_stats.json is still missing or invalid"
    else
      status=$?
      echo "[FAIL] $dataset_name failed with exit code $status at resolution=$resolution"
    fi
  done

  echo "[ERR ] $dataset_name could not be completed after trying all resolutions"
  FAILED+=("$dataset_name")
  return 1
}

print_summary() {
  echo
  echo "Summary"
  echo "Skipped  : ${#SKIPPED[@]}"
  for item in "${SKIPPED[@]}"; do
    echo "  - $item"
  done

  echo "Completed: ${#COMPLETED[@]}"
  for item in "${COMPLETED[@]}"; do
    echo "  - $item"
  done

  echo "Failed   : ${#FAILED[@]}"
  for item in "${FAILED[@]}"; do
    echo "  - $item"
  done
}

main() {
  print_header

  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config not found: $CONFIG_FILE" >&2
    exit 2
  fi

  if [[ ! -d "$ODINW_ROOT" ]]; then
    echo "ODINW_ROOT not found: $ODINW_ROOT" >&2
    exit 2
  fi

  if [[ ! -f "$BPE_PATH" ]]; then
    echo "BPE_PATH not found: $BPE_PATH" >&2
    exit 2
  fi

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "CKPT_PATH not found: $CKPT_PATH" >&2
    exit 2
  fi

  mkdir -p "$OUT_DIR/logs"

  local idx
  for idx in "${!DATASETS[@]}"; do
    run_dataset "${DATASETS[$idx]}" "$idx"
  done

  print_summary

  if [[ "${#FAILED[@]}" -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
