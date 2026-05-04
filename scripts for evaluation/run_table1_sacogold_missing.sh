#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/server/common.sh"

activate_venv
run_from_project_root
ensure_var CHECKPOINT_PATH

RUN_NAME="${RUN_NAME:-table1_saco_gold}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
SACO_GOLD_ANN="${SACO_GOLD_ANN:-${DATA_ROOT}/saco_gold/annotations}"
SACO_GOLD_METACLIP_IMG="${SACO_GOLD_METACLIP_IMG:-${DATA_ROOT}/saco_gold/metaclip_images}"
SACO_GOLD_SA1B_IMG="${SACO_GOLD_SA1B_IMG:-${DATA_ROOT}/saco_gold/sa1b_images}"
SACO_GOLD_RESOLUTION="${SACO_GOLD_RESOLUTION:-896}"
CKPT_PATH="${CHECKPOINT_PATH}"
RESOLUTIONS="${RESOLUTIONS:-${SACO_GOLD_RESOLUTION}}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASETS=(
  "metaclip_nps|configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml|${PROJECT_ROOT}/sam3/train/configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml"
  "sa1b_nps|configs/gold_image_evals/sam3_gold_image_sa1b_nps.yaml|${PROJECT_ROOT}/sam3/train/configs/gold_image_evals/sam3_gold_image_sa1b_nps.yaml"
  "attributes|configs/gold_image_evals/sam3_gold_image_attributes.yaml|${PROJECT_ROOT}/sam3/train/configs/gold_image_evals/sam3_gold_image_attributes.yaml"
  "crowded|configs/gold_image_evals/sam3_gold_image_crowded.yaml|${PROJECT_ROOT}/sam3/train/configs/gold_image_evals/sam3_gold_image_crowded.yaml"
  "wiki_common|configs/gold_image_evals/sam3_gold_image_wiki_common.yaml|${PROJECT_ROOT}/sam3/train/configs/gold_image_evals/sam3_gold_image_wiki_common.yaml"
  "fg_food|configs/gold_image_evals/sam3_gold_image_fg_food.yaml|${PROJECT_ROOT}/sam3/train/configs/gold_image_evals/sam3_gold_image_fg_food.yaml"
  "fg_sports_equipment|configs/gold_image_evals/sam3_gold_image_fg_sports.yaml|${PROJECT_ROOT}/sam3/train/configs/gold_image_evals/sam3_gold_image_fg_sports.yaml"
)

SKIPPED=()
COMPLETED=()
FAILED=()

require_dir() {
  local dir_path="$1"
  local label="$2"
  if [[ ! -d "${dir_path}" ]]; then
    echo "Missing ${label} at ${dir_path}" >&2
    exit 2
  fi
}

print_header() {
  echo "Project root: ${PROJECT_ROOT}"
  echo "OUT_DIR     : ${OUT_DIR}"
  echo "Annotations : ${SACO_GOLD_ANN}"
  echo "MetaCLIP    : ${SACO_GOLD_METACLIP_IMG}"
  echo "SA-1B       : ${SACO_GOLD_SA1B_IMG}"
  echo "BPE_PATH    : ${BPE_PATH}"
  echo "CKPT_PATH   : ${CKPT_PATH}"
  echo "Workers     : ${NUM_WORKERS}"
  echo "Batch size  : ${VAL_BATCH_SIZE}"
  echo "Resolutions : ${RESOLUTIONS}"
  echo
}

pred_file_for_subset() {
  local subset_name="$1"
  echo "${OUT_DIR}/gold_${subset_name}/dumps/gold_${subset_name}/coco_predictions_segm.json"
}

bbox_file_for_subset() {
  local subset_name="$1"
  echo "${OUT_DIR}/gold_${subset_name}/dumps/gold_${subset_name}/coco_predictions_bbox.json"
}

has_valid_prediction_json() {
  local json_path="$1"
  if [[ ! -s "${json_path}" ]]; then
    return 1
  fi

  "${PYTHON_BIN}" - "${json_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as f:
    obj = json.load(f)

if not isinstance(obj, (list, dict)):
    raise SystemExit(1)
PY
}

backup_invalid_file_if_needed() {
  local json_path="$1"
  if [[ ! -e "${json_path}" ]]; then
    return 0
  fi

  if has_valid_prediction_json "${json_path}"; then
    return 0
  fi

  local backup_path="${json_path}.invalid.$(date +%Y%m%d_%H%M%S).bak"
  mv "${json_path}" "${backup_path}"
  echo "Backed up invalid prediction file: ${backup_path}"
}

print_log_tail() {
  local log_path="$1"
  local line_count="${2:-80}"

  if [[ ! -f "${log_path}" ]]; then
    return 0
  fi

  echo "[LOG ] Last ${line_count} lines from ${log_path}"
  tail -n "${line_count}" "${log_path}"
}

ensure_bbox_pred() {
  local subset_name="$1"
  local segm_file
  local bbox_file
  segm_file="$(pred_file_for_subset "${subset_name}")"
  bbox_file="$(bbox_file_for_subset "${subset_name}")"

  if [[ -f "${bbox_file}" ]]; then
    return 0
  fi

  if [[ -f "${segm_file}" ]]; then
    cp "${segm_file}" "${bbox_file}"
  fi
}

run_subset() {
  local subset_name="$1"
  local config_arg="$2"
  local config_file="$3"
  local pred_file
  local attempt_dir

  pred_file="$(pred_file_for_subset "${subset_name}")"
  attempt_dir="${OUT_DIR}/logs/${subset_name}"

  if has_valid_prediction_json "${pred_file}"; then
    echo "[SKIP] ${subset_name} already has a valid prediction dump"
    ensure_bbox_pred "${subset_name}"
    SKIPPED+=("${subset_name}")
    return 0
  fi

  mkdir -p "${attempt_dir}"
  backup_invalid_file_if_needed "${pred_file}"

  local resolution
  local status=1

  for resolution in ${RESOLUTIONS}; do
    local attempt_log="${attempt_dir}/run_resolution_${resolution}.log"
    echo "[RUN ] ${subset_name} | resolution=${resolution} | batch_size=${VAL_BATCH_SIZE}"

    if env PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
      "${PYTHON_BIN}" sam3/train/train.py \
        -c "${config_arg}" \
        --use-cluster 0 \
        --num-gpus "${NUM_GPUS}" \
        "paths.base_experiment_log_dir=${OUT_DIR}" \
        "paths.base_annotation_path=${SACO_GOLD_ANN}" \
        "paths.metaclip_img_path=${SACO_GOLD_METACLIP_IMG}" \
        "paths.sa1b_img_path=${SACO_GOLD_SA1B_IMG}" \
        "paths.bpe_path=${BPE_PATH}" \
        "paths.checkpoint_path=${CKPT_PATH}" \
        "scratch.val_batch_size=${VAL_BATCH_SIZE}" \
        "scratch.num_val_workers=${NUM_WORKERS}" \
        "scratch.gather_pred_via_filesys=${GATHER_PRED_VIA_FILESYS}" \
        "scratch.resolution=${resolution}" >"${attempt_log}" 2>&1; then
      if has_valid_prediction_json "${pred_file}"; then
        echo "[ OK ] ${subset_name} finished successfully"
        ensure_bbox_pred "${subset_name}"
        COMPLETED+=("${subset_name}")
        return 0
      fi

      echo "[WARN] ${subset_name} finished but prediction dump is still missing or invalid"
      echo "[WARN] Check log: ${attempt_log}"
      print_log_tail "${attempt_log}" 60
    else
      status=$?
      echo "[FAIL] ${subset_name} failed with exit code ${status} at resolution=${resolution}"
      echo "[FAIL] Check log: ${attempt_log}"
      print_log_tail "${attempt_log}" 80
    fi
  done

  echo "[ERR ] ${subset_name} could not be completed after trying all resolutions"
  FAILED+=("${subset_name}")
  return 1
}

all_subsets_ready() {
  local item
  local subset_name

  for item in "${DATASETS[@]}"; do
    IFS="|" read -r subset_name _ _ <<<"${item}"
    if ! has_valid_prediction_json "$(pred_file_for_subset "${subset_name}")"; then
      return 1
    fi
  done

  return 0
}

run_aggregate_eval() {
  local item
  local subset_name

  echo
  echo "Running aggregated SA-Co/Gold evaluation"

  "${PYTHON_BIN}" scripts/eval/gold/eval_sam3.py \
    --gt-folder "${SACO_GOLD_ANN}" \
    --pred-folder "${OUT_DIR}" \
    --iou-type segm

  for item in "${DATASETS[@]}"; do
    IFS="|" read -r subset_name _ _ <<<"${item}"
    ensure_bbox_pred "${subset_name}"
  done

  "${PYTHON_BIN}" scripts/eval/gold/eval_sam3.py \
    --gt-folder "${SACO_GOLD_ANN}" \
    --pred-folder "${OUT_DIR}" \
    --iou-type bbox
}

print_summary() {
  echo
  echo "Summary"
  echo "Skipped  : ${#SKIPPED[@]}"
  for item in "${SKIPPED[@]}"; do
    echo "  - ${item}"
  done

  echo "Completed: ${#COMPLETED[@]}"
  for item in "${COMPLETED[@]}"; do
    echo "  - ${item}"
  done

  echo "Failed   : ${#FAILED[@]}"
  for item in "${FAILED[@]}"; do
    echo "  - ${item}"
  done
}

main() {
  print_header

  require_dir "${SACO_GOLD_ANN}" "SA-Co/Gold annotations"
  require_dir "${SACO_GOLD_METACLIP_IMG}" "SA-Co/Gold MetaCLIP images"
  require_dir "${SACO_GOLD_SA1B_IMG}" "SA-Co/Gold SA-1B images"
  ensure_dir "${OUT_DIR}/logs"

  if [[ ! -f "${BPE_PATH}" ]]; then
    echo "BPE_PATH not found: ${BPE_PATH}" >&2
    exit 2
  fi

  local item
  local subset_name
  local config_arg
  local config_file

  for item in "${DATASETS[@]}"; do
    IFS="|" read -r subset_name config_arg config_file <<<"${item}"

    if [[ ! -f "${config_file}" ]]; then
      echo "Config not found: ${config_file}" >&2
      FAILED+=("${subset_name}")
      continue
    fi

    if ! run_subset "${subset_name}" "${config_arg}" "${config_file}"; then
      continue
    fi
  done

  print_summary

  if all_subsets_ready; then
    run_aggregate_eval
  else
    echo
    echo "Skipping aggregated eval because some subsets are still missing predictions."
  fi

  if [[ "${#FAILED[@]}" -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
