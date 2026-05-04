#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../server/common.sh"

RUN_NAME="table1_saco_gold"

SACO_GOLD_ANN="${SACO_GOLD_ANN:-${DATA_ROOT}/saco_gold/annotations}"
SACO_GOLD_METACLIP_IMG="${SACO_GOLD_METACLIP_IMG:-${DATA_ROOT}/saco_gold/metaclip_images}"
SACO_GOLD_SA1B_IMG="${SACO_GOLD_SA1B_IMG:-${DATA_ROOT}/saco_gold/sa1b_images}"
SACO_GOLD_RESOLUTION="${SACO_GOLD_RESOLUTION:-896}"

CONFIGS=(
  "configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml"
  "configs/gold_image_evals/sam3_gold_image_sa1b_nps.yaml"
  "configs/gold_image_evals/sam3_gold_image_attributes.yaml"
  "configs/gold_image_evals/sam3_gold_image_crowded.yaml"
  "configs/gold_image_evals/sam3_gold_image_wiki_common.yaml"
  "configs/gold_image_evals/sam3_gold_image_fg_food.yaml"
  "configs/gold_image_evals/sam3_gold_image_fg_sports.yaml"
)

SUBSETS=(
  "metaclip_nps"
  "sa1b_nps"
  "attributes"
  "crowded"
  "wiki_common"
  "fg_food"
  "fg_sports_equipment"
)

require_dir() {
  local dir_path="$1"
  local label="$2"
  if [[ ! -d "${dir_path}" ]]; then
    echo "Missing ${label} at ${dir_path}" >&2
    exit 2
  fi
}

ensure_bbox_pred() {
  local pred_dir="$1"
  local segm_file="${pred_dir}/coco_predictions_segm.json"
  local bbox_file="${pred_dir}/coco_predictions_bbox.json"
  if [[ -f "${bbox_file}" ]]; then
    return 0
  fi
  if [[ -f "${segm_file}" ]]; then
    cp "${segm_file}" "${bbox_file}"
  fi
}

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  activate_venv
fi
run_from_project_root
ensure_var CHECKPOINT_PATH
ensure_dir "${OUT_ROOT}/${RUN_NAME}"
LOG_DIR="${OUT_ROOT}/${RUN_NAME}/logs"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
ensure_dir "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

require_dir "${SACO_GOLD_ANN}" "SA-Co/Gold annotations"
require_dir "${SACO_GOLD_METACLIP_IMG}" "SA-Co/Gold MetaCLIP images"
require_dir "${SACO_GOLD_SA1B_IMG}" "SA-Co/Gold SA-1B images"

for cfg in "${CONFIGS[@]}"; do
  "${PYTHON_BIN}" sam3/train/train.py \
    -c "${cfg}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    paths.base_experiment_log_dir="${OUT_ROOT}/${RUN_NAME}" \
    paths.base_annotation_path="${SACO_GOLD_ANN}" \
    paths.metaclip_img_path="${SACO_GOLD_METACLIP_IMG}" \
    paths.sa1b_img_path="${SACO_GOLD_SA1B_IMG}" \
    paths.bpe_path="${BPE_PATH}" \
    paths.checkpoint_path="${CHECKPOINT_PATH}" \
    scratch.val_batch_size="${VAL_BATCH_SIZE}" \
    scratch.num_val_workers="${NUM_WORKERS}" \
    scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
    scratch.resolution="${SACO_GOLD_RESOLUTION}"
done

"${PYTHON_BIN}" scripts/eval/gold/eval_sam3.py \
  --gt-folder "${SACO_GOLD_ANN}" \
  --pred-folder "${OUT_ROOT}/${RUN_NAME}" \
  --iou-type segm

for subset in "${SUBSETS[@]}"; do
  ensure_bbox_pred "${OUT_ROOT}/${RUN_NAME}/gold_${subset}/dumps/gold_${subset}"
done

"${PYTHON_BIN}" scripts/eval/gold/eval_sam3.py \
  --gt-folder "${SACO_GOLD_ANN}" \
  --pred-folder "${OUT_ROOT}/${RUN_NAME}" \
  --iou-type bbox
