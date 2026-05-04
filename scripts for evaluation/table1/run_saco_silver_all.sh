#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../server/common.sh"

RUN_NAME="table1_saco_silver"

SACO_GOLD_ANN="${SACO_GOLD_ANN:-${DATA_ROOT}/saco_gold/annotations}"
SACO_GOLD_METACLIP_IMG="${SACO_GOLD_METACLIP_IMG:-${DATA_ROOT}/saco_gold/metaclip_images}"
SACO_GOLD_SA1B_IMG="${SACO_GOLD_SA1B_IMG:-${DATA_ROOT}/saco_gold/sa1b_images}"
SACO_SILVER_ANN="${SACO_SILVER_ANN:-${DATA_ROOT}/saco_silver/annotations}"
SACO_SILVER_IMG="${SACO_SILVER_IMG:-${DATA_ROOT}/saco_silver/images}"

CONFIGS=(
  "configs/silver_image_evals/sam3_silver_image_bdd100k.yaml"
  "configs/silver_image_evals/sam3_silver_image_droid.yaml"
  "configs/silver_image_evals/sam3_silver_image_ego4d.yaml"
  "configs/silver_image_evals/sam3_silver_image_fathomnet.yaml"
  "configs/silver_image_evals/sam3_silver_image_food_rec.yaml"
  "configs/silver_image_evals/sam3_silver_image_geode.yaml"
  "configs/silver_image_evals/sam3_silver_image_inaturalist.yaml"
  "configs/silver_image_evals/sam3_silver_image_nga.yaml"
  "configs/silver_image_evals/sam3_silver_image_sav.yaml"
  "configs/silver_image_evals/sam3_silver_image_yt1b.yaml"
)

declare -A SILVER_GTS=(
  [bdd100k]="silver_bdd100k_merged_test.json"
  [droid]="silver_droid_merged_test.json"
  [ego4d]="silver_ego4d_merged_test.json"
  [fathomnet]="silver_fathomnet_test.json"
  [food_rec]="silver_food_rec_merged_test.json"
  [geode]="silver_geode_merged_test.json"
  [inaturalist]="silver_inaturalist_merged_test.json"
  [nga_art]="silver_nga_art_merged_test.json"
  [sav]="silver_sav_merged_test.json"
  [yt1b]="silver_yt1b_merged_test.json"
)

require_dir() {
  local dir_path="$1"
  local label="$2"
  if [[ ! -d "${dir_path}" ]]; then
    echo "Missing ${label} at ${dir_path}" >&2
    exit 2
  fi
}

get_pred_file() {
  local pred_dir="$1"
  local iou_type="$2"
  local preferred="${pred_dir}/coco_predictions_${iou_type}.json"
  if [[ -f "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi
  if [[ "${iou_type}" == "bbox" ]]; then
    local fallback="${pred_dir}/coco_predictions_segm.json"
    if [[ -f "${fallback}" ]]; then
      echo "${fallback}"
      return 0
    fi
  fi
  return 1
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
require_dir "${SACO_SILVER_ANN}" "SA-Co/Silver annotations"
require_dir "${SACO_SILVER_IMG}" "SA-Co/Silver images"

for cfg in "${CONFIGS[@]}"; do
  "${PYTHON_BIN}" sam3/train/train.py \
    -c "${cfg}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    paths.base_experiment_log_dir="${OUT_ROOT}/${RUN_NAME}" \
    paths.base_annotation_path="${SACO_GOLD_ANN}" \
    paths.base_annotation_path_silver="${SACO_SILVER_ANN}" \
    paths.metaclip_img_path="${SACO_GOLD_METACLIP_IMG}" \
    paths.sa1b_img_path="${SACO_GOLD_SA1B_IMG}" \
    paths.silver_img_path="${SACO_SILVER_IMG}" \
    paths.bpe_path="${BPE_PATH}" \
    paths.checkpoint_path="${CHECKPOINT_PATH}" \
    scratch.val_batch_size="${VAL_BATCH_SIZE}" \
    scratch.num_val_workers="${NUM_WORKERS}" \
    scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
    scratch.resolution="${SAM3_RESOLUTION}"
done

for subset in "${!SILVER_GTS[@]}"; do
  gt_file="${SACO_SILVER_ANN}/${SILVER_GTS[${subset}]}"
  pred_dir="${OUT_ROOT}/${RUN_NAME}/silver_${subset}/dumps/silver_${subset}"

  if [[ ! -f "${gt_file}" ]]; then
    echo "Missing GT file: ${gt_file}" >&2
    exit 2
  fi

  for iou_type in segm bbox; do
    pred_file="$(get_pred_file "${pred_dir}" "${iou_type}")"
    if [[ -z "${pred_file}" ]]; then
      echo "Missing predictions in ${pred_dir} for ${subset} (${iou_type})" >&2
      exit 2
    fi

    "${PYTHON_BIN}" scripts/eval/standalone_cgf1.py \
      --pred_file "${pred_file}" \
      --gt_files "${gt_file}" \
      --iou_type "${iou_type}"
  done
done
