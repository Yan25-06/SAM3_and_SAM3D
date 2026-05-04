#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MODE="${1:-text}"
GT_FILE="${GT_FILE:-${DATA_ROOT}/lvis/annotations/lvis_v1_val.json}"
IMG_DIR="${IMG_DIR:-${DATA_ROOT}/coco}"
CATEGORY_CHUNK_SIZE="${CATEGORY_CHUNK_SIZE:-20}"
OUT_DIR_OVERRIDE="${OUT_DIR:-}"

normalize_mode() {
  local mode_key="${1,,}"
  mode_key="${mode_key// /_}"
  mode_key="${mode_key//+/_}"
  mode_key="${mode_key//-/_}"
  echo "${mode_key}"
}

mode_to_config() {
  local mode_key
  mode_key="$(normalize_mode "$1")"
  case "${mode_key}" in
    text|text_only)
      echo "configs/external_image_evals/sam3_table3_lvis_text_only.yaml|table3_lvis_text_only"
      ;;
    visual|visual_only|image|image_only)
      echo "configs/external_image_evals/sam3_table3_lvis_visual_only.yaml|table3_lvis_visual_only"
      ;;
    text_visual|text_and_visual|textvisual)
      echo "configs/external_image_evals/sam3_table3_lvis_text_and_visual.yaml|table3_lvis_text_and_visual"
      ;;
    *)
      return 1
      ;;
  esac
}

run_one_mode() {
  local requested_mode="$1"
  local config_pair
  config_pair="$(mode_to_config "${requested_mode}")" || {
    echo "Usage: $0 [text|visual|text_visual|all]" >&2
    return 2
  }

  local config_path="${config_pair%%|*}"
  local run_name="${config_pair##*|}"
  local out_dir
  if [[ -n "${OUT_DIR_OVERRIDE}" ]]; then
    if [[ "$(normalize_mode "${MODE}")" == "all" ]]; then
      out_dir="${OUT_DIR_OVERRIDE}/${run_name}"
    else
      out_dir="${OUT_DIR_OVERRIDE}"
    fi
  else
    out_dir="${OUT_ROOT}/${run_name}"
  fi
  local log_dir="${out_dir}/logs"
  local log_file="${log_dir}/run_$(date +%Y%m%d_%H%M%S).log"
  local pred_file="${out_dir}/dumps/table3_lvis/coco_predictions_bbox.json"

  ensure_bundled_config_path "${config_path}"
  ensure_dir "${log_dir}"

  echo "[table3_lvis] mode=${requested_mode}"
  echo "[table3_lvis] config=${config_path}"
  echo "[table3_lvis] output=${out_dir}"

  (
    exec > >(tee -a "${log_file}") 2>&1

    echo "[table3_lvis] Step 1/2: running model inference via sam3/train/train.py"
    "${PYTHON_BIN}" sam3/train/train.py \
      -c "${config_path}" \
      --use-cluster 0 \
      --num-gpus "${NUM_GPUS}" \
      paths.base_experiment_log_dir="${OUT_ROOT}" \
      paths.experiment_log_dir="${out_dir}" \
      paths.img_path="${IMG_DIR}" \
      paths.coco_gt="${GT_FILE}" \
      paths.bpe_path="${BPE_PATH}" \
      paths.checkpoint_path="${CHECKPOINT_PATH}" \
      scratch.val_batch_size="${VAL_BATCH_SIZE}" \
      scratch.num_val_workers="${NUM_WORKERS}" \
      scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
      scratch.resolution="${SAM3_RESOLUTION}" \
      scratch.category_chunk_size="${CATEGORY_CHUNK_SIZE}"

    if [[ ! -f "${pred_file}" ]]; then
      echo "Missing LVIS predictions after inference: ${pred_file}" >&2
      exit 2
    fi

    echo "[table3_lvis] Step 2/2: running official LVIS bbox evaluation from dumped prediction file"
    "${PYTHON_BIN}" - "${GT_FILE}" "${pred_file}" <<'PY'
import sys
from sam3.eval.lvis_eval_offline import LvisEvaluatorOfflineWithPredFileEvaluators

gt_path, pred_path = sys.argv[1:3]
evaluator = LvisEvaluatorOfflineWithPredFileEvaluators(
    gt_path=gt_path,
    iou_type="bbox",
    max_dets=300,
)
outs = evaluator.evaluate(pred_path)
for key in sorted(outs):
    print(f"{key}: {outs[key]}")
PY
  )
}

main() {
  if [[ "${SKIP_VENV:-0}" != "1" ]]; then
    activate_venv
  fi
  run_from_project_root
  ensure_var CHECKPOINT_PATH

  if [[ ! -f "${GT_FILE}" ]]; then
    echo "Missing LVIS GT file: ${GT_FILE}" >&2
    exit 2
  fi

  if [[ ! -d "${IMG_DIR}" ]]; then
    echo "Missing LVIS image directory: ${IMG_DIR}" >&2
    exit 2
  fi

  local img_parent_dir
  img_parent_dir="$(dirname "${IMG_DIR}")"
  if [[ ! -d "${IMG_DIR}/train2017" && ! -d "${IMG_DIR}/val2017" \
     && ! -d "${img_parent_dir}/train2017" && ! -d "${img_parent_dir}/val2017" ]]; then
    echo "LVIS expects IMG_DIR to be the COCO image root containing train2017/ and val2017/." >&2
    echo "Current IMG_DIR: ${IMG_DIR}" >&2
    exit 2
  fi

  local mode_key
  mode_key="$(normalize_mode "${MODE}")"
  case "${mode_key}" in
    all)
      local overall_status=0
      run_one_mode text || overall_status=$?
      run_one_mode visual || overall_status=$?
      run_one_mode text_visual || overall_status=$?
      exit "${overall_status}"
      ;;
    *)
      run_one_mode "${MODE}"
      ;;
  esac
}

main "$@"
