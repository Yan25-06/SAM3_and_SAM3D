#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../server/common.sh"

RUN_NAME="table1_lvis"
GT_FILE="${GT_FILE:-${DATA_ROOT}/lvis/annotations/lvis_v1_val.json}"
IMG_DIR="${IMG_DIR:-${DATA_ROOT}/coco}"
CONFIG_PATH="${CONFIG_PATH:-configs/external_image_evals/sam3_table1_lvis.yaml}"
CATEGORY_CHUNK_SIZE="${CATEGORY_CHUNK_SIZE:-20}"
MAX_EVAL_IMAGES="${MAX_EVAL_IMAGES:-20000}"

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  activate_venv
fi
run_from_project_root
ensure_bundled_config_path "${CONFIG_PATH}"

LOG_DIR="${OUT_ROOT}/${RUN_NAME}/logs"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
ensure_dir "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ ! -f "${GT_FILE}" ]]; then
  echo "Missing LVIS GT file: ${GT_FILE}" >&2
  exit 2
fi

if [[ ! -d "${IMG_DIR}" ]]; then
  echo "Missing LVIS image directory: ${IMG_DIR}" >&2
  exit 2
fi

IMG_PARENT_DIR="$(dirname "${IMG_DIR}")"
if [[ ! -d "${IMG_DIR}/train2017" && ! -d "${IMG_DIR}/val2017" \
   && ! -d "${IMG_PARENT_DIR}/train2017" && ! -d "${IMG_PARENT_DIR}/val2017" ]]; then
  echo "LVIS expects IMG_DIR to be the COCO image root containing train2017/ and val2017/." >&2
  echo "Current IMG_DIR: ${IMG_DIR}" >&2
  exit 2
fi

echo "[table1_lvis] Step 1/2: running model inference via sam3/train/train.py in val mode"
echo "[table1_lvis] Predictions will be dumped under ${OUT_ROOT}/${RUN_NAME}/dumps/${RUN_NAME}"
"${PYTHON_BIN}" sam3/train/train.py \
  -c "${CONFIG_PATH}" \
  --use-cluster 0 \
  --num-gpus "${NUM_GPUS}" \
  paths.base_experiment_log_dir="${OUT_ROOT}" \
  paths.img_path="${IMG_DIR}" \
  paths.coco_gt="${GT_FILE}" \
  paths.bpe_path="${BPE_PATH}" \
  paths.checkpoint_path="${CHECKPOINT_PATH}" \
  scratch.val_batch_size="${VAL_BATCH_SIZE}" \
  scratch.num_val_workers="${NUM_WORKERS}" \
  scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
  scratch.resolution="${SAM3_RESOLUTION}" \
  scratch.category_chunk_size="${CATEGORY_CHUNK_SIZE}" \
  scratch.max_eval_images="${MAX_EVAL_IMAGES}"

echo "[table1_lvis] Step 2/2: running official LVIS evaluation from dumped prediction files"
for iou_type in bbox segm; do
  PRED_FILE="${OUT_ROOT}/${RUN_NAME}/dumps/${RUN_NAME}/coco_predictions_${iou_type}.json"
  if [[ ! -f "${PRED_FILE}" ]]; then
    echo "Missing LVIS predictions after inference: ${PRED_FILE}" >&2
    exit 2
  fi

  "${PYTHON_BIN}" - "${GT_FILE}" "${PRED_FILE}" "${iou_type}" <<'PY'
import sys
from sam3.eval.lvis_eval_offline import LvisEvaluatorOfflineWithPredFileEvaluators

gt_path, pred_path, iou_type = sys.argv[1:4]
evaluator = LvisEvaluatorOfflineWithPredFileEvaluators(
    gt_path=gt_path,
    iou_type=iou_type,
    max_dets=300,
)
outs = evaluator.evaluate(pred_path)
for k, v in outs.items():
    print(f"{k}: {v}")
PY
done
