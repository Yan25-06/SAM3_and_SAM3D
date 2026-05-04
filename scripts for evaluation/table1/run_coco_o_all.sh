#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../server/common.sh"

RUN_NAME="table1_coco_o"
GT_FILE="${GT_FILE:-${DATA_ROOT}/coco_o/annotations/instances_val2017.json}"
IMG_DIR="${IMG_DIR:-${DATA_ROOT}/coco_o/val2017}"
CONFIG_PATH="${CONFIG_PATH:-configs/external_image_evals/sam3_table1_coco_o.yaml}"
CATEGORY_CHUNK_SIZE="${CATEGORY_CHUNK_SIZE:-20}"

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
  echo "Missing COCO-O GT file: ${GT_FILE}" >&2
  exit 2
fi

if [[ ! -d "${IMG_DIR}" ]]; then
  echo "Missing COCO-O image directory: ${IMG_DIR}" >&2
  exit 2
fi

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
  scratch.category_chunk_size="${CATEGORY_CHUNK_SIZE}"

PRED_FILE="${OUT_ROOT}/${RUN_NAME}/dumps/${RUN_NAME}/coco_predictions_bbox.json"
if [[ ! -f "${PRED_FILE}" ]]; then
  echo "Missing COCO-O predictions after inference: ${PRED_FILE}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${GT_FILE}" "${PRED_FILE}" "bbox" <<'PY'
import sys
from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators

gt_path, pred_path, iou_type = sys.argv[1:4]
evaluator = CocoEvaluatorOfflineWithPredFileEvaluators(
    gt_path=gt_path,
    tide=False,
    iou_type=iou_type,
    positive_split=False,
)
outs = evaluator.evaluate(pred_path)
for k, v in outs.items():
    print(f"{k}: {v}")
PY
