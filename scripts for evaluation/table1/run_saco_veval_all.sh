#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../server/common.sh"

RUN_NAME="table1_saco_veval"

SACO_VEVAL_ANN_DIR="${SACO_VEVAL_ANN_DIR:-${DATA_ROOT}/annotation}"
SACO_VEVAL_MEDIA_DIR="${SACO_VEVAL_MEDIA_DIR:-${DATA_ROOT}/media}"
SACO_VEVAL_SAV_DIR="${SACO_VEVAL_SAV_DIR:-${SACO_VEVAL_MEDIA_DIR}/saco_sav/JPEGImages_24fps}"
SACO_VEVAL_YT1B_DIR="${SACO_VEVAL_YT1B_DIR:-${SACO_VEVAL_MEDIA_DIR}/saco_yt1b/JPEGImages_6fps}"
SACO_VEVAL_SG_DIR="${SACO_VEVAL_SG_DIR:-${SACO_VEVAL_MEDIA_DIR}/saco_sg/JPEGImages_6fps}"

DATASETS=(
  "saco_veval_sav_test|configs/saco_video_evals/saco_veval_sav_test.yaml|${SACO_VEVAL_ANN_DIR}/saco_veval_sav_test.json|${SACO_VEVAL_SAV_DIR}"
  "saco_veval_sav_val|configs/saco_video_evals/saco_veval_sav_val.yaml|${SACO_VEVAL_ANN_DIR}/saco_veval_sav_val.json|${SACO_VEVAL_SAV_DIR}"
  "saco_veval_yt1b_test|configs/saco_video_evals/saco_veval_yt1b_test.yaml|${SACO_VEVAL_ANN_DIR}/saco_veval_yt1b_test.json|${SACO_VEVAL_YT1B_DIR}"
  "saco_veval_yt1b_val|configs/saco_video_evals/saco_veval_yt1b_val.yaml|${SACO_VEVAL_ANN_DIR}/saco_veval_yt1b_val.json|${SACO_VEVAL_YT1B_DIR}"
  "saco_veval_smartglasses_test|configs/saco_video_evals/saco_veval_smartglasses_test.yaml|${SACO_VEVAL_ANN_DIR}/saco_veval_smartglasses_test.json|${SACO_VEVAL_SG_DIR}"
  "saco_veval_smartglasses_val|configs/saco_video_evals/saco_veval_smartglasses_val.yaml|${SACO_VEVAL_ANN_DIR}/saco_veval_smartglasses_val.json|${SACO_VEVAL_SG_DIR}"
)

require_dir() {
  local dir_path="$1"
  local label="$2"
  if [[ ! -d "${dir_path}" ]]; then
    echo "Missing ${label} at ${dir_path}" >&2
    exit 2
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

require_dir "${SACO_VEVAL_ANN_DIR}" "SA-Co/VEval annotations"
require_dir "${SACO_VEVAL_SAV_DIR}" "SA-Co/VEval SA-V frames"
require_dir "${SACO_VEVAL_YT1B_DIR}" "SA-Co/VEval YT1B frames"
require_dir "${SACO_VEVAL_SG_DIR}" "SA-Co/VEval SmartGlasses frames"

PRED_DIR="${OUT_ROOT}/${RUN_NAME}/preds"
EVAL_DIR="${OUT_ROOT}/${RUN_NAME}/eval"
ensure_dir "${PRED_DIR}"
ensure_dir "${EVAL_DIR}"

for item in "${DATASETS[@]}"; do
  IFS="|" read -r dataset_name config_path gt_path media_dir <<<"${item}"

  if [[ ! -f "${gt_path}" ]]; then
    echo "Missing GT file: ${gt_path}" >&2
    exit 2
  fi

  exp_dir="${OUT_ROOT}/${RUN_NAME}/${dataset_name}"
  ensure_dir "${exp_dir}"

  "${PYTHON_BIN}" sam3/train/train.py \
    -c "${config_path}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    paths.experiment_log_dir="${exp_dir}" \
    paths.ytvis_json="${gt_path}" \
    paths.ytvis_dir="${media_dir}" \
    paths.bpe_path="${BPE_PATH}" \
    paths.dump_file_name="${dataset_name}_preds" \
    trainer.model.checkpoint_path="${CHECKPOINT_PATH}" \
    scratch.val_batch_size="${VAL_BATCH_SIZE}" \
    scratch.num_val_workers="${NUM_WORKERS}" \
    scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
    scratch.resolution="${SAM3_RESOLUTION}"

  pred_src="${exp_dir}/preds/${dataset_name}_preds.json"
  if [[ ! -f "${pred_src}" ]]; then
    echo "Missing predictions: ${pred_src}" >&2
    exit 2
  fi

  cp "${pred_src}" "${PRED_DIR}/${dataset_name}_preds.json"
done

"${PYTHON_BIN}" sam3/eval/saco_veval_eval.py all \
  --gt_annot_dir "${SACO_VEVAL_ANN_DIR}" \
  --pred_dir "${PRED_DIR}" \
  --eval_res_dir "${EVAL_DIR}"
