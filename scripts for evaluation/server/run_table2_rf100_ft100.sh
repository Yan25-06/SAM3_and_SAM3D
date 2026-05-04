#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

START_IDX="${1:-0}"
END_IDX="${2:-99}"
RUN_NAME="table2_rf100_ft100"

activate_venv
run_from_project_root
ensure_dir "${OUT_ROOT}/${RUN_NAME}"
ensure_rf100_ready full
ensure_var CHECKPOINT_PATH
ensure_file_exists "${CHECKPOINT_PATH}"

for idx in $(seq "${START_IDX}" "${END_IDX}"); do
  python sam3/train/train.py \
    -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    paths.roboflow_vl_100_root="${RF100_ROOT}" \
    paths.experiment_log_dir="${OUT_ROOT}/${RUN_NAME}" \
    paths.bpe_path="${BPE_PATH}" \
    +trainer.model.checkpoint_path="${CHECKPOINT_PATH}" \
    submitit.job_array.task_index="${idx}" \
    scratch.train_batch_size="${TRAIN_BATCH_SIZE}" \
    scratch.val_batch_size="${VAL_BATCH_SIZE}" \
    scratch.num_train_workers="${TRAIN_WORKERS}" \
    scratch.num_val_workers="${NUM_WORKERS}" \
    scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
    scratch.resolution="${SAM3_RESOLUTION}"
done
