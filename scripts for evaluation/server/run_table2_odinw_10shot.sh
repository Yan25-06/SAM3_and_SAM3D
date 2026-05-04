#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SEED_TAG="${1:-300}"
START_IDX="${2:-0}"
END_IDX="${3:-12}"

case "${SEED_TAG}" in
  300)
    TRAIN_FILE="fewshot_train_shot10_seed300"
    ;;
  30)
    TRAIN_FILE="fewshot_train_shot10_seed30"
    ;;
  3)
    TRAIN_FILE="fewshot_train_shot10_seed3"
    ;;
  *)
    echo "Usage: $0 [300|30|3] [start_idx] [end_idx]" >&2
    exit 1
    ;;
esac

RUN_NAME="table2_odinw_10shot_seed${SEED_TAG}"
RESOLUTIONS="${RESOLUTIONS:-840 768 672}"
VAL_CATEGORY_CHUNK_SIZE="${VAL_CATEGORY_CHUNK_SIZE:-4}"
FREEZE_IMAGE_TOWER="${FREEZE_IMAGE_TOWER:-NoFreeze}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

activate_venv
run_from_project_root
ensure_dir "${OUT_ROOT}/${RUN_NAME}"
ensure_var CHECKPOINT_PATH

for idx in $(seq "${START_IDX}" "${END_IDX}"); do
  status=1
  for resolution in ${RESOLUTIONS}; do
    echo "[RUN ] task_index=${idx} | resolution=${resolution} | train_batch_size=${TRAIN_BATCH_SIZE} | val_category_chunk_size=${VAL_CATEGORY_CHUNK_SIZE} | freeze_image_tower=${FREEZE_IMAGE_TOWER}"
    if env PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
      python sam3/train/train.py \
        -c configs/odinw13/odinw_text_only_train.yaml \
        --use-cluster 0 \
        --num-gpus "${NUM_GPUS}" \
        paths.odinw_data_root="${ODINW_ROOT}" \
        paths.experiment_log_dir="${OUT_ROOT}/${RUN_NAME}" \
        paths.bpe_path="${BPE_PATH}" \
        +trainer.model.checkpoint_path="${CHECKPOINT_PATH}" \
        submitit.job_array.task_index="${idx}" \
        odinw_train.train_file="${TRAIN_FILE}" \
        scratch.train_batch_size="${TRAIN_BATCH_SIZE}" \
        scratch.val_batch_size="${VAL_BATCH_SIZE}" \
        scratch.num_train_workers="${TRAIN_WORKERS}" \
        scratch.num_val_workers="${NUM_WORKERS}" \
        scratch.val_category_chunk_size="${VAL_CATEGORY_CHUNK_SIZE}" \
        scratch.freeze_image_tower="${FREEZE_IMAGE_TOWER}" \
        scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
        scratch.resolution="${resolution}"; then
      status=0
      break
    else
      status=$?
      echo "[WARN] task_index=${idx} failed at resolution=${resolution} with exit code ${status}"
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "[ERR ] task_index=${idx} failed for all resolutions: ${RESOLUTIONS}" >&2
    exit "${status}"
  fi
done
