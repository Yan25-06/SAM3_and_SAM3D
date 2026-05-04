#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MODE="${1:-text}"
START_IDX="${2:-0}"
END_IDX="${3:-12}"

case "${MODE}" in
  text)
    CONFIG="configs/odinw13/odinw_text_only_positive.yaml"
    RUN_NAME="table3_odinw_text_only"
    ;;
  visual)
    CONFIG="configs/odinw13/odinw_visual_only.yaml"
    RUN_NAME="table3_odinw_visual_only"
    ;;
  text_visual)
    CONFIG="configs/odinw13/odinw_text_and_visual.yaml"
    RUN_NAME="table3_odinw_text_and_visual"
    ;;
  *)
    echo "Usage: $0 [text|visual|text_visual] [start_idx] [end_idx]" >&2
    exit 1
    ;;
esac

activate_venv
run_from_project_root
ensure_dir "${OUT_ROOT}/${RUN_NAME}"
ensure_var CHECKPOINT_PATH

for idx in $(seq "${START_IDX}" "${END_IDX}"); do
  python sam3/train/train.py \
    -c "${CONFIG}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    paths.odinw_data_root="${ODINW_ROOT}" \
    paths.experiment_log_dir="${OUT_ROOT}/${RUN_NAME}" \
    paths.bpe_path="${BPE_PATH}" \
    +trainer.model.checkpoint_path="${CHECKPOINT_PATH}" \
    submitit.job_array.task_index="${idx}" \
    scratch.val_batch_size="${VAL_BATCH_SIZE}" \
    scratch.num_val_workers="${NUM_WORKERS}" \
    scratch.gather_pred_via_filesys="${GATHER_PRED_VIA_FILESYS}" \
    scratch.resolution="${SAM3_RESOLUTION}"
done
