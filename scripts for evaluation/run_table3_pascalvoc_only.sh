#!/usr/bin/env bash

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

MODE="${1:-all}"
ODINW_ROOT="${ODINW_ROOT:-$REPO_ROOT/data/odinw13}"
BPE_PATH="${BPE_PATH:-$REPO_ROOT/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints/sam3.pt}"
OUT_ROOT="${OUT_ROOT:-/home/svo/Downloads/sam3/outputs}"
NUM_GPUS="${NUM_GPUS:-1}"
USE_CLUSTER="${USE_CLUSTER:-0}"
PASCALVOC_TASK_INDEX="${PASCALVOC_TASK_INDEX:-6}"
PASCALVOC_BATCH_SIZE="${PASCALVOC_BATCH_SIZE:-1}"
PASCALVOC_CATEGORY_CHUNK_SIZE="${PASCALVOC_CATEGORY_CHUNK_SIZE:-5}"
PASCALVOC_RESOLUTIONS="${PASCALVOC_RESOLUTIONS:-1008}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GATHER_PRED_VIA_FILESYS="${GATHER_PRED_VIA_FILESYS:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

normalize_mode() {
  local mode_key="${1,,}"
  mode_key="${mode_key// /_}"
  mode_key="${mode_key//+/_}"
  mode_key="${mode_key//-/_}"
  echo "$mode_key"
}

mode_to_config() {
  local mode_key
  mode_key="$(normalize_mode "$1")"
  case "$mode_key" in
    text|text_only)
      echo "configs/odinw13/odinw_text_only_positive.yaml|table3_odinw_text_only"
      ;;
    visual|visual_only|image|image_only)
      echo "configs/odinw13/odinw_visual_only.yaml|table3_odinw_visual_only"
      ;;
    text_visual|text_and_visual|textvisual)
      echo "configs/odinw13/odinw_text_and_visual.yaml|table3_odinw_text_and_visual"
      ;;
    *)
      return 1
      ;;
  esac
}

print_header() {
  echo "Repo root : $REPO_ROOT"
  echo "Mode      : $MODE"
  echo "ODINW_ROOT: $ODINW_ROOT"
  echo "OUT_ROOT  : $OUT_ROOT"
  echo "BPE_PATH  : $BPE_PATH"
  echo "CKPT_PATH : $CKPT_PATH"
  echo "Task index: $PASCALVOC_TASK_INDEX"
  echo "Batch size: $PASCALVOC_BATCH_SIZE"
  echo "Chunk size: $PASCALVOC_CATEGORY_CHUNK_SIZE"
  echo "Resolutions: $PASCALVOC_RESOLUTIONS"
  echo "Workers   : $NUM_WORKERS"
  echo "Gather fs : $GATHER_PRED_VIA_FILESYS"
  echo
}

run_one_mode() {
  local requested_mode="$1"
  local config_pair
  config_pair="$(mode_to_config "$requested_mode")" || {
    echo "Usage: $0 [text|visual|text+visual|all]" >&2
    return 2
  }

  local config_arg="${config_pair%%|*}"
  local run_name="${config_pair##*|}"
  local out_dir="${OUT_DIR:-$OUT_ROOT/$run_name}"
  local config_file="$REPO_ROOT/sam3/train/$config_arg"
  local log_dir="$out_dir/logs/PascalVOC"

  if [[ ! -f "$config_file" ]]; then
    echo "Config not found: $config_file" >&2
    return 2
  fi

  mkdir -p "$log_dir"

  local resolution
  local status=1

  for resolution in $PASCALVOC_RESOLUTIONS; do
    local attempt_log="$log_dir/pascalvoc_${requested_mode}_resolution_${resolution}.log"
    echo "[RUN ] mode=$requested_mode | task_index=$PASCALVOC_TASK_INDEX | resolution=$resolution | batch_size=$PASCALVOC_BATCH_SIZE | chunk_size=$PASCALVOC_CATEGORY_CHUNK_SIZE"

    if env PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
      "$PYTHON_BIN" sam3/train/train.py \
        -c "$config_arg" \
        --use-cluster "$USE_CLUSTER" \
        --num-gpus "$NUM_GPUS" \
        "paths.odinw_data_root=$ODINW_ROOT" \
        "paths.experiment_log_dir=$out_dir" \
        "paths.bpe_path=$BPE_PATH" \
        "submitit.job_array.task_index=$PASCALVOC_TASK_INDEX" \
        "scratch.val_batch_size=$PASCALVOC_BATCH_SIZE" \
        "scratch.num_val_workers=$NUM_WORKERS" \
        "scratch.gather_pred_via_filesys=$GATHER_PRED_VIA_FILESYS" \
        "scratch.resolution=$resolution" \
        "trainer.data.val.dataset.coco_json_loader.category_chunk_size=$PASCALVOC_CATEGORY_CHUNK_SIZE" \
        "+trainer.model.checkpoint_path=$CKPT_PATH" \
        "+trainer.model.load_from_HF=false" 2>&1 | tee "$attempt_log"; then
      echo "[ OK ] mode=$requested_mode finished"
      echo "[LOG ] $attempt_log"
      status=0
      break
    else
      status=$?
      echo "[FAIL] mode=$requested_mode failed with exit code $status at resolution=$resolution"
      echo "[LOG ] $attempt_log"
    fi
  done

  return "$status"
}

main() {
  print_header

  if [[ ! -d "$ODINW_ROOT" ]]; then
    echo "ODINW_ROOT not found: $ODINW_ROOT" >&2
    exit 2
  fi

  if [[ ! -f "$BPE_PATH" ]]; then
    echo "BPE_PATH not found: $BPE_PATH" >&2
    exit 2
  fi

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "CKPT_PATH not found: $CKPT_PATH" >&2
    exit 2
  fi

  local mode_key
  mode_key="$(normalize_mode "$MODE")"

  case "$mode_key" in
    all)
      local overall_status=0
      run_one_mode text || overall_status=$?
      run_one_mode visual || overall_status=$?
      run_one_mode text+visual || overall_status=$?
      exit "$overall_status"
      ;;
    *)
      run_one_mode "$MODE"
      ;;
  esac
}

main "$@"
