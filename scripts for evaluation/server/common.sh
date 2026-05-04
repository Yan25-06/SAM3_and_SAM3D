#!/usr/bin/env bash
set -euo pipefail

SERVER_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SERVER_SCRIPTS_DIR}/../.." && pwd)"

SAM3_ENV_FILE="${SAM3_ENV_FILE:-${SERVER_SCRIPTS_DIR}/sam3_server.env}"
if [[ -f "${SAM3_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SAM3_ENV_FILE}"
  set +a
fi

PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-sam3}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
HF_MODEL_REPO="${HF_MODEL_REPO:-facebook/sam3}"
ODINW_ROOT="${ODINW_ROOT:-${DATA_ROOT}/odinw13}"
RF100_ROOT="${RF100_ROOT:-${DATA_ROOT}/rf100-vl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/sam3.pt}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
SAM3_RESOLUTION="${SAM3_RESOLUTION:-1008}"
GATHER_PRED_VIA_FILESYS="${GATHER_PRED_VIA_FILESYS:-false}"
AUTO_DOWNLOAD_RF100="${AUTO_DOWNLOAD_RF100:-false}"

BPE_PATH="${BPE_PATH:-${PROJECT_ROOT}/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"

ensure_dir() {
  mkdir -p "$1"
}

ensure_var() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required variable: ${var_name}" >&2
    exit 1
  fi
}

ensure_file_exists() {
  local file_path="$1"
  if [[ ! -f "${file_path}" ]]; then
    echo "Required file not found: ${file_path}" >&2
    exit 1
  fi
}

validate_roboflow_api_key() {
  ensure_var ROBOFLOW_API_KEY

  if [[ "${ROBOFLOW_API_KEY}" == "..." || "${ROBOFLOW_API_KEY}" == *"YOUR_"* || "${ROBOFLOW_API_KEY}" == *"your_"* ]]; then
    echo "ROBOFLOW_API_KEY looks like a placeholder, not a real private API key." >&2
    echo "Get a real key from https://app.roboflow.com/settings/api" >&2
    exit 1
  fi

  local http_code
  http_code="$(
    curl -sS -o /dev/null -w "%{http_code}" \
      "https://api.roboflow.com/?api_key=${ROBOFLOW_API_KEY}"
  )"

  if [[ "${http_code}" != "200" ]]; then
    echo "Roboflow API key validation failed with HTTP ${http_code}." >&2
    echo "Use a valid private API key from https://app.roboflow.com/settings/api" >&2
    exit 1
  fi
}

rf100_dataset_ready() {
  if [[ ! -d "${RF100_ROOT}" ]]; then
    return 1
  fi

  find "${RF100_ROOT}" -mindepth 3 -maxdepth 3 -type f -name "_annotations.coco.json" -print -quit | grep -q .
}

download_rf100_variant() {
  local rf100_variant="${1:-full}"

  activate_venv
  ensure_dir "${RF100_ROOT}"
  validate_roboflow_api_key
  export RF100_ROOT
  export ROBOFLOW_API_KEY

  case "${rf100_variant}" in
    full)
      run_from_project_root
      python scripts/server/download_rf100vl_direct.py \
        --root "${RF100_ROOT}" \
        --variant full
      ;;
    fsod)
      run_from_project_root
      python scripts/server/download_rf100vl_direct.py \
        --root "${RF100_ROOT}" \
        --variant fsod
      ;;
    *)
      echo "Usage: $0 [full|fsod]" >&2
      exit 1
      ;;
  esac
}

ensure_rf100_ready() {
  local rf100_variant="${1:-full}"

  if rf100_dataset_ready; then
    return 0
  fi

  if [[ "${AUTO_DOWNLOAD_RF100}" == "true" ]]; then
    echo "RF100-VL dataset not found under ${RF100_ROOT}. Downloading variant '${rf100_variant}'..."
    download_rf100_variant "${rf100_variant}"

    if rf100_dataset_ready; then
      return 0
    fi

    echo "RF100-VL download finished but dataset layout was not detected under ${RF100_ROOT}" >&2
    exit 1
  fi

  cat >&2 <<EOF
RF100-VL dataset not found under ${RF100_ROOT}.

Download it first with:
  bash scripts/server/download_rf100vl.sh ${rf100_variant}

Or rerun the Table 2 script with automatic download enabled:
  AUTO_DOWNLOAD_RF100=true ROBOFLOW_API_KEY=... bash $0
EOF
  exit 1
}

activate_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtualenv not found at ${VENV_DIR}. Run setup_sam3_env.sh first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

run_from_project_root() {
  cd "${PROJECT_ROOT}"
}

ensure_bundled_config_path() {
  local config_path="$1"
  local normalized="${config_path#./}"
  local train_rel=""

  if [[ -f "${PROJECT_ROOT}/${normalized}" ]]; then
    return
  fi

  if [[ "${normalized}" == sam3/train/* ]]; then
    train_rel="${normalized#sam3/train/}"
  elif [[ "${normalized}" == configs/* ]]; then
    train_rel="${normalized}"
  else
    return
  fi

  local repo_target="${PROJECT_ROOT}/sam3/train/${train_rel}"
  if [[ -f "${repo_target}" ]]; then
    return
  fi

  local config_rel="${train_rel#configs/}"
  local archive_group="${config_rel%%/*}"
  local archive_path="${PROJECT_ROOT}/sam3/train/configs/${archive_group}.zip"

  if [[ -f "${archive_path}" ]]; then
    echo "Extracting bundled config archive ${archive_path}"
    "${PYTHON_BIN}" - "${archive_path}" "${PROJECT_ROOT}/sam3/train/configs" <<'PY'
import sys
import zipfile
from pathlib import Path

archive_path = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(target_dir)
PY
  elif [[ "${archive_group}" == "external_image_evals" ]]; then
    echo "Bundled archive missing; recreating ${archive_group} configs from script fallback"
    "${PYTHON_BIN}" - "${PROJECT_ROOT}/sam3/train/configs" <<'PY'
from pathlib import Path
import sys

config_root = Path(sys.argv[1])
target_dir = config_root / "external_image_evals"
target_dir.mkdir(parents=True, exist_ok=True)

files = {
    "sam3_table1_coco.yaml": """# @package _global_
defaults:
  - /configs/eval_base.yaml
  - _self_

paths:
  experiment_log_dir: ${paths.base_experiment_log_dir}/table1_coco/
  img_path: <YOUR_COCO_IMG_DIR>
  coco_gt: <YOUR_COCO_GT_FILE>

scratch:
  category_chunk_size: 20

trainer:
  data:
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: ${scratch.category_chunk_size}
          _partial_: true
        img_folder: ${paths.img_path}
        ann_file:
          _target_: sam3.eval.coco_reindex.reindex_coco_to_temp
          input_json_path: ${paths.coco_gt}
        transforms: ${scratch.base_val_transform}
        max_ann_per_img: 100000
        multiplier: 1
        training: false

      shuffle: False
      batch_size: ${scratch.val_batch_size}
      num_workers: ${scratch.num_val_workers}
      pin_memory: False
      drop_last: False
      collate_fn:
        _target_: sam3.train.data.collator.collate_fn_api
        _partial_: true
        repeats: ${scratch.hybrid_repeats}
        dict_key: table1_coco

  meters:
    val:
      table1_coco:
        detection:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "bbox"
          dump_dir: ${launcher.experiment_log_dir}/dumps/table1_coco
          merge_predictions: True
          postprocessor: ${scratch.original_box_postprocessor}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
          maxdets: 100
          pred_file_evaluators:
            - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
              gt_path:
                _target_: sam3.eval.coco_reindex.reindex_coco_to_temp
                input_json_path: ${paths.coco_gt}
              tide: False
              iou_type: "bbox"
              positive_split: False
""",
    "sam3_table1_coco_o.yaml": """# @package _global_
defaults:
  - /configs/eval_base.yaml
  - _self_

paths:
  experiment_log_dir: ${paths.base_experiment_log_dir}/table1_coco_o/
  img_path: <YOUR_COCO_O_IMG_DIR>
  coco_gt: <YOUR_COCO_O_GT_FILE>

scratch:
  category_chunk_size: 20

trainer:
  data:
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: ${scratch.category_chunk_size}
          _partial_: true
        img_folder: ${paths.img_path}
        ann_file:
          _target_: sam3.eval.coco_reindex.reindex_coco_to_temp
          input_json_path: ${paths.coco_gt}
        transforms: ${scratch.base_val_transform}
        max_ann_per_img: 100000
        multiplier: 1
        training: false

      shuffle: False
      batch_size: ${scratch.val_batch_size}
      num_workers: ${scratch.num_val_workers}
      pin_memory: False
      drop_last: False
      collate_fn:
        _target_: sam3.train.data.collator.collate_fn_api
        _partial_: true
        repeats: ${scratch.hybrid_repeats}
        dict_key: table1_coco_o

  meters:
    val:
      table1_coco_o:
        detection:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "bbox"
          dump_dir: ${launcher.experiment_log_dir}/dumps/table1_coco_o
          merge_predictions: True
          postprocessor: ${scratch.original_box_postprocessor}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
          maxdets: 100
          pred_file_evaluators:
            - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
              gt_path:
                _target_: sam3.eval.coco_reindex.reindex_coco_to_temp
                input_json_path: ${paths.coco_gt}
              tide: False
              iou_type: "bbox"
              positive_split: False
""",
    "sam3_table1_lvis.yaml": """# @package _global_
defaults:
  - /configs/eval_base.yaml
  - _self_

paths:
  experiment_log_dir: ${paths.base_experiment_log_dir}/table1_lvis/
  img_path: <YOUR_LVIS_IMG_DIR>
  coco_gt: <YOUR_LVIS_GT_FILE>

scratch:
  category_chunk_size: 20

trainer:
  data:
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: ${scratch.category_chunk_size}
          _partial_: true
        img_folder: ${paths.img_path}
        ann_file:
          _target_: sam3.eval.coco_reindex.reindex_coco_to_temp
          input_json_path: ${paths.coco_gt}
        transforms: ${scratch.base_val_transform}
        max_ann_per_img: 100000
        multiplier: 1
        training: false
        load_segmentation: true

      shuffle: False
      batch_size: ${scratch.val_batch_size}
      num_workers: ${scratch.num_val_workers}
      pin_memory: False
      drop_last: False
      collate_fn:
        _target_: sam3.train.data.collator.collate_fn_api
        _partial_: true
        repeats: ${scratch.hybrid_repeats}
        dict_key: table1_lvis

  meters:
    val:
      table1_lvis:
        detection:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "bbox"
          dump_dir: ${launcher.experiment_log_dir}/dumps/table1_lvis
          merge_predictions: True
          postprocessor: ${scratch.original_box_postprocessor}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
          maxdets: 300
          pred_file_evaluators:
            - _target_: sam3.eval.lvis_eval_offline.LvisEvaluatorOfflineWithPredFileEvaluators
              gt_path: ${paths.coco_gt}
              iou_type: "bbox"
              max_dets: 300
        segmentation:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "segm"
          dump_dir: ${launcher.experiment_log_dir}/dumps/table1_lvis
          merge_predictions: True
          postprocessor: ${scratch.mask_postprocessor_thresholded}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
          maxdets: 300
          pred_file_evaluators:
            - _target_: sam3.eval.lvis_eval_offline.LvisEvaluatorOfflineWithPredFileEvaluators
              gt_path: ${paths.coco_gt}
              iou_type: "segm"
              max_dets: 300
""",
}

for name, content in files.items():
    (target_dir / name).write_text(content, encoding="utf-8")
PY
  else
    echo "Missing config file ${config_path} and no bundled archive found at ${archive_path}" >&2
    exit 2
  fi

  if [[ ! -f "${repo_target}" ]]; then
    echo "Config file still missing after extracting archive: ${repo_target}" >&2
    exit 2
  fi
}
