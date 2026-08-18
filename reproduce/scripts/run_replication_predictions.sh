#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
RESULT_ROOT="${WORKSPACE_DIR}/replicate_result"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RESET_RESULTS="${RESET_RESULTS:-0}"

GPU0_MODELS="resnet50,resnet152,efficientnet_b0,swin_v2_t,swin_v2_b,openclip_h"
GPU1_MODELS="resnet50_deepaugment_augmix,efficientnet_b3,swin_v2_s,openclip_b,dinov2_b,dinov2_g"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "${RESET_RESULTS}" == "1" ]]; then
  if [[ "${RESULT_ROOT}" != "${WORKSPACE_DIR}/replicate_result" ]]; then
    echo "Refusing to reset unexpected result root: ${RESULT_ROOT}" >&2
    exit 2
  fi
  rm -rf -- "${RESULT_ROOT}"
elif [[ "${RESET_RESULTS}" != "0" ]]; then
  echo "RESET_RESULTS must be 0 or 1" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"

run_dataset() {
  local dataset_name="$1"
  local roi_config="$2"
  local source_root="${WORKSPACE_DIR}/data/replication/${dataset_name}"
  local dataset_result_root="${RESULT_ROOT}/${dataset_name}"
  local cropped_root="${WORKSPACE_DIR}/data/replication/${dataset_name}_roi"
  local roi_review_root="${dataset_result_root}/roi_review"
  local crop_overwrite=()

  if [[ "${RESET_RESULTS}" == "1" ]]; then
    crop_overwrite=(--overwrite)
  fi

  mkdir -p "${dataset_result_root}/logs"

  "${PYTHON_BIN}" -m reproduce_in_ae.preview_replication_roi \
    --source-root "${source_root}" \
    --config "${roi_config}" \
    --output-dir "${roi_review_root}" \
    --overwrite

  "${PYTHON_BIN}" -m reproduce_in_ae.prepare_replication \
    --source-root "${source_root}" \
    --output-root "${cropped_root}" \
    --roi-config "${roi_config}" \
    "${crop_overwrite[@]}"

  "${PYTHON_BIN}" -m reproduce_in_ae.audit_replication \
    --dataset-root "${cropped_root}" \
    --result-root "${dataset_result_root}" \
    --preflight-only \
    --required-gpus 2

  "${PYTHON_BIN}" -m reproduce_in_ae.evaluate_replication \
    --models "${GPU0_MODELS}" \
    --dataset-root "${cropped_root}" \
    --result-root "${dataset_result_root}" \
    --device "cuda:${GPU0}" \
    --workers "${NUM_WORKERS}" \
    >"${dataset_result_root}/logs/gpu${GPU0}.log" 2>&1 &
  local pid0=$!

  "${PYTHON_BIN}" -m reproduce_in_ae.evaluate_replication \
    --models "${GPU1_MODELS}" \
    --dataset-root "${cropped_root}" \
    --result-root "${dataset_result_root}" \
    --device "cuda:${GPU1}" \
    --workers "${NUM_WORKERS}" \
    >"${dataset_result_root}/logs/gpu${GPU1}.log" 2>&1 &
  local pid1=$!

  echo "${dataset_name} GPU ${GPU0} log: ${dataset_result_root}/logs/gpu${GPU0}.log"
  echo "${dataset_name} GPU ${GPU1} log: ${dataset_result_root}/logs/gpu${GPU1}.log"
  echo "${dataset_name} worker PIDs: ${pid0}, ${pid1}"

  local status0=0
  local status1=0
  wait "${pid0}" || status0=$?
  wait "${pid1}" || status1=$?
  if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
    echo "${dataset_name} inference failed: gpu${GPU0}=${status0}, gpu${GPU1}=${status1}" >&2
    echo "Completed per-model outputs are retained; rerun without RESET_RESULTS=1." >&2
    exit 1
  fi

  "${PYTHON_BIN}" -m reproduce_in_ae.audit_replication \
    --dataset-root "${cropped_root}" \
    --result-root "${dataset_result_root}" \
    --required-gpus 2 \
    --require-complete

  echo "${dataset_name} predictions complete: 12 models x 1500 records."
}

run_dataset \
  "replicated_capture" \
  "${PROJECT_DIR}/configs/replication_roi_replicated_capture.json"

run_dataset \
  "dpi600" \
  "${PROJECT_DIR}/configs/replication_roi_dpi600.json"

echo "All predictions complete under ${RESULT_ROOT}."
