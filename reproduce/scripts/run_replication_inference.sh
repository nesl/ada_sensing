#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

DATASET="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

GPU0_MODELS="resnet50,resnet152,efficientnet_b0,swin_v2_t,swin_v2_b,openclip_h"
GPU1_MODELS="resnet50_deepaugment_augmix,efficientnet_b3,swin_v2_s,openclip_b,dinov2_b,dinov2_g"

if [[ "${DATASET}" != "replicated_capture" && "${DATASET}" != "dpi600" ]]; then
  echo "Usage: $0 {replicated_capture|dpi600}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_ROOT="${WORKSPACE_DIR}/data/replication/${DATASET}_roi"
RESULT_ROOT="${WORKSPACE_DIR}/replicate_result/${DATASET}"
ROI_CONFIG="${PROJECT_DIR}/configs/replication_roi_${DATASET}.json"

mkdir -p "${RESULT_ROOT}/logs"

"${PYTHON_BIN}" -m reproduce_in_ae.review_replication_roi \
  --dataset "${DATASET}" \
  --config "${ROI_CONFIG}" \
  --check-approval

"${PYTHON_BIN}" -m reproduce_in_ae.audit_replication \
  --dataset-root "${DATASET_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --preflight-only \
  --required-gpus 2

"${PYTHON_BIN}" -m reproduce_in_ae.evaluate_replication \
  --models "${GPU0_MODELS}" \
  --dataset-root "${DATASET_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --device "cuda:${GPU0}" \
  --workers "${NUM_WORKERS}" \
  --resume \
  >"${RESULT_ROOT}/logs/gpu${GPU0}.log" 2>&1 &
pid0=$!

"${PYTHON_BIN}" -m reproduce_in_ae.evaluate_replication \
  --models "${GPU1_MODELS}" \
  --dataset-root "${DATASET_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --device "cuda:${GPU1}" \
  --workers "${NUM_WORKERS}" \
  --resume \
  >"${RESULT_ROOT}/logs/gpu${GPU1}.log" 2>&1 &
pid1=$!

echo "${DATASET} GPU ${GPU0} log: ${RESULT_ROOT}/logs/gpu${GPU0}.log"
echo "${DATASET} GPU ${GPU1} log: ${RESULT_ROOT}/logs/gpu${GPU1}.log"
echo "${DATASET} worker PIDs: ${pid0}, ${pid1}"

status0=0
status1=0
wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "${DATASET} inference failed: gpu${GPU0}=${status0}, gpu${GPU1}=${status1}" >&2
  echo "Completed per-model outputs are retained; run this same command again to resume." >&2
  exit 1
fi

"${PYTHON_BIN}" -m reproduce_in_ae.audit_replication \
  --dataset-root "${DATASET_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --required-gpus 2 \
  --require-complete

echo "${DATASET} inference complete: 12 models x $(wc -l < "${DATASET_ROOT}/crop_manifest.jsonl") records."
