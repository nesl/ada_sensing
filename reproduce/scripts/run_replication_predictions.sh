#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_DIR}/data/replication/replicated_capture_roi}"
RESULT_ROOT="${RESULT_ROOT:-${WORKSPACE_DIR}/replicate_result}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

GPU0_MODELS="resnet50,resnet152,efficientnet_b0,swin_v2_t,swin_v2_b,openclip_h"
GPU1_MODELS="resnet50_deepaugment_augmix,efficientnet_b3,swin_v2_s,openclip_b,dinov2_b,dinov2_g"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "${RESULT_ROOT}/logs"

"${PYTHON_BIN}" -m reproduce_in_ae.prepare_replication \
  --output-root "${DATASET_ROOT}"

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
  >"${RESULT_ROOT}/logs/gpu${GPU0}.log" 2>&1 &
PID0=$!

"${PYTHON_BIN}" -m reproduce_in_ae.evaluate_replication \
  --models "${GPU1_MODELS}" \
  --dataset-root "${DATASET_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --device "cuda:${GPU1}" \
  --workers "${NUM_WORKERS}" \
  >"${RESULT_ROOT}/logs/gpu${GPU1}.log" 2>&1 &
PID1=$!

echo "GPU ${GPU0} log: ${RESULT_ROOT}/logs/gpu${GPU0}.log"
echo "GPU ${GPU1} log: ${RESULT_ROOT}/logs/gpu${GPU1}.log"
echo "Worker PIDs: ${PID0}, ${PID1}"

set +e
wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?
set -e

if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  echo "Replication inference failed: gpu${GPU0}=${STATUS0}, gpu${GPU1}=${STATUS1}" >&2
  echo "Completed per-model outputs are retained; rerun this command after inspecting logs." >&2
  exit 1
fi

"${PYTHON_BIN}" -m reproduce_in_ae.audit_replication \
  --dataset-root "${DATASET_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --required-gpus 2 \
  --require-complete

echo "Replication predictions complete: 12 models x 600 records."
