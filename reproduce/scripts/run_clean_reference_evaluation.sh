#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-${WORKSPACE_DIR}/replicate_result}"
DEVICE="${DEVICE:-cpu}"
MODELS="${MODELS:-resnet50,resnet50_deepaugment_augmix,resnet152,efficientnet_b0,efficientnet_b3,swin_v2_t,swin_v2_s,swin_v2_b,openclip_b,openclip_h,dinov2_b,dinov2_g}"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"${PYTHON_BIN}" -m reproduce_in_ae.evaluate_clean_reference \
  --result-root "${RESULT_ROOT}/comparison/clean_reference" \
  --models "${MODELS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m reproduce_in_ae.compare_replication \
  --workspace-root "${WORKSPACE_DIR}" \
  --result-root "${RESULT_ROOT}" \
  --models "${MODELS}"

echo "Five-image clean-reference evaluation complete."
