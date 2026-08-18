#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-${WORKSPACE_DIR}/replicate_result}"
MODELS="${MODELS:-resnet50,resnet50_deepaugment_augmix,resnet152,efficientnet_b0,efficientnet_b3,swin_v2_t,swin_v2_s,swin_v2_b,openclip_b,openclip_h,dinov2_b,dinov2_g}"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_individual_analysis() {
  local dataset_name="$1"
  "${PYTHON_BIN}" -m reproduce_in_ae.analyze_replication \
    --source-root "${WORKSPACE_DIR}/data/replication/${dataset_name}" \
    --dataset-root "${WORKSPACE_DIR}/data/replication/${dataset_name}_roi" \
    --result-root "${RESULT_ROOT}/${dataset_name}" \
    --models "${MODELS}"
}

run_individual_analysis "replicated_capture"
run_individual_analysis "dpi600"

"${PYTHON_BIN}" -m reproduce_in_ae.compare_replication \
  --workspace-root "${WORKSPACE_DIR}" \
  --result-root "${RESULT_ROOT}" \
  --models "${MODELS}"

echo "Four-group analysis complete: ${RESULT_ROOT}/comparison/report.md"
