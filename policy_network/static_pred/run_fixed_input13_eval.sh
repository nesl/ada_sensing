#!/usr/bin/env bash

set -euo pipefail

EXP_ID="${1:?Usage: run_fixed_input13_eval.sh [B|C|D|E]}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

ANALYZE_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_best_index_predictions.py"
EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/eval_policy_selected_visit_acc.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"

case "${EXP_ID}" in
  B)
    CHECKPOINT="${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/policy_net_fixed13_head_hard.pth"
    DATA_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"
    ANALYSIS_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/test_analysis_fixed13.json"
    COMPARE_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/test_policy_selected_visit_acc_fixed13.json"
    ;;
  C)
    CHECKPOINT="${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/policy_net_fixed13_head_hard.pth"
    DATA_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
    ANALYSIS_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/test_analysis_fixed13.json"
    COMPARE_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/test_oracle_compare_fixed13.json"
    ;;
  D)
    CHECKPOINT="${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth"
    DATA_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
    ANALYSIS_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/test_analysis_fixed13.json"
    COMPARE_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/test_oracle_soft_compare_fixed13.json"
    ;;
  E)
    CHECKPOINT="${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/policy_net_fixed13_part_soft.pth"
    DATA_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
    ANALYSIS_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/test_analysis_fixed13.json"
    COMPARE_JSON="${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/test_oracle_part_soft_compare_fixed13.json"
    ;;
  *)
    echo "Unsupported experiment id: ${EXP_ID}"
    exit 1
    ;;
esac

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --data_json "${DATA_JSON}" \
  --output_json "${ANALYSIS_JSON}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${CHECKPOINT}" \
  --data_json "${DATA_JSON}" \
  --output_json "${COMPARE_JSON}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"
