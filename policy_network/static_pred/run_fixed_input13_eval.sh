#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

ANALYZE_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_best_index_predictions.py"
EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/eval_policy_selected_visit_acc.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
LENS_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

# A
"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/policy_net_ori.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/test_analysis_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/policy_net_ori.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/test_policy_selected_visit_acc_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# B
"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/policy_net_fixed13_head_hard.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/test_analysis_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/policy_net_fixed13_head_hard.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/test_policy_selected_visit_acc_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# C
"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/policy_net_fixed13_head_hard.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/test_analysis_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/policy_net_fixed13_head_hard.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/test_oracle_compare_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# D
"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/test_analysis_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/test_oracle_soft_compare_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# E
"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/policy_net_fixed13_part_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/test_analysis_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/policy_net_fixed13_part_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/test_oracle_part_soft_compare_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# F
"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/policy_net_fixed13_full_hard.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/test_analysis_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/policy_net_fixed13_full_hard.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/test_oracle_full_hard_compare_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# G
"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/policy_net_fixed13_full_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/test_analysis_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/policy_net_fixed13_full_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/test_oracle_full_soft_compare_fixed13.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"
