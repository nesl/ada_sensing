#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
TOPK="${TOPK:-5}"

SCRIPT_PATH="${ROOT_DIR}/policy_network/static_pred/debug/downstream_eval/analyze_topk_downstream_candidates.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
LENS_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

# A
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/policy_net_ori.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/topk_downstream_debug_A.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"

# B
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/policy_net_fixed13_head_hard.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/topk_downstream_debug_B.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"

# C
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/policy_net_fixed13_head_hard.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/topk_downstream_debug_C.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"

# D
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/topk_downstream_debug_D.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"

# E
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/policy_net_fixed13_part_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/topk_downstream_debug_E.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"

# F
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/policy_net_fixed13_full_hard.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/topk_downstream_debug_F.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"

# G
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/policy_net_fixed13_full_soft.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/topk_downstream_debug_G.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"
