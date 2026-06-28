#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

ANALYZE_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/index_prediction/analyze_best_index_predictions.py"

# "${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
#   --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/policy_net_ori.pth" \
#   --data_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json" \
#   --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/index_test_analysis.json" \
#   --image_size "${IMAGE_SIZE}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --device "${DEVICE}"

# "${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
#   --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/policy_net_fixed13_head_hard.pth" \
#   --data_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json" \
#   --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/index_test_analysis.json" \
#   --image_size "${IMAGE_SIZE}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --device "${DEVICE}"

# "${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
#   --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/policy_net_fixed13_head_hard.pth" \
#   --data_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
#   --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/index_test_analysis.json" \
#   --image_size "${IMAGE_SIZE}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --device "${DEVICE}"

# "${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
#   --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth" \
#   --data_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
#   --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/index_test_analysis.json" \
#   --image_size "${IMAGE_SIZE}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --device "${DEVICE}"

# "${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
#   --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/policy_net_fixed13_part_soft.pth" \
#   --data_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
#   --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/index_test_analysis.json" \
#   --image_size "${IMAGE_SIZE}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --device "${DEVICE}"

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/policy_net_fixed13_full_hard.pth" \
  --data_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/index_test_analysis.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/policy_net_fixed13_full_soft.pth" \
  --data_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
  --output_json "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/index_test_analysis.json" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"
