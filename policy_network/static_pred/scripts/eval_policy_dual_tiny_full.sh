#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/policy_network/results_dual_tiny_conv_scratch}"
ANALYZE_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/downstream_eval/analyze_topk_downstream_candidates.py"
HISTORY_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/policy_visualization/visualize_train_history_distribution.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
LENS_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/lens_full_hard/best_checkpoint.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/lens_full_hard/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/lens_full_hard/last_checkpoint.pth" \
  --data_json "${LENS_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/lens_full_hard/downstream_test_last.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/oracle_full_hard/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/oracle_full_hard/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/oracle_full_hard/last_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/oracle_full_hard/downstream_test_last.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/oracle_full_soft/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/oracle_full_soft/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/oracle_full_soft/last_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/oracle_full_soft/downstream_test_last.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${HISTORY_SCRIPT}" \
  --history_json "${RESULTS_DIR}/lens_full_hard/train_history.json" \
  --output_png "${RESULTS_DIR}/lens_full_hard/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${HISTORY_SCRIPT}" \
  --history_json "${RESULTS_DIR}/oracle_full_hard/train_history.json" \
  --output_png "${RESULTS_DIR}/oracle_full_hard/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${HISTORY_SCRIPT}" \
  --history_json "${RESULTS_DIR}/oracle_full_soft/train_history.json" \
  --output_png "${RESULTS_DIR}/oracle_full_soft/train_history_distribution_summary.png"
