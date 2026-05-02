#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/policy_network/results_hist3_multiview}"
ANALYZE_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_topk_downstream_candidates.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/H1_hist3_only/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/H1_hist3_only/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/H2_ae_hist3/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/H2_ae_hist3/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/R1_random3_seed0/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/R1_random3_seed0/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/R1_random3_seed1/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/R1_random3_seed1/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${RESULTS_DIR}/R1_random3_seed2/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${RESULTS_DIR}/R1_random3_seed2/downstream_test_best.json" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk 5
