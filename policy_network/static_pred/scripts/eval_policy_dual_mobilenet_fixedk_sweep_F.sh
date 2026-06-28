#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/downstream_eval/analyze_topk_downstream_candidates.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/policy_network/results_dual_mobilenet_v3_small_fixedk_sweep}"

BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOPK="${TOPK:-5}"

for ENV_OPTION_ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26; do
  RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${RESULTS_DIR}" "${ENV_OPTION_ID}")"

  "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --manifest "${MANIFEST_JSON}" \
    --checkpoint "${RUN_DIR}/best_checkpoint.pth" \
    --data_json "${ORACLE_TEST_JSON}" \
    --output_json "${RUN_DIR}/downstream_test_best.json" \
    --image_size 224 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --topk "${TOPK}"

  "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --manifest "${MANIFEST_JSON}" \
    --checkpoint "${RUN_DIR}/last_checkpoint.pth" \
    --data_json "${ORACLE_TEST_JSON}" \
    --output_json "${RUN_DIR}/downstream_test_last.json" \
    --image_size 224 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --topk "${TOPK}"
done
