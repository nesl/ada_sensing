#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_topk_downstream_candidates.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
LABEL_DIR="${LABEL_DIR:-${EXPERIMENT_DIR}/outputs/labels}"
RESULTS_DIR="${RESULTS_DIR:-${EXPERIMENT_DIR}/outputs/results_single_fixedk}"

ORACLE_TEST_JSON="${LABEL_DIR}/oracle_policy_test_labels.json"

K="${K:-24}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
TOPK="${TOPK:-1}"

RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${RESULTS_DIR}" "${K}")"
CHECKPOINT="${RUN_DIR}/${CHECKPOINT_KIND}_checkpoint.pth"
OUTPUT_JSON="${RUN_DIR}/downstream_top1_test_${CHECKPOINT_KIND}.json"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --manifest "${MANIFEST_JSON}" \
  --checkpoint "${CHECKPOINT}" \
  --data_json "${ORACLE_TEST_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --topk "${TOPK}"
