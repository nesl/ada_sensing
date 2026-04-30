#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
TRAIN_SCRIPT="${ROOT_DIR}/policy_network/static_pred/train_policy.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/policy_network/results_dual_mobilenet_v3_small_fixedk_sweep}"

ORACLE_TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json"
ORACLE_VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-2e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"

for ENV_OPTION_ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26; do
  RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${RESULTS_DIR}" "${ENV_OPTION_ID}")"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --train_json "${ORACLE_TRAIN_JSON}" \
    --val_json "${ORACLE_VAL_JSON}" \
    --test_json "${ORACLE_TEST_JSON}" \
    --save_dir "${RUN_DIR}" \
    --manifest_json "${MANIFEST_JSON}" \
    --image_size 224 \
    --backbone mobilenet_v3_small \
    --input_mode dual \
    --input_variant real \
    --env_option_id "${ENV_OPTION_ID}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --backbone_lr "${BACKBONE_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --device "${DEVICE}" \
    --num_workers "${NUM_WORKERS}" \
    --pretrained \
    --trainable_scope full_finetune \
    --loss_type hard_ce
done
