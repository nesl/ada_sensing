#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
TRAIN_SCRIPT="${ROOT_DIR}/policy_network/static_pred/train_policy.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
LABEL_DIR="${LABEL_DIR:-${EXPERIMENT_DIR}/outputs/labels}"
RESULTS_DIR="${RESULTS_DIR:-${EXPERIMENT_DIR}/outputs/results_single_fixedk}"

ORACLE_TRAIN_JSON="${LABEL_DIR}/oracle_policy_train_labels.json"
ORACLE_VAL_JSON="${LABEL_DIR}/oracle_policy_val_labels.json"
ORACLE_TEST_JSON="${LABEL_DIR}/oracle_policy_test_labels.json"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-2e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_K="${START_K:-24}"
END_K="${END_K:-24}"

for ENV_OPTION_ID in $(seq "${START_K}" "${END_K}"); do
  RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${RESULTS_DIR}" "${ENV_OPTION_ID}")"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --train_json "${ORACLE_TRAIN_JSON}" \
    --val_json "${ORACLE_VAL_JSON}" \
    --test_json "${ORACLE_TEST_JSON}" \
    --save_dir "${RUN_DIR}" \
    --manifest_json "${MANIFEST_JSON}" \
    --image_size 224 \
    --backbone mobilenet_v3_small \
    --input_mode single \
    --single_input_source env \
    --input_variant real \
    --env_input_variant real \
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
