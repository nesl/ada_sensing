#!/usr/bin/env bash

set -euo pipefail

EXP_ID="${1:?Usage: run_fixed_input13_train.sh [B|C|D|E]}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
EPOCHS="${EPOCHS:-20}"
FIXED_OPTION_ID="${FIXED_OPTION_ID:-13}"

TRAIN_SCRIPT="${ROOT_DIR}/policy_network/static_pred/train_policy.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"

COMMON_ARGS=(
  --manifest_json "${MANIFEST_JSON}"
  --image_size "${IMAGE_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr 1e-4
  --backbone_lr 5e-6
  --weight_decay 1e-4
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --pretrained
  --train_input_sampling fixed_option
  --train_input_option_id "${FIXED_OPTION_ID}"
)

case "${EXP_ID}" in
  B)
    TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_train_labels.json"
    VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_val_labels.json"
    TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"
    SAVE_DIR="${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard"
    EXTRA_ARGS=(
      --save_dir "${SAVE_DIR}"
      --checkpoint_name policy_net_fixed13_head_hard.pth
      --trainable_scope head_only
      --loss_type hard_ce
    )
    ;;
  C)
    TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json"
    VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json"
    TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
    SAVE_DIR="${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard"
    EXTRA_ARGS=(
      --save_dir "${SAVE_DIR}"
      --checkpoint_name policy_net_fixed13_head_hard.pth
      --trainable_scope head_only
      --loss_type hard_ce
    )
    ;;
  D)
    TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json"
    VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json"
    TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
    SAVE_DIR="${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft"
    EXTRA_ARGS=(
      --save_dir "${SAVE_DIR}"
      --checkpoint_name policy_net_fixed13_head_soft.pth
      --trainable_scope head_only
      --loss_type soft_kl
    )
    ;;
  E)
    TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json"
    VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json"
    TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
    SAVE_DIR="${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft"
    RESUME_CKPT="${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth"
    EXTRA_ARGS=(
      --save_dir "${SAVE_DIR}"
      --checkpoint_name policy_net_fixed13_part_soft.pth
      --resume_checkpoint "${RESUME_CKPT}"
      --trainable_scope partial_unfreeze
      --loss_type soft_kl
    )
    ;;
  *)
    echo "Unsupported experiment id: ${EXP_ID}"
    exit 1
    ;;
esac

exec "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${TRAIN_JSON}" \
  --val_json "${VAL_JSON}" \
  --test_json "${TEST_JSON}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
