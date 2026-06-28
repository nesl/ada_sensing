#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
TRAIN_SCRIPT="${ROOT_DIR}/policy_network/static_pred/train_policy.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
ENV_OPTION_ID="${ENV_OPTION_ID:-13}"

LENS_TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_train_labels.json"
LENS_VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_val_labels.json"
LENS_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"

ORACLE_TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json"
ORACLE_VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

DINO_RESULTS_DIR="${ROOT_DIR}/policy_network/results_dual_dinov2_vits14"
TINY_RESULTS_DIR="${ROOT_DIR}/policy_network/results_dual_tiny_conv_scratch"

DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-8}"
TINY_BATCH_SIZE="${TINY_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-50}"
TINY_EPOCHS="${TINY_EPOCHS:-80}"
TINY_LR="${TINY_LR:-3e-4}"
TINY_WEIGHT_DECAY="${TINY_WEIGHT_DECAY:-1e-4}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# DINOv2-S/14 A. Lens label + full finetune + hard label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${LENS_TRAIN_JSON}" \
  --val_json "${LENS_VAL_JSON}" \
  --test_json "${LENS_TEST_JSON}" \
  --save_dir "${DINO_RESULTS_DIR}/A_lens_ft_all_hard" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone dinov2_vits14 \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${DINO_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr 1e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type hard_ce

# DINOv2-S/14 B. Lens label + head only + hard label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${LENS_TRAIN_JSON}" \
  --val_json "${LENS_VAL_JSON}" \
  --test_json "${LENS_TEST_JSON}" \
  --save_dir "${DINO_RESULTS_DIR}/B_lens_head_hard" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone dinov2_vits14 \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${DINO_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr 1e-4 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained \
  --trainable_scope head_only \
  --loss_type hard_ce

# DINOv2-S/14 C. Oracle label + head only + hard label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${DINO_RESULTS_DIR}/C_oracle_head_hard" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone dinov2_vits14 \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${DINO_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr 1e-4 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained \
  --trainable_scope head_only \
  --loss_type hard_ce

# DINOv2-S/14 D. Oracle label + head only + soft label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${DINO_RESULTS_DIR}/D_oracle_head_soft" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone dinov2_vits14 \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${DINO_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr 1e-4 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained \
  --trainable_scope head_only \
  --loss_type soft_kl

# DINOv2-S/14 E. Oracle label + partial unfreeze + soft label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${DINO_RESULTS_DIR}/E_oracle_partial_soft" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone dinov2_vits14 \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${DINO_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr 5e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained \
  --resume_checkpoint "${DINO_RESULTS_DIR}/D_oracle_head_soft/best_checkpoint.pth" \
  --trainable_scope partial_unfreeze \
  --loss_type soft_kl

# DINOv2-S/14 F. Oracle label + full finetune + hard label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${DINO_RESULTS_DIR}/F_oracle_full_hard" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone dinov2_vits14 \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${DINO_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr 1e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type hard_ce

# DINOv2-S/14 G. Oracle label + full finetune + soft label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${DINO_RESULTS_DIR}/G_oracle_full_soft" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone dinov2_vits14 \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${DINO_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr 1e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type soft_kl

# Tiny conv scratch. Lens dataset + full training + hard label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${LENS_TRAIN_JSON}" \
  --val_json "${LENS_VAL_JSON}" \
  --test_json "${LENS_TEST_JSON}" \
  --save_dir "${TINY_RESULTS_DIR}/lens_full_hard" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone tiny_conv_scratch \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${TINY_BATCH_SIZE}" \
  --epochs "${TINY_EPOCHS}" \
  --lr "${TINY_LR}" \
  --backbone_lr "${TINY_LR}" \
  --weight_decay "${TINY_WEIGHT_DECAY}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --trainable_scope full_finetune \
  --loss_type hard_ce

# Tiny conv scratch. Oracle dataset + full training + hard label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${TINY_RESULTS_DIR}/oracle_full_hard" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone tiny_conv_scratch \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${TINY_BATCH_SIZE}" \
  --epochs "${TINY_EPOCHS}" \
  --lr "${TINY_LR}" \
  --backbone_lr "${TINY_LR}" \
  --weight_decay "${TINY_WEIGHT_DECAY}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --trainable_scope full_finetune \
  --loss_type hard_ce

# Tiny conv scratch. Oracle dataset + full training + soft label + dual auto/fixed input
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${TINY_RESULTS_DIR}/oracle_full_soft" \
  --manifest_json "${MANIFEST_JSON}" \
  --image_size 224 \
  --backbone tiny_conv_scratch \
  --input_mode dual \
  --input_variant real \
  --env_option_id "${ENV_OPTION_ID}" \
  --batch_size "${TINY_BATCH_SIZE}" \
  --epochs "${TINY_EPOCHS}" \
  --lr "${TINY_LR}" \
  --backbone_lr "${TINY_LR}" \
  --weight_decay "${TINY_WEIGHT_DECAY}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --trainable_scope full_finetune \
  --loss_type soft_kl
