#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${ROOT_DIR}/policy_network/static_pred/train_policy.py"
POLICY_BACKBONE="${POLICY_BACKBONE:-resnet18}"
RESULTS_DIR="${ROOT_DIR}/policy_network/results_${POLICY_BACKBONE}"

LENS_TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_train_labels.json"
LENS_VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_val_labels.json"
LENS_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"

ORACLE_TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json"
ORACLE_VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

# A. Lens label + full finetune + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${LENS_TRAIN_JSON}" \
  --val_json "${LENS_VAL_JSON}" \
  --test_json "${LENS_TEST_JSON}" \
  --save_dir "${RESULTS_DIR}/A_lens_ft_all_hard" \
  --image_size 224 \
  --backbone "${POLICY_BACKBONE}" \
  --batch_size 16 \
  --epochs 50 \
  --lr 2e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type hard_ce

# B. Lens label + head only + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${LENS_TRAIN_JSON}" \
  --val_json "${LENS_VAL_JSON}" \
  --test_json "${LENS_TEST_JSON}" \
  --save_dir "${RESULTS_DIR}/B_lens_head_hard" \
  --image_size 224 \
  --backbone "${POLICY_BACKBONE}" \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope head_only \
  --loss_type hard_ce

# C. Oracle label + head only + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${RESULTS_DIR}/C_oracle_head_hard" \
  --image_size 224 \
  --backbone "${POLICY_BACKBONE}" \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope head_only \
  --loss_type hard_ce

# D. Oracle label + head only + soft label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${RESULTS_DIR}/D_oracle_head_soft" \
  --image_size 224 \
  --backbone "${POLICY_BACKBONE}" \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope head_only \
  --loss_type soft_kl

# E. Oracle label + partial unfreeze + soft label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${RESULTS_DIR}/E_oracle_partial_soft" \
  --image_size 224 \
  --backbone "${POLICY_BACKBONE}" \
  --batch_size 16 \
  --epochs 50 \
  --lr 5e-5 \
  --backbone_lr 2e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --resume_checkpoint "${RESULTS_DIR}/D_oracle_head_soft/best_checkpoint.pth" \
  --trainable_scope partial_unfreeze \
  --loss_type soft_kl

# F. Oracle label + full finetune + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${RESULTS_DIR}/F_oracle_full_hard" \
  --image_size 224 \
  --backbone "${POLICY_BACKBONE}" \
  --batch_size 16 \
  --epochs 50 \
  --lr 2e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type hard_ce

# G. Oracle label + full finetune + soft label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --save_dir "${RESULTS_DIR}/G_oracle_full_soft" \
  --image_size 224 \
  --backbone "${POLICY_BACKBONE}" \
  --batch_size 16 \
  --epochs 50 \
  --lr 2e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type soft_kl
