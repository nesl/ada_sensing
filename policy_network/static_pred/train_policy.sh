#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${ROOT_DIR}/policy_network/static_pred/train_policy.py"

LENS_TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_train_labels.json"
LENS_VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_val_labels.json"
LENS_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"

ORACLE_TRAIN_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json"
ORACLE_VAL_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"

# A. Lens label + full finetune + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${LENS_TRAIN_JSON}" \
  --val_json "${LENS_VAL_JSON}" \
  --test_json "${LENS_TEST_JSON}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard" \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type hard_ce \
  --checkpoint_name policy_net_ori.pth \
  --train_input_sampling fixed_option \
  --train_input_option_id 13

# B. Lens label + head only + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${LENS_TRAIN_JSON}" \
  --val_json "${LENS_VAL_JSON}" \
  --test_json "${LENS_TEST_JSON}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard" \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope head_only \
  --loss_type hard_ce \
  --checkpoint_name policy_net_fixed13_head_hard.pth \
  --train_input_sampling fixed_option \
  --train_input_option_id 13

# C. Oracle label + head only + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard" \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope head_only \
  --loss_type hard_ce \
  --checkpoint_name policy_net_fixed13_head_hard.pth \
  --train_input_sampling fixed_option \
  --train_input_option_id 13

# D. Oracle label + head only + soft label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft" \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope head_only \
  --loss_type soft_kl \
  --checkpoint_name policy_net_fixed13_head_soft.pth \
  --train_input_sampling fixed_option \
  --train_input_option_id 13

# E. Oracle label + partial unfreeze + soft label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft" \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --resume_checkpoint "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth" \
  --trainable_scope partial_unfreeze \
  --loss_type soft_kl \
  --checkpoint_name policy_net_fixed13_part_soft.pth \
  --train_input_sampling fixed_option \
  --train_input_option_id 13

# F. Oracle label + full finetune + hard label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard" \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type hard_ce \
  --checkpoint_name policy_net_fixed13_full_hard.pth \
  --train_input_sampling fixed_option \
  --train_input_option_id 13

# G. Oracle label + full finetune + soft label + fixed input-13
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --train_json "${ORACLE_TRAIN_JSON}" \
  --val_json "${ORACLE_VAL_JSON}" \
  --test_json "${ORACLE_TEST_JSON}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft" \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type soft_kl \
  --checkpoint_name policy_net_fixed13_full_soft.pth \
  --train_input_sampling fixed_option \
  --train_input_option_id 13
