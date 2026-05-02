#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/policy_network/results_hist3_multiview}"

mkdir -p "${RESULTS_DIR}"

# "${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/select_brightness_histogram_probes.py" \
#   --manifest "${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json" \
#   --data_json \
#     "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json" \
#     "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json" \
#     "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
#   --output_json "${RESULTS_DIR}/brightness_histogram_probes.json" \
#   --bins 64 \
#   --resize 128 \
#   --random_seeds 0 1 2

# H1: hist3 only. hist3 = option_id [11, 7, 2] = param_3, param_25, param_20.
"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/train_policy.py" \
  --train_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json" \
  --val_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json" \
  --test_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
  --save_dir "${RESULTS_DIR}/H1_hist3_only" \
  --manifest_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json" \
  --image_size 224 \
  --backbone mobilenet_v3_small \
  --input_mode multiview \
  --input_variant real \
  --env_option_ids 11,7,2 \
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

# H2: AE + hist3.
"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/train_policy.py" \
  --train_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json" \
  --val_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json" \
  --test_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
  --save_dir "${RESULTS_DIR}/H2_ae_hist3" \
  --manifest_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json" \
  --image_size 224 \
  --backbone mobilenet_v3_small \
  --input_mode multiview \
  --input_variant real \
  --env_option_ids 11,7,2 \
  --include_ae_input \
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

# R1 seed 0: random3 = option_id [12, 24, 13] = param_2, param_1, param_14.
"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/train_policy.py" \
  --train_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json" \
  --val_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json" \
  --test_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
  --save_dir "${RESULTS_DIR}/R1_random3_seed0" \
  --manifest_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json" \
  --image_size 224 \
  --backbone mobilenet_v3_small \
  --input_mode multiview \
  --input_variant real \
  --env_option_ids 12,24,13 \
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

# R1 seed 1: random3 = option_id [4, 18, 25] = param_18, param_21, param_12.
"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/train_policy.py" \
  --train_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json" \
  --val_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json" \
  --test_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
  --save_dir "${RESULTS_DIR}/R1_random3_seed1" \
  --manifest_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json" \
  --image_size 224 \
  --backbone mobilenet_v3_small \
  --input_mode multiview \
  --input_variant real \
  --env_option_ids 4,18,25 \
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

# R1 seed 2: random3 = option_id [1, 2, 11] = param_17, param_20, param_3.
"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/train_policy.py" \
  --train_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_train_labels.json" \
  --val_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_val_labels.json" \
  --test_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json" \
  --save_dir "${RESULTS_DIR}/R1_random3_seed2" \
  --manifest_json "${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json" \
  --image_size 224 \
  --backbone mobilenet_v3_small \
  --input_mode multiview \
  --input_variant real \
  --env_option_ids 1,2,11 \
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
