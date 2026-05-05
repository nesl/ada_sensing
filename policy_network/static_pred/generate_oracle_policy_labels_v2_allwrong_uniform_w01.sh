#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/generate_oracle_policy_labels.py" \
  --manifest "${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json" \
  --output_dir "${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels_v2_allwrong_uniform_w01" \
  --split_source_dir "${ROOT_DIR}/data/ImageNet-ES-Diverse/policy_labels" \
  --model resnet50 \
  --image_size 224 \
  --num_candidates 27 \
  --device "${DEVICE:-cuda}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --soft_label_mode confidence_correct \
  --all_wrong_soft_target_mode uniform \
  --all_wrong_sample_weight 0.1
