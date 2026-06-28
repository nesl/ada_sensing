#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/policy_network/results_hist3_multiview}"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/policy_visualization/visualize_train_history_distribution.py" \
  --history_json "${RESULTS_DIR}/H1_hist3_only/train_history.json" \
  --output_png "${RESULTS_DIR}/H1_hist3_only/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/policy_visualization/visualize_train_history_distribution.py" \
  --history_json "${RESULTS_DIR}/H2_ae_hist3/train_history.json" \
  --output_png "${RESULTS_DIR}/H2_ae_hist3/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/policy_visualization/visualize_train_history_distribution.py" \
  --history_json "${RESULTS_DIR}/R1_random3_seed0/train_history.json" \
  --output_png "${RESULTS_DIR}/R1_random3_seed0/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/policy_visualization/visualize_train_history_distribution.py" \
  --history_json "${RESULTS_DIR}/R1_random3_seed1/train_history.json" \
  --output_png "${RESULTS_DIR}/R1_random3_seed1/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/policy_visualization/visualize_train_history_distribution.py" \
  --history_json "${RESULTS_DIR}/R1_random3_seed2/train_history.json" \
  --output_png "${RESULTS_DIR}/R1_random3_seed2/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/multiview_probes/visualize_multiview_probe_images.py" \
  --manifest "${MANIFEST_JSON}" \
  --probe_json "${RESULTS_DIR}/brightness_histogram_probes.json" \
  --output_dir "${RESULTS_DIR}/probe_visualizations" \
  --envs l1,l2,l3,l4,l6,l7 \
  --image_size 180

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/index_prediction/visualize_pred_vs_gt_index_distribution.py" \
  --checkpoint "${RESULTS_DIR}/H1_hist3_only/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --manifest "${MANIFEST_JSON}" \
  --output_png "${RESULTS_DIR}/H1_hist3_only/pred_vs_gt_index_distribution.png" \
  --output_json "${RESULTS_DIR}/H1_hist3_only/pred_vs_gt_index_distribution.json" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/index_prediction/visualize_pred_vs_gt_index_distribution.py" \
  --checkpoint "${RESULTS_DIR}/H2_ae_hist3/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --manifest "${MANIFEST_JSON}" \
  --output_png "${RESULTS_DIR}/H2_ae_hist3/pred_vs_gt_index_distribution.png" \
  --output_json "${RESULTS_DIR}/H2_ae_hist3/pred_vs_gt_index_distribution.json" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/index_prediction/visualize_pred_vs_gt_index_distribution.py" \
  --checkpoint "${RESULTS_DIR}/R1_random3_seed0/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --manifest "${MANIFEST_JSON}" \
  --output_png "${RESULTS_DIR}/R1_random3_seed0/pred_vs_gt_index_distribution.png" \
  --output_json "${RESULTS_DIR}/R1_random3_seed0/pred_vs_gt_index_distribution.json" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/index_prediction/visualize_pred_vs_gt_index_distribution.py" \
  --checkpoint "${RESULTS_DIR}/R1_random3_seed1/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --manifest "${MANIFEST_JSON}" \
  --output_png "${RESULTS_DIR}/R1_random3_seed1/pred_vs_gt_index_distribution.png" \
  --output_json "${RESULTS_DIR}/R1_random3_seed1/pred_vs_gt_index_distribution.json" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/index_prediction/visualize_pred_vs_gt_index_distribution.py" \
  --checkpoint "${RESULTS_DIR}/R1_random3_seed2/best_checkpoint.pth" \
  --data_json "${ORACLE_TEST_JSON}" \
  --manifest "${MANIFEST_JSON}" \
  --output_png "${RESULTS_DIR}/R1_random3_seed2/pred_vs_gt_index_distribution.png" \
  --output_json "${RESULTS_DIR}/R1_random3_seed2/pred_vs_gt_index_distribution.json" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"
