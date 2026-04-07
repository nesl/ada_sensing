#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

bash "${ROOT_DIR}/policy_network/static_pred/run_fixed_input13_eval.sh"
bash "${ROOT_DIR}/policy_network/static_pred/debug/run_topk_downstream_debug.sh"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/visualize_train_history_distribution.py" \
  --history_json "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/train_history.json" \
  --output_png "${ROOT_DIR}/policy_network/results_fixed_input13/A_lens_ft_all_hard/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/visualize_train_history_distribution.py" \
  --history_json "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/train_history.json" \
  --output_png "${ROOT_DIR}/policy_network/results_fixed_input13/B_lens_head_hard/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/visualize_train_history_distribution.py" \
  --history_json "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/train_history.json" \
  --output_png "${ROOT_DIR}/policy_network/results_fixed_input13/C_oracle_head_hard/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/visualize_train_history_distribution.py" \
  --history_json "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/train_history.json" \
  --output_png "${ROOT_DIR}/policy_network/results_fixed_input13/D_oracle_head_soft/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/visualize_train_history_distribution.py" \
  --history_json "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/train_history.json" \
  --output_png "${ROOT_DIR}/policy_network/results_fixed_input13/E_oracle_partial_soft/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/visualize_train_history_distribution.py" \
  --history_json "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/train_history.json" \
  --output_png "${ROOT_DIR}/policy_network/results_fixed_input13/F_oracle_full_hard/train_history_distribution_summary.png"

"${PYTHON_BIN}" "${ROOT_DIR}/policy_network/static_pred/debug/visualize_train_history_distribution.py" \
  --history_json "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/train_history.json" \
  --output_png "${ROOT_DIR}/policy_network/results_fixed_input13/G_oracle_full_soft/train_history_distribution_summary.png"
