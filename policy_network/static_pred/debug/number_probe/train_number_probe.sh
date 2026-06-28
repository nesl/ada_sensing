#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/number_probe/train_number_probe.py"
LABEL_KIND="${LABEL_KIND:-oracle}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/policy_network/results_number_probe/${LABEL_KIND}}"

if [[ "${LABEL_KIND}" != "oracle" && "${LABEL_KIND}" != "policy" ]]; then
  echo "Unsupported LABEL_KIND=${LABEL_KIND}. Use oracle or policy." >&2
  exit 1
fi

COMMON_ARGS=(
  --label_kind "${LABEL_KIND}"
  --epochs "${EPOCHS:-100}"
  --device "${DEVICE:-cuda}"
  --seed "${SEED:-0}"
)

for FEATURE_MODE in lightning_class lightning class; do
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --feature_mode "${FEATURE_MODE}" \
    --save_dir "${RESULTS_ROOT}/${FEATURE_MODE}"
done
