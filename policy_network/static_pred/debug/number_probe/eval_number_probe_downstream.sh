#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/number_probe/eval_number_probe_downstream.py"

"${PYTHON_BIN}" "${SCRIPT}" \
  --label_kind "${LABEL_KIND:-oracle}" \
  --device "${DEVICE:-cuda}" \
  --model "${MODEL:-resnet50}"
