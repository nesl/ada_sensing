#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET="${1:-}"
PORT="${2:-8765}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"

if [[ "${DATASET}" != "replicated_capture" && "${DATASET}" != "dpi600" ]]; then
  echo "Usage: $0 {replicated_capture|dpi600} [port]" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" -m reproduce_in_ae.review_replication_roi \
  --dataset "${DATASET}" \
  --port "${PORT}"
