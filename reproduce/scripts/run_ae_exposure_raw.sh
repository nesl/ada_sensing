#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RAW_DIR="${RAW_DIR:-${PROJECT_DIR}/results/ae_exposure_raw}"
LUMINANCE_CSV="${LUMINANCE_CSV:-${PROJECT_DIR}/results/ae_luminance/per_image.csv}"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${LUMINANCE_CSV}" ]]; then
  echo "Luminance index not found: ${LUMINANCE_CSV}" >&2
  echo "Run the first-stage luminance analysis before starting the raw sweep." >&2
  exit 2
fi

mkdir -p "${RAW_DIR}"

cleanup() {
  local status=$?
  trap - INT TERM EXIT
  if [[ -n "${PID0:-}" ]]; then kill "${PID0}" 2>/dev/null || true; fi
  if [[ -n "${PID1:-}" ]]; then kill "${PID1}" 2>/dev/null || true; fi
  wait "${PID0:-}" 2>/dev/null || true
  wait "${PID1:-}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup INT TERM EXIT

echo "Starting resumable AE exposure sweep"
echo "  GPU ${GPU0}: dinov2_g"
echo "  GPU ${GPU1}: remaining 11 models"
echo "  raw results: ${RAW_DIR}"

CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON_BIN}" \
  -m reproduce_in_ae.evaluate_exposure \
  --models dinov2_g \
  --datasets all \
  --modes all \
  --device cuda:0 \
  --workers "${NUM_WORKERS}" \
  --luminance-csv "${LUMINANCE_CSV}" \
  --output-dir "${RAW_DIR}" &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" "${PYTHON_BIN}" \
  -m reproduce_in_ae.evaluate_exposure \
  --models resnet50,resnet50_deepaugment_augmix,resnet152,efficientnet_b0,efficientnet_b3,swin_v2_t,swin_v2_s,swin_v2_b,openclip_b,openclip_h,dinov2_b \
  --datasets all \
  --modes all \
  --device cuda:0 \
  --workers "${NUM_WORKERS}" \
  --luminance-csv "${LUMINANCE_CSV}" \
  --output-dir "${RAW_DIR}" &
PID1=$!

wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?
PID0=
PID1=
trap - INT TERM EXIT

"${PYTHON_BIN}" -m reproduce_in_ae.audit_exposure \
  --raw-dir "${RAW_DIR}" \
  --baseline-dir "${PROJECT_DIR}/results/raw" \
  --output "${RAW_DIR}/audit.json"
AUDIT_STATUS=$?

if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 || "${AUDIT_STATUS}" -ne 0 ]]; then
  echo "One or more workers failed. Re-run this same command to resume." >&2
  exit 1
fi

echo "Raw sweep workers finished. Inspect ${RAW_DIR}/audit.json for completeness."
