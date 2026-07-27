#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
DOWNLOAD_DIR="${PROJECT_DIR}/downloads"
ARCHIVE="${DOWNLOAD_DIR}/ImageNet-ES.zip"
HF_LOCAL_DIR="${DOWNLOAD_DIR}/hf"
HF_ARCHIVE="${HF_LOCAL_DIR}/ImageNet-ES.zip"
DATA_DIR="${WORKSPACE_DIR}/data"
EXPECTED_ROOT="${DATA_DIR}/ImageNet-ES"
URL="https://huggingface.co/datasets/Edw2n/ImageNet-ES/resolve/main/ImageNet-ES.zip"

mkdir -p "${DOWNLOAD_DIR}" "${DATA_DIR}"

if [[ -d "${EXPECTED_ROOT}/es-test/auto_exposure" ]]; then
  echo "ImageNet-ES already exists at ${EXPECTED_ROOT}"
else
  VALID_ARCHIVE=""
  if [[ -f "${HF_ARCHIVE}" ]] && unzip -tq "${HF_ARCHIVE}" >/dev/null 2>&1; then
    VALID_ARCHIVE="${HF_ARCHIVE}"
  elif [[ -f "${ARCHIVE}" ]] && unzip -tq "${ARCHIVE}" >/dev/null 2>&1; then
    VALID_ARCHIVE="${ARCHIVE}"
  elif command -v hf >/dev/null 2>&1; then
    mkdir -p "${HF_LOCAL_DIR}"
    hf download Edw2n/ImageNet-ES ImageNet-ES.zip \
      --repo-type dataset \
      --local-dir "${HF_LOCAL_DIR}" \
      --max-workers 8
    VALID_ARCHIVE="${HF_ARCHIVE}"
  else
    wget --continue --output-document="${ARCHIVE}" "${URL}"
    VALID_ARCHIVE="${ARCHIVE}"
  fi
  unzip -tq "${VALID_ARCHIVE}" >/dev/null
  unzip -n -q "${VALID_ARCHIVE}" -d "${DATA_DIR}"
fi

PYTHONPATH="${PROJECT_DIR}/src" python -m reproduce_in_ae.validate_data
