#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
CHECKPOINT="${CHECKPOINT_DIR}/deepaugment_and_augmix.pth.tar"
GOOGLE_DRIVE_ID="1QKmc_p6-qDkh51WvsaS9HKFv8bX5jLnP"

mkdir -p "${CHECKPOINT_DIR}"

if [[ -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint already exists at ${CHECKPOINT}"
else
  python -m gdown --id "${GOOGLE_DRIVE_ID}" --output "${CHECKPOINT}"
fi

sha256sum "${CHECKPOINT}"

