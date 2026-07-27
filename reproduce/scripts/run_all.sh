#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m reproduce_in_ae.validate_data
python -m reproduce_in_ae.evaluate "$@"
python -m reproduce_in_ae.report
python -m reproduce_in_ae.audit_results
