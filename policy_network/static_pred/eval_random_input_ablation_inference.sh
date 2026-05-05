#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
INDEX_EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_best_index_predictions.py"
DOWNSTREAM_EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_topk_downstream_candidates.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

SINGLE_RESULTS_DIR="${SINGLE_RESULTS_DIR:-${ROOT_DIR}/policy_network/results_single_mobilenet_v3_small_fixedk_sweep}"
DUAL_RESULTS_DIR="${DUAL_RESULTS_DIR:-${ROOT_DIR}/policy_network/results_dual_mobilenet_v3_small_fixedk_sweep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/policy_network/results_random_input_ablation_inference}"

BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOPK="${TOPK:-5}"
NOISE_SEED="${NOISE_SEED:-0}"
START_K="${START_K:-0}"
END_K="${END_K:-26}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
RUN_SINGLE="${RUN_SINGLE:-1}"
RUN_DUAL="${RUN_DUAL:-1}"

run_eval() {
  local checkpoint_path="$1"
  local output_dir="$2"
  local single_source="$3"
  local ae_variant="$4"
  local env_variant="$5"

  mkdir -p "${output_dir}"

  "${PYTHON_BIN}" "${INDEX_EVAL_SCRIPT}" \
    --checkpoint "${checkpoint_path}" \
    --data_json "${ORACLE_TEST_JSON}" \
    --output_json "${output_dir}/index_test_${CHECKPOINT_KIND}.json" \
    --manifest "${MANIFEST_JSON}" \
    --image_size 224 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --eval_single_input_source "${single_source}" \
    --eval_ae_input_variant "${ae_variant}" \
    --eval_env_input_variant "${env_variant}" \
    --eval_noise_seed "${NOISE_SEED}"

  "${PYTHON_BIN}" "${DOWNSTREAM_EVAL_SCRIPT}" \
    --manifest "${MANIFEST_JSON}" \
    --checkpoint "${checkpoint_path}" \
    --data_json "${ORACLE_TEST_JSON}" \
    --output_json "${output_dir}/downstream_test_${CHECKPOINT_KIND}.json" \
    --image_size 224 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --topk "${TOPK}" \
    --eval_single_input_source "${single_source}" \
    --eval_ae_input_variant "${ae_variant}" \
    --eval_env_input_variant "${env_variant}" \
    --eval_noise_seed "${NOISE_SEED}"
}

for ENV_OPTION_ID in $(seq "${START_K}" "${END_K}"); do
  if [[ "${RUN_SINGLE}" == "1" ]]; then
    SINGLE_RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${SINGLE_RESULTS_DIR}" "${ENV_OPTION_ID}")"
    SINGLE_CKPT="${SINGLE_RUN_DIR}/${CHECKPOINT_KIND}_checkpoint.pth"
    if [[ -f "${SINGLE_CKPT}" ]]; then
      OUT_TRUE="$(printf "%s/single_fixedk/true/fixed_k_%02d/F_oracle_full_hard" "${OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
      OUT_RANDOM="$(printf "%s/single_fixedk/random/fixed_k_%02d/F_oracle_full_hard" "${OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
      run_eval "${SINGLE_CKPT}" "${OUT_TRUE}" env real real
      run_eval "${SINGLE_CKPT}" "${OUT_RANDOM}" env real random_noise_per_sample
    else
      echo "Missing single checkpoint, skipping: ${SINGLE_CKPT}" >&2
    fi
  fi

  if [[ "${RUN_DUAL}" == "1" ]]; then
    DUAL_RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${DUAL_RESULTS_DIR}" "${ENV_OPTION_ID}")"
    DUAL_CKPT="${DUAL_RUN_DIR}/${CHECKPOINT_KIND}_checkpoint.pth"
    if [[ -f "${DUAL_CKPT}" ]]; then
      OUT_TT="$(printf "%s/dual_fixedk/trueAE_trueK/fixed_k_%02d/F_oracle_full_hard" "${OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
      OUT_RT="$(printf "%s/dual_fixedk/randomAE_trueK/fixed_k_%02d/F_oracle_full_hard" "${OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
      OUT_TR="$(printf "%s/dual_fixedk/trueAE_randomK/fixed_k_%02d/F_oracle_full_hard" "${OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
      OUT_RR="$(printf "%s/dual_fixedk/randomAE_randomK/fixed_k_%02d/F_oracle_full_hard" "${OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
      run_eval "${DUAL_CKPT}" "${OUT_TT}" baseline real real
      run_eval "${DUAL_CKPT}" "${OUT_RT}" baseline random_noise_per_sample real
      run_eval "${DUAL_CKPT}" "${OUT_TR}" baseline real random_noise_per_sample
      run_eval "${DUAL_CKPT}" "${OUT_RR}" baseline random_noise_per_sample random_noise_per_sample
    else
      echo "Missing dual checkpoint, skipping: ${DUAL_CKPT}" >&2
    fi
  fi
done
