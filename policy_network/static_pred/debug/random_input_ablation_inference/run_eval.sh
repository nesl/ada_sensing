#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python}"
INDEX_EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_best_index_predictions.py"
DOWNSTREAM_EVAL_SCRIPT="${ROOT_DIR}/policy_network/static_pred/debug/analyze_topk_downstream_candidates.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_results.py"
CACHE_SCRIPT="${SCRIPT_DIR}/precompute_downstream_cache.py"
MANIFEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/manifest_all.json"
ORACLE_TEST_JSON="${ROOT_DIR}/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json"

SINGLE_RESULTS_DIR="${SINGLE_RESULTS_DIR:-${ROOT_DIR}/policy_network/results/results_single_mobilenet_v3_small_fixedk_sweep}"
DUAL_RESULTS_DIR="${DUAL_RESULTS_DIR:-${ROOT_DIR}/policy_network/results/results_dual_mobilenet_v3_small_fixedk_sweep}"

BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNSTREAM_MODEL="${DOWNSTREAM_MODEL:-resnet50}"
TOPK="${TOPK:-1}"
INDEX_TOPK="${INDEX_TOPK:-1}"
NOISE_SEEDS="${NOISE_SEEDS:-$(seq -s ' ' 0 19)}"
START_K="${START_K:-0}"
END_K="${END_K:-26}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
RUN_SINGLE="${RUN_SINGLE:-1}"
RUN_DUAL="${RUN_DUAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_DOWNSTREAM_CACHE="${RUN_DOWNSTREAM_CACHE:-1}"
RUN_REAL_ONLY_SEED0="${RUN_REAL_ONLY_SEED0:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/policy_network/results/results_random_input_ablation_inference}"
DOWNSTREAM_CACHE_JSON="${DOWNSTREAM_CACHE_JSON:-${ROOT_DIR}/policy_network/results/downstream_candidate_cache/${DOWNSTREAM_MODEL}_im224_oracle_test.json}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/summary_${CHECKPOINT_KIND}.csv}"
RAW_SUMMARY_CSV="${RAW_SUMMARY_CSV:-}"

if [[ "${RUN_DOWNSTREAM_CACHE}" == "1" && ! -f "${DOWNSTREAM_CACHE_JSON}" ]]; then
  "${PYTHON_BIN}" "${CACHE_SCRIPT}" \
    --manifest "${MANIFEST_JSON}" \
    --data_json "${ORACLE_TEST_JSON}" \
    --output_json "${DOWNSTREAM_CACHE_JSON}" \
    --model "${DOWNSTREAM_MODEL}" \
    --image_size 224 \
    --device "${DEVICE}"
fi

run_eval() {
  local checkpoint_path="$1"
  local output_dir="$2"
  local single_source="$3"
  local ae_variant="$4"
  local env_variant="$5"
  local noise_seed="$6"

  mkdir -p "${output_dir}"

  if [[ "${SKIP_EXISTING}" == "1" \
      && -f "${output_dir}/index_test_${CHECKPOINT_KIND}.json" \
      && -f "${output_dir}/downstream_test_${CHECKPOINT_KIND}.json" ]]; then
    echo "Existing eval outputs found, skipping: ${output_dir}"
    return
  fi

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
    --eval_noise_seed "${noise_seed}" \
    --topk "${INDEX_TOPK}"

  DOWNSTREAM_ARGS=(
    --manifest "${MANIFEST_JSON}" \
    --checkpoint "${checkpoint_path}" \
    --data_json "${ORACLE_TEST_JSON}" \
    --output_json "${output_dir}/downstream_test_${CHECKPOINT_KIND}.json" \
    --image_size 224 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --model "${DOWNSTREAM_MODEL}" \
    --topk "${TOPK}" \
    --eval_single_input_source "${single_source}" \
    --eval_ae_input_variant "${ae_variant}" \
    --eval_env_input_variant "${env_variant}" \
    --eval_noise_seed "${noise_seed}"
  )
  if [[ "${RUN_DOWNSTREAM_CACHE}" == "1" ]]; then
    DOWNSTREAM_ARGS+=(--downstream_cache_json "${DOWNSTREAM_CACHE_JSON}")
  fi
  "${PYTHON_BIN}" "${DOWNSTREAM_EVAL_SCRIPT}" "${DOWNSTREAM_ARGS[@]}"
}

for NOISE_SEED in ${NOISE_SEEDS}; do
  SEED_OUTPUT_ROOT="${OUTPUT_ROOT}/seed${NOISE_SEED}"
  echo "Running inference random-input ablation with eval_noise_seed=${NOISE_SEED}"

  for ENV_OPTION_ID in $(seq "${START_K}" "${END_K}"); do
    if [[ "${RUN_SINGLE}" == "1" ]]; then
      SINGLE_RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${SINGLE_RESULTS_DIR}" "${ENV_OPTION_ID}")"
      SINGLE_CKPT="${SINGLE_RUN_DIR}/${CHECKPOINT_KIND}_checkpoint.pth"
      if [[ -f "${SINGLE_CKPT}" ]]; then
        OUT_TRUE="$(printf "%s/single_fixedk/true/fixed_k_%02d/F_oracle_full_hard" "${SEED_OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
        OUT_RANDOM="$(printf "%s/single_fixedk/random/fixed_k_%02d/F_oracle_full_hard" "${SEED_OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
        if [[ "${RUN_REAL_ONLY_SEED0}" != "1" || "${NOISE_SEED}" == "0" ]]; then
          run_eval "${SINGLE_CKPT}" "${OUT_TRUE}" env real real "${NOISE_SEED}"
        fi
        run_eval "${SINGLE_CKPT}" "${OUT_RANDOM}" env real random_noise_per_sample "${NOISE_SEED}"
      else
        echo "Missing single checkpoint, skipping: ${SINGLE_CKPT}" >&2
      fi
    fi

    if [[ "${RUN_DUAL}" == "1" ]]; then
      DUAL_RUN_DIR="$(printf "%s/fixed_k_%02d/F_oracle_full_hard" "${DUAL_RESULTS_DIR}" "${ENV_OPTION_ID}")"
      DUAL_CKPT="${DUAL_RUN_DIR}/${CHECKPOINT_KIND}_checkpoint.pth"
      if [[ -f "${DUAL_CKPT}" ]]; then
        OUT_TT="$(printf "%s/dual_fixedk/trueAE_trueK/fixed_k_%02d/F_oracle_full_hard" "${SEED_OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
        OUT_RT="$(printf "%s/dual_fixedk/randomAE_trueK/fixed_k_%02d/F_oracle_full_hard" "${SEED_OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
        OUT_TR="$(printf "%s/dual_fixedk/trueAE_randomK/fixed_k_%02d/F_oracle_full_hard" "${SEED_OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
        OUT_RR="$(printf "%s/dual_fixedk/randomAE_randomK/fixed_k_%02d/F_oracle_full_hard" "${SEED_OUTPUT_ROOT}" "${ENV_OPTION_ID}")"
        if [[ "${RUN_REAL_ONLY_SEED0}" != "1" || "${NOISE_SEED}" == "0" ]]; then
          run_eval "${DUAL_CKPT}" "${OUT_TT}" baseline real real "${NOISE_SEED}"
        fi
        run_eval "${DUAL_CKPT}" "${OUT_RT}" baseline random_noise_per_sample real "${NOISE_SEED}"
        run_eval "${DUAL_CKPT}" "${OUT_TR}" baseline real random_noise_per_sample "${NOISE_SEED}"
        run_eval "${DUAL_CKPT}" "${OUT_RR}" baseline random_noise_per_sample random_noise_per_sample "${NOISE_SEED}"
      else
        echo "Missing dual checkpoint, skipping: ${DUAL_CKPT}" >&2
      fi
    fi
  done
done

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  SUMMARY_ARGS=(
    --results_root "${OUTPUT_ROOT}" \
    --checkpoint_kind "${CHECKPOINT_KIND}" \
    --output_csv "${SUMMARY_CSV}" \
    --aggregate_seed_dirs
  )
  if [[ -n "${RAW_SUMMARY_CSV}" ]]; then
    SUMMARY_ARGS+=(--raw_output_csv "${RAW_SUMMARY_CSV}")
  fi
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" "${SUMMARY_ARGS[@]}"
fi
