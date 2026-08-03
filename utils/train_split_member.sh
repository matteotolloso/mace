#!/bin/bash

# Shared implementation sourced by experiment train scripts.

run_split_member() {
  if (( $# != 3 )); then
    echo "Usage: $0 <split_seed:0-4> <member_seed:0-9> <gpu_id>" >&2
    echo "Example: bash $0 2 7 4" >&2
    return 2
  fi

  local split_seed=$1
  local member_seed=$2
  local gpu_id=$3

  if [[ ! ${split_seed} =~ ^[0-4]$ ]]; then
    echo "Invalid split seed '${split_seed}'; expected 0 through 4." >&2
    return 2
  fi
  if [[ ! ${member_seed} =~ ^[0-9]$ ]]; then
    echo "Invalid member seed '${member_seed}'; expected 0 through 9." >&2
    return 2
  fi
  if [[ ! ${gpu_id} =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id '${gpu_id}'; expected a non-negative integer." >&2
    return 2
  fi

  local python_bin=${PYTHON_BIN:-python}
  local dataset_dir="${REPO_ROOT}/${DATASET_PREFIX}_${split_seed}"
  local train_file="${dataset_dir}/${TRAIN_RELATIVE_PATH}"
  local valid_file="${dataset_dir}/${VALID_RELATIVE_PATH}"
  local test_file="${dataset_dir}/${TEST_RELATIVE_PATH}"
  local model_dir="${REPO_ROOT}/${EXPERIMENT_DIR}/models_${split_seed}"
  local log_dir="${REPO_ROOT}/${EXPERIMENT_DIR}/logs_${split_seed}"
  local checkpoints_dir="${REPO_ROOT}/${EXPERIMENT_DIR}/checkpoints_${split_seed}"
  local results_dir="${REPO_ROOT}/${EXPERIMENT_DIR}/results_${split_seed}"

  for dataset_file in "${train_file}" "${valid_file}" "${test_file}"; do
    if [[ ! -f ${dataset_file} ]]; then
      echo "Missing dataset file: ${dataset_file}" >&2
      return 1
    fi
  done

  mkdir -p "${model_dir}" "${log_dir}" "${checkpoints_dir}" "${results_dir}"

  local command=(
    "${python_bin}" "${REPO_ROOT}/mace/cli/run_train.py"
    --config "${REPO_ROOT}/${CONFIG_FILE}"
    --train_file "${train_file}"
    --valid_file "${valid_file}"
    --test_file "${test_file}"
    --model_dir "${model_dir}"
    --log_dir "${log_dir}"
    --checkpoints_dir "${checkpoints_dir}"
    --results_dir "${results_dir}"
    --seed "${member_seed}"
    --wandb_name "split_${split_seed}_mace_seed_${member_seed}"
  )

  if [[ -n ${FOUNDATION_EXPERIMENT_DIR:-} ]]; then
    local foundation_model="${REPO_ROOT}/${FOUNDATION_EXPERIMENT_DIR}/checkpoints_${split_seed}/mace_run-${member_seed}.model"
    if [[ ! -f ${foundation_model} ]]; then
      echo "Missing matching foundation model: ${foundation_model}" >&2
      return 1
    fi
    command+=(--foundation_model "${foundation_model}")
  fi

  echo "Experiment: ${EXPERIMENT_DIR}"
  echo "Dataset split seed: ${split_seed}"
  echo "Ensemble member seed: ${member_seed}"
  echo "GPU: ${gpu_id}"
  echo "Outputs: ${EXPERIMENT_DIR}/{models,logs,checkpoints,results}_${split_seed}"

  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES=${gpu_id} "${command[@]}"
}
