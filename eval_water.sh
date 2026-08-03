#!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-6}

run_eval() {
  local script_path=$1
  echo "Running ${script_path} on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} bash "${script_path}"
}

run_eval experiment_wA/eval_wA.sh
run_eval experiment_wB/eval_wB.sh
run_eval experiment_wC/eval_wC.sh

echo "All water evaluations completed."
