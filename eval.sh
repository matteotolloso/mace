#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if [[ ${1:-} == --plots-only && $# == 1 ]]; then
  # Rebuild aggregate figures directly from split CSVs: no model loading or GPU.
  for experiment in experiment_{A,B,C,D,E,F,wA,wB,wC}; do
    if [[ ! -d "$experiment/evaluation/cache/split_0" ]]; then
      printf 'Skipping %s: no evaluation cache yet.\n' "$experiment"
      continue
    fi
    args=()
    if [[ $experiment == experiment_w* ]]; then
      args+=(--log-log-reliability --reliability-axis-min 1e-4 --reliability-axis-max 1e-1)
    fi
    "${EVAL_PYTHON:-python}" -B eval/aggregate_replicates.py \
      --cache-dir "$experiment/evaluation/cache" \
      --output-dir "$experiment/evaluation" --split-seeds 0 1 2 3 4 "${args[@]}"
  done
  shopt -s nullglob
  for summary in active_learning/ani_energy/runs/aggregate*/summary_ci95.json \
                 active_learning/ani_energy/runs/aggregate*/common_evaluator/summary_ci95.json; do
    "${EVAL_PYTHON:-python}" -B active_learning/ani_energy/plot_results.py --summary "$summary"
  done
  exit 0
elif (( $# != 0 )); then
  echo "Usage: bash eval.sh [--plots-only]" >&2
  exit 2
fi

scripts=(
  "experiment_A/eval_A.sh"
  "experiment_B/eval_B.sh"
  "experiment_C/eval_C.sh"
  "experiment_D/eval_D.sh"
  "experiment_E/eval_E.sh"
  "experiment_F/eval_F.sh"
)

gpus=(6 7 5 4 3 2)

pids=()

for i in "${!scripts[@]}"; do
  script=${scripts[$i]}
  bash "$script" "${gpus[$i]}" &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Evaluation failed: ${scripts[$i]}" >&2
    status=1
  fi
done

exit "$status"
