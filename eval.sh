#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

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
