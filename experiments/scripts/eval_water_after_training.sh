#!/usr/bin/env bash
# Wait for train_water_splits.py to finish, check that all 150 water members are
# complete, then evaluate wA, wB and wC over all five splits on at most two GPUs:
# wA on GPU_A and wC on GPU_B in parallel, then wB on GPU_A. Each eval_wX.sh runs
# its five splits sequentially and then aggregates them with confidence intervals.
#
# Usage (from the repository root):
#   bash experiments/scripts/eval_water_after_training.sh <scheduler_pid> <gpu_a> <gpu_b>

set -uo pipefail

SCHEDULER_PID=$1
GPU_A=$2
GPU_B=$3
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
LOG_DIR=experiments/runs/water
ENV_BIN=${ENV_BIN:-${CONDA_PREFIX:?activate the mace conda environment or set ENV_BIN}/bin}
export PATH="$ENV_BIN:$PATH" MPLCONFIGDIR=/tmp/mpl-cache EVAL_PYTHON="$ENV_BIN/python"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

while kill -0 "$SCHEDULER_PID" 2>/dev/null; do sleep 60; done
echo "$(stamp) training scheduler finished"

missing=$(./check_experiments.sh 2>/dev/null | grep -E "^experiment_w[ABC] split_[0-4]: INCOMPLETE" || true)
if [[ -n $missing ]]; then
  echo "$(stamp) NOT evaluating: water members are incomplete:"
  echo "$missing"
  exit 1
fi
echo "$(stamp) all water members complete; starting evaluation"

bash experiment_wA/eval_wA.sh "$GPU_A" > "$LOG_DIR/eval_wA.log" 2>&1 &
pid_a=$!
bash experiment_wC/eval_wC.sh "$GPU_B" > "$LOG_DIR/eval_wC.log" 2>&1 &
pid_c=$!
wait "$pid_a"; status_a=$?
echo "$(stamp) eval_wA exit $status_a"
bash experiment_wB/eval_wB.sh "$GPU_A" > "$LOG_DIR/eval_wB.log" 2>&1
status_b=$?
echo "$(stamp) eval_wB exit $status_b"
wait "$pid_c"; status_c=$?
echo "$(stamp) eval_wC exit $status_c"

echo "WATER_EVAL_DONE wA=$status_a wB=$status_b wC=$status_c"
