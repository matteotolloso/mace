#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "Usage: bash active_learning/ani_energy/run_all.sh <gpu_number> [--aggregate-only --wait] [--seed N] [--epochs N] [--run-tag NAME] [--common-evaluator]" >&2
  exit 2
fi
GPU="$1"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHON="${PYTHON:-python}"
exec "$PYTHON" -B "$SCRIPT_DIR/five_splits.py" --gpu "$GPU" "$@"
