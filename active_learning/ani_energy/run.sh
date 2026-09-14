#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "Usage: bash active_learning/ani_energy/run.sh <gpu_number> [workflow options]" >&2
  exit 2
fi

GPU="$1"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -B "$SCRIPT_DIR/workflow.py" "$@"
