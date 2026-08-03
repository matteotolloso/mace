#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_DIR="${SCRIPT_DIR}/water_0"

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(0 1 2 3 4)
fi

if [[ ! -d ${SOURCE_DIR} ]]; then
  echo "Missing source water dataset: ${SOURCE_DIR}" >&2
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  if [[ ! ${seed} =~ ^[0-4]$ ]]; then
    echo "Invalid split seed '${seed}'; expected an integer from 0 to 4." >&2
    exit 2
  fi

  target_dir="${SCRIPT_DIR}/water_${seed}"
  if [[ ${seed} != 0 ]]; then
    for theory_dir in "${SOURCE_DIR}"/*; do
      [[ -d ${theory_dir} ]] || continue
      theory_name=$(basename "${theory_dir}")
      mkdir -p "${target_dir}/${theory_name}"
      cp -p "${theory_dir}/input.data" "${target_dir}/${theory_name}/input.data"
      if [[ -f ${theory_dir}/AFQMC-energies-stderr.txt ]]; then
        cp -p \
          "${theory_dir}/AFQMC-energies-stderr.txt" \
          "${target_dir}/${theory_name}/AFQMC-energies-stderr.txt"
      fi
    done
  fi

  echo "Generating water split for seed ${seed}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_water_n2p2_to_extxyz.py" \
    --root "${target_dir}" \
    --train 65 \
    --val 10 \
    --test 25 \
    --seed "${seed}"
done
