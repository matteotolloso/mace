#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(0 1 2 3 4)
fi

for seed in "${SEEDS[@]}"; do
  if [[ ! ${seed} =~ ^[0-4]$ ]]; then
    echo "Invalid split seed '${seed}'; expected an integer from 0 to 4." >&2
    exit 2
  fi

  echo "Generating ANI-1x system split for seed ${seed}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/ani1x_system_splitter.py" \
    --h5 "${SCRIPT_DIR}/ani1x-release.h5" \
    --outdir "${SCRIPT_DIR}/ani1x_system_split_${seed}" \
    --dft-key wb97x_tz.energy \
    --cc-key 'ccsd(t)_cbs.energy' \
    --dft-forces-key ' ' \
    --cc-forces-key ' ' \
    --p-seen 0.6 \
    --p-train 0.6 --p-val 0.2 --p-test-id 0.2 \
    --max-per-system 64 \
    --n-train-dft 50000 --n-val-dft 10000 --n-test-id-dft 50000 --n-test-ood-dft 50000 \
    --n-train-cc 5000 --n-val-cc 1000 --n-test-id-cc 5000 --n-test-ood-cc 5000 \
    --seed "${seed}"

  echo "Generating ANI-1x energy split for seed ${seed}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/ani1x_energy_splitter.py" \
    --h5 "${SCRIPT_DIR}/ani1x-release.h5" \
    --outdir "${SCRIPT_DIR}/ani1x_energy_split_${seed}" \
    --energy-key wb97x_tz.energy \
    --dft-key wb97x_tz.energy \
    --cc-key 'ccsd(t)_cbs.energy' \
    --dft-forces-key ' ' \
    --cc-forces-key ' ' \
    --q-low 0.5 \
    --q-high 0.55 \
    --p-train 0.6 --p-val 0.2 --p-test-id 0.2 \
    --max-per-system 64 \
    --n-train-dft 50000 --n-val-dft 10000 --n-test-id-dft 50000 --n-test-ood-dft 50000 \
    --n-train-cc 5000 --n-val-cc 1000 --n-test-id-cc 5000 --n-test-ood-cc 5000 \
    --seed "${seed}"
done
