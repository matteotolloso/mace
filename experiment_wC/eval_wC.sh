#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source eval/split_cache.sh

if (( $# != 1 )); then
  printf 'Usage: %s <gpu_number>\n' "$0" >&2
  exit 2
fi
if [[ ! $1 =~ ^[0-9]+$ ]]; then
  printf 'gpu_number must be a non-negative integer, got: %s\n' "$1" >&2
  exit 2
fi

GPU_ID=$1
EVAL_SPLIT_SEEDS="0 1 2 3 4"

run_split() {
local split_seed=$1

mkdir -p experiment_wC/evaluation

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/train_curves.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --experiment-name mace \
  --train-split dataset/water_${split_seed}/ccsdt/train.xyz \
  --validation-split dataset/water_${split_seed}/ccsdt/val.xyz \
  --energy-key-train REF_energy \
  --energy-key-val REF_energy \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_wC/evaluation/cache/split_${split_seed}/train_curves.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/train_curves.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --results-dir experiment_wC/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/water_${split_seed}/ccsdt/val.xyz \
  --test-split dataset/water_${split_seed}/ccsdt/test.xyz \
  --energy-key-val REF_energy \
  --energy-key-test REF_energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --log-log-scale \
  --axis-min 1e-4 \
  --axis-max 1e-1 \
  --output-csv-raw experiment_wC/evaluation/cache/split_${split_seed}/reliability_test_cal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/cache/split_${split_seed}/reliability_test_cal_bins.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/reliability_test_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --results-dir experiment_wC/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/water_${split_seed}/ccsdt/val.xyz \
  --test-split dataset/water_${split_seed}/ccsdt/test.xyz \
  --energy-key-val REF_energy \
  --energy-key-test REF_energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --log-log-scale \
  --axis-min 1e-4 \
  --axis-max 1e-1 \
  --output-csv-raw experiment_wC/evaluation/cache/split_${split_seed}/reliability_test_nocal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/cache/split_${split_seed}/reliability_test_nocal_bins.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/reliability_test_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --results-dir experiment_wC/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/water_${split_seed}/ccsdt/val.xyz \
  --test-split dataset/water_${split_seed}/ccsdt/train.xyz \
  --energy-key-val REF_energy \
  --energy-key-test REF_energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --log-log-scale \
  --axis-min 1e-4 \
  --axis-max 1e-1 \
  --output-csv-raw experiment_wC/evaluation/cache/split_${split_seed}/reliability_train_cal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/cache/split_${split_seed}/reliability_train_cal_bins.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/reliability_train_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --results-dir experiment_wC/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/water_${split_seed}/ccsdt/val.xyz \
  --test-split dataset/water_${split_seed}/ccsdt/train.xyz \
  --energy-key-val REF_energy \
  --energy-key-test REF_energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --log-log-scale \
  --axis-min 1e-4 \
  --axis-max 1e-1 \
  --output-csv-raw experiment_wC/evaluation/cache/split_${split_seed}/reliability_train_nocal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/cache/split_${split_seed}/reliability_train_nocal_bins.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/reliability_train_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/epoch_raw.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --experiment-name mace \
  --split-path dataset/water_${split_seed}/ccsdt/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 10 \
  --output-csv experiment_wC/evaluation/cache/split_${split_seed}/epoch_raw_train.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/epoch_raw_train.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/epoch_quality.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --experiment-name mace \
  --split-path dataset/water_${split_seed}/ccsdt/test.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wC/evaluation/cache/split_${split_seed}/epoch_quality_test.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/epoch_quality_test.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/epoch_quality.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --experiment-name mace \
  --split-path dataset/water_${split_seed}/ccsdt/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wC/evaluation/cache/split_${split_seed}/epoch_quality_train.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/epoch_quality_train.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/distribution.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --results-dir experiment_wC/results_${split_seed} \
  --experiment-name mace \
  --split-path dataset/water_${split_seed}/ccsdt/test.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_wC/evaluation/cache/split_${split_seed}/distribution_test.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/distribution_test.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/distribution.py \
  --checkpoints-dir experiment_wC/checkpoints_${split_seed} \
  --results-dir experiment_wC/results_${split_seed} \
  --experiment-name mace \
  --split-path dataset/water_${split_seed}/ccsdt/train.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_wC/evaluation/cache/split_${split_seed}/distribution_train.csv \
  --output-plot experiment_wC/evaluation/cache/split_${split_seed}/distribution_train.png
}

for split_seed in $EVAL_SPLIT_SEEDS; do
  printf 'Evaluating experiment_wC on dataset split %s\n' "$split_seed"
  run_split "$split_seed"
done

aggregate_args=()
aggregate_args+=(--log-log-reliability --reliability-axis-min 1e-4 --reliability-axis-max 1e-1)
aggregate_split_caches experiment_wC "${aggregate_args[@]}"
