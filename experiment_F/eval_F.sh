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
mkdir -p experiment_F/evaluation
# ensemble train/val loss curves

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/train_curves.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --experiment-name mace \
  --train-split dataset/ani1x_energy_split_${split_seed}/cc_train.xyz \
  --validation-split dataset/ani1x_energy_split_${split_seed}/cc_val.xyz \
  --energy-key-train "ccsd(t)_cbs.energy" \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/train_curves.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/train_curves.png

# reliability diagram ID

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split_${split_seed}/cc_val.xyz \
  --test-split dataset/ani1x_energy_split_${split_seed}/cc_test_id.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --output-csv-raw experiment_F/evaluation/cache/split_${split_seed}/reliability_id_cal_raw.csv \
  --output-csv-bins experiment_F/evaluation/cache/split_${split_seed}/reliability_id_cal_bins.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/reliability_id_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split_${split_seed}/cc_val.xyz \
  --test-split dataset/ani1x_energy_split_${split_seed}/cc_test_id.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --output-csv-raw experiment_F/evaluation/cache/split_${split_seed}/reliability_id_nocal_raw.csv \
  --output-csv-bins experiment_F/evaluation/cache/split_${split_seed}/reliability_id_nocal_bins.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/reliability_id_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# reliability diagram OOD

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split_${split_seed}/cc_val.xyz \
  --test-split dataset/ani1x_energy_split_${split_seed}/cc_test_ood.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --output-csv-raw experiment_F/evaluation/cache/split_${split_seed}/reliability_ood_cal_raw.csv \
  --output-csv-bins experiment_F/evaluation/cache/split_${split_seed}/reliability_ood_cal_bins.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/reliability_ood_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/reliability.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split_${split_seed}/cc_val.xyz \
  --test-split dataset/ani1x_energy_split_${split_seed}/cc_test_ood.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --output-csv-raw experiment_F/evaluation/cache/split_${split_seed}/reliability_ood_nocal_raw.csv \
  --output-csv-bins experiment_F/evaluation/cache/split_${split_seed}/reliability_ood_nocal_bins.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/reliability_ood_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# train AU/EU diagram from checkpoints

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/epoch_raw.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split_${split_seed}/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 10 \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/epoch_raw_train.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/epoch_raw_train.png

# train diagram metrics

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/epoch_quality.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split_${split_seed}/cc_test_id.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/epoch_quality_id.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/epoch_quality_id.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/epoch_quality.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split_${split_seed}/cc_test_ood.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/epoch_quality_ood.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/epoch_quality_ood.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/epoch_quality.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split_${split_seed}/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/epoch_quality_train.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/epoch_quality_train.png

# distributions

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/distribution.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split_${split_seed}/cc_test_id.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/distribution_id.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/distribution_id.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/distribution.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split_${split_seed}/cc_test_ood.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/distribution_ood.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/distribution_ood.png

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/distribution.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split_${split_seed}/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_F/evaluation/cache/split_${split_seed}/distribution_train.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/distribution_train.png


# uncertainty vs energy for energy-OOD test sets

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/energy_ood.py \
  --checkpoints-dir experiment_F/checkpoints_${split_seed} \
  --results-dir experiment_F/results_${split_seed} \
  --experiment-name mace \
  --test-id-split dataset/ani1x_energy_split_${split_seed}/cc_test_id.xyz \
  --test-ood-split dataset/ani1x_energy_split_${split_seed}/cc_test_ood.xyz \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --device cuda \
  --batch-size 256 \
  --output-csv-raw experiment_F/evaluation/cache/split_${split_seed}/energy_ood_raw.csv \
  --output-csv-bins experiment_F/evaluation/cache/split_${split_seed}/energy_ood_bins.csv \
  --output-plot experiment_F/evaluation/cache/split_${split_seed}/energy_ood.png
}

for split_seed in $EVAL_SPLIT_SEEDS; do
  printf 'Evaluating experiment_F on dataset split %s\n' "$split_seed"
  run_split "$split_seed"
done

aggregate_args=()
aggregate_split_caches experiment_F "${aggregate_args[@]}"
