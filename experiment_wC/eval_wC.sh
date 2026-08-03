#!/bin/bash

mkdir -p experiment_wC/evaluation

CUDA_VISIBLE_DEVICES=6 python utils/train_curves.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --experiment-name mace \
  --train-split dataset/water/ccsdt/train.xyz \
  --validation-split dataset/water/ccsdt/val.xyz \
  --energy-key-train REF_energy \
  --energy-key-val REF_energy \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_wC/evaluation/train_curves.csv \
  --output-plot experiment_wC/evaluation/train_curves.png

CUDA_VISIBLE_DEVICES=6 python utils/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --results-dir experiment_wC/results \
  --experiment-name mace \
  --validation-split dataset/water/ccsdt/val.xyz \
  --test-split dataset/water/ccsdt/test.xyz \
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
  --output-csv-raw experiment_wC/evaluation/reliability_test_cal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/reliability_test_cal_bins.csv \
  --output-plot experiment_wC/evaluation/reliability_test_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=6 python utils/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --results-dir experiment_wC/results \
  --experiment-name mace \
  --validation-split dataset/water/ccsdt/val.xyz \
  --test-split dataset/water/ccsdt/test.xyz \
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
  --output-csv-raw experiment_wC/evaluation/reliability_test_nocal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/reliability_test_nocal_bins.csv \
  --output-plot experiment_wC/evaluation/reliability_test_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=6 python utils/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --results-dir experiment_wC/results \
  --experiment-name mace \
  --validation-split dataset/water/ccsdt/val.xyz \
  --test-split dataset/water/ccsdt/train.xyz \
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
  --output-csv-raw experiment_wC/evaluation/reliability_train_cal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/reliability_train_cal_bins.csv \
  --output-plot experiment_wC/evaluation/reliability_train_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=6 python utils/reliability.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --results-dir experiment_wC/results \
  --experiment-name mace \
  --validation-split dataset/water/ccsdt/val.xyz \
  --test-split dataset/water/ccsdt/train.xyz \
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
  --output-csv-raw experiment_wC/evaluation/reliability_train_nocal_raw.csv \
  --output-csv-bins experiment_wC/evaluation/reliability_train_nocal_bins.csv \
  --output-plot experiment_wC/evaluation/reliability_train_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=6 python utils/epoch_raw.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 10 \
  --output-csv experiment_wC/evaluation/epoch_raw_train.csv \
  --output-plot experiment_wC/evaluation/epoch_raw_train.png

CUDA_VISIBLE_DEVICES=6 python utils/epoch_quality.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/test.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wC/evaluation/epoch_quality_test.csv \
  --output-plot experiment_wC/evaluation/epoch_quality_test.png

CUDA_VISIBLE_DEVICES=6 python utils/epoch_quality.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wC/evaluation/epoch_quality_train.csv \
  --output-plot experiment_wC/evaluation/epoch_quality_train.png

CUDA_VISIBLE_DEVICES=6 python utils/distribution.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --results-dir experiment_wC/results \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/test.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_wC/evaluation/distribution_test.csv \
  --output-plot experiment_wC/evaluation/distribution_test.png

CUDA_VISIBLE_DEVICES=6 python utils/distribution.py \
  --checkpoints-dir experiment_wC/checkpoints \
  --results-dir experiment_wC/results \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/train.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_wC/evaluation/distribution_train.csv \
  --output-plot experiment_wC/evaluation/distribution_train.png
