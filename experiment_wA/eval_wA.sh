#!/bin/bash

mkdir -p experiment_wA/evaluation

CUDA_VISIBLE_DEVICES=6 python utils/train_curves.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --experiment-name mace \
  --train-split dataset/water/blyp/train.xyz \
  --validation-split dataset/water/blyp/val.xyz \
  --energy-key-train REF_energy \
  --energy-key-val REF_energy \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_wA/evaluation/train_curves.csv \
  --output-plot experiment_wA/evaluation/train_curves.png

CUDA_VISIBLE_DEVICES=6 python utils/reliability.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --results-dir experiment_wA/results \
  --experiment-name mace \
  --validation-split dataset/water/blyp/val.xyz \
  --test-split dataset/water/blyp/test.xyz \
  --energy-key-val REF_energy \
  --energy-key-test REF_energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --output-csv-raw experiment_wA/evaluation/reliability_test_cal_raw.csv \
  --output-csv-bins experiment_wA/evaluation/reliability_test_cal_bins.csv \
  --output-plot experiment_wA/evaluation/reliability_test_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=6 python utils/reliability.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --results-dir experiment_wA/results \
  --experiment-name mace \
  --validation-split dataset/water/blyp/val.xyz \
  --test-split dataset/water/blyp/test.xyz \
  --energy-key-val REF_energy \
  --energy-key-test REF_energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --output-csv-raw experiment_wA/evaluation/reliability_test_nocal_raw.csv \
  --output-csv-bins experiment_wA/evaluation/reliability_test_nocal_bins.csv \
  --output-plot experiment_wA/evaluation/reliability_test_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=6 python utils/epoch_raw.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/blyp/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 10 \
  --output-csv experiment_wA/evaluation/epoch_raw_train.csv \
  --output-plot experiment_wA/evaluation/epoch_raw_train.png

CUDA_VISIBLE_DEVICES=6 python utils/epoch_quality.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/blyp/test.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wA/evaluation/epoch_quality_test.csv \
  --output-plot experiment_wA/evaluation/epoch_quality_test.png

CUDA_VISIBLE_DEVICES=6 python utils/epoch_quality.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/blyp/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wA/evaluation/epoch_quality_train.csv \
  --output-plot experiment_wA/evaluation/epoch_quality_train.png

CUDA_VISIBLE_DEVICES=6 python utils/distribution.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --results-dir experiment_wA/results \
  --experiment-name mace \
  --split-path dataset/water/blyp/test.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_wA/evaluation/distribution_test.csv \
  --output-plot experiment_wA/evaluation/distribution_test.png

CUDA_VISIBLE_DEVICES=6 python utils/distribution.py \
  --checkpoints-dir experiment_wA/checkpoints \
  --results-dir experiment_wA/results \
  --experiment-name mace \
  --split-path dataset/water/blyp/train.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_wA/evaluation/distribution_train.csv \
  --output-plot experiment_wA/evaluation/distribution_train.png
