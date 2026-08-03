#!/bin/bash

mkdir -p experiment_wB/evaluation

CUDA_VISIBLE_DEVICES=7 python utils/train_curves.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --experiment-name mace \
  --train-split dataset/water/ccsdt/train.xyz \
  --validation-split dataset/water/ccsdt/val.xyz \
  --energy-key-train REF_energy \
  --energy-key-val REF_energy \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_wB/evaluation/train_curves.csv \
  --output-plot experiment_wB/evaluation/train_curves.png \
  --batch-size 16

CUDA_VISIBLE_DEVICES=7 python utils/reliability.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --results-dir experiment_wB/results \
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
  --output-csv-raw experiment_wB/evaluation/reliability_test_cal_raw.csv \
  --output-csv-bins experiment_wB/evaluation/reliability_test_cal_bins.csv \
  --output-plot experiment_wB/evaluation/reliability_test_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 16

CUDA_VISIBLE_DEVICES=7 python utils/reliability.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --results-dir experiment_wB/results \
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
  --output-csv-raw experiment_wB/evaluation/reliability_test_nocal_raw.csv \
  --output-csv-bins experiment_wB/evaluation/reliability_test_nocal_bins.csv \
  --output-plot experiment_wB/evaluation/reliability_test_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 16

CUDA_VISIBLE_DEVICES=7 python utils/epoch_raw.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 16 \
  --every-n-epochs 10 \
  --output-csv experiment_wB/evaluation/epoch_raw_train.csv \
  --output-plot experiment_wB/evaluation/epoch_raw_train.png

CUDA_VISIBLE_DEVICES=7 python utils/epoch_quality.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/test.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 16 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wB/evaluation/epoch_quality_test.csv \
  --output-plot experiment_wB/evaluation/epoch_quality_test.png

CUDA_VISIBLE_DEVICES=7 python utils/epoch_quality.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 16 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_wB/evaluation/epoch_quality_train.csv \
  --output-plot experiment_wB/evaluation/epoch_quality_train.png

CUDA_VISIBLE_DEVICES=7 python utils/epoch_quality_finetune_same_dataset.py \
  --pretrain-checkpoints-dir experiment_wA/checkpoints \
  --pretrain-experiment-name mace \
  --finetune-checkpoints-dir experiment_wB/checkpoints \
  --finetune-experiment-name mace \
  --split-path dataset/water/ccsdt/test.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 16 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --free-scale \
  --output-csv experiment_wB/evaluation/epoch_quality_finetune_same_test.csv \
  --output-plot experiment_wB/evaluation/epoch_quality_finetune_same_test.png \
  --title "Epoch Quality Test: BLYP Pretrain + CCSDT Finetune"

CUDA_VISIBLE_DEVICES=7 python utils/epoch_quality_finetune_same_dataset.py \
  --pretrain-checkpoints-dir experiment_wA/checkpoints \
  --pretrain-experiment-name mace \
  --finetune-checkpoints-dir experiment_wB/checkpoints \
  --finetune-experiment-name mace \
  --split-path dataset/water/ccsdt/train.xyz \
  --energy-key REF_energy \
  --device cuda \
  --batch-size 16 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --free-scale \
  --output-csv experiment_wB/evaluation/epoch_quality_finetune_same_train.csv \
  --output-plot experiment_wB/evaluation/epoch_quality_finetune_same_train.png \
  --title "Epoch Quality Train: BLYP Pretrain + CCSDT Finetune"

CUDA_VISIBLE_DEVICES=7 python utils/epoch_quality_finetune_mixed_dataset.py \
  --pretrain-checkpoints-dir experiment_wA/checkpoints \
  --pretrain-experiment-name mace \
  --pretrain-split-path dataset/water/blyp/test.xyz \
  --pretrain-energy-key REF_energy \
  --finetune-checkpoints-dir experiment_wB/checkpoints \
  --finetune-experiment-name mace \
  --finetune-split-path dataset/water/ccsdt/test.xyz \
  --finetune-energy-key REF_energy \
  --device cuda \
  --batch-size 16 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --free-scale \
  --output-csv experiment_wB/evaluation/epoch_quality_finetune_mixed_test.csv \
  --output-plot experiment_wB/evaluation/epoch_quality_finetune_mixed_test.png \
  --title "Epoch Quality Test: BLYP Pretrain + CCSDT Finetune"

CUDA_VISIBLE_DEVICES=7 python utils/epoch_quality_finetune_mixed_dataset.py \
  --pretrain-checkpoints-dir experiment_wA/checkpoints \
  --pretrain-experiment-name mace \
  --pretrain-split-path dataset/water/blyp/train.xyz \
  --pretrain-energy-key REF_energy \
  --finetune-checkpoints-dir experiment_wB/checkpoints \
  --finetune-experiment-name mace \
  --finetune-split-path dataset/water/ccsdt/train.xyz \
  --finetune-energy-key REF_energy \
  --device cuda \
  --batch-size 16 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --free-scale \
  --output-csv experiment_wB/evaluation/epoch_quality_finetune_mixed_train.csv \
  --output-plot experiment_wB/evaluation/epoch_quality_finetune_mixed_train.png \
  --title "Epoch Quality Train: BLYP Pretrain + CCSDT Finetune"

CUDA_VISIBLE_DEVICES=7 python utils/distribution.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --results-dir experiment_wB/results \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/test.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 16 \
  --output-csv experiment_wB/evaluation/distribution_test.csv \
  --output-plot experiment_wB/evaluation/distribution_test.png

CUDA_VISIBLE_DEVICES=7 python utils/distribution.py \
  --checkpoints-dir experiment_wB/checkpoints \
  --results-dir experiment_wB/results \
  --experiment-name mace \
  --split-path dataset/water/ccsdt/train.xyz \
  --energy-key REF_energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 16 \
  --output-csv experiment_wB/evaluation/distribution_train.csv \
  --output-plot experiment_wB/evaluation/distribution_train.png
