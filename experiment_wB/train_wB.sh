#!/bin/bash

set -e

GPU_ID=${GPU_ID:-7}

for seed in {0..9}; do
  CUDA_VISIBLE_DEVICES=${GPU_ID} python mace/cli/run_train.py \
    --config experiment_wB/config_wB.yml \
    --foundation_model "experiment_wA/checkpoints/mace_run-${seed}.model" \
    --seed "${seed}" \
    --wandb_name "mace_seed_${seed}"
done
