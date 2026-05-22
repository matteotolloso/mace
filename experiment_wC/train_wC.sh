#!/bin/bash

set -e

GPU_ID=${GPU_ID:-6}

for seed in {0..9}; do
  CUDA_VISIBLE_DEVICES=${GPU_ID} python mace/cli/run_train.py \
    --config experiment_wC/config_wC.yml \
    --seed "${seed}" \
    --wandb_name "mace_seed_${seed}"
done
