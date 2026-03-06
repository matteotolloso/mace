#!/bin/bash

python utils/eval_ae.py \
  --inputs outputs/results/mace-mv_run-0_epoch_outputs.txt outputs/results/mace-mv_run-1_epoch_outputs.txt outputs/results/mace-mv_run-2_epoch_outputs.txt outputs/results/mace-mv_run-3_epoch_outputs.txt outputs/results/mace-mv_run-4_epoch_outputs.txt \
  --split train \
  --clip_percentile 90 \
  --drop_first_k_epochs 0 \
  --output_csv outputs/results/ae_vs_epoch.csv \
  --output_plot outputs/results/ae_vs_epoch.png \
  --per_atom \
  --plot_log_variance
