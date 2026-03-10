#!/bin/bash

python utils/plt_epoch_au_eu.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split train \
  --clip_percentile 80 \
  --drop_first_k_epochs 0 \
  --output_csv outputs/plots/unc_vs_epoch_train-mv-sp.csv \
  --output_plot outputs/plots/unc_vs_epoch_train-mv-sp.png \
  --per_atom \
  --plot_log_variance

python utils/plt_epoch_au_eu.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split valid \
  --clip_percentile 80 \
  --drop_first_k_epochs 0 \
  --output_csv outputs/plots/unc_vs_epoch_valid-mv-sp.csv \
  --output_plot outputs/plots/unc_vs_epoch_valid-mv-sp.png \
  --per_atom \
  --plot_log_variance

python utils/plt_epoch_au_eu.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split test \
  --clip_percentile 80 \
  --drop_first_k_epochs 0 \
  --output_csv outputs/plots/unc_vs_epoch_test-mv-sp.csv \
  --output_plot outputs/plots/unc_vs_epoch_test-mv-sp.png \
  --per_atom \
  --plot_log_variance


# python utils/analyze_au_spikes.py \
#   --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
#   --split test \
#   --loader Default_Default \
#   --per_atom \
#   --top_k 20 \
#   --summary_csv outputs/plots/au_spike_summary_test.csv \
#   --details_csv outputs/plots/au_spike_top_configs_test.csv

python utils/plt_epoch_au_members_configs.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split train \
  --per_atom \
  --output_csv outputs/plots/au_members_configs_train-mv-sp.csv \
  --output_plot outputs/plots/au_members_configs_train-mv-sp.png \
  --plot_log_variance

python utils/plt_epoch_au_members_configs.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split valid \
  --per_atom \
  --output_csv outputs/plots/au_members_configs_valid-mv-sp.csv \
  --output_plot outputs/plots/au_members_configs_valid-mv-sp.png \
  --plot_log_variance

python utils/plt_epoch_au_members_configs.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split test \
  --per_atom \
  --output_csv outputs/plots/au_members_configs_test-mv-sp.csv \
  --output_plot outputs/plots/au_members_configs_test-mv-sp.png \
  --plot_log_variance
