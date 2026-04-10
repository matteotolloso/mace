#!/bin/bash

# Plot AU and EU per epoch

python utils/epoch_raw.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split train \
  --clip_percentile 100 \
  --drop_first_k_epochs 0 \
  --output_csv outputs/plots/unc_vs_epoch_train-mv-sp.csv \
  --output_plot outputs/plots/unc_vs_epoch_train-mv-sp.png \
  --plot_log_variance

# python utils/epoch_raw.py \
#   --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
#   --split valid \
#   --clip_percentile 100 \
#   --drop_first_k_epochs 0 \
#   --output_csv outputs/plots/unc_vs_epoch_valid-mv-sp.csv \
#   --output_plot outputs/plots/unc_vs_epoch_valid-mv-sp.png \
#   --plot_log_variance

# python utils/epoch_raw.py \
#   --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
#   --split test \
#   --clip_percentile 100 \
#   --drop_first_k_epochs 0 \
#   --output_csv outputs/plots/unc_vs_epoch_test-mv-sp.csv \
#   --output_plot outputs/plots/unc_vs_epoch_test-mv-sp.png \
#   --plot_log_variance


# Plot AU for each member of the ensemble per epoch

python utils/plt_epoch_au_members_configs.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --split train \
  --output_csv outputs/plots/au_members_configs_train-mv-sp.csv \
  --output_plot outputs/plots/au_members_configs_train-mv-sp.png \
  --plot_log_variance

# python utils/plt_epoch_au_members_configs.py \
#   --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
#   --split valid \
#   --output_csv outputs/plots/au_members_configs_valid-mv-sp.csv \
#   --output_plot outputs/plots/au_members_configs_valid-mv-sp.png \
#   --plot_log_variance

# python utils/plt_epoch_au_members_configs.py \
#   --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
#   --split test \
#   --output_csv outputs/plots/au_members_configs_test-mv-sp.csv \
#   --output_plot outputs/plots/au_members_configs_test-mv-sp.png \
#   --plot_log_variance


# Plot reliability diagram 

python utils/reliability.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --selection_split valid \
  --plot_split test \
  --num_bins 40 \
  --isotonic_calibration \
  --output_csv_raw outputs/plots/unc_vs_error_raw.csv \
  --output_csv_bins outputs/plots/unc_vs_error_bins.csv \
  --output_plot outputs/plots/unc_vs_error.png


# Plot single config diagram

python utils/plt_member_pred_var_grid.py \
  --inputs outputs/results/mace-mv-sp_run-0_epoch_outputs.txt outputs/results/mace-mv-sp_run-1_epoch_outputs.txt outputs/results/mace-mv-sp_run-2_epoch_outputs.txt outputs/results/mace-mv-sp_run-3_epoch_outputs.txt outputs/results/mace-mv-sp_run-4_epoch_outputs.txt \
  --selection_split valid \
  --plot_split test \
  --output_csv outputs/plots/member_pred_var_grid.csv \
  --output_plot outputs/plots/member_pred_var_grid.png
