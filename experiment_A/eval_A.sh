

CUDA_VISIBLE_DEVICES=1 python utils/plt_unc_vs_error.py \
  --checkpoints-dir experiment_A/checkpoints \
  --experiment-name mace \
  --validation-split dataset/ani1x_system_split/dft_val.xyz \
  --test-split dataset/ani1x_system_split/dft_test_id.xyz \
  --energy-key-val wb97x_tz.energy \
  --energy-key-test wb97x_tz.energy \
  --num-bins 50 \
  --isotonic-calibration \
  --output-csv-raw experiment_A/results/unc_vs_error_raw.csv \
  --output-csv-bins experiment_A/results/unc_vs_error_bins.csv \
  --output-plot experiment_A/results/unc_vs_error.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

python utils/plt_epoch_au_eu.py \
  --inputs experiment_A/results/mace_run-0_epoch_outputs.txt experiment_A/results/mace_run-1_epoch_outputs.txt experiment_A/results/mace_run-2_epoch_outputs.txt experiment_A/results/mace_run-3_epoch_outputs.txt experiment_A/results/mace_run-4_epoch_outputs.txt \
  --split valid \
  --clip_percentile 100 \
  --drop_first_k_epochs 0 \
  --output_csv experiment_A/results/unc_vs_epoch_valid-mv-sp.csv \
  --output_plot experiment_A/results/unc_vs_epoch_valid-mv-sp.png \
  --plot_log_variance