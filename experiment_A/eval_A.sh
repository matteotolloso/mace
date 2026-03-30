

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
