
mkdir -p experiment_C/evaluation

# ensemble train/val loss curves

CUDA_VISIBLE_DEVICES=5 python utils/train_curves.py \
  --checkpoints-dir experiment_C/checkpoints \
  --experiment-name mace \
  --train-split dataset/ani1x_energy_split/dft_train.xyz \
  --validation-split dataset/ani1x_energy_split/dft_val.xyz \
  --energy-key-train wb97x_tz.energy \
  --energy-key-val wb97x_tz.energy \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_C/evaluation/train_curves.csv \
  --output-plot experiment_C/evaluation/train_curves.png

# reliability digram ID

# rmse + calibration
CUDA_VISIBLE_DEVICES=5 python utils/reliability.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/dft_val.xyz \
  --test-split dataset/ani1x_energy_split/dft_test_id.xyz \
  --energy-key-val wb97x_tz.energy \
  --energy-key-test wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --output-csv-raw experiment_C/evaluation/reliability_id_cal_raw.csv \
  --output-csv-bins experiment_C/evaluation/reliability_id_cal_bins.csv \
  --output-plot experiment_C/evaluation/reliability_id_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# rmse + no calibration
  CUDA_VISIBLE_DEVICES=5 python utils/reliability.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/dft_val.xyz \
  --test-split dataset/ani1x_energy_split/dft_test_id.xyz \
  --energy-key-val wb97x_tz.energy \
  --energy-key-test wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --output-csv-raw experiment_C/evaluation/reliability_id_nocal_raw.csv \
  --output-csv-bins experiment_C/evaluation/reliability_id_nocal_bins.csv \
  --output-plot experiment_C/evaluation/reliability_id_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256


# Reliability diagram OOD

# rmse + calibration
CUDA_VISIBLE_DEVICES=5 python utils/reliability.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/dft_val.xyz \
  --test-split dataset/ani1x_energy_split/dft_test_ood.xyz \
  --energy-key-val wb97x_tz.energy \
  --energy-key-test wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --output-csv-raw experiment_C/evaluation/reliability_ood_cal_raw.csv \
  --output-csv-bins experiment_C/evaluation/reliability_ood_cal_bins.csv \
  --output-plot experiment_C/evaluation/reliability_ood_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# rmse + no calibration
  CUDA_VISIBLE_DEVICES=5 python utils/reliability.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/dft_val.xyz \
  --test-split dataset/ani1x_energy_split/dft_test_ood.xyz \
  --energy-key-val wb97x_tz.energy \
  --energy-key-test wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --output-csv-raw experiment_C/evaluation/reliability_ood_nocal_raw.csv \
  --output-csv-bins experiment_C/evaluation/reliability_ood_nocal_bins.csv \
  --output-plot experiment_C/evaluation/reliability_ood_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# -------------------------

# train AU/EU diagram from checkpoints


  CUDA_VISIBLE_DEVICES=5 python utils/epoch_raw.py \
  --checkpoints-dir experiment_C/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/dft_train.xyz \
  --energy-key wb97x_tz.energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 10 \
  --output-csv experiment_C/evaluation/epoch_raw_train.csv \
  --output-plot experiment_C/evaluation/epoch_raw_train.png


# train diagram metrics 

# ID

CUDA_VISIBLE_DEVICES=5 python utils/epoch_quality.py \
  --checkpoints-dir experiment_C/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/dft_test_id.xyz \
  --energy-key wb97x_tz.energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_C/evaluation/epoch_quality_id.csv \
  --output-plot experiment_C/evaluation/epoch_quality_id.png


  # OOD

  CUDA_VISIBLE_DEVICES=5 python utils/epoch_quality.py \
  --checkpoints-dir experiment_C/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/dft_test_ood.xyz \
  --energy-key wb97x_tz.energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_C/evaluation/epoch_quality_ood.csv \
  --output-plot experiment_C/evaluation/epoch_quality_ood.png

  # on train set

  CUDA_VISIBLE_DEVICES=5 python utils/epoch_quality.py \
  --checkpoints-dir experiment_C/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/dft_train.xyz \
  --energy-key wb97x_tz.energy \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_C/evaluation/epoch_quality_train.csv \
  --output-plot experiment_C/evaluation/epoch_quality_train.png 


# distributions

CUDA_VISIBLE_DEVICES=5 python utils/distribution.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/dft_test_id.xyz \
  --energy-key wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_C/evaluation/distribution_id.csv \
  --output-plot experiment_C/evaluation/distribution_id.png

CUDA_VISIBLE_DEVICES=5 python utils/distribution.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/dft_test_ood.xyz \
  --energy-key wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_C/evaluation/distribution_ood.csv \
  --output-plot experiment_C/evaluation/distribution_ood.png

CUDA_VISIBLE_DEVICES=5 python utils/distribution.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/dft_train.xyz \
  --energy-key wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_C/evaluation/distribution_train.csv \
  --output-plot experiment_C/evaluation/distribution_train.png


# uncertainty vs energy for energy-OOD test sets

CUDA_VISIBLE_DEVICES=5 python utils/energy_ood.py \
  --checkpoints-dir experiment_C/checkpoints \
  --results-dir experiment_C/results \
  --experiment-name mace \
  --test-id-split dataset/ani1x_energy_split/dft_test_id.xyz \
  --test-ood-split dataset/ani1x_energy_split/dft_test_ood.xyz \
  --energy-key-test wb97x_tz.energy \
  --selection-key loss \
  --selection-mode min \
  --device cuda \
  --batch-size 256 \
  --output-csv-raw experiment_C/evaluation/energy_ood_raw.csv \
  --output-csv-bins experiment_C/evaluation/energy_ood_bins.csv \
  --output-plot experiment_C/evaluation/energy_ood.png
