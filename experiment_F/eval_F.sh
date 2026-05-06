mkdir -p experiment_F/evaluation

# ensemble train/val loss curves

CUDA_VISIBLE_DEVICES=2 python utils/train_curves.py \
  --checkpoints-dir experiment_F/checkpoints \
  --experiment-name mace \
  --train-split dataset/ani1x_energy_split/cc_train.xyz \
  --validation-split dataset/ani1x_energy_split/cc_val.xyz \
  --energy-key-train "ccsd(t)_cbs.energy" \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_F/evaluation/train_curves.csv \
  --output-plot experiment_F/evaluation/train_curves.png

# reliability diagram ID

CUDA_VISIBLE_DEVICES=2 python utils/reliability.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/cc_val.xyz \
  --test-split dataset/ani1x_energy_split/cc_test_id.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --output-csv-raw experiment_F/evaluation/reliability_id_cal_raw.csv \
  --output-csv-bins experiment_F/evaluation/reliability_id_cal_bins.csv \
  --output-plot experiment_F/evaluation/reliability_id_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=2 python utils/reliability.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/cc_val.xyz \
  --test-split dataset/ani1x_energy_split/cc_test_id.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --output-csv-raw experiment_F/evaluation/reliability_id_nocal_raw.csv \
  --output-csv-bins experiment_F/evaluation/reliability_id_nocal_bins.csv \
  --output-plot experiment_F/evaluation/reliability_id_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# reliability diagram OOD

CUDA_VISIBLE_DEVICES=2 python utils/reliability.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/cc_val.xyz \
  --test-split dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration True \
  --output-csv-raw experiment_F/evaluation/reliability_ood_cal_raw.csv \
  --output-csv-bins experiment_F/evaluation/reliability_ood_cal_bins.csv \
  --output-plot experiment_F/evaluation/reliability_ood_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

CUDA_VISIBLE_DEVICES=2 python utils/reliability.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --validation-split dataset/ani1x_energy_split/cc_val.xyz \
  --test-split dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --num-bins 15 \
  --trim 0.005 \
  --isotonic-calibration False \
  --output-csv-raw experiment_F/evaluation/reliability_ood_nocal_raw.csv \
  --output-csv-bins experiment_F/evaluation/reliability_ood_nocal_bins.csv \
  --output-plot experiment_F/evaluation/reliability_ood_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# train AU/EU diagram from checkpoints

CUDA_VISIBLE_DEVICES=2 python utils/epoch_raw.py \
  --checkpoints-dir experiment_F/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 10 \
  --output-csv experiment_F/evaluation/epoch_raw_train.csv \
  --output-plot experiment_F/evaluation/epoch_raw_train.png

# train diagram metrics

CUDA_VISIBLE_DEVICES=2 python utils/epoch_quality.py \
  --checkpoints-dir experiment_F/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_id.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_F/evaluation/epoch_quality_id.csv \
  --output-plot experiment_F/evaluation/epoch_quality_id.png

CUDA_VISIBLE_DEVICES=2 python utils/epoch_quality.py \
  --checkpoints-dir experiment_F/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_F/evaluation/epoch_quality_ood.csv \
  --output-plot experiment_F/evaluation/epoch_quality_ood.png

CUDA_VISIBLE_DEVICES=2 python utils/epoch_quality.py \
  --checkpoints-dir experiment_F/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --free-scale \
  --output-csv experiment_F/evaluation/epoch_quality_train.csv \
  --output-plot experiment_F/evaluation/epoch_quality_train.png

# distributions

CUDA_VISIBLE_DEVICES=2 python utils/distribution.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_id.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_F/evaluation/distribution_id.csv \
  --output-plot experiment_F/evaluation/distribution_id.png

CUDA_VISIBLE_DEVICES=2 python utils/distribution.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_F/evaluation/distribution_ood.csv \
  --output-plot experiment_F/evaluation/distribution_ood.png

CUDA_VISIBLE_DEVICES=2 python utils/distribution.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_F/evaluation/distribution_train.csv \
  --output-plot experiment_F/evaluation/distribution_train.png


# uncertainty vs energy for energy-OOD test sets

CUDA_VISIBLE_DEVICES=4 python utils/energy_ood.py \
  --checkpoints-dir experiment_F/checkpoints \
  --results-dir experiment_F/results \
  --experiment-name mace \
  --test-id-split dataset/ani1x_energy_split/cc_test_id.xyz \
  --test-ood-split dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --device cuda \
  --batch-size 256 \
  --output-csv-raw experiment_F/evaluation/energy_ood_raw.csv \
  --output-csv-bins experiment_F/evaluation/energy_ood_bins.csv \
  --output-plot experiment_F/evaluation/energy_ood.png
