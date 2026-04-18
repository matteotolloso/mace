mkdir -p experiment_D/evaluation

# ensemble train/val loss curves

CUDA_VISIBLE_DEVICES=4 python utils/train_curves.py \
  --checkpoints-dir experiment_D/checkpoints \
  --experiment-name mace \
  --train-split dataset/ani1x_energy_split/cc_train.xyz \
  --validation-split dataset/ani1x_energy_split/cc_val.xyz \
  --energy-key-train "ccsd(t)_cbs.energy" \
  --energy-key-val "ccsd(t)_cbs.energy" \
  --every-n-epochs 5 \
  --device cuda \
  --output-csv experiment_D/evaluation/train_curves.csv \
  --output-plot experiment_D/evaluation/train_curves.png

# reliability digram ID

# rmse + calibration
CUDA_VISIBLE_DEVICES=4 python utils/reliability.py \
  --checkpoints-dir experiment_D/checkpoints \
  --results-dir experiment_D/results \
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
  --output-csv-raw experiment_D/evaluation/reliability_id_cal_raw.csv \
  --output-csv-bins experiment_D/evaluation/reliability_id_cal_bins.csv \
  --output-plot experiment_D/evaluation/reliability_id_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# rmse + no calibration
CUDA_VISIBLE_DEVICES=4 python utils/reliability.py \
  --checkpoints-dir experiment_D/checkpoints \
  --results-dir experiment_D/results \
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
  --output-csv-raw experiment_D/evaluation/reliability_id_nocal_raw.csv \
  --output-csv-bins experiment_D/evaluation/reliability_id_nocal_bins.csv \
  --output-plot experiment_D/evaluation/reliability_id_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256


# Reliability diagram OOD

# rmse + calibration
CUDA_VISIBLE_DEVICES=4 python utils/reliability.py \
  --checkpoints-dir experiment_D/checkpoints \
  --results-dir experiment_D/results \
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
  --output-csv-raw experiment_D/evaluation/reliability_ood_cal_raw.csv \
  --output-csv-bins experiment_D/evaluation/reliability_ood_cal_bins.csv \
  --output-plot experiment_D/evaluation/reliability_ood_cal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# rmse + no calibration
CUDA_VISIBLE_DEVICES=4 python utils/reliability.py \
  --checkpoints-dir experiment_D/checkpoints \
  --results-dir experiment_D/results \
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
  --output-csv-raw experiment_D/evaluation/reliability_ood_nocal_raw.csv \
  --output-csv-bins experiment_D/evaluation/reliability_ood_nocal_bins.csv \
  --output-plot experiment_D/evaluation/reliability_ood_nocal.png \
  --log-level INFO \
  --log-every-batches 1 \
  --device cuda \
  --batch-size 256

# -------------------------

# train AU/EU diagram from checkpoints

CUDA_VISIBLE_DEVICES=4 python utils/epoch_raw.py \
  --checkpoints-dir experiment_D/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 10 \
  --output-csv experiment_D/evaluation/epoch_raw_train.csv \
  --output-plot experiment_D/evaluation/epoch_raw_train.png


# train diagram metrics

# ID
CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality.py \
  --checkpoints-dir experiment_D/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_id.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --output-csv experiment_D/evaluation/epoch_quality_id.csv \
  --output-plot experiment_D/evaluation/epoch_quality_id.png

# OOD
CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality.py \
  --checkpoints-dir experiment_D/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --output-csv experiment_D/evaluation/epoch_quality_ood.csv \
  --output-plot experiment_D/evaluation/epoch_quality_ood.png

# on train set
CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality.py \
  --checkpoints-dir experiment_D/checkpoints \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs 5 \
  --output-csv experiment_D/evaluation/epoch_quality_train.csv \
  --output-plot experiment_D/evaluation/epoch_quality_train.png


# concatenated pretrain + finetune epoch-quality on the same target dataset

CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality_finetune_same_dataset.py \
  --pretrain-checkpoints-dir experiment_C/checkpoints \
  --pretrain-experiment-name mace \
  --finetune-checkpoints-dir experiment_D/checkpoints \
  --finetune-experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_id.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --output-csv experiment_D/evaluation/epoch_quality_finetune_same_id.csv \
  --output-plot experiment_D/evaluation/epoch_quality_finetune_same_id.png \
  --title "Epoch Quality ID: Same Dataset Pretrain + Finetune"

CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality_finetune_same_dataset.py \
  --pretrain-checkpoints-dir experiment_C/checkpoints \
  --pretrain-experiment-name mace \
  --finetune-checkpoints-dir experiment_D/checkpoints \
  --finetune-experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --output-csv experiment_D/evaluation/epoch_quality_finetune_same_ood.csv \
  --output-plot experiment_D/evaluation/epoch_quality_finetune_same_ood.png \
  --title "Epoch Quality OOD: Same Dataset Pretrain + Finetune"

CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality_finetune_same_dataset.py \
  --pretrain-checkpoints-dir experiment_C/checkpoints \
  --pretrain-experiment-name mace \
  --finetune-checkpoints-dir experiment_D/checkpoints \
  --finetune-experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --output-csv experiment_D/evaluation/epoch_quality_finetune_same_train.csv \
  --output-plot experiment_D/evaluation/epoch_quality_finetune_same_train.png \
  --title "Epoch Quality Train: Same Dataset Pretrain + Finetune"


# concatenated pretrain + finetune epoch-quality across DFT -> CC datasets

CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality_finetune_mixed_dataset.py \
  --pretrain-checkpoints-dir experiment_C/checkpoints \
  --pretrain-experiment-name mace \
  --pretrain-split-path dataset/ani1x_energy_split/dft_test_id.xyz \
  --pretrain-energy-key wb97x_tz.energy \
  --finetune-checkpoints-dir experiment_D/checkpoints \
  --finetune-experiment-name mace \
  --finetune-split-path dataset/ani1x_energy_split/cc_test_id.xyz \
  --finetune-energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --output-csv experiment_D/evaluation/epoch_quality_finetune_mixed_id.csv \
  --output-plot experiment_D/evaluation/epoch_quality_finetune_mixed_id.png \
  --title "Epoch Quality ID: DFT Pretrain + CC Finetune"

CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality_finetune_mixed_dataset.py \
  --pretrain-checkpoints-dir experiment_C/checkpoints \
  --pretrain-experiment-name mace \
  --pretrain-split-path dataset/ani1x_energy_split/dft_test_ood.xyz \
  --pretrain-energy-key wb97x_tz.energy \
  --finetune-checkpoints-dir experiment_D/checkpoints \
  --finetune-experiment-name mace \
  --finetune-split-path dataset/ani1x_energy_split/cc_test_ood.xyz \
  --finetune-energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --output-csv experiment_D/evaluation/epoch_quality_finetune_mixed_ood.csv \
  --output-plot experiment_D/evaluation/epoch_quality_finetune_mixed_ood.png \
  --title "Epoch Quality OOD: DFT Pretrain + CC Finetune"

CUDA_VISIBLE_DEVICES=4 python utils/epoch_quality_finetune_mixed_dataset.py \
  --pretrain-checkpoints-dir experiment_C/checkpoints \
  --pretrain-experiment-name mace \
  --pretrain-split-path dataset/ani1x_energy_split/dft_train.xyz \
  --pretrain-energy-key wb97x_tz.energy \
  --finetune-checkpoints-dir experiment_D/checkpoints \
  --finetune-experiment-name mace \
  --finetune-split-path dataset/ani1x_energy_split/cc_train.xyz \
  --finetune-energy-key "ccsd(t)_cbs.energy" \
  --device cuda \
  --batch-size 256 \
  --every-n-epochs-pretrain 5 \
  --every-n-epochs-finetune 5 \
  --output-csv experiment_D/evaluation/epoch_quality_finetune_mixed_train.csv \
  --output-plot experiment_D/evaluation/epoch_quality_finetune_mixed_train.png \
  --title "Epoch Quality Train: DFT Pretrain + CC Finetune"


# pretrain vs finetune AU/EU summary
CUDA_VISIBLE_DEVICES=4 python utils/finetune.py \
  --pretrain-checkpoints-dir experiment_C/checkpoints \
  --pretrain-results-dir experiment_C/results \
  --pretrain-experiment-name mace \
  --finetune-checkpoints-dir experiment_D/checkpoints \
  --finetune-results-dir experiment_D/results \
  --finetune-experiment-name mace \
  --test-id-split dataset/ani1x_energy_split/cc_test_id.xyz \
  --test-ood-split dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key-test "ccsd(t)_cbs.energy" \
  --selection-key-pretrain loss \
  --selection-key-finetune loss \
  --selection-mode-pretrain min \
  --selection-mode-finetune min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_D/evaluation/finetune.csv \
  --output-plot experiment_D/evaluation/finetune.png


# distributions

CUDA_VISIBLE_DEVICES=4 python utils/distribution.py \
  --checkpoints-dir experiment_D/checkpoints \
  --results-dir experiment_D/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_id.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_D/evaluation/distribution_id.csv \
  --output-plot experiment_D/evaluation/distribution_id.png

CUDA_VISIBLE_DEVICES=4 python utils/distribution.py \
  --checkpoints-dir experiment_D/checkpoints \
  --results-dir experiment_D/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_test_ood.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_D/evaluation/distribution_ood.csv \
  --output-plot experiment_D/evaluation/distribution_ood.png

CUDA_VISIBLE_DEVICES=4 python utils/distribution.py \
  --checkpoints-dir experiment_D/checkpoints \
  --results-dir experiment_D/results \
  --experiment-name mace \
  --split-path dataset/ani1x_energy_split/cc_train.xyz \
  --energy-key "ccsd(t)_cbs.energy" \
  --selection-key loss \
  --selection-mode min \
  --trim 0.005 \
  --device cuda \
  --batch-size 256 \
  --output-csv experiment_D/evaluation/distribution_train.csv \
  --output-plot experiment_D/evaluation/distribution_train.png
