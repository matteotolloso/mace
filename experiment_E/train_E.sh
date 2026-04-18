#!/bin/bash


# Store PIDs of background processes
# CUDA_VISIBLE_DEVICES=0 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 0 --wandb_name mace_seed_0 &
# PID1=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=0 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 1 --wandb_name mace_seed_1 &
# PID2=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=1 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 2 --wandb_name mace_seed_2 &
# PID3=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=1 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 3 --wandb_name mace_seed_3 &
# PID4=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=2 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 4 --wandb_name mace_seed_4 &
# PID5=$!


# trap "kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null" INT

# wait


# ----------

CUDA_VISIBLE_DEVICES=7 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 5 --wandb_name mace_seed_5 &
PID1=$!
sleep 60

CUDA_VISIBLE_DEVICES=7 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 6 --wandb_name mace_seed_6 &
PID2=$!
sleep 60

CUDA_VISIBLE_DEVICES=6 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 7 --wandb_name mace_seed_7 &
PID3=$!
sleep 60

CUDA_VISIBLE_DEVICES=6 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 8 --wandb_name mace_seed_8 &
PID4=$!
sleep 60

CUDA_VISIBLE_DEVICES=5 python mace/cli/run_train.py --config experiment_E/config_E.yml --seed 9 --wandb_name mace_seed_9 &
PID5=$!


trap "kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null" INT

wait
