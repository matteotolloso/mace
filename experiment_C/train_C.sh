#!/bin/bash


# # Store PIDs of background processes
# CUDA_VISIBLE_DEVICES=5 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 0 --wandb_name mace_seed_0 &
# PID1=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=4 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 1 --wandb_name mace_seed_1 &
# PID2=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=4 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 2 --wandb_name mace_seed_2 &
# PID3=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=3  python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 3 --wandb_name mace_seed_3 &
# PID4=$!
# sleep 300

# CUDA_VISIBLE_DEVICES=3 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 4 --wandb_name mace_seed_4 &
# PID5=$!


# # Kill both processes on Ctrl+C
# trap "kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null" INT

# wait


# -------------------------

# Store PIDs of background processes
CUDA_VISIBLE_DEVICES=5 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 5 --wandb_name mace_seed_5 &
PID1=$!
sleep 300

CUDA_VISIBLE_DEVICES=4 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 6 --wandb_name mace_seed_6 &
PID2=$!
sleep 300

CUDA_VISIBLE_DEVICES=4 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 7 --wandb_name mace_seed_7 &
PID3=$!
sleep 300

CUDA_VISIBLE_DEVICES=3  python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 8 --wandb_name mace_seed_8 &
PID4=$!
sleep 300

CUDA_VISIBLE_DEVICES=3 python mace/cli/run_train.py --config experiment_C/config_C.yml --seed 9 --wandb_name mace_seed_9 &
PID5=$!


# Kill both processes on Ctrl+C
trap "kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null" INT

wait