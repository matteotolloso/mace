#!/bin/bash


# Store PIDs of background processes
CUDA_VISIBLE_DEVICES=7 python mace/cli/run_train.py --config experiment_B/config_B.yml --foundation_model experiment_A/checkpoints/mace_run-0.model --seed 0 --wandb_name mace_seed_0 &
PID1=$!
sleep 180

CUDA_VISIBLE_DEVICES=7 python mace/cli/run_train.py --config experiment_B/config_B.yml --foundation_model experiment_A/checkpoints/mace_run-1.model --seed 1 --wandb_name mace_seed_1 &
PID2=$!
sleep 180

CUDA_VISIBLE_DEVICES=6 python mace/cli/run_train.py --config experiment_B/config_B.yml --foundation_model experiment_A/checkpoints/mace_run-2.model --seed 2 --wandb_name mace_seed_2 &
PID3=$!
sleep 180   

CUDA_VISIBLE_DEVICES=6  python mace/cli/run_train.py --config experiment_B/config_B.yml --foundation_model experiment_A/checkpoints/mace_run-3.model --seed 3 --wandb_name mace_seed_3 &
PID4=$!
sleep 180

CUDA_VISIBLE_DEVICES=5 python mace/cli/run_train.py --config experiment_B/config_B.yml --foundation_model experiment_A/checkpoints/mace_run-4.model --seed 4 --wandb_name mace_seed_4 &
PID5=$!


# Kill both processes on Ctrl+C
trap "kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null" INT

wait
