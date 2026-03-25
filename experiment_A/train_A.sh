#!/bin/bash


# Store PIDs of background processes
CUDA_VISIBLE_DEVICES=2 python mace/cli/run_train.py --config experiment_A/config_A.yml --seed 0 &
PID1=$!
CUDA_VISIBLE_DEVICES=3 python mace/cli/run_train.py --config experiment_A/config_A.yml --seed 1 &
PID2=$!
CUDA_VISIBLE_DEVICES=4 python mace/cli/run_train.py --config experiment_A/config_A.yml --seed 2 &
PID3=$!
CUDA_VISIBLE_DEVICES=5  python mace/cli/run_train.py --config experiment_A/config_A.yml --seed 3 &
PID4=$!
CUDA_VISIBLE_DEVICES=6 python mace/cli/run_train.py --config experiment_A/config_A.yml --seed 4 &
PID5=$!


# Kill both processes on Ctrl+C
trap "kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null" INT

wait