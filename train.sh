#!/bin/bash

# rm -rf outputs/*

# Store PIDs of background processes
CUDA_VISIBLE_DEVICES=2 python mace/cli/run_train.py --config config/config-mv.yml --seed 0 &
PID1=$!
CUDA_VISIBLE_DEVICES=3 python mace/cli/run_train.py --config config/config-mv.yml --seed 1 &
PID2=$!
CUDA_VISIBLE_DEVICES=4 python mace/cli/run_train.py --config config/config-mv.yml --seed 2 &
PID3=$!
CUDA_VISIBLE_DEVICES=5  python mace/cli/run_train.py --config config/config-mv.yml --seed 3 &
PID4=$!
CUDA_VISIBLE_DEVICES=6 python mace/cli/run_train.py --config config/config-mv.yml --seed 4 &
PID5=$!
# CUDA_VISIBLE_DEVICES=7 python mace/cli/run_train.py --config config/config-default.yml --seed 0 &
# PID6=$!

# Kill both processes on Ctrl+C
trap "kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null" INT

wait