#!/bin/bash

rm -rf outputs/*

# Store PIDs of background processes
CUDA_VISIBLE_DEVICES=4 python mace/cli/run_train.py --config config/config-mv.yml &
PID1=$!
CUDA_VISIBLE_DEVICES=5 python mace/cli/run_train.py --config config/config-default.yml &
PID2=$!

# Kill both processes on Ctrl+C
trap "kill $PID1 $PID2 2>/dev/null" INT

wait