#!/usr/bin/env bash

# Sourced by experiment eval scripts to aggregate their five split caches.

aggregate_split_caches() {
    local experiment_dir=$1
    shift
    command "${EVAL_PYTHON:-python}" eval/aggregate_replicates.py \
        --cache-dir "${experiment_dir}/evaluation/cache" \
        --output-dir "${experiment_dir}/evaluation" \
        --split-seeds ${EVAL_SPLIT_SEEDS} \
        "$@"
}
