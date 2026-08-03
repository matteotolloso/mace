#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS=(A B C D E F wA wB wC)
SPLITS=(0 1 2 3 4)
MEMBERS=(0 1 2 3 4 5 6 7 8 9)

total_runs=0
complete_runs=0
partial_runs=0
missing_runs=0
complete_models=0
incomplete_models=()

printf 'Checking experiment outputs in %s\n\n' "$ROOT_DIR"

for experiment in "${EXPERIMENTS[@]}"; do
    experiment_dir="$ROOT_DIR/experiment_${experiment}"

    if [[ ! -d "$experiment_dir" ]]; then
        printf 'experiment_%s: MISSING DIRECTORY\n\n' "$experiment"
        missing_runs=$((missing_runs + ${#SPLITS[@]} * ${#MEMBERS[@]}))
        total_runs=$((total_runs + ${#SPLITS[@]} * ${#MEMBERS[@]}))
        for split in "${SPLITS[@]}"; do
            incomplete_models+=("${experiment}_${split}")
        done
        continue
    fi

    for split in "${SPLITS[@]}"; do
        checkpoint_dir="$experiment_dir/checkpoints_${split}"
        results_dir="$experiment_dir/results_${split}"
        split_complete=0
        partial_members=()
        missing_members=()

        for member in "${MEMBERS[@]}"; do
            total_runs=$((total_runs + 1))
            final_model="$checkpoint_dir/mace_run-${member}.model"
            train_result="$results_dir/mace_run-${member}_train.txt"

            if [[ -f "$final_model" && -f "$train_result" ]]; then
                split_complete=$((split_complete + 1))
                complete_runs=$((complete_runs + 1))
            elif [[ -f "$final_model" || -f "$train_result" ]] \
                || compgen -G "$checkpoint_dir/mace_run-${member}_epoch-*.pt" >/dev/null; then
                partial_members+=("$member")
                partial_runs=$((partial_runs + 1))
            else
                missing_members+=("$member")
                missing_runs=$((missing_runs + 1))
            fi
        done

        if (( split_complete == ${#MEMBERS[@]} )); then
            printf 'experiment_%s split_%s: COMPLETE (10/10)\n' "$experiment" "$split"
            complete_models=$((complete_models + 1))
        else
            printf 'experiment_%s split_%s: INCOMPLETE (%d/10 complete)' \
                "$experiment" "$split" "$split_complete"
            if (( ${#partial_members[@]} > 0 )); then
                printf ' | partial: %s' "${partial_members[*]}"
            fi
            if (( ${#missing_members[@]} > 0 )); then
                printf ' | missing: %s' "${missing_members[*]}"
            fi
            printf '\n'
            incomplete_models+=("${experiment}_${split}")
        fi
    done
    printf '\n'
done

total_models=$((${#EXPERIMENTS[@]} * ${#SPLITS[@]}))

printf 'Summary\n'
printf '  Complete member runs: %d/%d\n' "$complete_runs" "$total_runs"
printf '  Partial member runs:  %d\n' "$partial_runs"
printf '  Missing member runs:  %d\n' "$missing_runs"
printf '  Complete MVE models:   %d/%d\n' "$complete_models" "$total_models"

if (( ${#incomplete_models[@]} > 0 )); then
    printf '  Incomplete MVE models: %s\n' "${incomplete_models[*]}"
    exit 1
fi

printf '  All experiments are complete.\n'
