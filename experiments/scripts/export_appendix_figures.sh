#!/usr/bin/env bash
# Convert the five-split evaluation SVGs used in the ICLR27 appendix to PDF.
#
# Sources are the aggregates in experiment_X/evaluation/ (eval/aggregate_replicates.py);
# which ones may back a claim is listed in experiments/evaluation_figures.md. Energy-split
# OOD epoch-quality curves (C, D, F) are unfiltered and are deliberately not exported.
# Needs cairosvg (pip install cairosvg); pass its path as CAIROSVG if it is not on PATH.
#
# Usage (from the repository root):
#   CAIROSVG=/path/to/cairosvg bash experiments/scripts/export_appendix_figures.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$ROOT/ICLR27-UQ-MF/figures/appendix}"
CAIROSVG="${CAIROSVG:-cairosvg}"
mkdir -p "$OUT"

convert() {  # experiment, figure name
  "$CAIROSVG" "$ROOT/experiment_$1/evaluation/$2.svg" -o "$OUT/$1_$2.pdf"
}

for e in A B C D E F; do
  for t in id ood; do
    convert "$e" "reliability_${t}_nocal"
    convert "$e" "reliability_${t}_cal"
  done
  convert "$e" train_curves
done
for t in train id ood; do
  convert E "epoch_quality_${t}-log-scale"
  convert B "epoch_quality_finetune_mixed_${t}-log-scale"
done
for t in train id; do
  convert F "epoch_quality_${t}-log-scale"
  convert D "epoch_quality_finetune_mixed_${t}-log-scale"
done
ls "$OUT" | wc -l
