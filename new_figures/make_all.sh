#!/usr/bin/env bash
# Regenerate every figure from the caches (CPU only). See README.md for the one-off
# geometry cache and the GPU per-epoch predictions that Figs. 3 and 5 need.
set -euo pipefail
cd "$(dirname "$0")"
main_text="fig2_final_summary fig3_hf_dynamics fig4_rank_vs_calibration fig5_decomposition fig6_acquisition"
appendix="figA_recalibration figA_eu_au_ratio figA_acquisition_id"
for fig in $main_text $appendix; do
  PYTHONDONTWRITEBYTECODE=1 python -B "$fig.py"
done
