#!/usr/bin/env bash
# Regenerate every figure from the caches (CPU only). See README.md for the one-off
# geometry cache and the GPU per-epoch predictions that Figs. 3 and 5 need.
set -euo pipefail
cd "$(dirname "$0")"
main_text="fig2_final_summary fig3_hf_dynamics fig4_rank_vs_calibration fig5_decomposition fig6_acquisition"
appendix="figA_recalibration figA_eu_au_ratio figA_acquisition_id figA_reliability figA_dynamics figA_train_curves"  # figA_reliability and figA_dynamics write 4 figures each
water="figW_final_summary figW_reliability figW_dynamics"  # not in the paper yet
for fig in $main_text $appendix $water; do
  PYTHONDONTWRITEBYTECODE=1 python -B "$fig.py"
done
