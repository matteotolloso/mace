# Evaluation figures in `experiment_X/evaluation/`

What every figure produced by `experiment_X/eval_X.sh` shows, which data it uses,
and whether it can back a claim in the paper. Written 2026-09-23 from the scripts
and the current outputs; if a script changes, update this file.

## General conventions

- **Aggregates**: `experiment_X/evaluation/<name>.svg` plus `<name>.csv`, redrawn by
  `eval/aggregate_replicates.py` from the five per-split caches. Lines are
  five-split means with shaded 95% Student-t bands (df=4); on log axes, geometric
  means with log-space bands. Rebuild on CPU with `bash eval.sh --plots-only`.
- **Per-split diagnostics**: `experiment_X/evaluation/cache/split_<s>/<name>.svg`,
  one dataset split each, drawn by the producing script. Do not use them in the
  paper; they are single replicates.
- **Epoch-quality variants**: `<name>.svg` has fixed axis ranges (values outside
  are clipped and marked `x`), `<name>-free-scale.svg` is autoscaled, and
  `<name>-log-scale.svg` uses shared log ranges. **Prefer `-log-scale`**: the
  fixed-scale version clips ENCE at 2 and GNLL at +/-0.5, which hides the
  post-fine-tuning AU ENCE rise (to ~80) and the whole GNLL curve.
- Style: large diagnostic figures (e.g. 14 x 26 in for epoch quality), SVG only,
  default matplotlib colours. They are not paper-styled; paper-ready versions of
  the key results are built by `new_figures/` (PDF/SVG/PNG in `new_figures/out/`).
- Metrics are per atom. Signals: AU (blue), EU (orange), Total/TU (green).
- **Filtering**: only the `reliability_*` aggregates apply the geometry-only
  training-support filter (`eval/support_filter.py`). Everything else is
  unfiltered. That is harmless on the system split (the filter removes nothing
  there) but, on the **energy split (C, D, F), OOD curves are dominated by the
  out-of-support configurations** (13-16 of 5000 in each CC OOD file, 68-120 of
  50 000 in each DFT OOD file; e.g. D's final Energy-OOD TU AUSE is 0.044 in the
  cached curves vs 0.143 filtered).

## Catalogue

| Figure (`evaluation/…`) | Experiments | What it shows | Data | Paper use |
|---|---|---|---|---|
| `reliability_{id,ood}_nocal.svg` | A-F | RMSE vs RMV reliability diagram (15 equal-count bins) for AU, EU, TU with the ideal diagonal; right-hand text lists RMSE, Spearman, AUSE, ENCE (mean [95% CI], n=5) | Best-validation checkpoint per member; ID or OOD test file of the experiment's own target (DFT for A/C, CC otherwise) | **Yes.** Support-filtered, all metrics from one population. Numbers also in `reliability_*_summary.csv`. The CI bands are very wide in the few highest-RMV bins. |
| `reliability_{id,ood}_cal.svg` | A-F | Same after isotonic recalibration of each signal, fitted on the validation split | as above | **Yes**, same caveats. Summarised compactly by `new_figures/out/figA_recalibration`. |
| `epoch_quality_{train,id,ood}[-free-scale,-log-scale].svg` | A-F | Rows: Spearman, AUSE, ENCE, mean AU/EU magnitude, RMSE, Gaussian NLL, against epoch (every 5), for the ensemble at each common epoch | The experiment's own checkpoints only: 300 epochs for A, C, E, F; **only the 100 fine-tuning epochs for B, D**. Train/ID/OOD files of the experiment's target | System split (A, B, E): **yes** (use `-log-scale`). Energy split (C, D, F): **ID and train yes; OOD no** (unfiltered, outlier-dominated). Use `new_figures/out/fig3_hf_dynamics` for the HF stage instead. |
| `epoch_quality_finetune_mixed_{train,id,ood}[…].svg` | B, D | Same six rows over the whole LF->HF trajectory: pretraining epochs 0-299 (parent A/C) then fine-tuning 300-399 (B/D), vertical line at 300 | Pretraining stage on the **DFT** files and labels (`dft_train`, `dft_test_id`, `dft_test_ood`), fine-tuning stage on the **CC** files; so RMSE and GNLL jump at 300 partly because the target changes | System (B): **yes** (`-log-scale`). Energy (D): train and ID yes, **OOD no** (unfiltered). The original dynamics figures `dynamic-sytem-shift.pdf` (B) and `dynamic-energy-shift.pdf` (D) were built from these files. `new_figures/out/fig5_decomposition` is the filtered, paper-styled version of all three. |
| `epoch_quality_finetune_same_*.csv` | B, D | Legacy (no figure is drawn) | fine-tuning targets for both stages | **No.** Skipped by aggregation. |
| `train_curves.svg` | A-F | Ensemble Gaussian NLL loss (top) and ensemble RMSE per atom (bottom) on the train and validation sets, every 5 epochs | The experiment's own train/validation files | Supporting only (convergence, over-fitting). |
| `epoch_raw_train.svg` | A-F | Mean AU, EU, TU variance (top) and ensemble RMSE (bottom) on the training set, every 10 epochs | Training file of the experiment | Supporting only. Linear axes; early spikes dominate. |
| `distribution_{train,id,ood}.svg` | A-F | Histograms (density) of AU, EU, signed error and absolute error per configuration, best-validation ensemble | train / ID / OOD file of the experiment | **No** as is: linear axes are dominated by the tails, so most of the mass falls in the first bin. Unfiltered, no trim. |
| `finetune.svg` | B, D | Bars: mean AU, EU, TU and RMSE for the pretrained (A/C) and fine-tuned (B/D) ensembles on ID and OOD | Both ensembles evaluated on the **CC** test files | **No.** The pretrained-ensemble RMSE (~0.93 eV/atom) is an artifact of scoring a DFT model against CC labels (different reference energies), and it hides the uncertainty bars. The AU/EU means themselves are valid. |
| `energy_ood.svg` | C, D, F | Bars of mean AU and EU in Energy-ID and in Energy-OOD within-system energy-quantile bins | ID and OOD files of the experiment's target; bins from each configuration's `energy_quantile` | **No.** System-balanced *means of variances*, carried by a few out-of-support configurations (D top bin: EU ~1.5e4 with a CI spanning zero), and the bins are drawn in string order (`0.90-1.00` second). Use `new_figures/out/figA_eu_au_ratio` panel (b) (medians, filtered). |

## Water experiments (wA, wB, wC)

Same scripts with one `test` split instead of ID/OOD: `reliability_{train,test}_{nocal,cal}`,
`epoch_quality_{train,test}`, `epoch_quality_finetune_mixed_{train,test}` (wB),
`distribution_{train,test}`, `epoch_raw_train`, `train_curves`. Reliability plots use
common log-log bounds `[1e-4, 1e-1]`. Water is not support-filtered and still passes
`--trim 0.005` for the per-split diagnostics. The five-split water aggregates were produced on
2026-09-23 (ledger entry in `results/results.md`; the aggregate uses every test
configuration, not the trim), and the BLYP and CCSDT splits are not aligned (see "Audit notes and known issues" in `README.md`).

## Paper-styled replacements (`new_figures/out/`)

| Paper figure | Replaces / summarises |
|---|---|
| `fig2_final_summary` | the `reliability_*_nocal` summary numbers of B, D, E, F |
| `fig3_hf_dynamics` | AU/EU/TU AUSE and ENCE from `epoch_quality_ood` of E, F (HF-only) and B, D (fine-tuning stage), filtered for energy |
| `fig4_rank_vs_calibration` | ID -> OOD movement of the `reliability_*_nocal` summaries |
| `fig5_decomposition` | `epoch_quality_finetune_mixed_{train,id,ood}` of B and D, filtered for energy OOD |
| `figA_recalibration` | `reliability_ood_{nocal,cal}` ENCE |
| `figA_dynamics_{system,energy}_{hf,lfhf}` | `epoch_quality_{train,id,ood}` of E, F and `epoch_quality_finetune_mixed_{train,id,ood}` of B, D (`-log-scale` variants); Energy-OOD support-filtered |
| `figA_train_curves` | `train_curves` of A-F |
| `figA_reliability_{system,energy}[_cal]` | `reliability_{id,ood}_{nocal,cal}.svg` of A-F (paper-sized, log-log, support-filtered) |
| `figA_eu_au_ratio` | the magnitude row of `epoch_quality_finetune_mixed_ood`, and `energy_ood` |
| `figW_final_summary`, `figW_reliability` | `reliability_test_nocal` of wB and wC |
| `figW_dynamics` | `epoch_quality_test` of wB, wC and `epoch_quality_finetune_mixed_{train,test}` of wB |
