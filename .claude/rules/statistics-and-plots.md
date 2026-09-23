# Statistics and plotting contract

- Linear summaries use arithmetic means and two-sided 95% Student-t intervals.
  With five splits, df=4 and `t_.975 = 2.776...`.
- Strictly positive quantities on log axes use geometric means and log-space
  Student-t intervals, back-transformed with `exp`. Do not plot arithmetic
  symmetric intervals on log axes and do not add epsilon/drop splits silently.
- Reliability metrics (RMSE, Spearman, AUSE, ENCE) are all computed from the
  same support-filtered per-configuration rows in `eval/aggregate_replicates.py`,
  using the geometry-only filter in `eval/support_filter.py`.
  `reliability.py --trim` is off for A-F: it selected on total uncertainty, the
  quantity under evaluation. Do not re-enable it for paper numbers.
- Uncertainty definitions: AU is the mean member-predicted variance; EU is the
  population variance of member energy means (`unbiased=False`); TU = AU + EU.
  Evaluation normally uses per-atom energies and variances (`variance / N_atoms^2`).
- ENCE uses 15 equal-count bins in variance form, `mean_b |MV_b - MSE_b| / MV_b`.
- Evaluation figures under `experiment_X/evaluation/` are SVG-only. Epoch-quality
  plots have three variants: standard fixed scale (`.svg`), autoscaled
  (`-free-scale.svg`), and shared logarithmic scale (`-log-scale.svg`).
- Current shared log-view limits are Spearman `[-0.2, .8]`, AUSE `[1e-2, 1]`,
  ENCE `[1e-1, 1e2]`, uncertainty `[1e-7, 1e5]`, RMSE `[1e-4, 1e2]`, and GNLL
  `[-8, 4]`. Spearman and GNLL remain linear because they can be negative.
- A center outside a fixed range is clipped to the nearest boundary and marked
  with `x`; confidence bands are clipped to the visible range.
- Labels use direction arrows where meaningful: Spearman up; AUSE, ENCE, RMSE,
  and GNLL down. Uncertainty magnitude has no intrinsic better direction.
- Fine-tuning epoch-quality output is mixed-target only. The legacy
  `epoch_quality_finetune_same*` caches are retained but skipped by aggregation.
- Water reliability uses common log-log bounds `[1e-4, 1e-1]`.
- Paper figures in `new_figures/` follow the same statistics and are written as
  PDF/SVG/PNG to `new_figures/out/`; see `new_figures/README.md`.

Verify statistics or plotting changes with:

```bash
cd eval && MPLCONFIGDIR=/tmp/mpl-cache \
  /raid/m.tolloso/miniconda3/envs/mace/bin/python -B -m unittest -v test_plotting
```
