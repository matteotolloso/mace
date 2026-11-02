---
paths:
  - "uq/evaluation/**"
  - "papers/**"
  - "experiments/**/evaluation/**"
---

# Statistics and plotting contract

Full definitions: `docs/concepts.md`. Keep these when touching evaluation code,
aggregates or figures.

- Linear summaries: arithmetic mean and two-sided 95% Student-t interval over the five
  splits (`df = 4`, `t = 2.776`). Strictly positive quantities on log axes: geometric
  mean with a log-space interval, back-transformed with `exp`; never arithmetic
  symmetric intervals on log axes; never add epsilon or drop splits silently.
- RMSE, Spearman, AUSE and ENCE of an aggregate all come from the same
  support-filtered per-configuration rows (`uq/evaluation/aggregate.py` +
  `support_filter.py`). Keep `trim: 0.0` for reportable numbers.
- AU = mean member variance; EU = population variance of member means
  (`unbiased=False`); TU = AU + EU; per atom by default (`variance / N_atoms^2`).
- ENCE: 15 equal-count bins, variance form `mean_b |MV_b - MSE_b| / MV_b`.
- Diagnostic outputs under `experiments/<id>/evaluation/` are SVG-only; epoch-quality
  plots come fixed-scale, `-free-scale` and `-log-scale` (shared log ranges: Spearman
  `[-0.2, 0.8]` linear, AUSE `[1e-2, 1]`, ENCE `[1e-1, 1e2]`, uncertainty `[1e-7, 1e5]`,
  RMSE `[1e-4, 1e2]`, GNLL `[-8, 4]` linear). Out-of-range centers are clipped and
  marked `x`; bands are clipped to the visible range. Water per-split reliability plots
  use log-log `[1e-4, 1e-1]`.
- Labels carry direction arrows where meaningful: Spearman up; AUSE, ENCE, RMSE, GNLL
  down. Matplotlib rotates y-label glyphs: write `←` to draw ↓ and `→` to draw ↑ in a
  rotated label. Calibration panels emphasize TU (AU/EU faded); ranking panels draw all
  three signals equally.
- Paper figures (`papers/<venue>/figures/`) follow the same statistics and write
  PDF/SVG/PNG to `figures/out/`. After changing evaluation or figure code, rebuild
  (`uq aggregate`, `uq figures iclr27`) and confirm unchanged numbers (CSV bytes,
  `paper_numbers.py` output) or record the change in the ledger.

Verify with `python -m unittest uq.tests.test_evaluation`.
