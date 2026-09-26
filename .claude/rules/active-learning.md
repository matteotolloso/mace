---
paths:
  - "uq/active_learning/**"
  - "uq/tests/active_learning/**"
  - "runs/active-learning/**"
---

# Active learning (ANI-1x Energy-OOD proof of concept)

Read `docs/active-learning.md` before any AL change (formulas, folder schema, resume
rules, common-evaluator control, leakage safeguards).

- Code in `uq/active_learning/`, runs in `runs/active-learning/ani1x-energy/<name>/`
  (`common.RUNS`). AL never writes into experiments, datasets or their evaluation
  outputs. It compares HF-only (initial members of `ani1x-energy/mace-mve/hf-only`, F)
  and LF->HF (`.../lf-to-hf`, D), each with Random/AU/EU/TU acquisition, budget 500.
- AL learning rate 0.001 for both regimes; the code default is 100 additional epochs,
  but the completed runs are `split_{0..4}_epochs_50` + `aggregate_epochs_50` (explicit
  `--epochs 50`).
- The OOD file is split system-stratified 50/50 into pool and held-out test; the
  held-out half is not higher in energy.
- `C1H5N1:887` was deleted post hoc from split 3's held-out test and cached predictions
  (2499 rows, no backup); raw split-3 results depend on it.
- `python -m uq.active_learning.support_filter` is the geometry-only domain-of-validity
  analysis; report raw and filtered results. The filtered aggregate shares
  `aggregate_epochs_50/` with the raw one; `five_splits` refuses to overwrite it unless
  `--overwrite-filtered` is passed deliberately.
- Runs are provenance-locked (SHA-256 of every input, recorded as absolute paths).
  Pre-restructure paths resolve through `uq.legacy`; resuming an old run fails by design
  (code changed) — use a new `--name`/`--run-tag`. Never hand-edit or copy partial run
  trees; one runner per split (run directories are locked).

Verify with `python -m unittest uq.tests.active_learning.test_workflow
uq.tests.active_learning.test_five_splits uq.tests.active_learning.test_reporting` (24 tests, CPU).
