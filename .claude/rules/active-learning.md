---
paths:
  - "active_learning/**"
---

# Active learning (ANI energy-OOD proof of concept)

Read the "ANI energy-OOD active learning" section of `README.md` before any AL
change; it has the formulas, folder schema, resume rules, common-evaluator
control and leakage safeguards.

- The AL code is isolated under `active_learning/ani_energy/` and must not write
  into A-F datasets or results. It compares HF-only (initial F) and LF->HF
  (initial D), each with Random/AU/EU/TU acquisition at budget 500.
- Both regimes use AL LR 0.001; the code default is 100 additional epochs. The
  completed scientific runs are `split_0_epochs_50` ... `split_4_epochs_50` with
  aggregate `aggregate_epochs_50`; do not mistake their explicit 50-epoch
  override for the current default.
- The OOD source file is split system-stratified 50/50 into acquisition pool and
  held-out test. The held-out half is not a higher-energy half.
- `C1H5N1:887` was deleted post hoc from split 3's held-out AL test and all ten
  matching cached OOD predictions after diagnosis of a catastrophic LF->HF
  extrapolation. Split 3 has 2499 held-out rows; raw split-3 results are
  conditional on it and no backup of the original predictions exists.
- `support_filter.py` is a geometry-only domain-of-validity analysis based on
  each split's minimum training interatomic distance; it never reads prediction
  errors. Report raw and support-filtered AL results and disclose the
  restriction. The same rule lives in `eval/support_filter.py` for A-F; the AL
  copy owns the AL runs.
- Support-filtered output shares `aggregate_epochs_50/` with the raw aggregator.
  `five_splits.py` refuses to overwrite a filtered aggregate: rerun
  `support_filter.py` to refresh it, or pass `--overwrite-filtered` only to
  deliberately replace it with unfiltered numbers.
- AL caches use hashes and manifests. Do not hand-edit or copy partial cache
  trees without updating the dependency graph and verifying fingerprints.
- Do not start two runners for the same split; run directories are locked.

Verify AL changes with:

```bash
/raid/m.tolloso/miniconda3/envs/mace/bin/python -B active_learning/ani_energy/test_workflow.py
/raid/m.tolloso/miniconda3/envs/mace/bin/python -B active_learning/ani_energy/test_five_splits.py
/raid/m.tolloso/miniconda3/envs/mace/bin/python -B active_learning/ani_energy/test_reporting.py
bash -n active_learning/ani_energy/run.sh active_learning/ani_energy/run_all.sh
```
