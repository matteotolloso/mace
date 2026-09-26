# Known issues and audit notes

Pipeline audit of 2026-09-22 (dataset generation to figures), plus findings of the
2026-09-26 restructure. Paths are current; the audit ran on the old layout.

## Verified sound (2026-09-22)

- Split integrity, CC-subset-of-DFT, disjoint OOD systems, energy ordering
  ([datasets.md](datasets.md#ani-1x)).
- MVE head and Gaussian NLL, AU/EU/TU with per-atom normalization and population EU,
  ENCE (manuscript Eq. 8, 15 bins), AUSE ([concepts.md](concepts.md)).
- Fine-tuning copies body, readouts and scale/shift from the matching parent member and
  re-fits E0s; checkpoint selection uses validation Gaussian NLL only.
- All 450 members complete; unit tests pass (6 evaluation, 24 active learning).

## Open

1. **Water fidelities are not aligned** (`water-random`). Each level of theory is
   shuffled with its own seed, so 31-37 of the 50 CCSD(T) test geometries per split sit
   in the BLYP training set of the LF->HF parent (80-91 of 130 CCSD(T) train
   geometries too). Water LF->HF vs HF-only comparisons are biased toward LF->HF.
   Fix: a configuration-level split shared by all theories (new dataset) and retraining.
   Water specs also keep the legacy `trim: 0.005` (per-split diagnostics only) and have
   no support filter.
2. **GPU inference is not bit-reproducible.** Re-running the same evaluation changes
   per-configuration EU by up to ~5% relative (AU ~1e-7) on small-variance
   configurations (float32 reductions); predictions agree to printed precision.
   Regenerated per-split outputs therefore never match byte for byte, and old-vs-new
   code checks must compare code or command lines, not GPU outputs.
3. **Active-learning run provenance vs current code.** Runs record the SHA-256 of their
   source files. After 2026-09-22 (`mace/modules/models.py` fix) and the 2026-09
   restructure, resuming a completed run fails by design; re-aggregation and plotting
   still verify ([active-learning.md](active-learning.md#provenance-and-the-2026-09-restructure)).
4. **Split-3 AL caveat.** `C1H5N1:887` was deleted post hoc from split 3's held-out test
   and cached predictions, with no backup; raw split-3 numbers depend on it (the support
   filter removes it anyway).
5. **Epoch checkpoints were pruned (2026-09-26).** Only epochs divisible by 5, each
   member's best-validation checkpoint and the final `.model` files remain
   (`runs/<id>/split_<s>/checkpoints/PRUNED.json`). Every tracked output can be regenerated
   from them (up to GPU nondeterminism); epoch analyses at a finer resolution or re-selection by another criterion
   need retraining, and the evaluation code at tag `iclr27-submission` refuses the pruned
   folders (it requires a checkpoint for every logged epoch).

## Manuscript-side (not code)

- **EU denominator**: code uses the population variance (`1/M`, intended); the
  manuscript's EU equation shows `1/(M-1)`.
- **Old dynamics figure with rotated rows** (NeurIPS draft Fig. 4,
  `dynamic-sytem-shift.pdf`): in its ID and OOD columns the row labelled "Spearman"
  plots AUSE, "AUSE" plots ENCE and "ENCE" plots Spearman (r = 1.000 against the split-0
  caches of experiment B); the train column is correct. The ICLR27 figures are rebuilt
  from `papers/iclr27/figures/` and do not have this problem.

## Fixed

- 2026-09-22 — **One population for every reliability metric**: `--trim 0.0` for ANI and
  aggregates rebuilt from support-filtered rows (previously ENCE came from trimmed bins,
  the others from untrimmed rows). Energy-OOD numbers changed substantially (experiment D
  OOD RMSE 10.00 +/- 16.09 eV/atom -> 0.0587 +/- 0.0040 eV/atom; TU AUSE 0.044 -> 0.143);
  system-split numbers moved slightly (B OOD TU ENCE 0.313 -> 0.324).
- 2026-09-22 — **MVE variance of the unused `MACE` class** now matches `ScaleShiftMACE`
  (softplus per-atom variances, summed, deterministic terms zero). No trained model was
  affected: every config builds `ScaleShiftMACE`.
- 2026-09-22 — **AL aggregate overwrite hazard**: `five_splits` refuses to replace a
  support-filtered aggregate without `--overwrite-filtered`.
- 2026-09-26 — **Result tables were silently git-ignored** (`*.txt`): `results/tables/`
  is now tracked. **`eval.sh` used six GPUs**: replaced by `uq evaluate` (one GPU) and
  `uq train` (at most two).
