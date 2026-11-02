# Active learning on ANI-1x Energy-OOD (proof of concept)

Single-round acquisition study: code in `uq/active_learning/`, runs in
`runs/active-learning/ani1x-energy/<name>/` (git-ignored). It is **not** part of the
ICLR27 main text except paper Fig. 6 / appendix figA_acquisition_id; keep its claims
separate unless a paper is extended.

## Commands

```bash
uq al run-all 2 --epochs 50                       # run/resume splits 0-4 on GPU 2, then aggregate
uq al run-all 2 --aggregate-only --epochs 50      # aggregate completed reports only (no GPU)
uq al run-all 2 --aggregate-only --wait           # wait for running splits, then aggregate
uq al run 7 --split-seed 1 --epochs 50            # one split on GPU 7
python -m uq.active_learning.support_filter --output aggregate_epochs_50   # support-filtered aggregate
python -m uq.active_learning.plot_results --summary runs/active-learning/ani1x-energy/aggregate_epochs_50/summary_ci95.json
```

The completed scientific runs are `split_{0..4}_epochs_50` with the support-filtered
aggregate `aggregate_epochs_50` (paper Fig. 6 reads it). Without `--epochs` the
directories are `split_<s>` / `aggregate`; `--run-tag NAME` makes an independent repeat.

> **Overwrite hazard.** `run-all` writes the *unfiltered* aggregate to the same
> directory as the support-filtered one. `five_splits` refuses to replace a filtered
> aggregate unless `--overwrite-filtered` is given; after any deliberate overwrite,
> rerun the support-filter and plot commands above.

Stages of one split (rerunning reuses completed work; `--case`/`--members` select parts):

```bash
python -B -m uq.active_learning.workflow --stage prepare --device cpu
uq al run 7 --stage acquire
uq al run 7 --stage train [--case hf_only_random --members 0 1]
uq al run 7 --stage evaluate
python -B -m uq.active_learning.workflow --stage report
```

Cases: `hf_only_{random,au,eu,tu}`, `lf_hf_{random,au,eu,tu}`; optional control
`--common-evaluator` adds `lf_hf_from_hf_only_tu` (D ensemble trained on the HF-only
TU batch), reported under `common_evaluator/`.

## Design

| Regime | Initial ensemble (final members, per-member validation-NLL checkpoint) |
|---|---|
| HF-only | `ani1x-energy/mace-mve/hf-only` (F) |
| LF->HF | `ani1x-energy/mace-mve/lf-to-hf` (D) |

- Only `cc_test_ood.xyz` of the energy split is repartitioned: 2500 configurations form
  the acquisition pool and 2500 the held-out test (system-stratified halves of the same
  OOD domain; the held-out half is not higher in energy). All other files are unchanged
  and checked for ID overlap.
- Budget B = 500 whole configurations per condition: random (shared by both regimes)
  or the global top 500 of AU, EU or TU (stable ID tie-break). Scores are the per-atom
  AU/EU/TU of [concepts.md](concepts.md), one scalar per configuration.
- The pool file holds geometry only; CC labels are revealed from an offline oracle
  (`data/oracle.json`, pool IDs only) after the selection is persisted.
- Each member continues from its frozen initial checkpoint on original HF train + the
  500 revealed configurations: learning rate 0.001 for every condition, at most 100
  epochs (the completed runs used 50), fresh optimizer/scheduler, E0s/heads/buffers
  preserved, `mace.tools.train`, best checkpoint by validation Gaussian NLL.
- Held-out Energy-OOD and the unchanged `cc_test_id.xyz` are scored; RMSE is that of
  the ensemble-mean energy per atom (meV/atom).
- `improvement (%) = 100 (RMSE_before - RMSE_after) / RMSE_before`;
  `gain (meV/atom) = RMSE_random - RMSE_method`; LF->HF gain minus HF-only gain per
  method. Five splits are the replicates (approximate 95% t intervals).

## Support filter for AL

`python -m uq.active_learning.support_filter` recomputes every metric without held-out
configurations below the split's training-support bound (0.825-0.846 A; removes 6-10 of
~2500 held-out Energy-OOD and 0-2 Energy-ID configurations per split). Insensitive to the
threshold between 0.80 and 1.00 A. Effect (five splits): LF->HF TU improvement
20.3% +/- 74.8 raw -> 14.2% +/- 3.7; LF->HF EU gain 699 [-558, 1957] -> 2.50 [1.89, 3.11]
meV/atom; all six AU/EU/TU-over-random OOD gains exclude zero after filtering, none did
before. `support_filter.json` records the thresholds and every excluded ID.

## Provenance and the 2026-09 restructure

Each run records every input as an absolute path with its SHA-256 and re-verifies them.
Runs made before the restructure recorded old paths (`/.../active_learning/ani_energy/...`,
`/.../experiment_D/...`); `verify_inventory` follows them to their current location via
`uq.legacy` and still requires identical hashes, so completed runs can be re-aggregated
and re-plotted (all five reports, the 100 inputs of `aggregate_epochs_50` and every
manifest artifact were verified after the move). **Resuming** an old run fails on
purpose: its recorded code/config sources changed (moved modules, config path fields,
and `mace/modules/models.py`, fixed on 2026-09-22 before the restructure). Use a new
`--name` or `--run-tag` for new work.

## Known caveats

- `split_3_epochs_50` had configuration `C1H5N1:887` deleted post hoc from its held-out
  test and cached predictions (no backup); raw split-3 numbers are conditional on that
  deletion. The support filter excludes it anyway (filtered results identical).
- The pool comes from an already-labelled OOD set; labels are hidden from the workflow,
  not from dataset construction: an offline POC, not a prospective benchmark.
- Companion models contain CUDA TorchScript blocks: acquisition, training and
  evaluation need a GPU (preparation works on CPU).
- Tests (CPU, synthetic data): `python -m unittest uq.tests.active_learning.test_workflow
  uq.tests.active_learning.test_five_splits uq.tests.active_learning.test_reporting`.
