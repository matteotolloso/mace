# Evaluation

```bash
uq evaluate B --gpu 6 --dry-run        # list the per-split commands
uq evaluate B --gpu 6                  # all splits on one GPU, then the five-split aggregate
uq evaluate B --gpu 6 --splits 2 --steps reliability_ood_nocal --no-aggregate
uq aggregate [B ...]                   # rebuild aggregates from per-split outputs (CPU only)
uq summary B --test ood                # headline metrics from the tracked aggregates
```

The steps come from the experiment spec (`uq/evaluation/pipeline.py`); each is a
module run as `python -m uq.evaluation.<module>`. For split `s`, in order:

| Step (output name) | Module | What it computes |
|---|---|---|
| `train_curves` | `train_curves` | ensemble train/val loss and RMSE every 5 epochs |
| `reliability_<set>_{cal,nocal}` | `reliability` | final ensemble (per-member validation-NLL checkpoints): per-configuration rows `*_raw.csv` (prediction, error, AU, EU, TU), reliability bins `*_bins.csv`, uncalibrated and isotonic-recalibrated |
| `epoch_raw_train` | `epoch_raw` | mean AU/EU and RMSE on the train set every 10 epochs |
| `epoch_quality_<set>` | `epoch_quality` | Spearman, AUSE, ENCE, AU/EU magnitudes, RMSE, GNLL of the ensemble every 5 epochs |
| `epoch_quality_finetune_mixed_<set>` | `epoch_quality_mixed` | (LF->HF) parent epochs 0-299 on LF files/labels, then fine-tuning 300-399 on HF |
| `finetune` | `finetune_comparison` | (`finetune_comparison: true`) parent vs fine-tuned ensemble on the HF tests |
| `distribution_<set>` | `distribution` | per-configuration AU, EU and error histograms |
| `energy_ood` | `energy_ood` | (`energy_quantiles: true`) AU/EU by within-system energy quantile |

`<set>` runs over `evaluation.reliability` (reliability) or `evaluation.sets`
(the others). Outputs go to `experiments/<id>/evaluation/split_<s>/`; after all
splits, `aggregate` writes the five-split `*.csv`/`*.svg` files (mean, SD, 95%
interval, n; geometric columns for positive quantities) to
`experiments/<id>/evaluation/`. What every output shows and whether it can back a
claim: [evaluation-outputs.md](evaluation-outputs.md).

**Reportable numbers come from the aggregates.** Every reliability metric of an
aggregate (`reliability_<set>_<cal>_summary.csv`) is recomputed from the
support-filtered per-configuration rows, so RMSE, Spearman, AUSE and ENCE describe
the same configurations. Per-split bins/SVGs are unfiltered diagnostics.

## Reusing and regenerating outputs

- The step modules reuse compatible CSV outputs and only redraw plots; to force a
  clean evaluation, move the split's outputs away first (never delete tracked
  results without recording why).
- GPU inference is not bit-reproducible (EU changes by up to a few percent relative
  run to run); rerunning `uq evaluate` therefore changes tracked files slightly.
  `uq aggregate` on unchanged per-split files is byte-reproducible for CSVs.
- `uq aggregate` with no argument also re-renders the active-learning plots.

## Per-epoch predictions (paper Figs. 3 and 5)

The epoch-quality outputs keep only per-epoch summaries, which cannot be
support-filtered afterwards. `uq.evaluation.epoch_predictions` stores the
per-configuration error, AU and EU of every sampled epoch:

```bash
python -m uq.evaluation.epoch_predictions all C D F --gpu 6 --streams 2   # ~3.5 h on one A100
```

Output: `experiments/<id>/evaluation/split_<s>/epoch_predictions_cc_test_ood.npz`
(C is scored on the CC OOD geometries against its own DFT labels, so parent and child
share geometries).

## Recalibration, statistics, filter

Definitions are in [concepts.md](concepts.md). The water specs keep the legacy
`trim: 0.005` and log-log per-split reliability axes (`1e-4` to `1e-1`); the trim only
affects the per-split diagnostics.

## Extras

`uq/evaluation/extra/` holds exploratory plotters that are not part of the pipeline
(member variance grids, AU of individual configurations over epochs, idealized
AU/EU illustration). Run them with `python -m uq.evaluation.extra.<name> --help`.
