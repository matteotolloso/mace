# Concepts: models, uncertainty, metrics, statistics

The definitions every experiment, table and figure in this repository relies on.
Code locations are given so each statement can be checked.

## Terminology

| Term | Meaning |
|---|---|
| LF / HF | low / high fidelity labels. ANI-1x: DFT `wb97x_tz.energy` / CC `ccsd(t)_cbs.energy`. Water: BLYP / CCSD(T), both stored as `REF_energy` |
| Protocol | `lf-only` (train on LF), `hf-only` (train on HF from scratch), `lf-to-hf` (each member pretrained on LF, then fine-tuned on HF: DFT pretraining + CC fine-tuning) |
| Shift / test role | `id` / `ood` test sets of a dataset (System-ID/OOD, Energy-ID/OOD); water has a single `test` role |
| Split | one statistical replicate: an independently seeded data split, seeds `0..4` |
| Member | one of the ten MVE models of a split's ensemble, seeds `0..9`; the ten members form **one** ensemble |
| Experiment | dataset x model/UQ method x protocol, e.g. `ani1x-energy/mace-mve/lf-to-hf` (legacy letter `D`); see [experiments.md](experiments.md) |
| AU / EU / TU | aleatoric / epistemic / total uncertainty, defined below |

## Model: MACE with a mean-variance (MVE) head

`model: "MACE"` in the experiment configs builds `ScaleShiftMACE`
(`mace/tools/model_script_utils.py::_build_model`) with `predict_mve: True`
(`mace/modules/models.py`, code between `### MVE ###` fences):

- the last readout emits a mean and a raw variance parameter per atom;
- per-atom variance = `softplus(raw) + 1e-12` (strictly positive);
- per-atom variances are summed over atoms (variance of a sum of independent
  contributions) and scaled by `scale^2`; the deterministic E0 baseline adds no variance.

Training minimizes the Gaussian NLL (`mace/modules/loss.py`):
`0.5 * (err^2 / var + log(var) + log(2*pi)) / N_atoms`, weighted by the
configuration and energy weights. Training is energy-only (`forces_weight: 0.0`,
`compute_forces: False`). Every epoch is checkpointed (`save_all_checkpoints: True`); after
evaluation, `uq prune-checkpoints` keeps every 5th epoch and each member's best
([experiments.md](experiments.md#pruning-epoch-checkpoints)).

## Uncertainty decomposition (`mve-ensemble`)

Per configuration, per atom by default (`uq/evaluation/reliability.py::evaluate_split`,
mirrored in `epoch_quality.py` and the active-learning runtime), with `M = 10` members:

```text
energy_m   = member m total energy / N_atoms
var_m      = member m predicted variance / N_atoms^2
prediction = mean_m(energy_m)
AU         = mean_m(var_m)
EU         = population variance over members of energy_m   (unbiased=False, denominator M)
TU         = AU + EU
```

The manuscript's EU equation shows `1/(M-1)`; the code (the intended estimator)
uses `1/M` — see [known-issues.md](known-issues.md).

## Baseline: plain MSE ensemble (`mse-ensemble`, experiments `*/mace-mse/*`)

Same architecture, data, seeds and hyperparameters as the MVE twin, but
`predict_mve: False` and `loss: "energy_only"`: members minimize the per-atom energy
MSE `((E_ref - E_pred) / N_atoms)^2` and have no variance output. The evaluation code is
unchanged: a member without `energy_var` contributes variance 0, so

```text
AU = 0          (constant; its rank metrics are undefined and must not be reported)
EU = population variance over members of energy_m   (as above)
TU = EU
```

The ablation compares, per split, MSE-ensemble EU with MVE EU (effect of the loss)
and with MVE TU (effect of adding AU). EU of an MSE ensemble is not calibrated by
construction; compare ranking metrics (Spearman, AUSE) directly and calibration only
after the same recalibration for both methods.

## Checkpoint selection

Final ensembles use, **per member**, the checkpoint with the lowest **validation
loss** in its MACE results log (`--selection-key loss --selection-mode min`): the
Gaussian NLL for MVE members, the energy MSE for `mse-ensemble` members.
ID/OOD test sets are never used for selection. Epoch-quality analyses instead
evaluate the whole ensemble at every common epoch (every 5th).

## Metrics (`uq/evaluation/reliability.py`, `epoch_quality.py`)

| Metric | Definition | Better |
|---|---|---|
| RMSE | of the ensemble-mean per-atom energy, eV/atom in files, meV/atom in figures | lower |
| Spearman | rank correlation between an uncertainty signal and the squared error | higher |
| AUSE | MAE sparsification error, normalized by the all-configuration MAE, integrated over the removed fraction (oracle = true absolute errors) | lower |
| ENCE | 15 equal-count bins ordered by the signal, variance form `mean_b |MV_b - MSE_b| / MV_b` (manuscript Eq. 8) | lower |
| GNLL | Gaussian NLL of the ensemble with TU variance (epoch-quality outputs) | lower |
| Isotonic recalibration | fitted on the validation split (predicted variance -> squared error), applied to the test split (`*_cal` outputs) | — |

Ranking metrics: Spearman, AUSE. Calibration: ENCE. Only TU is expected to be
calibrated (the predictive variance is AU + EU), which is why calibration figures
emphasize TU.

## Training-support filter

Held-out test sets can contain geometries compressed below the smallest
interatomic distance of the split's own training data; models extrapolate
catastrophically there and one such configuration can dominate a mean of squares.
Every reliability metric of the five-split aggregates is computed after removing
test configurations whose minimum interatomic distance is below the smallest one
in the split's train and val files of every fidelity. The rule reads geometry only
(never predictions, errors or uncertainties), applies to every model and signal
alike, and is opt-in per dataset (`support_filter: true` in `dataset.yaml`; ANI-1x
yes, water no). Code: `uq/evaluation/support_filter.py`; cache and report:
`uq data support`. ANI-1x effect: 13-16 of 5000 Energy-OOD and 0-2 Energy-ID
configurations removed per split, essentially none on the system split.

The older `reliability --trim` option (drop the 0.5% lowest/highest *total-variance*
configurations) selects on the quantity being evaluated; it is **off** (`--trim 0.0`)
for ANI-1x and must not be used for reportable numbers. Water specs still pass the
legacy `0.005`, which only affects the per-split diagnostic plots, not the rows that
the aggregate reads.

## Statistics contract

- Five splits are the replicates; ten members make one ensemble and are never
  treated as independent samples. Never aggregate fewer than five splits or ten
  members silently.
- Linear summaries: arithmetic mean with a two-sided 95% Student-t interval
  over splits (`df = 4`, `t = 2.776`, sample SD with `ddof = 1`).
- Strictly positive quantities on log axes (ENCE, RMSE, ratios): geometric mean
  with a log-space Student-t interval `exp(mean(log x) +/- t * sd(log x) / sqrt(n))`;
  if any replicate is non-positive or missing, fall back to arithmetic intervals
  on a linear axis (no epsilon, no dropped split).
- Metrics are computed within each split, then summarized across splits.
- Direction arrows on axis labels: Spearman up; AUSE, ENCE, RMSE, GNLL down.
- Diagnostic epoch-quality plots come in three variants: fixed scale (`.svg`),
  autoscaled (`-free-scale.svg`) and shared log scale (`-log-scale.svg`; prefer this one).
  Shared log ranges: Spearman `[-0.2, 0.8]` (linear), AUSE `[1e-2, 1]`, ENCE
  `[1e-1, 1e2]`, uncertainty `[1e-7, 1e5]`, RMSE `[1e-4, 1e2]`, GNLL `[-8, 4]` (linear).
- GPU inference is not bit-reproducible: re-running the same evaluation changes
  per-configuration EU by up to a few percent (relative) on small-variance
  configurations, so regenerated per-split outputs are never byte-identical.
