# <span style="font-size:larger;">MACE</span>

## Project-specific experiment pipeline

This repository contains the original MACE codebase and a project-specific
multi-fidelity uncertainty pipeline for ANI-1x and periodic liquid water. The
pipeline trains mean-variance estimation (MVE) ensembles, separates aleatoric
uncertainty (AU) from ensemble epistemic uncertainty (EU), and repeats every
experiment on five independently seeded dataset splits for statistical
significance. It also contains an isolated single-round active-learning (AL)
proof of concept on the ANI energy split, documented in its own section below.

The statistical design is:

- 9 experiment families: `A` through `F`, and `wA` through `wC`.
- 5 dataset splits per experiment, with split seeds `0` through `4`.
- 10 independently initialized ensemble members per dataset split, with member
  seeds `0` through `9`.
- 50 trained members per experiment family and 450 member runs in total.
- Each 10-member ensemble produces one split-level estimate of AU, EU, total
  uncertainty, calibration, and error metrics.
- Final statistics use the five dataset splits as independent replicates. The
  10 members inside one split are not treated as 10 independent statistical
  samples.

Current training state (`./check_experiments.sh`): ANI-1x `A`-`F` are complete
(300/300 members). Water `wA`/`wB`/`wC` are complete
(150/150 members) and evaluated on all five splits (2026-09-23; see
`results/results.md`). See
[Audit notes and known issues](#audit-notes-and-known-issues).

All project commands below are intended to be run from the repository root.

### Repository layout

The project-specific files are organized as follows:

```text
CLAUDE.md, .claude/               Agent instructions: rules/ (reproducibility, statistics,
                                  code changes, paper writing, AL) and skills/
mace/                             MACE package with the MVE head and Gaussian NLL
dataset/                          Dataset generation and converted data
experiment_A/ ... experiment_F/   ANI-1x experiments (configs, launchers, runs, evaluation)
experiment_wA/ ... experiment_wC/ Water experiments
eval/                             Evaluation, plotting, aggregation, support filter
active_learning/ani_energy/       Isolated ANI energy-OOD active-learning POC
new_figures/                      Rebuilt paper figures (support-filtered, 5 splits)
utils/train_split_member.sh       Shared single-member training launcher
check_experiments.sh              Training-completeness checker
eval.sh                           ANI-1x evaluation launcher
eval_water.sh                     Water evaluation launcher
experiments/                      Index of experiment locations; ad-hoc launch scripts
results/results.md                Research ledger: the only source for citable results
results/tables/, results/figures/ Curated tables and figures that back a claim
notes/                            Research notes and the previous NeurIPS manuscript
ICLR27-UQ-MF/                     Overleaf-synced manuscript (separate Git repository)
```

`experiment_X/` and the code directories stay at the root because the launchers
resolve the repository root from their own location and every cache uses fixed
relative paths; `experiments/README.md` maps each piece.

The manuscript lives in its own repository,
`<paper-repository-url>`, which Overleaf syncs through
GitHub. It is cloned inside this project as `ICLR27-UQ-MF/` and ignored by this
repository's Git; commit and push paper changes from inside that folder:

```bash
git clone <paper-repository-url> ICLR27-UQ-MF   # once
git -C ICLR27-UQ-MF pull --ff-only                                          # update
```

Every citable number must be recorded in `results/results.md` with its commit,
command, configuration, dataset split, seeds, metrics and output paths.

Each experiment directory contains:

```text
config_X.yml                     Base MACE configuration
train_X.sh                       Train one split/member pair
eval_X.sh                        Evaluate all five splits on one GPU
checkpoints_<split_seed>/        Epoch checkpoints and final companion models
models_<split_seed>/             Serialized model output
results_<split_seed>/            Per-member training JSONL logs
logs_<split_seed>/               Training logs
evaluation/cache/split_<seed>/   Expensive per-split evaluation caches
evaluation/                      Five-split aggregate CSV files and plots
```

The paths ending in `_0` inside a `config_X.yml` are defaults. The training
wrapper overrides dataset and output paths from the requested split seed, so a
config file does not need to be edited before training splits `_1` through `_4`.

### Dataset generation

#### ANI-1x

ANI-1x generation expects `dataset/ani1x-release.h5`. The script creates both
system-OOD and energy-OOD datasets for split seeds `0` through `4`:

```bash
bash dataset/make_dataset.sh
```

Specific seeds can be regenerated by passing them as arguments:

```bash
bash dataset/make_dataset.sh 2 4
```

The generated folders are:

```text
dataset/ani1x_system_split_0 ... dataset/ani1x_system_split_4
dataset/ani1x_energy_split_0 ... dataset/ani1x_energy_split_4
```

System-OOD splits separate molecular systems between the seen and OOD sets.
Energy-OOD splits separate configurations using within-system relative-energy
quantiles. Every folder contains low-fidelity DFT and high-fidelity CC data:

```text
dft_train.xyz       cc_train.xyz
dft_val.xyz         cc_val.xyz
dft_test_id.xyz     cc_test_id.xyz
dft_test_ood.xyz    cc_test_ood.xyz
```

The DFT target is `wb97x_tz.energy`; the high-fidelity target is
`ccsd(t)_cbs.energy`. `split_summary.json` and `system_assignments.csv` record
the generated split metadata. `dataset/ani1x-processed` is a legacy export and
is not used by the current generators, training scripts, or evaluation scripts.

Splitter settings actually used by `dataset/make_dataset.sh` (these are not the
standalone splitters' argument defaults):

- **System split (`A`/`B`/`E`).** 60% of systems form the seen domain; their
  configurations are partitioned 60/20/20 into candidate train/validation/ID
  pools. The remaining systems supply OOD configurations. This tests
  generalization to unseen systems, not a high-energy threshold.
- **Energy split (`C`/`D`/`F`).** Configurations are ranked by DFT energy
  (`wb97x_tz.energy`) *within each system*. Empirical ranks at or below
  `--q-low 0.5` form the low-energy domain, ranks at or above `--q-high 0.55`
  form the high-energy OOD domain, and the band between is discarded. The
  low-energy configurations are partitioned 60/20/20 into candidate
  train/validation/ID pools. The ranking never uses CC labels.
- Systems with fewer than four valid ranking energies are dropped. With exactly
  four, the highest-energy configuration goes to OOD and the other three are
  assigned one each to train/validation/ID. This is why the seen domain contains
  a few quantile values above 0.5 (up to 0.667): they come from very small
  systems. Within every system the seen-domain configurations still lie strictly
  below every OOD configuration (verified: 0 violations in 2148 shared systems).
- Final files are sampled from the candidate pools with system-stratified quotas
  and `max_per_system=64` as a **soft weighting cap**, not a hard per-file limit.
  DFT train/validation/ID/OOD sizes are 50000/10000/50000/50000; CC sizes are
  5000/1000/5000/5000. CC configurations are sampled inside the corresponding
  DFT selection, so each CC file is an exact subset of the DFT file with the
  same role (verified for all four roles).
- Forces are deliberately not exported (`--dft-forces-key ' '`), and training is
  energy-only (`forces_weight: 0.0`, `compute_forces: False`).

#### Periodic liquid water

`dataset/water_0` is the source tree. Its theory subdirectories contain n2p2
`input.data` files, including `blyp`, `ccsdt`, `ccsd`, `hf`, `revpbe-d3`, and
`afqmc`. The converter preserves periodic cells and periodic boundary
conditions in extended XYZ format.

Generate all five 65/10/25 train/validation/test splits with:

```bash
bash dataset/make_dataset_w.sh
```

Specific seeds can be regenerated with:

```bash
bash dataset/make_dataset_w.sh 1 3
```

This creates `dataset/water_0` through `dataset/water_4`. For seeds `_1`
through `_4`, the script copies the source `input.data` files from `water_0`
and performs a new seeded split. Each theory directory then contains:

```text
input.data
train.xyz
val.xyz
test.xyz
```

Water extended XYZ files use `REF_energy` as the energy key.

> **Caveat.** `convert_water_n2p2_to_extxyz.py` shuffles **each theory folder
> with its own derived seed** (`seed + sum(ord(c) for c in theory_name)`), and
> the theories contain different numbers of configurations. BLYP and CCSDT
> splits are therefore not aligned: in `water_0`, 37 of the 50 CCSDT test
> geometries also appear in the BLYP training set. See
> [Audit notes and known issues](#audit-notes-and-known-issues).

Set `PYTHON_BIN` when dataset scripts should use a specific Python executable:

```bash
PYTHON_BIN=/path/to/python bash dataset/make_dataset_w.sh
```

### MVE model and uncertainty definitions

`model: "MACE"` in the experiment configs builds a `ScaleShiftMACE` (see
`mace/tools/model_script_utils.py::_build_model`) with `predict_mve: True`.
In that model (`mace/modules/models.py`):

- the last readout emits `2 x n_heads` scalars, a mean and a raw variance
  parameter per atom;
- the per-atom variance is `softplus(raw) + 1e-12`, so it is strictly positive;
- per-atom variances are summed over atoms, which is the variance of a sum of
  independent per-atom contributions, and scaled by `scale^2`;
- the `E0` baseline is deterministic, so `energy_var` equals the interaction
  variance. Deterministic readouts contribute exactly zero variance.

Training minimizes the Gaussian NLL of `mace/modules/loss.py`:

```text
0.5 * (err^2 / var + log(var) + log(2*pi)) / N_atoms
```

weighted by the configuration and energy weights. Forces are off
(`forces_weight: 0.0`), so the loss is energy-only.

Ensemble uncertainty (`eval/reliability.py::evaluate_split`, mirrored in
`eval/epoch_quality.py` and the AL code) is computed per configuration, by
default per atom:

```text
energy_m   = member m total energy / N_atoms
var_m      = member m predicted variance / N_atoms^2
AU         = mean_m(var_m)
EU         = population variance over members of energy_m   (unbiased=False)
TU         = AU + EU
prediction = mean_m(energy_m)
```

Note that EU uses the **population** variance (denominator `M`), while Eq. 3 of
the current manuscript shows `M-1`.

Reported metrics (`eval/reliability.py`) are:

- **RMSE** of the ensemble-mean per-atom energy.
- **Spearman** rank correlation between uncertainty and squared error.
- **AUSE**: MAE-based sparsification error, normalized by the all-configuration
  MAE, integrated against the removed fraction, with the oracle ordering given
  by the true absolute errors.
- **ENCE**: 15 equal-count bins ordered by uncertainty, with
  `mean_b |MV_b - MSE_b| / MV_b`. This matches Eq. 8 of the manuscript, which is
  written in variance units rather than the more common RMV/RMSE form.
- **Gaussian NLL** per configuration (epoch-quality outputs only).
- Optional **isotonic recalibration**, fitted on the *validation* split mapping
  predicted variance to squared error, then applied to the test split.

### Experiment definitions

The ANI-1x experiment families are:

| Experiment | Dataset split | Training target | Initialization |
| --- | --- | --- | --- |
| `A` | system-OOD | low-fidelity DFT | from scratch |
| `B` | system-OOD | high-fidelity CC | fine-tune matching `A` member |
| `C` | energy-OOD | low-fidelity DFT | from scratch |
| `D` | energy-OOD | high-fidelity CC | fine-tune matching `C` member |
| `E` | system-OOD | high-fidelity CC | from scratch |
| `F` | energy-OOD | high-fidelity CC | from scratch |

The water experiment families are:

| Experiment | Training target | Initialization |
| --- | --- | --- |
| `wA` | low-fidelity BLYP | from scratch pretraining |
| `wB` | high-fidelity CCSDT | fine-tune matching `wA` member |
| `wC` | high-fidelity CCSDT | from scratch, high-fidelity-only baseline |

For fine-tuning, matching means the same dataset split seed and member seed.
For example, `wB` split 3 member 7 requires:

```text
experiment_wA/checkpoints_3/mace_run-7.model
```

The training launcher checks this dependency before starting.

What fine-tuning inherits (`mace/tools/finetuning_utils.py`,
`mace/tools/model_script_utils.py`): the architecture is extracted from the
parent model, including `predict_mve`; the embedding, interaction, product and
**readout** weights are copied (`load_readout=args.foundation_filter_elements`,
default `True`), as are the `scale_shift` buffers. Because
`multiheads_finetuning: False`, the atomic reference energies are **re-fitted on
the fine-tuning target** (`E0s: "average"` on `cc_train`), which absorbs the
DFT-to-CC reference offset. Fine-tuning uses 1/10 of the pretraining learning
rate and 100 epochs against 300.

### Training

One training command runs exactly one ensemble member. Its interface is:

```bash
bash experiment_X/train_X.sh <split_seed:0-4> <member_seed:0-9> <gpu_id>
```

For example, train experiment `A`, split 2, member 7 on physical GPU 4:

```bash
bash experiment_A/train_A.sh 2 7 4
```

Train the corresponding pretraining member before starting `B`, `D`, or `wB`:

```bash
bash experiment_wA/train_wA.sh 2 7 4
bash experiment_wB/train_wB.sh 2 7 4
```

Members are intentionally launched one at a time. This makes it possible to
assign runs manually according to GPU availability. `CUDA_VISIBLE_DEVICES` is
set internally from `gpu_id`. Set `PYTHON_BIN` to override the Python command:

```bash
PYTHON_BIN=/path/to/python bash experiment_wC/train_wC.sh 4 9 2
```

For a complete experiment family, repeat training for every split/member pair:

```bash
for split_seed in 0 1 2 3 4; do
  for member_seed in 0 1 2 3 4 5 6 7 8 9; do
    bash experiment_wC/train_wC.sh "$split_seed" "$member_seed" 0
  done
done
```

This example is sequential and uses GPU 0. Runs may instead be distributed
manually across GPUs, provided each split/member pair is trained exactly once.

The member seed is passed as `--seed`, so members of one split differ by weight
initialization and batch order only. `save_all_checkpoints: True` keeps every
epoch, which the epoch-quality analyses require. SWA/stage-two and EMA are off.

### Checking training completeness

Use the checker before evaluation:

```bash
./check_experiments.sh
```

It scans all 9 experiment families, 5 split seeds, and 10 member seeds. A member
is complete only when both files exist:

```text
checkpoints_<split>/mace_run-<member>.model
results_<split>/mace_run-<member>_train.txt
```

An epoch checkpoint or result log without the final companion model is reported
as partial. A run with no member-specific artifacts is reported as missing. The
script returns exit code `0` only when all 450 runs are complete; otherwise it
returns `1` after printing the full report.

### Evaluation and statistical significance

Each `eval_X.sh` takes exactly one argument: the physical GPU number.

```bash
experiment_X/eval_X.sh <gpu_number>
```

For example:

```bash
experiment_wA/eval_wA.sh 6
```

The script sequentially evaluates split seeds `0`, `1`, `2`, `3`, and `4` on
GPU 6. For split `s`, it explicitly reads:

```text
experiment_wA/checkpoints_s/
experiment_wA/results_s/
dataset/water_s/...
```

and writes reusable per-split inference outputs under:

```text
experiment_wA/evaluation/cache/split_s/
```

After all five split evaluations succeed, `eval/aggregate_replicates.py` writes
the final tables and figures under `experiment_wA/evaluation/`. If any split
evaluation fails or required training output is missing, `set -e` stops the
script and aggregate plots are not rebuilt.

**Checkpoint selection.** Every `eval_X.sh` passes `--selection-key loss
--selection-mode min`, so each member independently contributes the checkpoint
with the lowest **validation Gaussian NLL** recorded in its results JSONL. The
ID and OOD test sets are never used for selection. (The training config's
`test_file` is the ID test set; it is scored once at the end of training for the
log only.) Epoch-quality analyses instead evaluate the whole 10-member ensemble
at each common epoch.

The aggregation procedure computes each metric independently for each
10-member ensemble, then reports across the five dataset splits:

```text
<metric>_mean
<metric>_std
<metric>_ci95_low
<metric>_ci95_high
<metric>_n
```

Confidence intervals are two-sided 95% Student-t intervals. With five finite
replicates, the calculation uses 4 degrees of freedom (`t = 2.776`, sample SD
with `ddof=1`). Line plots show the five-split arithmetic mean and a shaded 95%
confidence band on linear axes. Bar and distribution plots use equivalent
replicate-level intervals. Reliability summaries compute RMSE, Spearman
correlation, AUSE, and ENCE per split before averaging them.

For strictly positive quantities displayed on logarithmic axes, aggregate plots
instead show the **geometric mean and a 95% log-space Student-t interval**:
`exp(mean(log(x)) +/- t * sd(log(x)) / sqrt(n))`. These multiplicative bounds
are symmetric in log coordinates and stay positive. They estimate the geometric
mean, not the arithmetic mean; affected axes are labeled accordingly. Existing
arithmetic CSV columns are preserved, with extra `<metric>_geometric_*` columns.
If any replicate is zero, negative or missing, that panel uses arithmetic
intervals on a linear axis; values are not dropped or replaced by an epsilon.
Reliability text summaries remain arithmetic. AL OOD RMSE plots also default
to geometric means with log-space t intervals; ID RMSE and signed gains remain
linear with arithmetic intervals. AL aggregate reports are unchanged; separate
`plots/rmse_display.json` records the displayed estimates and their estimators.

**Training-support filter, not trimming.** Every reliability metric in the
five-split aggregate is computed from the same population: the per-configuration
rows restricted to the region where the models have training support. A test
configuration is excluded when its smallest interatomic distance falls below the
smallest one anywhere in that split's own `cc_train`, `cc_val`, `dft_train` and
`dft_val`. The rule is implemented in [`eval/support_filter.py`](eval/support_filter.py),
reads geometry only, and applies identically to every model and every signal, so
it cannot select on the quantity under evaluation. It removes 13-16 of 5000
Energy-OOD and 0-2 Energy-ID configurations per split, and 0-8 of 50000 on the
system split. Water and any non-ANI family are never filtered.

The filter needs a one-off geometry cache (CPU, about 5 minutes), stored in
`eval/cache/geometry/` and shared with `new_figures/`:

```bash
python -B eval/support_filter.py --build     # build the cache
python -B eval/support_filter.py --report    # per-split bound and removal counts
```

`eval/aggregate_replicates.py` rebuilds the reliability bins from the filtered
rows, so ENCE, Spearman, AUSE and RMSE all describe the same configurations.

`reliability.py --trim` still exists but is **off** (`--trim 0.0`) in the A-F
eval scripts. It dropped the 0.5% lowest and highest **total-variance**
configurations, which selected on the quantity being evaluated and applied a
TU-based selection to the AU and EU rows as well. The per-split
`reliability_*_bins.csv` caches and their SVGs remain unfiltered per-split
diagnostics; the aggregate is the reportable artifact. The water eval scripts
still pass `--trim 0.005`, since those experiments are out of scope for now.

What every figure in `experiment_X/evaluation/` shows, and which ones can back a
paper claim, is catalogued in
[`experiments/evaluation_figures.md`](experiments/evaluation_figures.md).

Evaluation programs live in [`eval/`](eval) and include:

- `epoch_quality.py`: epoch-wise correlation, AUSE, ENCE, uncertainty
  magnitude, RMSE, and Gaussian NLL. It evaluates the full ensemble at epochs
  available for every member, and applies no trimming.
- `epoch_quality_finetune_mixed_dataset.py`: concatenated low-fidelity pretrain
  and high-fidelity fine-tune trajectories evaluated on their respective
  targets.
- `epoch_quality_finetune_same_dataset.py`: retained as a shared implementation
  dependency of the mixed version, but no experiment script runs the same-target
  evaluation. Legacy `epoch_quality_finetune_same*.csv` caches are preserved and
  skipped by aggregation.
- `reliability.py`: calibrated or uncalibrated RMV-versus-RMSE reliability
  diagrams, and the per-configuration prediction caches every other analysis
  reuses.
- `train_curves.py`, `epoch_raw.py`, `distribution.py`, `finetune.py`, and
  `energy_ood.py`: supporting loss, uncertainty, distribution, transfer, and
  OOD analyses.
- `aggregate_replicates.py` and `replicate_statistics.py`: five-split
  aggregation and Student-t confidence intervals.

Epoch-quality outputs retain both common fixed-scale plots and autoscaled
`-free-scale` plots. Water reliability plots use log-log axes with common limits
of `1e-4` to `1e-1`, allowing direct visual comparison across water
experiments.

### Evaluation caching

The expensive inference CSVs are cached independently for each dataset split:

```text
experiment_X/evaluation/cache/split_0/
...
experiment_X/evaluation/cache/split_4/
```

Rerunning an evaluation reuses compatible CSV caches and regenerates plots or
missing columns when possible. To force a clean evaluation for one experiment,
remove its split caches before running its `eval_X.sh` again:

```bash
rm -rf experiment_wA/evaluation/cache/split_{0,1,2,3,4}
experiment_wA/eval_wA.sh 6
```

Aggregate files in `evaluation/` are overwritten after all five fresh split
evaluations complete.

### Evaluating multiple experiments

Evaluate all three water experiments sequentially on one GPU:

```bash
GPU_ID=6 bash eval_water.sh
```

This runs `wA`, `wB`, and `wC`, and each experiment evaluates all five splits.

Evaluate all six ANI-1x experiment families with:

```bash
bash eval.sh
```

`eval.sh` launches the six experiment evaluations in parallel using its GPU
map: `A=6`, `B=7`, `C=5`, `D=4`, `E=3`, and `F=2`. Edit the `gpus` array in
`eval.sh` when a different assignment is required.

To regenerate only the final figures and aggregate tables from existing caches,
without model inference, training or a GPU:

```bash
bash eval.sh --plots-only
```

This handles A-F and water experiments with existing caches, plus available AL
aggregate reports. Experiments with no split-0 cache are announced and skipped;
incomplete five-split caches fail rather than silently using fewer splits.
Per-split diagnostic figures can be regenerated with the individual `eval_X.sh`
scripts (they reuse CSV caches). Finetuning epoch-quality figures are mixed-only.

### Recommended end-to-end order

1. Generate all five dataset splits.
2. Train all direct/pretraining experiments: `A`, `C`, `E`, `F`, `wA`, and
   `wC`.
3. Train fine-tuning experiments `B`, `D`, and `wB` using matching pretrained
   split/member models.
4. Run `./check_experiments.sh` and resolve every partial or missing member.
5. Run each experiment's `eval_X.sh <gpu_number>`.
6. Use the aggregate CSVs and plots in `experiment_X/evaluation/` for final
   comparisons and statistical reporting.

### Practical notes

- Dataset split seeds and ensemble member seeds have different roles. Split
  seeds define statistical replicates; member seeds define one ensemble within
  each replicate.
- Evaluation requires all 10 members of each requested split because EU is
  computed from the ensemble disagreement.
- Fine-tuning experiments require matching parent models from the same split
  and member seed.
- The water XYZ files contain periodic cells and PBC metadata and should not be
  converted to a non-periodic format before MACE training.
- Evaluation figures are SVG-only, with larger fonts. Old PNG/PDF/SVG figures
  may be removed without removing the CSV/JSON caches used to redraw them.

## ANI energy-OOD active learning

Isolated, single-round proof of concept under
[`active_learning/ani_energy/`](active_learning/ani_energy). Nothing in
`experiment_*`, `dataset/`, `eval/`, or the shared MACE training code is
modified, and all generated files live under
`active_learning/ani_energy/runs/`, which Git ignores. This study is a later
addition and is **not** part of the manuscript's main sections or listed
appendix figures; keep its claims separate unless the paper is extended.

### Five splits and confidence intervals

One command runs or resumes splits 0-4 sequentially, evaluates all eight
conditions per split, then writes the aggregate reports and plots:

```bash
conda activate mace
bash active_learning/ani_energy/run_all.sh 2
```

Replace `2` with the GPU number. The default batch compares **Random-500,
AU-500, EU-500 and TU-500 for both HF-only and LF->HF**. All eight conditions
use AL learning rate **0.001** and at most **100 additional epochs** per member
(the code default; the completed scientific runs used `--epochs 50`). Across
five splits this trains 400 members sequentially, with no background training
jobs. The single Random-500 batch is shared between regimes and is the baseline
for AU, EU and TU.

Default run directories are `runs/split_0/` ... `runs/split_4/` with aggregate
`runs/aggregate/`. An explicit `--epochs N` switches to
`runs/split_<seed>_epochs_<N>/` and `runs/aggregate_epochs_<N>/`, keeping
budgets separate; use the same option when resuming. The existing completed runs
are therefore `split_0_epochs_50` ... `split_4_epochs_50` with
`aggregate_epochs_50`. Use `--run-tag NAME` for an independently named repeat.

```bash
bash active_learning/ani_energy/run_all.sh 2 --epochs 20
```

Completed split reports are reused without launching training or GPU inference,
and incomplete splits resume through the single-split runner. Do not start two
runners for the same splits. If the splits are already running, use wait-only
mode, which uses no GPU and polls every 60 seconds until all five reports exist
and their locks are released:

```bash
bash active_learning/ani_energy/run_all.sh 2 --aggregate-only --wait
```

Omit `--wait` to require all reports immediately. To aggregate completed reports
only, with no training or GPU inference:

```bash
bash active_learning/ani_energy/run_all.sh <gpu> --aggregate-only --epochs 50
```

Outputs are `summary_ci95.{csv,json,md}` in the aggregate directory. They record
all five split values, their mean and an approximate 95% Student-t interval,
`mean +/- 2.776445105 * sample_SD / sqrt(5)`. The unit of repetition is the
dataset split, not an ensemble member. Improvements, AU/EU/TU-over-random gains
and the LF->HF-minus-HF-only gain are computed **within** each split before
aggregation. Missing reports, mismatched settings and non-finite statistics fail
validation; undefined percentage improvements yield no CI rather than a
smaller-sample CI. Reports are cached by input hashes. Overlapping datasets and
only five repetitions limit the independence and normality assumptions: these
are approximate intervals, not an automatic significance claim.

`--epochs N` and `--common-evaluator` must match across all five runs. The
acquisition RNG defaults to 0 and is recorded in metadata for reproducibility;
it is not another layer of repetitions. Ensemble member seeds remain 0-9.

### Graphical results

`run_all.sh` invokes `plot_results.py` after aggregation, including in
`--aggregate-only` mode, writing SVGs to `<aggregate>/plots/`:

- `rmse`: before acquisition, Random-500, AU-500, EU-500 and TU-500, for both
  regimes and both tests.
- `improvement`: relative improvements from the within-split baseline.
- `acquisition_gain`: paired AU/EU/TU-over-random gains, in meV/atom and
  percentage points.
- `regime_contrast`: LF->HF's gain minus HF-only's gain, per signal.
- `common_evaluator`: the optional control, when available.

Diamonds are means with approximate 95% Student-t intervals; grey points are the
five individual split values. No split is dropped and no value is trimmed by its
error; the geometric support filter below is the only exclusion, and it never
reads predictions. OOD RMSE defaults to a logarithmic axis with a **geometric
mean and log-space Student-t interval**, which changes the plotted estimator and
not merely the axis. Both OOD panels fall back to linear arithmetic intervals if
any value is nonpositive or missing; no epsilon, clipping or dropped split is
used. ID RMSE, improvements, gains and contrasts stay linear and arithmetic,
since signed quantities are not log-transformed. Scales are shared between
regimes within a test. `summary_ci95.*` keeps its arithmetic statistics, and
`plots/rmse_display.json` records the displayed estimates with explicit
estimator labels.

To redraw only the plots (no GPU), or to choose a linear OOD RMSE axis:

```bash
python -B active_learning/ani_energy/plot_results.py
python -B active_learning/ani_energy/plot_results.py --ood-rmse-scale linear
```

`plot_results.py` defaults to `runs/aggregate/`, so pass `--summary` explicitly
for an `--epochs` run. Plot caching fingerprints the aggregate report, the plot
code and the display settings.

### Geometric support filter

Held-out Energy-OOD contains configurations compressed far below anything in the
training data: minimum interatomic distances reach **0.646 A**, while no training
file goes below about **0.83 A**. The models extrapolate catastrophically there.
Because RMSE averages squares, one such configuration can supply over 99% of a
reported OOD RMSE, and which of them land in the held-out half is close to a coin
flip. That produced the very wide split-to-split scatter in the raw
`aggregate_epochs_50` plots; it is an extrapolation artifact, not variance between
experimental repetitions.

`support_filter.py` recomputes every metric restricted to the region where the
models have training support:

```bash
conda activate mace
python -B active_learning/ani_energy/support_filter.py --output aggregate_epochs_50
python -B active_learning/ani_energy/plot_results.py \
    --summary active_learning/ani_energy/runs/aggregate_epochs_50/summary_ci95.json
```

For each dataset split the threshold is the smallest interatomic distance
occurring anywhere in **that split's own** `cc_train`, `cc_val`, `dft_train` and
`dft_val` (0.825-0.846 A). Held-out configurations below it are excluded. This
removes 6-10 of ~2500 held-out Energy-OOD configurations per split (~0.3%) and
0-2 Energy-ID configurations.

The rule reads training geometries alone. It never inspects predictions, errors
or uncertainties, and the same configurations are removed for every condition,
regime and test within a split, so the paired within-split improvements and gains
remain valid. The result is insensitive to the threshold: any cut between 0.80
and 1.00 A gives the same answer to within 0.4 percentage points, even though
1.00 A discards 37% of the test set. Only the handful of configurations below
0.80 A change anything.

Effect on the five-split Energy-OOD statistics:

| Metric | Raw | Support-filtered |
|---|---|---|
| LF->HF TU improvement | 20.3% +/- 74.8 | 14.2% +/- 3.7 |
| LF->HF EU gain over random | 699 [-558, 1957] meV/atom | 2.50 [1.89, 3.11] meV/atom |
| LF->HF pre-acquisition RMSE | 2777 meV/atom | 58.9 meV/atom |
| HF-only pre-acquisition RMSE | 148 meV/atom | 75.5 meV/atom |

All six AU/EU/TU-over-random OOD gains have intervals excluding zero after
filtering; none did before. Energy-ID results are unchanged to within 0.0003
meV/atom, since the ID test contains almost nothing below the support bound.

Per-split runs, cached predictions, models and metrics are never modified. The
cached predictions are verified against the SHA-256 hashes recorded in each run's
`metrics/` before use, and no model inference, training or GPU is required.
`runs/aggregate_epochs_50/support_filter.json` records the per-split threshold
and every excluded ID.

> **The support-filtered results currently occupy the default aggregate
> directory, `runs/aggregate_epochs_50/`, which `run_all.sh` and
> `five_splits.py` also write. Those write the *unfiltered* aggregate and will
> silently overwrite the filtered summary and plots.** After any `run_all.sh`
> invocation, including `--aggregate-only`, rerun the two commands above to
> restore the filtered results. The raw aggregate is never lost: it regenerates
> in seconds from the five per-split reports.

Report both the raw and the support-filtered numbers, and state the restriction
as a domain-of-validity caveat: these models are not characterized below the
smallest interatomic distance they were trained on. The same rule, applied to
the main A-F evaluations, is implemented in `new_figures/common.py`.

**Unrelated earlier exclusion.** `split_3_epochs_50` additionally had the single
configuration `C1H5N1:887` deleted from `data/heldout_ood.xyz`, `data/split.json`
and its ten cached Energy-OOD prediction files before this filter existed, so that
split holds 2499 rather than 2500 held-out configurations. Its minimum interatomic
distance is 0.646 A, so the support filter would exclude it anyway and the filtered
split-3 numbers are identical either way (2493 retained with or without it). The
raw split-3 values, however, are conditional on that deletion, and no backup of the
original predictions exists.

### Running a single split

```bash
conda activate mace
bash active_learning/ani_energy/run.sh 7
```

This prepares the data, scores the pool, selects and reveals labels, trains all
eight 10-member ensembles **sequentially**, evaluates them and writes the report.
GPU 7 is exposed as local `cuda:0`; `PYTHON=/path/to/python` overrides the
interpreter. No ANI HDF5 file, original evaluation cache or new dependency is
needed; the energy-split XYZ files, D/F checkpoints and training logs are.

Defaults: dataset split 0, B=500, 10 members (seeds 0-9), 50/50 OOD pool/test
partition, output `runs/split_0/`. Other splits use separate directories:

```bash
bash active_learning/ani_energy/run.sh 7 --split-seed 1
```

### Stages and restarting

```bash
# CPU-only preparation and freezing of initial checkpoints.
python -B active_learning/ani_energy/workflow.py --stage prepare --device cpu

bash active_learning/ani_energy/run.sh 7 --stage acquire
bash active_learning/ani_energy/run.sh 7 --stage train
bash active_learning/ani_energy/run.sh 7 --stage evaluate
python -B active_learning/ani_energy/workflow.py --stage report
```

Rerunning the same command reuses completed work. Predictions, selections,
augmented datasets and completed member models are cached, and source/artifact
SHA-256 checks prevent silently mixing datasets, code, checkpoints or settings.
Changing the inference device or batch size after scoring requires a new
`--name`. Pass the same split and optional epoch override to every stage, and
keep acquisition RNG settings unchanged so persisted batches stay reproducible.

Interrupted member training restarts that member from its frozen initial model
with a fresh optimizer in a new `attempt_NNN/`; completed members are skipped and
old attempts are retained. The runner locks a run directory, so do not run two
processes against the same run concurrently.

For manual scheduling, train selected cases and members in successive calls:

```bash
bash active_learning/ani_energy/run.sh 7 --stage train --case hf_only_random --members 0 1
bash active_learning/ani_energy/run.sh 6 --stage train --case hf_only_random --members 2 3
```

Available cases: `hf_only_random`, `hf_only_au`, `hf_only_eu`, `hf_only_tu`,
`lf_hf_random`, `lf_hf_au`, `lf_hf_eu`, `lf_hf_tu`. Evaluation requires all 10
members of the requested case; `--case` also works with `--stage evaluate`, and
the final report requires all eight default cases. Follow live progress with
`tail -f` on the printed member `console.log` path.

### Experimental choices

| Regime | Initial ensemble | Additional training protocol |
|---|---|---|
| HF-only | `experiment_F/checkpoints_<split>` | AL LR 0.001, up to 100 epochs |
| LF->HF | `experiment_D/checkpoints_<split>` | AL LR 0.001, up to 100 epochs |

**Both regimes start post-acquisition training at LR 0.001**, overriding the
learning rate inherited from their source configs; this also applies to the
optional common-evaluator control. The schedulers can subsequently reduce the LR.
Original F/D training configs are unchanged, and other training settings are
inherited from them. Manifests record the shared AL learning rate and effective
epoch limit, and the runner raises a protocol-mismatch error rather than
silently reusing models trained under different settings. It never deletes,
updates or retrains those models.

Initial checkpoints are selected independently per member by **minimum original
HF validation loss**, exactly as in `eval_D.sh` / `eval_F.sh`. The selected
checkpoint and its companion architecture are frozen as byte-identical AL-local
copies, then loaded strictly for acquisition and training. LF->HF already
contains the LF pretraining from experiment C, so C need not be evaluated or
retrained first.

Both acquisition branches of a regime start from identical member weights and
use the same original HF validation set, member seeds, Gaussian NLL loss,
optimizer parameter groups, batching, scheduler, clipping and epoch limit.
Training calls the existing `mace.tools.train` implementation with all
parameters trainable. Mean/variance heads, E0s and normalization buffers are
preserved; the model is not rebuilt through the foundation-model CLI.
Optimizer/scheduler state is deliberately reset for this round, identically
across methods. No OOD or ID test loader is passed to training.

**Epoch budgets and checkpoint selection are different things.** Before
acquisition, each of the 10 members independently uses its best original HF
validation checkpoint, so their epoch numbers can differ. After acquisition,
each member trains on the original HF training data plus the selected 500
configurations, and its best additional-training checkpoint is again selected
independently by minimum validation Gaussian NLL (not validation RMSE) and saved
as `best.model`. The same selected ensemble is evaluated on both the held-out
OOD test and the existing ID test; neither test influences selection, and
evaluation never falls back to the last epoch or to a common epoch. The 5
dataset splits, not the 10 members, are the replicates behind the intervals.

### Data and hidden labels

- Only `cc_test_ood.xyz` of the selected ANI energy split is repartitioned. With
  5000 configurations, 2500 form the acquisition pool and 2500 the held-out test.
  The original file is never changed.
- The existing ANI splitter's quota allocator and sampler preserve the source OOD
  system composition, using proportional half-sampling per system and seeded
  remainder allocation. Singleton systems cannot appear in both halves. No new
  energy threshold or label-based ranking is introduced.
- A system contributing 100 original OOD configurations supplies 50 randomly
  chosen configurations to the pool and 50 to the held-out test; odd counts use
  reproducible rounding to keep the pool at 2500. **The held-out test is not the
  higher-energy half of OOD.** Both halves sample the same original OOD energy
  domain, and pool energies are not systematically lower.
- All original LF/HF train, validation and ID-test XYZ files remain unchanged.
  Preparation checks that OOD IDs do not overlap any of those six inputs.
- Stable IDs are `system:conf_idx`; pool and held-out IDs, source indices and
  system counts are saved. The held-out set is never acquired or appended.
- Public `pool.xyz` contains geometry, cell/PBC and IDs only: no CC/DFT energies,
  forces, calculators, relative energies or quantiles. The offline CC oracle is
  stored separately in `data/oracle.json` and contains **pool IDs only**.
- Acquisition uses only public geometries and model predictions. The shared
  evaluator's missing-label adapter uses a zero-weight placeholder, not CC labels;
  placeholder errors and references are discarded and never used for selection.
- Selection is persisted **before** the oracle is read to append the chosen 500
  labels. Augmented files start from an unchanged copy of the original HF training
  file, and new configurations use its training `config_type` and equal weights.
- Random acquisition is uniform without replacement and shared between regimes.
  Each AU/EU/TU acquisition is the global top 500 of its own score (stable ID
  tie-break), not system-capped. Selected batches may overlap: they are
  alternative single-round conditions, not sequential acquisitions.

```text
Original low-energy train / validation / ID test: unchanged
Original CC Energy-OOD file: 5000 configurations
    +-- 2500 acquisition-pool configurations (random, stratified by system)
    |       +-- acquire 500 by AU, EU, TU or random sampling; append their CC labels
    +-- 2500 held-out OOD test configurations (never acquired or trained on)
```

This tests acquisition within the original OOD domain. It does not test
acquisition from a lower-energy pool followed by testing on an even higher-energy
domain; that would require a separate partition.

### Acquisition scores and RMSE

The runner reuses `eval/reliability.py` for prediction and uncertainty:

```text
AU = mean_m(variance_m / N_atoms^2)
EU = population_variance_m(energy_mean_m / N_atoms)    # unbiased=False
TU = AU + EU
```

Each score is **one scalar per configuration**, from the same cached ensemble
predictions, with no extra inference per signal. `energy_mean_m` and
`variance_m` are member `m`'s predicted total configuration energy and its
variance, and `N_atoms` is that configuration's atom count; "per-atom" refers
only to the `N_atoms^2` normalization, not to per-atom acquisition decisions.
The 2500 whole configurations are ranked and 500 whole configurations selected.

Unnormalized total-energy TU would be
`mean_m(variance_m) + population_variance_m(energy_mean_m)`, i.e. the current
score times `N_atoms^2`. Both give one score per configuration but can rank
differently when atom counts differ. The current normalization matches the
project's uncertainty and energy-per-atom RMSE definitions; switching would be a
different acquisition experiment, not a change of terminology.

There is no calibration, trimming, error-based selection or label-based
filtering. Non-finite values fail the run instead of silently excluding
configurations.

All conditions use the same held-out Energy-OOD test and the unchanged
`cc_test_id.xyz`. RMSE is that of the **ensemble-mean energy per atom**, averaged
equally over configurations, not an average of member RMSEs. Units are meV/atom.
Individual predictions and AU/EU/TU are retained in the inference caches.

```text
relative improvement (%) = 100 * (RMSE_before - RMSE_after) / RMSE_before
method gain (meV/atom)   = RMSE_random - RMSE_method
method gain (pp)         = improvement_method - improvement_random
# method is AU, EU or TU
```

Positive gain favors the uncertainty method over random. The report also gives
LF->HF's gain minus HF-only's gain for each method and test: positive values
support the acquisition-quality hypothesis. Negative improvements and gains are
retained. A zero baseline RMSE produces `null` percentage improvements, not a
division by zero.

### Optional common evaluator

```bash
bash active_learning/ani_energy/run.sh 7 --common-evaluator
```

This adds `lf_hf_from_hf_only_tu`: the same D initial ensemble and protocol
trained on the HF-only TU batch. Its comparator is the existing `lf_hf_tu`
branch, so the control adds 10 members rather than 20. It is reported separately
under `common_evaluator/`, can be enabled after the default run finishes, and
compares TU acquisitions only. Include `--common-evaluator` in the subsequent
train/evaluate/report stages.

### AL artifacts

```text
runs/<name>/
  manifest.json                 settings, source hashes, versions, checkpoint provenance
  data/                         public pool, held-out OOD, oracle, split/ID manifest
  initial/{hf_only,lf_hf}/      frozen 10-member ensembles, configs and source logs
  inference/                    cached pool and test predictions, input fingerprints
  acquisition/{random,hf_only_au,hf_only_eu,hf_only_tu,lf_hf_au,lf_hf_eu,lf_hf_tu}/
    selection.json              selected IDs, method, seed, score provenance
    train.xyz                   original HF train + 500 revealed configurations
    augmented.json              counts and file hashes
  cases/<case>/member_<seed>/
    attempt_NNN/                local config, resolved arguments, logs, checkpoints, best model
    complete.json               validated completion/cache marker
  metrics/                      per-condition/test RMSE, counts and provenance
  report/summary.{csv,json,md}  default comparisons and optional control
```

Each attempt retains every epoch checkpoint, matching the main protocol; budget
disk space accordingly. Original checkpoint and config sources are read-only. No
W&B run, shared evaluation cache or existing result directory is written.

### AL scope and verification

Each `run.sh` invocation is a single-round, single-split POC. `run_all.sh`
aggregates five such splits with approximate confidence intervals, not a blanket
significance claim. The 10 members form one uncertainty estimator, not 10
independent AL repetitions. GPU floating-point kernels can prevent bitwise
equality across machines despite the recorded seeds, versions and hashes.

Some existing companion models contain CUDA-serialized TorchScript blocks, so
even a CPU `torch.load` of them requires an available GPU. Preparation copies
bytes without deserializing models and works on CPU; run acquisition, training
and evaluation in a GPU-enabled session for those checkpoints.

The pool comes from an already labeled, CC-available, previously used OOD
dataset. Labels are hidden from this workflow, not retroactively from past
experiments or from dataset construction. Treat the result as an offline POC,
not a pristine prospective benchmark. The common-evaluator control helps
separate acquisition quality from downstream-regime differences.

Focused CPU tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B active_learning/ani_energy/test_workflow.py
python -B active_learning/ani_energy/test_five_splits.py
python -B active_learning/ani_energy/test_reporting.py
bash -n active_learning/ani_energy/run.sh
bash -n active_learning/ani_energy/run_all.sh
```

They cover deterministic balanced partitioning, hidden-label invariance, oracle
reveal and leakage guards, selection budgets and ties, cache invalidation,
reporting formulas, the shared TU calculation, and one epoch of genuine small
MVE checkpoint continuation through the existing MACE training loop. A separate
orchestration test uses synthetic predictions and a stub trainer to verify all
eight branches, resume behavior, shared tests and the optional control; it
produces no scientific results. Reporting tests cover folder names, recursive
path/hash migration, unchanged numerical values, paired confidence intervals and
nonblank/cached plots.

## Audit notes and known issues

Pipeline audit of 2026-09-22, from dataset generation to figures. Checks were
run against the current working tree; no file was modified by the audit.

**Verified sound**

- **Split integrity.** Across system and energy splits (seeds 0 and 3 checked
  exhaustively by `system:conf_idx`): no configuration is shared between train,
  validation, ID test and OOD test; every CC file is an exact subset of the DFT
  file with the same role; system-split OOD systems are disjoint from the seen
  systems (0 of 1460 shared); energy-split OOD configurations lie above every
  seen-domain configuration of the same system (0 violations in 2148 systems).
- **MVE head and loss** as described in
  [MVE model and uncertainty definitions](#mve-model-and-uncertainty-definitions).
- **AU/EU/TU** match the documented formulas, including per-atom normalization
  (`var / N^2`) and the population variance for EU.
- **ENCE** matches manuscript Eq. 8 exactly, with 15 equal-count bins.
- **AUSE** is the standard normalized sparsification-error area.
- **Fine-tuning** copies body, readouts and scale/shift buffers from the matching
  parent member and re-fits E0s on the high-fidelity training set; the launcher
  refuses to start without the matching split/member parent.
- **Checkpoint selection** uses validation Gaussian NLL only, per member.
- **Completeness.** ANI `A`-`F`: 300/300 members, 5/5 split caches each.
- **Tests.** `eval/test_plotting` (6), AL `test_workflow` (8),
  `test_five_splits` (11), `test_reporting` (5) all pass.

**Open issues (water only; out of scope for the current paper)**

1. **Water fidelity splits are not aligned.**
   `dataset/convert_water_n2p2_to_extxyz.py` shuffles each theory folder with its
   own derived seed, and the theories hold different configuration counts. In
   `water_0`, 37 of the 50 CCSDT test geometries are also in the BLYP training
   set (and 80 of 130 CCSDT training geometries are). `wB` is therefore
   fine-tuned from a `wA` model that saw most of `wB`'s own test geometries at
   low fidelity, while `wC` saw none. Any `wB`-versus-`wC` comparison on the
   water ID test is biased in favour of the multi-fidelity protocol. Fixing this
   requires one shared configuration-level split reused by all theories, and
   retraining water.
2. **Water has five splits (resolved 2026-09-23).** All 150 members are trained and
   evaluated; the overlap above holds in every split (31-37 of 50 CCSDT test
   geometries in BLYP train; `results/tables/water_split_overlap.txt`).

The water eval scripts still pass `--trim 0.005` and are not support-filtered,
because water has no ANI geometry cache and is excluded from the current paper.

**Manuscript-side, not fixable in code**

3. **EU denominator.** The code uses the population variance (`unbiased=False`,
   denominator `M`), while Eq. 3 of the current manuscript shows `M-1`. Align the
   text with the code (the code is the intended estimator).
4. **Fig. 4 panel rows are rotated in the ID and OOD columns.** Extracting the
   plotted points from the PDF and correlating them with
   `experiment_B/evaluation/cache/split_0/epoch_quality_finetune_mixed_*.csv`
   gives `r = 1.000` for: row labelled "Spearman" = AUSE, row labelled "AUSE" =
   ENCE, row labelled "ENCE" = Spearman. Those panels also carry the train
   column's tick labels while using their own range. The train column is correct
   and the underlying data match the caches exactly, so only the figure and its
   panel references need redrawing.
5. **Split-3 AL caveat.** `C1H5N1:887` was deleted post hoc from split 3's
   held-out test and its cached predictions; raw split-3 numbers are conditional
   on that deletion, with no backup. The support filter would exclude it anyway.

**Fixed on 2026-09-22**

- **One population for every reliability metric.** The TU trim is off for A-F
  (`--trim 0.0`), and `eval/aggregate_replicates.py` now rebuilds the reliability
  bins from support-filtered per-configuration rows, so RMSE, Spearman, AUSE and
  ENCE describe the same configurations. Previously ENCE came from trimmed bins
  while the other three came from untrimmed rows. The filter itself moved into
  the shared `eval/support_filter.py`, which `new_figures/` also imports, and the
  regenerated aggregates agree with the independently written figure code to all
  printed digits (experiment D, Energy-OOD: RMSE 58.689 meV/atom, TU AUSE
  0.143451, TU ENCE 3.071831, TU Spearman 0.640044).
  The Energy-OOD tables change substantially, as expected: experiment D's
  five-split OOD RMSE goes from 10.00 +/- 16.09 eV/atom, dominated by a few
  out-of-support geometries, to 0.0587 +/- 0.0040 eV/atom, and its TU AUSE from
  0.044 to 0.143. System-split numbers move only slightly (experiment B OOD TU
  ENCE 0.313 -> 0.324), since the filter removes nothing there and only the trim
  was dropped.
- **MVE variance in the unused `MACE` class.** It treated the head output as a
  log-variance, summed log-variances over atoms (a product of per-atom variances)
  and gave each deterministic contribution `exp(0) = 1`, a constant variance floor
  of about 3 energy units squared. It now matches `ScaleShiftMACE`: softplus
  per-atom variances, summed over atoms, deterministic terms exactly zero.
  Verified by running both classes on identical weights and inputs: identical
  `energy_var`, equal to the sum of the per-atom variances. No trained model was
  affected, since every experiment config builds `ScaleShiftMACE`.
- **AL aggregate overwrite hazard.** `five_splits.py` now refuses to replace a
  support-filtered aggregate with a raw one, naming the command to refresh it;
  `--overwrite-filtered` is the deliberate escape hatch.

[![GitHub release](https://img.shields.io/github/release/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/releases/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPs2022-blue)](https://openreview.net/forum?id=YPpSngE-ZU)
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/issues/)
[![Documentation Status](https://readthedocs.org/projects/mace/badge/)](https://mace-docs.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/505964914.svg)](https://doi.org/10.5281/zenodo.14103332)

## Table of contents

- [MACE](#mace)
  - [Table of contents](#table-of-contents)
  - [About MACE](#about-mace)
  - [Documentation](#documentation)
  - [Installation](#installation)
    - [pip installation](#installation-from-pypi)
    - [pip installation from source](#installation-from-source)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Tutorials](#tutorials)
  - [CUDA acceleration with cuEquivariance](#cuda-acceleration-with-cuequivariance)
  - [Weights and Biases for experiment tracking](#weights-and-biases-for-experiment-tracking)
  - [Pretrained Foundation Models](#pretrained-foundation-models)
    - [MACE-MP: Materials Project Force Fields](#mace-mp-materials-project-force-fields)
      - [Example usage in ASE](#example-usage-in-ase)
    - [MACE-OFF: Transferable Organic Force Fields](#mace-off-transferable-organic-force-fields)
      - [Example usage in ASE](#example-usage-in-ase-1)
    - [Finetuning foundation models](#finetuning-foundation-models)
    - [Latest recommended foundation models](#latest-recommended-foundation-models)
  - [Caching](#caching)
  - [Development](#development)
  - [References](#references)
  - [Contact](#contact)
  - [License](#license)

## About MACE

MACE provides fast and accurate machine learning interatomic potentials with higher order equivariant message passing.

This repository contains the MACE reference implementation developed by
Ilyes Batatia, Gregor Simm, David Kovacs, and the group of Gabor Csanyi, and friends (see Contributors).

Also available:

- [MACE in JAX](https://github.com/ACEsuit/mace-jax), currently about 2x times faster at evaluation, but training is recommended in Pytorch for optimal performances.
- [MACE layers](https://github.com/ACEsuit/mace-layer) for constructing higher order equivariant graph neural networks for arbitrary 3D point clouds.

## Documentation

A partial documentation is available at: https://mace-docs.readthedocs.io

## Installation

### 1. Requirements

- Python >= 3.7  (for openMM, use Python = 3.9)
- [PyTorch](https://pytorch.org/) >= 1.12 **(training with float64 is not supported with PyTorch 2.1 but is supported with 2.2 and later, Pytorch 2.4.1 is not supported)**

**Make sure to install PyTorch.** Please refer to the [official PyTorch installation](https://pytorch.org/get-started/locally/) for the installation instructions. Select the appropriate options for your system.

### Installation from PyPI

This is the recommended way to install MACE.

```sh
pip install --upgrade pip
pip install mace-torch
```

**Note:** The homonymous package on [PyPI](https://pypi.org/project/MACE/) has nothing to do with this one.

### Installation from source

```sh
git clone https://github.com/ACEsuit/mace.git
pip install ./mace
```

## Usage

### Training

To train a MACE model, you can use the `mace_run_train` script, which should be in the usual place that pip places binaries (or you can explicitly run `python3 <path_to_cloned_dir>/mace/cli/run_train.py`)

```sh
mace_run_train \
    --name="MACE_model" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --stage_two \
    --start_stage_two=1200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
```

To give a specific validation set, use the argument `--valid_file`. To set a larger batch size for evaluating the validation set, specify `--valid_batch_size`.

To control the model's size, you need to change `--hidden_irreps`. For most applications, the recommended default model size is `--hidden_irreps='256x0e'` (meaning 256 invariant messages) or `--hidden_irreps='128x0e + 128x1o'`. If the model is not accurate enough, you can include higher order features, e.g., `128x0e + 128x1o + 128x2e`, or increase the number of channels to `256`. It is also possible to specify the model using the     `--num_channels=128` and `--max_L=1`keys.

It is usually preferred to add the isolated atoms to the training set, rather than reading in their energies through the command line like in the example above. To label them in the training set, set `config_type=IsolatedAtom` in their info fields. If you prefer not to use or do not know the energies of the isolated atoms, you can use the option `--E0s="average"` which estimates the atomic energies using least squares regression. Note that using fitted E0s corresponds to fitting the deviations of the atomic energies from the average, rather than fitting the atomization energy (which is the case when using isolated-atom E0s), and this will most likely result in less stable potentials for molecular dynamics applications.

If the keyword `--stage_two` (previously called swa) is enabled, the energy weight of the loss is increased for the last ~20% of the training epochs (from `--start_stage_two` epochs). This setting usually helps lower the energy errors.

The precision can be changed using the keyword `--default_dtype`, the default is `float64` but `float32` gives a significant speed-up (usually a factor of x2 in training).

The keywords `--batch_size` and `--max_num_epochs` should be adapted based on the size of the training set. The batch size should be increased when the number of training data increases, and the number of epochs should be decreased. An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $\text{max-num-epochs}*\frac{\text{num-configs-training}}{\text{batch-size}}$.

The code can handle training set with heterogeneous labels, for example containing both bulk structures with stress and isolated molecules. In this example, to make the code ignore stress on molecules, append to your molecules configuration a `config_stress_weight = 0.0`.

By default, a figure displaying the progression of loss and RMSEs during training, along with a scatter plot of the model's inferences on the train, validation, and test sets, will be generated in the results folder at the end of training. This can be disabled using `--plot False`. To track these metrics throughout training (excluding inference on the test set), you can enable periodic plotting for the train and validation sets by specifying `--plot_frequency N`, which updates the plots every Nth epoch.

#### Apple Silicon GPU acceleration

To use Apple Silicon GPU acceleration make sure to install the latest PyTorch version and specify `--device=mps`.

#### Multi-GPU training

For multi-GPU training, use the `--distributed` flag. This will use PyTorch's DistributedDataParallel module to train the model on multiple GPUs. Combine with on-line data loading for large datasets (see below). An example slurm script can be found in `mace/scripts/distributed_example.sbatch`.

#### YAML configuration

Option to parse all or some arguments using a YAML is available. For example, to train a model using the arguments above, you can create a YAML file `your_configs.yaml` with the following content:

```yaml
name: nacl
seed: 2024
train_file: train.xyz
stage_two: yes
start_stage_two: 1200
max_num_epochs: 1500
device: cpu
test_file: test.xyz
E0s:
  41: -1029.2809654211628
  38: -1484.1187695035828
  8: -2042.0330099956639
config_type_weights:
  Default: 1.0

```

And append to the command line `--config="your_configs.yaml"`. Any argument specified in the command line will overwrite the one in the YAML file.

### Evaluation

To evaluate your MACE model on an XYZ file, run the `mace_eval_configs`:

```sh
mace_eval_configs \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"
```

## Tutorials

You can run our [Colab tutorial](https://colab.research.google.com/drive/1D6EtMUjQPey_GkuxUAbPgld6_9ibIa-V?authuser=1#scrollTo=Z10787RE1N8T) to quickly get started with MACE.

We also have a more detailed Colab tutorials on:

- [Introduction to MACE training and evaluation](https://colab.research.google.com/drive/1ZrTuTvavXiCxTFyjBV4GqlARxgFwYAtX)
- [Introduction to MACE active learning and fine-tuning](https://colab.research.google.com/drive/1oCSVfMhWrqHTeHbKgUSQN9hTKxLzoNyb)
- [MACE theory and code (advanced)](https://colab.research.google.com/drive/1AlfjQETV_jZ0JQnV5M3FGwAM2SGCl2aU)

## CUDA acceleration with cuEquivariance

MACE supports CUDA acceleration with the cuEquivariance library. To install the library and use the acceleration, see our documentation at https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html.

## On-line data loading for large datasets

If you have a large dataset that might not fit into the GPU memory it is recommended to preprocess the data on a CPU and use on-line dataloading for training the model. To preprocess your dataset specified as an xyz file run the `preprocess_data.py` script. An example is given here:

```sh
mkdir processed_data
python ./mace/scripts/preprocess_data.py \
    --train_file="/path/to/train_large.xyz" \
    --valid_fraction=0.05 \
    --test_file="/path/to/test_large.xyz" \
    --atomic_numbers="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]" \
    --r_max=4.5 \
    --h5_prefix="processed_data/" \
    --compute_statistics \
    --E0s="average" \
    --seed=123 \
```

To see all options and a little description of them run `python ./mace/scripts/preprocess_data.py --help` . The script will create a number of HDF5 files in the `processed_data` folder which can be used for training. There will be one folder for training, one for validation and a separate one for each `config_type` in the test set. To train the model use the `run_train.py` script as follows:

```sh
python ./mace/scripts/run_train.py \
    --name="MACE_on_big_data" \
    --num_workers=16 \
    --train_file="./processed_data/train.h5" \
    --valid_file="./processed_data/valid.h5" \
    --test_dir="./processed_data" \
    --statistics_file="./processed_data/statistics.json" \
    --model="ScaleShiftMACE" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --batch_size=32 \
    --valid_batch_size=32 \
    --max_num_epochs=100 \
    --stage_two \
    --start_stage_two=60 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --error_table='PerAtomMAE' \
    --device=cuda \
    --seed=123 \
```

## Weights and Biases for experiment tracking

If you would like to use MACE with Weights and Biases to log your experiments simply install with

```sh
pip install ./mace[wandb]
```

And specify the necessary keyword arguments (`--wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_name`, `--wandb_log_hypers`)

## Pretrained Foundation Models

We provide a series of pretrained foundation models for various applications. These models can be used directly for inference, or as a starting point for fine-tuning on a new dataset.
Foundation models are a rapidly evolving field. Please look at the [MACE-MP GitHub repository](https://github.com/ACEsuit/mace-foundations/releases) and the [MACE-OFF23 GitHub repository](https://github.com/ACEsuit/mace-off/releases) for the latest releases.

### Latest Recommended Foundation Models

| Model Name           | Elements Covered | Training Dataset | Level of Theory     | Target System     | Model Size                                                                                                                                                                                                                                                                                                                                                                        | GitHub Release | Notes                                                              | License |
| -------------------- | ---------------- | ---------------- | ------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------ | ------- |
| MACE-MP-0a           | 89               | MPTrj            | DFT (PBE+U)         | Materials         | [small](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model), [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model), [large](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model) | >=v0.3.6       | Initial release of foundation model.                               | MIT     |
| MACE-MP-0b3          | 89               | MPTrj            | DFT (PBE+U)         | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model)                                                                                                                                                                                                                                                                      | >=v0.3.10      | Improved high pressure stability and reference energies.           | MIT     |
| MACE-MPA-0           | 89               | MPTrj + sAlex    | DFT (PBE+U)         | Materials         | [medium-mpa-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model)                                                                                                                                                                                                                                                                  | >=v0.3.10      | Improved accuracy for materials, improved high pressure stability. | MIT     |
| MACE-OMAT-0          | 89               | OMAT             | DFT (PBE+U) VASP 54 | Materials         | [medium-omat-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model)                                                                                                                                                                                                                                                               | >=v0.3.10      |                                                                    | ASL     |
| MACE-OFF23           | 10               | SPICE v1         | DFT (wB97M+D3)      | Organic Chemistry | [small](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model), [medium](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_medium.model), [large](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model)                                                                                                    | >=v0.3.6       | Initial release covering neutral organic chemistry.                | ASL     |
| MACE-MATPES-PBE-0    | 89               | MATPES-PBE       | DFT (PBE)           | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model)                                                                                                                                                                                                                                                               | >=v0.3.10      | No +U correction.                                                  | ASL     |
| MACE-MATPES-r2SCAN-0 | 89               | MATPES-r2SCAN    | DFT (r2SCAN)        | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model)                                                                                                                                                                                                                                                            | >=v0.3.10      | Better functional for materials.                                   | ASL     |
| MACE-OMOL-0 | 89               | OMOL    | DFT (wB97M-VV10)        | Molecules/Transition metals/Cations         | [large](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model)                                                                                                                                                                                                                                                           | >=v0.3.14      | Charge/Spin embedding, very good molecular accuracy.                                   | ASL     |

### MACE-MP: Materials Project Force Fields

We have collaborated with the Materials Project (MP) to train a universal MACE potential covering 89 elements on 1.6 M bulk crystals in the [MPTrj dataset](https://figshare.com/articles/dataset/23713842) selected from MP relaxation trajectories.
The models are releaed on GitHub at https://github.com/ACEsuit/mace-foundations.
If you use them please cite [our paper](https://arxiv.org/abs/2401.00096) which also contains an large range of example applications and benchmarks.

> [!CAUTION]
> The MACE-MP models are trained on MPTrj raw DFT energies from VASP outputs, and are not directly comparable to the MP's DFT energies or CHGNet's energies, which have been applied MP2020Compatibility corrections for some transition metal oxides, fluorides (GGA/GGA+U mixing corrections), and 14 anions species (anion corrections). For more details, please refer to the [MP Documentation](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/thermodynamic-stability/anion-and-gga-gga+u-mixing) and [MP2020Compatibility.yaml](https://github.com/materialsproject/pymatgen/blob/master/pymatgen/entries/MP2020Compatibility.yaml).

#### Example usage in ASE

```py
from mace.calculators import mace_mp
from ase import build

atoms = build.molecule('H2O')
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

### MACE-OFF: Transferable Organic Force Fields

There is a series (small, medium, large) transferable organic force fields. These can be used for the simulation of organic molecules, crystals and molecular liquids, or as a starting point for fine-tuning on a new dataset. The models are released under the [ASL license](https://github.com/gabor1/ASL).
The models are releaed on GitHub at https://github.com/ACEsuit/mace-off.
If you use them please cite [our paper](https://arxiv.org/abs/2312.15211) which also contains detailed benchmarks and example applications.

#### Example usage in ASE

```py
from mace.calculators import mace_off
from ase import build

atoms = build.molecule('H2O')
calc = mace_off(model="medium", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

### Finetuning foundation models

To finetune one of the mace-mp-0 foundation model, you can use the `mace_run_train` script with the extra argument `--foundation_model=model_type`. For example to finetune the small model on a new dataset, you can use:

```sh
mace_run_train \
  --name="MACE" \
  --foundation_model="small" \
  --train_file="train.xyz" \
  --valid_fraction=0.05 \
  --test_file="test.xyz" \
  --energy_weight=1.0 \
  --forces_weight=1.0 \
  --E0s="average" \
  --lr=0.01 \
  --scaling="rms_forces_scaling" \
  --batch_size=2 \
  --max_num_epochs=6 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --default_dtype="float32" \
  --device=cuda \
  --seed=3
```

Other options are "medium" and "large", or the path to a foundation model.
If you want to finetune another model, the model will be loaded from the path provided `--foundation_model=$path_model`, all the hypers will be extracted automatically.

## Caching

By default automatically downloaded models, like mace_mp, mace_off and data for fine tuning, end up in `~/.cache/mace`. The path can be changed by using
the environment variable XDG_CACHE_HOME. When set, the new cache path expands to $XDG_CACHE_HOME/.cache/mace

## Development

This project uses [pre-commit](https://pre-commit.com/) to execute code formatting and linting on commit.
We also use `black`, `isort`, `pylint`, and `mypy`.
We recommend setting up your development environment by installing the `dev` packages
into your python environment:

```bash
pip install -e ".[dev]"
pre-commit install
```

The second line will initialise `pre-commit` to automaticaly run code checks on commit.
We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.

We are happy to accept pull requests under an [MIT license](https://choosealicense.com/licenses/mit/). Please copy/paste the license text as a comment into your pull request.

## References

If you use this code, please cite our papers:

```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr.

For bugs or feature requests, please use [GitHub Issues](https://github.com/ACEsuit/mace/issues).

## License

The MACE code is published and distributed under the [MIT License](MIT.md). (Note that some of the models linked above come with different licenses).
