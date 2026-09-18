# ANI Energy-OOD Active Learning

Isolated, single-round proof of concept. Nothing in `experiment_*`, `dataset/`,
`eval/`, or the shared MACE training code is modified. All generated files live
under `active_learning/ani_energy/runs/` and are ignored by Git.

## Five Splits and Confidence Intervals

One command runs/resumes splits 0..4 sequentially on GPU 2, evaluates all eight
conditions for each split, then produces the aggregate reports and plots:

```bash
conda activate mace
bash active_learning/ani_energy/run_all.sh 2
```

Replace `2` with the desired GPU number. The default batch compares **Random-500,
AU-500, EU-500 and TU-500 for both HF-only and LF->HF**. All eight conditions use
initial LR **0.001** and at most **100 additional epochs** per member. Across five
splits this trains **400 members**, sequentially (no background training jobs),
then aggregates the five-split results and generates the plots automatically.

Default run directories are `runs/split_0/` through `runs/split_4/`, with no
version suffix. The old AL results were removed before this fresh run.
TU and Random are trained again from the original selected D/F
checkpoints, not from the previous post-AL models. The pool/held-out partition
and acquisition RNG are unchanged, so the dataset setup remains comparable.
The single Random-500 batch is shared across regimes and serves as the baseline
for AU, EU and TU; no redundant Random training is needed for each metric.

For an independently named repeat use `--run-tag NAME`; by default there is no
tag. Rerunning the same command resumes the matching batch.

To choose the additional epoch limit for every member of both regimes:

```bash
bash active_learning/ani_energy/run_all.sh 2 --epochs 20
```

Replace `20` with the desired positive integer. Omitting `--epochs` uses 100.
Explicit overrides use `runs/split_<seed>_epochs_<N>/` and
`runs/aggregate_epochs_<N>/`, keeping different budgets separate. Use the same
option when resuming. The old HF-only AL protocol used 300 additional epochs;
that explains its 300 epoch checkpoints. Original F pretraining-from-scratch
checkpoints are separate and are not affected by the AL epoch option.

Completed split reports are reused without launching training or GPU inference.
Incomplete splits resume through the existing single-split runner.
Do not start this training command alongside another runner for the same splits.
If the splits are **already running**, leave them running and use wait-only mode:

```bash
bash active_learning/ani_energy/run_all.sh 2 --aggregate-only --wait
```

Wait-only mode does not use the GPU or launch training/evaluation. It polls every
60 seconds until all five reports are present and their run locks are released.
It also waits for split 0 if that run is on a different GPU. If a training job
fails, restart that job separately; the watcher keeps waiting and can be stopped
with Ctrl-C. Omit `--wait` to require all reports immediately.

Outputs: `runs/aggregate/summary_ci95.{csv,json,md}`. They include all five
individual values, their mean and approximate 95% Student-t confidence interval:
`mean +/- 2.776445105 * sample_SD / sqrt(5)`. The unit of repetition is the dataset
split, not an ensemble member. Improvements, AU/EU/TU-over-random gains, and the
LF->HF-minus-HF-only gain are computed within each split before aggregation.
Missing reports, mismatched settings and non-finite statistics fail validation;
undefined percentage improvements yield no CI rather than a smaller-sample CI.
Reports are cached by input hashes. Overlapping datasets and only five repetitions
limit the independence/normality assumptions: these are approximate intervals,
not an automatic statistical-significance claim.

`--epochs N` and `--common-evaluator` must match across all five runs.
The acquisition RNG defaults to 0 and is retained in metadata for reproducibility;
it is not another layer of experimental repetitions or folder names. The legacy
`--seed` option remains available, but changing it requires separate runs rather
than mixing old and new acquisitions. Ensemble member seeds remain 0..9.
The optional control has separate aggregate output under `common_evaluator/`.
Default runs are `runs/split_0/` through `runs/split_4/`. Their ten members remain
under `cases/<condition>/member_0/` through `member_9/`. Confidence intervals use
the five splits, not the members.

### Graphical Results

`run_all.sh` automatically invokes `plot_results.py` after aggregation, including
in `--aggregate-only` mode. The `runs/aggregate/plots/` folder contains SVG
versions of:

- `rmse`: before acquisition, Random-500, AU-500, EU-500 and TU-500, for both regimes and tests.
- `improvement`: relative improvements from the within-split baseline.
- `acquisition_gain`: paired AU/EU/TU-over-random gains in meV/atom and percentage points.
- `regime_contrast`: LF->HF's gain minus HF-only's gain, separately for AU/EU/TU.
- `common_evaluator`: the optional control, when those results are available.

Diamonds show means with approximate 95% Student-t intervals; grey points show
all five individual split values. No outliers are removed. OOD RMSE defaults to
a logarithmic axis with a **geometric mean and log-space Student-t interval**:
`exp(mean(log(RMSE)) +/- 2.776445105 * sample_SD(log(RMSE)) / sqrt(5))`.
This gives positive, multiplicative intervals symmetric in log coordinates.
It changes the displayed estimator, not merely the axis: the plotted center is
the geometric mean, not the arithmetic mean. Both OOD regime panels fall back
to linear arithmetic intervals if any required value is nonpositive, missing,
or cannot be represented safely. No epsilon, clipping or dropped splits is used.

ID RMSE, improvements, gains, and contrasts remain linear with arithmetic means
and intervals; signed quantities are not log-transformed. Scales are shared
between regimes within each test. The aggregate `summary_ci95.*` files retain
their original arithmetic statistics. Displayed RMSE estimates are additionally
saved to `plots/rmse_display.json` with explicit estimator labels.
`--ood-rmse-scale linear` or `--ood-rmse-scale symlog` restores arithmetic OOD
intervals; their visual asymmetry on a symmetric-log axis is expected.

To redraw only the plots (no GPU needed), or choose a linear OOD RMSE axis:

```bash
python -B active_learning/ani_energy/plot_results.py
python -B active_learning/ani_energy/plot_results.py --ood-rmse-scale linear
```

Plot caching fingerprints the aggregate report, plot code and display settings.
The previous five AL run directories and their aggregate results were deleted
at the user's request before restarting with the corrected protocol. Auxiliary
cache, verification and migration-backup directories were preserved; they are
not used as new experiment results. Original A-F artifacts and datasets were
not deleted.

## Run

Use the same Python/conda environment as the existing MACE experiments:

```bash
conda activate mace
bash active_learning/ani_energy/run.sh 7
```

This prepares the data, scores the pool, selects/reveals labels, trains all eight
10-member ensembles **sequentially**, evaluates them, and writes the report.
GPU 7 is exposed as local `cuda:0`. `PYTHON=/path/to/python` overrides the interpreter.
No ANI HDF5 file, original evaluation caches, or new dependencies are needed;
the existing energy-split XYZ files, D/F checkpoints and training logs are required.

Defaults: dataset split 0, B=500, 10 members (seeds 0..9), 50/50 OOD
pool/test partition. The default output is:

```text
active_learning/ani_energy/runs/split_0/
```

Other dataset splits use separate output directories:

```bash
bash active_learning/ani_energy/run.sh 7 --split-seed 1
```

## Stages and Restarting

```bash
# CPU-only preparation and freezing of initial checkpoints.
python -B active_learning/ani_energy/workflow.py --stage prepare --device cpu

bash active_learning/ani_energy/run.sh 7 --stage acquire
bash active_learning/ani_energy/run.sh 7 --stage train
bash active_learning/ani_energy/run.sh 7 --stage evaluate
python -B active_learning/ani_energy/workflow.py --stage report
```

Rerun the same command to reuse completed work. Predictions, selections, augmented
datasets, and completed member models are cached. Source and artifact SHA-256
checks prevent silently mixing datasets, code, checkpoints, or settings.
Changing inference device/batch size after scoring also requires a new `--name`.
Pass the same split and optional epoch override to every stage. Keep acquisition
RNG settings unchanged so that persisted batches remain reproducible.

Interrupted member training restarts that member from its frozen initial model
with a fresh optimizer in a new `attempt_NNN/`; completed members are skipped.
Old attempts are retained. The runner locks a run directory, so do not run two
processes against the same run concurrently.

For manual scheduling, train selected cases/members in successive invocations:

```bash
bash active_learning/ani_energy/run.sh 7 --stage train --case hf_only_random --members 0 1
bash active_learning/ani_energy/run.sh 6 --stage train --case hf_only_random --members 2 3
```

Available cases: `hf_only_random`, `hf_only_tu`, `lf_hf_random`, `lf_hf_tu`.
Evaluation requires all 10 members of the requested case. `--case` also works with
`--stage evaluate`; the final report requires all eight default cases.
Follow live progress with `tail -f` on the printed member `console.log` path.

## Experimental Choices

| Regime | Initial ensemble | Additional training protocol |
|---|---|---|
| HF-only | `experiment_F/checkpoints_<split>` | AL LR 0.001, up to 100 epochs |
| LF->HF | `experiment_D/checkpoints_<split>` | AL LR 0.001, up to 100 epochs |

**Both regimes start post-acquisition training at LR 0.001**, overriding the
learning rate inherited from their source configs. This also applies to the
optional common-evaluator control. Original F/D training configs are unchanged.
The schedulers can subsequently reduce the LR. Both regimes default to at most
100 additional epochs; `--epochs N` overrides this limit for both, including the
optional common-evaluator control. Other training settings remain inherited from F/D.

The old AL results (originally HF-only LR 0.01 / 300 epochs, LF->HF LR 0.001 /
100 epochs) were deleted; they are not relabeled as corrected results. Manifests
record the shared AL learning rate and effective additional epoch limit.
The runner rejects reuse of incompatible settings. Use `--name NAME` for a
custom single-split directory, or `run_all.sh --run-tag NAME` for a coordinated
five-split batch.
This protection means the runner raises a protocol-mismatch error instead of
silently treating old completed models as models trained with the new defaults.
It does not delete, update, or retrain those models.

Initial checkpoints are selected independently per member by **minimum original
HF validation loss**, exactly as in `eval_D.sh` / `eval_F.sh`. The selected checkpoint
and its companion architecture are frozen as byte-identical AL-local copies,
then loaded strictly for acquisition/training. LF->HF already contains the LF
pretraining from experiment C;
there is no need to evaluate or retrain C first.

Both acquisition branches of a regime start from identical member weights, use
the same original HF validation set, member seeds, Gaussian NLL loss, optimizer
parameter groups, batching, scheduler, clipping and epoch limit. Training calls
the existing `mace.tools.train` implementation. All model parameters are enabled
for training. Mean/variance heads, E0s and normalization buffers are preserved;
the model is not rebuilt through the foundation-model CLI. Optimizer/scheduler
state is deliberately reset for this new round, identically across methods.
No OOD or ID test loader is passed to training. Best post-acquisition epochs are
again chosen by minimum original validation loss.

**Epoch budgets and checkpoint selection are different things.** Each dataset
split has 10 ensemble members. Before acquisition, each member independently
uses its best original HF validation checkpoint, so their epoch numbers can
differ. After acquisition, each member trains on the original HF training data
plus the selected 500 configurations, and its best additional-training checkpoint
is again selected independently. "Best" here means minimum validation Gaussian
NLL, not minimum validation RMSE. The 5 dataset splits, not the 10 members, are
the replicates used for the reported confidence intervals.

**Post-AL evaluation uses these independently selected checkpoints:** for every
condition and every dataset split, each of the 10 members contributes its own
minimum-validation-NLL checkpoint, saved as `best.model`. Their selected epochs
may differ. The same selected ensemble is evaluated on both the held-out OOD
test and the existing ID test. Neither test is used to select checkpoints;
evaluation does not automatically use the last epoch or choose a common epoch
for all members.

The default epoch limit is 100 for both regimes. For a shorter run:

```bash
bash active_learning/ani_energy/run.sh 7 --epochs 20
```

This limits post-acquisition training to 20 additional epochs **per member in
both regimes**, instead of the shared default of 100. The option is spelled
`--epochs` (plural). It does not retrain the original models or force selection of the last
epoch: one member might be selected at additional epoch 7 and another at epoch
15. Equal epoch limits are not necessarily equal wall-clock or compute budgets.
For the default dataset split 0, this command automatically uses
`runs/split_0_epochs_20/`. It is a different experiment, not the default
protocol. The optional `--name NAME` chooses another isolated run directory.

### Original ANI Splits (Before Active Learning)

The settings below are those in [`dataset/make_dataset.sh`](../../dataset/make_dataset.sh),
not the standalone Python splitters' default arguments. Dataset seeds 0 through
4 generate the five versions of each split.

- **A/B/E use the system split:** approximately 60% of systems are assigned to
  the seen domain. Their configurations are randomly partitioned into candidate
  train/validation/ID-test pools in proportions 60%/20%/20%. The remaining
  systems supply OOD configurations. This tests generalization to unseen systems,
  not a high-energy threshold.
- **C/D/F use the energy split:** configurations are ranked by DFT energy
  (`wb97x_tz.energy`) separately within each system. Empirical ranks at or below
  0.50 supply the low-energy domain; ranks at or above 0.55 supply the high-energy
  OOD domain. The intermediate band is discarded. Low-energy configurations are
  randomly partitioned into candidate train/validation/ID pools in proportions
  60%/20%/20%. These are per-system ranks, not a global absolute-energy cutoff
  across molecules, and the ranking does not use CC labels.
- The energy splitter excludes systems with fewer than four valid ranking
  energies. For exactly four, the highest-energy configuration goes to OOD and
  the lowest three are randomly assigned one each to train/validation/ID.
- Final files are sampled from the candidate pools, with system-stratified
  quotas and `max_per_system=64` as a **soft weighting cap**, not a strict maximum
  file contribution. DFT train/validation/ID/OOD sizes are respectively
  50000/10000/50000/50000; CC sizes are 5000/1000/5000/5000. Available CC-labeled
  configurations are sampled within the corresponding DFT selection.

Thus the original energy experiments deliberately separate low-energy training
from higher-energy testing within systems. The AL partition below is a second,
different operation applied only to the already-created CC OOD file.

### Data and Hidden Labels

- Only `cc_test_ood.xyz` of the selected existing ANI energy split is repartitioned.
  With the current 5000 configurations, there are 2500 pool and 2500 held-out OOD
  configurations. The original file itself is never changed.
- The existing ANI splitter's quota allocator and sampler preserve the source
  OOD system composition, using proportional half-sampling per system and seeded
  remainder allocation. Singleton systems cannot occur in both halves. No new
  energy threshold or label-based ranking is introduced.
- In plain terms, a system contributing 100 original OOD configurations supplies
  50 randomly chosen configurations to the pool and 50 to the held-out test.
  Odd counts require reproducible rounding to keep the total pool size at 2500.
  **The held-out test is not the higher-energy half of OOD.** Both halves sample
  the same original OOD energy domain; their energy ranges can overlap, and pool
  energies are not systematically lower than held-out energies.
- All original LF/HF train, validation and ID-test XYZ files remain unchanged.
  Preparation checks that OOD IDs do not overlap any of those six inputs.
- Stable IDs are `system:conf_idx`; pool and held-out IDs/source indices/system
  counts are saved. The held-out set is never acquired or appended to training.
- Public `pool.xyz` contains geometry, cell/PBC and IDs only: no CC/DFT energies,
  forces, calculators, relative energies or quantiles. The offline CC oracle is
  stored separately in `data/oracle.json`, containing **pool IDs only**.
- Acquisition uses only public geometries and model predictions. The shared
  evaluator's missing-label adapter uses a zero-weight placeholder, not CC labels;
  placeholder errors/references are discarded and never used for selection.
- Selection is persisted **before** the oracle is read to append the chosen 500
  labels. Augmented files begin with an unchanged copy of the original HF training
  file. New configurations use its training `config_type` and are equally weighted.
- Random acquisition is uniform without replacement, shared between regimes.
  Each AU/EU/TU acquisition is the global top 500 of its own score (stable ID
  tie-break), not system-capped. The selected batches can overlap; they are
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
domain. Changing to that design would require a separate experimental partition.

### Acquisition Scores and RMSE

The runner directly reuses `eval/reliability.py` for prediction and uncertainty:

```text
AU = mean_m(variance_m / N_atoms^2)
EU = population_variance_m(energy_mean_m / N_atoms)    # unbiased=False
TU = AU + EU
```

Each acquisition score is **one scalar per configuration**: AU uses the raw
aleatoric variance, EU uses the population ensemble variance, and TU uses their
sum. All are variances of the configuration's energy per atom. They come from
the same cached ensemble predictions, with no extra inference per metric.
In the formulas above, `energy_mean_m`
and `variance_m` are member `m`'s predicted total configuration energy and its
variance; `N_atoms` is the atom count of that configuration. "Per-atom" refers
only to normalization by `N_atoms^2`, not to separate acquisition decisions for
individual atoms. The 2500 whole configurations are ranked by their scores and
500 whole configurations are selected.

Unnormalized total-energy TU would instead be
`mean_m(variance_m) + population_variance_m(energy_mean_m)`, equal to the current
score times `N_atoms^2`. Both choices give one score per configuration, but they
can produce different rankings when atom counts differ. The current normalization
matches the existing project uncertainty and energy-per-atom RMSE definitions.
Switching to total-energy TU would be a different acquisition experiment, not
just a change of terminology.

There is no calibration, trimming, error-based selection, or label-based
filtering. Non-finite values fail the run instead of silently excluding
configurations.

All conditions use the same held-out Energy-OOD test and the unchanged existing
`cc_test_id.xyz`. RMSE is that of the **ensemble-mean energy per atom**, averaged
equally over configurations, not an average of member RMSEs. Reported units are
meV/atom. Individual predictions and AU/EU/TU are retained in inference caches.

```text
relative improvement (%) = 100 * (RMSE_before - RMSE_after) / RMSE_before
method gain (meV/atom) = RMSE_random - RMSE_method
method gain (percentage points) = improvement_method - improvement_random
# method is AU, EU or TU
```

Positive gain favors the uncertainty method over random. The report also gives
LF->HF's gain minus HF-only's gain for each method and test: positive values
support the acquisition-quality hypothesis.
Negative improvements/gains are retained. A zero baseline RMSE produces `null`
percentage improvements, not a division by zero.

## Optional Common Evaluator

```bash
bash active_learning/ani_energy/run.sh 7 --common-evaluator
```

This adds `lf_hf_from_hf_only_tu`: the same D initial ensemble/protocol trained on
the HF-only TU batch. Its comparator is the existing `lf_hf_tu` branch, which is
already exactly the requested LF->HF evaluator on the LF->HF-selected batch.
Reusing that branch avoids 10 redundant, identically initialized training runs.
The control therefore adds 10 members, not 20. It is reported separately and can
be enabled after the default run finishes; the eight default cases are reused.
This optional control still compares TU acquisitions only.
Include `--common-evaluator` for subsequent control train/evaluate/report stages.

## Artifacts

```text
runs/<name>/
  manifest.json                 settings, source hashes, versions, checkpoint provenance
  data/                         public pool, held-out OOD, oracle, split/ID manifest
  initial/{hf_only,lf_hf}/       frozen 10-member ensembles, configs and source logs
  inference/                    cached pool and test predictions, input fingerprints
  acquisition/{random,hf_only_au,hf_only_eu,hf_only_tu,lf_hf_au,lf_hf_eu,lf_hf_tu}/
    selection.json              selected IDs, method, seed, score provenance
    train.xyz                   original HF train + 500 revealed configurations
    augmented.json              counts and file hashes
  cases/<case>/member_<seed>/
    attempt_NNN/                local config, resolved arguments, logs, checkpoints, best model
    complete.json               validated completion/cache marker
  metrics/                      per-condition/test RMSE, counts and provenance
  report/summary.{csv,json,md}   default comparisons and optional control
```

Each attempt retains every epoch checkpoint, matching the existing protocol;
budget disk space accordingly. Original checkpoint/config sources are read-only.
No W&B run, shared evaluation cache or existing result directory is written.

## Scope and Verification

Each `run.sh` invocation is a single-round, single-split POC. `run_all.sh` aggregates
five such splits with approximate confidence intervals, not a blanket claim of
statistical significance. The 10 members form one uncertainty estimator, not 10
independent AL repetitions. GPU floating-point kernels can prevent bitwise equality
across machines despite the recorded seeds, versions and artifact hashes.

Some existing companion models contain CUDA-serialized TorchScript blocks, so
even a CPU `torch.load` of them requires an available GPU. Preparation copies
bytes without deserializing models and works on CPU; run acquisition/training/
evaluation on a GPU-enabled session for those checkpoints. CPU inference remains
available for checkpoints that were serialized without CUDA-only blocks.

The pool comes from an already labeled, CC-available, previously used OOD dataset.
Labels are hidden from this acquisition workflow, not retroactively from past
experiments or from dataset construction. Treat the result as an offline POC,
not as a pristine prospective benchmark. The common-evaluator control helps
separate acquisition quality from downstream-regime differences.

Run the focused CPU tests with the existing MACE environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B active_learning/ani_energy/test_workflow.py
python -B active_learning/ani_energy/test_five_splits.py
python -B active_learning/ani_energy/test_reporting.py
bash -n active_learning/ani_energy/run.sh
bash -n active_learning/ani_energy/run_all.sh
```

Tests cover deterministic balanced partitioning, hidden-label invariance, oracle
reveal and leakage guards, selection budgets/ties, cache invalidation, reporting
formulas, the actual shared TU calculation, and one epoch of genuine small MVE
checkpoint continuation through the existing MACE training loop. A separate
orchestration test uses synthetic predictions and a stub trainer to verify all
eight branches, resume behavior, shared tests and the optional control; it does
not produce scientific results.
Reporting tests cover default folder names, recursive path/hash migration,
unchanged numerical values, paired confidence intervals, and nonblank/cached plots.
