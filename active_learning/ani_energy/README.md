# ANI Energy-OOD Active Learning

Isolated, single-round proof of concept. Nothing in `experiment_*`, `dataset/`,
`eval/`, or the shared MACE training code is modified. All generated files live
under `active_learning/ani_energy/runs/` and are ignored by Git.

## Five Splits and Confidence Intervals

One command runs/resumes splits 0..4 sequentially on GPU 2, evaluates all four
conditions for each split, then produces the aggregate reports and plots:

```bash
conda activate mace
bash active_learning/ani_energy/run_all.sh 2
```

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
split, not an ensemble member. Improvements, TU-over-random gains, and the
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
Renamed runs are `runs/split_0/` through `runs/split_4/`. Their ten members remain
under `cases/<condition>/member_0/` through `member_9/`. Confidence intervals use
the five splits, not the members.

### Graphical Results

`run_all.sh` automatically invokes `plot_results.py` after aggregation, including
in `--aggregate-only` mode. The `runs/aggregate/plots/` folder contains PNG and PDF
versions of:

- `rmse`: before acquisition, Random-500 and TU-500, for both regimes and tests.
- `improvement`: relative improvements from the within-split baseline.
- `tu_gain`: paired TU-over-random gains in meV/atom and percentage points.
- `regime_contrast`: LF->HF's TU gain minus HF-only's TU gain, the key comparison.
- `common_evaluator`: the optional control, when those results are available.

Diamonds show means with approximate 95% Student-t intervals; grey points show
all five individual split values. No outliers are removed. OOD RMSE uses a
symmetric-log display to retain the wide dynamic range and negative confidence
bounds; the intervals themselves are still calculated on the original scale.
ID RMSE is linear, with scales shared between regimes within each test.

To redraw only the plots (no GPU needed), or choose a linear OOD RMSE axis:

```bash
python -B active_learning/ani_energy/plot_results.py
python -B active_learning/ani_energy/plot_results.py --ood-rmse-scale linear
```

Plot caching fingerprints the aggregate report, plot code and display settings.
The September 2026 folder migration preserves the original metadata in
`runs/migration_backup/original_metadata.tar.gz` and records its changes in
`runs/migration_backup/complete.json`. Paths and dependent cache digests were
updated; checkpoints, selected configurations and numerical results were not.

## Run

Use the same Python/conda environment as the existing MACE experiments:

```bash
conda activate mace
bash active_learning/ani_energy/run.sh 7
```

This prepares the data, scores the pool, selects/reveals labels, trains all four
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
`--stage evaluate`; the final report requires all four default cases.
Follow live progress with `tail -f` on the printed member `console.log` path.

## Experimental Choices

| Regime | Initial ensemble | Additional training protocol |
|---|---|---|
| HF-only | `experiment_F/checkpoints_<split>` | F: LR 0.01, up to 300 epochs |
| LF->HF | `experiment_D/checkpoints_<split>` | D: LR 0.001, up to 100 epochs |

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

The default epoch budgets differ between regimes because they retain F/D's
existing protocols. For a shorter run or equal-budget sensitivity check:

```bash
bash active_learning/ani_energy/run.sh 7 --epochs 20
```

This sets 20 additional epochs for **both** regimes and automatically uses
`runs/split_0_epochs_20/`. It is a different experiment, not the default
protocol. The optional `--name NAME` chooses another isolated run directory.

### Data and Hidden Labels

- Only `cc_test_ood.xyz` of the selected existing ANI energy split is repartitioned.
  With the current 5000 configurations, there are 2500 pool and 2500 held-out OOD
  configurations. The original file itself is never changed.
- The existing ANI splitter's quota allocator and sampler preserve the source
  OOD system composition, using proportional half-sampling per system and seeded
  remainder allocation. Singleton systems cannot occur in both halves. No new
  energy threshold or label-based ranking is introduced.
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
  TU acquisition is the global top 500 (stable ID tie-break), not system-capped.

### TU and RMSE

The runner directly reuses `eval/reliability.py` for prediction and uncertainty:

```text
AU = mean_m(variance_m / N_atoms^2)
EU = population_variance_m(energy_mean_m / N_atoms)    # unbiased=False
TU = AU + EU
```

The acquisition score is **raw per-atom TU variance**, with no calibration,
trimming, error-based selection, or label-based filtering. Non-finite values fail
the run instead of silently excluding configurations.

All conditions use the same held-out Energy-OOD test and the unchanged existing
`cc_test_id.xyz`. RMSE is that of the **ensemble-mean energy per atom**, averaged
equally over configurations, not an average of member RMSEs. Reported units are
meV/atom. Individual predictions and AU/EU/TU are retained in inference caches.

```text
relative improvement (%) = 100 * (RMSE_before - RMSE_after) / RMSE_before
TU gain (meV/atom) = RMSE_random - RMSE_TU
TU gain (percentage points) = improvement_TU - improvement_random
```

Positive gain favors TU. The report also gives LF->HF's TU gain minus HF-only's TU
gain on each test: positive values support the acquisition-quality hypothesis.
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
be enabled after the default run finishes; the four default cases are reused.
Include `--common-evaluator` for subsequent control train/evaluate/report stages.

## Artifacts

```text
runs/<name>/
  manifest.json                 settings, source hashes, versions, checkpoint provenance
  data/                         public pool, held-out OOD, oracle, split/ID manifest
  initial/{hf_only,lf_hf}/       frozen 10-member ensembles, configs and source logs
  inference/                    cached pool and test predictions, input fingerprints
  acquisition/{random,hf_only_tu,lf_hf_tu}/
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
four branches, resume behavior, shared tests and the optional control; it does
not produce scientific results.
Reporting tests cover default folder names, recursive path/hash migration,
unchanged numerical values, paired confidence intervals, and nonblank/cached plots.
