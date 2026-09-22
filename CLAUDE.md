# Project context

This is a research fork of MACE for the paper **"Multi-Fidelity Training
Reshapes Uncertainty Decomposition in Atomistic Models"**. The project adds
mean-variance estimation (MVE), deep-ensemble uncertainty decomposition,
multi-fidelity ANI-1ccx and liquid-water experiments, five-split statistics,
and an isolated ANI active-learning proof of concept.

Start with this file. Read only the relevant parts of these canonical sources:

- [`README.md`](README.md): experiment matrix, datasets, training, evaluation,
  caching, normal commands, the complete AL protocol (section "ANI energy-OOD
  active learning"), and the pipeline audit ("Audit notes and known issues").
- [`_Nips26__UQ_MF-3.pdf`](_Nips26__UQ_MF-3.pdf): current paper narrative,
  figures, and appendix structure. Do not edit the manuscript unless asked.

This file is intentionally an index and decision log, not a replacement for
those documents. Explore the specific files involved before changing code.

## Non-negotiable experimental invariants

- There are five dataset splits (`0..4`). A split is one statistical replicate.
- Each split has ten independently initialized MVE members (`0..9`). Together
  they form one ensemble; **members are not independent statistical samples**.
- Final confidence intervals are computed over the five split-level results.
- Fine-tuning must pair the same split and member seed with its pretrained
  parent: B <- A, D <- C, and wB <- wA.
- ANI low fidelity is DFT `wb97x_tz.energy`; ANI high fidelity is CC
  `ccsd(t)_cbs.energy`. Water uses `REF_energy` and must retain cell/PBC data.
- AU is the mean member-predicted variance; EU is population variance of member
  energy means (`unbiased=False`); TU = AU + EU. Evaluation normally uses
  per-atom energies and variances (`variance / N_atoms^2`).
- Do not treat ID/OOD tests as checkpoint-selection data. Each member's best
  checkpoint is selected independently by validation loss (Gaussian NLL).
- Never silently use fewer than five splits or fewer than ten members.
- Do not overwrite, delete, rename, or retrain existing datasets, checkpoints,
  caches, or results unless the user explicitly requests that exact action.
  These artifacts occupy over 1 TB and are expensive to reproduce.
- The worktree commonly contains user-generated plots, paper files, and ongoing
  changes. Never reset or revert unrelated changes.

## Experiment matrix

| Family | Shift/data | Protocol | Target | Parent |
|---|---|---|---|---|
| A | ANI system-OOD | LF-only | DFT | none |
| B | ANI system-OOD | LF->HF | CC | matching A |
| C | ANI energy-OOD | LF-only | DFT | none |
| D | ANI energy-OOD | LF->HF | CC | matching C |
| E | ANI system-OOD | HF-only | CC | none |
| F | ANI energy-OOD | HF-only | CC | none |
| wA | periodic water | LF-only | BLYP | none |
| wB | periodic water | LF->HF | CCSDT | matching wA |
| wC | periodic water | HF-only | CCSDT | none |

Every experiment has `config_X.yml`, `train_X.sh`, and `eval_X.sh`. Output
directories carry the split suffix: `checkpoints_<s>`, `models_<s>`,
`results_<s>`, and `logs_<s>`. Expensive evaluation rows are cached under
`experiment_X/evaluation/cache/split_<s>/`; aggregate tables and SVGs are in
`experiment_X/evaluation/`.

## Code map

- MVE model/readout: `mace/modules/models.py`
- Gaussian NLL: `mace/modules/loss.py`
- MVE CLI/config wiring: `mace/tools/arg_parser.py`,
  `mace/tools/scripts_utils.py`, `mace/cli/run_train.py`
- Ensemble prediction: `eval/reliability.py` and `mace/cli/eval_configs.py`
- ANI splitters: `dataset/ani1x_system_splitter.py` and
  `dataset/ani1x_energy_splitter.py`; orchestration is `dataset/make_dataset.sh`
- Water conversion/splits: `dataset/convert_water_n2p2_to_extxyz.py` and
  `dataset/make_dataset_w.sh`
- Shared training launcher: `utils/train_split_member.sh`
- Per-experiment evaluation: `experiment_X/eval_X.sh`
- Epoch metrics: `eval/epoch_quality.py`; LF->HF mixed trajectories:
  `eval/epoch_quality_finetune_mixed_dataset.py`
- Reliability/calibration: `eval/reliability.py`
- Five-split statistics and plots: `eval/replicate_statistics.py` and
  `eval/aggregate_replicates.py`
- Top-level evaluators: `eval.sh` and `eval_water.sh`
- ANI AL implementation: `active_learning/ani_energy/`

`dataset/ani1x-processed/` is legacy and is not used by the current pipeline.

## Paper-to-repository map

Use this map when changing figures or checking reproducibility:

| Paper element | Repository implementation/artifacts |
|---|---|
| Sec. 2, MVE and AU/EU/TU | `mace/modules/models.py`, `mace/modules/loss.py`, `eval/reliability.py` |
| Sec. 3 / App. D, system split | `dataset/ani1x_system_splitter.py`; experiments A, B, E |
| Sec. 3 / App. E, energy split | `dataset/ani1x_energy_splitter.py`; experiments C, D, F |
| Sec. 3.1 / App. C, metrics | `eval/epoch_quality.py`, `eval/reliability.py` |
| Fig. 2, system reliability | A (LF), E (HF), B (LF->HF) |
| Fig. 3, energy reliability | C (LF), F (HF), D (LF->HF) |
| Fig. 4, system LF->HF dynamics | mixed A->B epoch-quality outputs |
| Fig. 5, energy LF->HF dynamics | mixed C->D epoch-quality outputs |
| Fig. 6, system HF vs LF->HF | E compared with A->B |
| App. A, water reliability/dynamics | wA (LF), wB (LF->HF), wC (HF) |
| App. G / Fig. 9, energy quantiles | `eval/energy_ood.py`; C, D, F |
| App. H / Figs. 10-12 | A, B, E train/ID/OOD dynamics |
| App. H / Figs. 13-15 | C, D, F train/ID/OOD dynamics |
| App. I / Figs. 16-17 | calibrated reliability from `eval/reliability.py` |
| App. F, hyperparameters | `experiment_*/config_*.yml` is implementation authority |

The active-learning study is a later proof of concept and is **not part of the
current paper's main Sections 1-5 or listed appendix figures**. Keep AL claims
and outputs separate unless the manuscript is explicitly extended.

## Statistics and plotting contract

- Linear summaries use arithmetic means and two-sided 95% Student-t intervals.
  With five splits, df=4 and `t=.975 = 2.776...`.
- Strictly positive quantities on log axes use geometric means and log-space
  Student-t intervals, back-transformed with `exp`. Do not plot arithmetic
  symmetric intervals on log axes and do not add epsilon/drop splits silently.
- Evaluation figures are SVG-only. Epoch-quality plots have three variants:
  standard fixed scale (`.svg`), autoscaled (`-free-scale.svg`), and shared
  logarithmic scale (`-log-scale.svg`).
- Current shared log-view limits are Spearman `[-0.2, .8]`, AUSE `[1e-2, 1]`,
  ENCE `[1e-1, 1e2]`, uncertainty `[1e-7, 1e5]`, RMSE `[1e-4, 1e2]`, and GNLL
  `[-8, 4]`. Spearman and GNLL remain linear because they can be negative.
- A center outside a fixed range is clipped to the nearest boundary and marked
  with `x`; confidence bands are clipped to the visible range.
- Reliability metrics (RMSE, Spearman, AUSE, ENCE) are all computed from the
  same support-filtered per-configuration rows in `aggregate_replicates.py`.
  `reliability.py --trim` is off for A-F: it selected on total uncertainty, the
  quantity under evaluation. Do not re-enable it for paper numbers.
- Labels use direction arrows where meaningful: Spearman up; AUSE, ENCE, RMSE,
  and GNLL down. Uncertainty magnitude has no intrinsic better direction.
- Fine-tuning epoch-quality output is mixed-target only. The legacy
  `epoch_quality_finetune_same*` caches are retained but skipped by aggregation.
- Water reliability uses common log-log bounds `[1e-4, 1e-1]`.

## Active-learning state and history

The ANI energy-OOD AL code is isolated under `active_learning/ani_energy/` and
must not write into A-F datasets/results. It compares HF-only (initial F) and
LF->HF (initial D), each with Random/AU/EU/TU acquisition at budget 500. Both
regimes use AL LR 0.001; the code default is 100 additional epochs. The existing
completed scientific runs are named `split_0_epochs_50` ...
`split_4_epochs_50`, with aggregate `aggregate_epochs_50`; do not mistake their
explicit 50-epoch override for the current default.

Important provenance:

- The OOD source file is split system-stratified 50/50 into acquisition pool
  and held-out test. The held-out half is not a higher-energy half.
- One configuration, `C1H5N1:887`, was deleted post hoc from split 3's held-out
  AL test and all ten matching cached OOD predictions after diagnosis of a
  catastrophic LF->HF extrapolation. Split 3 has 2499 held-out rows. Reports
  record the exclusion; raw split-3 results are conditional on it and no backup
  of the original predictions exists.
- `support_filter.py` provides a separate geometry-only domain-of-validity
  analysis based on each split's minimum training interatomic distance. It does
  not inspect prediction errors. Report raw and support-filtered AL results and
  disclose the restriction. The same rule now lives in `eval/support_filter.py`
  and is applied by `eval/aggregate_replicates.py` to the A-F reliability
  metrics; the AL copy still owns the AL runs.
- Support-filtered output shares `aggregate_epochs_50/` with the raw aggregator,
  but `five_splits.py` now refuses to overwrite it and says how to proceed.
  Rerun `support_filter.py` to refresh, or pass `--overwrite-filtered` to
  deliberately replace it with unfiltered numbers.
- AL caches use hashes and manifests. Do not hand-edit or copy partial cache
  trees without updating the dependency graph and verifying fingerprints.

Read the AL section of `README.md` before any AL mutation; it has the formulas,
folder schema, resume rules, common-evaluator control, and leakage safeguards.

## Commands

Run commands from the repository root and normally use the `mace` conda
environment. The known interpreter is
`/raid/m.tolloso/miniconda3/envs/mace/bin/python`.

```bash
# Train exactly one member
bash experiment_A/train_A.sh <split:0-4> <member:0-9> <gpu>

# Check all 450 main experiment members
./check_experiments.sh

# Evaluate one experiment (GPU inference, then five-split aggregation)
experiment_A/eval_A.sh <gpu>

# Rebuild all aggregate tables/plots from existing CSV caches; CPU-only
MPLCONFIGDIR=/tmp/mpl-cache \
EVAL_PYTHON=/raid/m.tolloso/miniconda3/envs/mace/bin/python \
bash eval.sh --plots-only

# Run/resume five AL splits sequentially on one GPU
bash active_learning/ani_energy/run_all.sh <gpu> --epochs 50

# Aggregate completed AL split reports only; no training or GPU inference
bash active_learning/ani_energy/run_all.sh <gpu> --aggregate-only --epochs 50
```

`bash eval.sh` without `--plots-only` launches A-F evaluations concurrently on
its hard-coded GPU map. Inspect/edit that map only when explicitly requested.
Dataset generation scripts can replace generated split files; do not run them
casually against completed experiments.

## Verification

Use focused checks proportional to the change; avoid expensive GPU work when a
cache-only or unit check is sufficient.

```bash
# Plot/statistics changes
cd eval && MPLCONFIGDIR=/tmp/mpl-cache \
  /raid/m.tolloso/miniconda3/envs/mace/bin/python -B -m unittest -v test_plotting

# AL logic/reporting
/raid/m.tolloso/miniconda3/envs/mace/bin/python -B active_learning/ani_energy/test_workflow.py
/raid/m.tolloso/miniconda3/envs/mace/bin/python -B active_learning/ani_energy/test_five_splits.py
/raid/m.tolloso/miniconda3/envs/mace/bin/python -B active_learning/ani_energy/test_reporting.py

# Shell syntax after launcher edits
bash -n eval.sh active_learning/ani_energy/run.sh active_learning/ani_energy/run_all.sh
```

When reporting completion, state whether work used cached data, CPU inference,
GPU inference, or retraining, and name the exact outputs changed.

## Manuscript consistency checks

Do not silently change code or old results to match the PDF. Raise these points
when editing methods, equations, captions, or reproducibility claims:

1. The implementation computes EU with population variance (`unbiased=False`,
   denominator M), while Equation 3 in the current PDF appears to show M-1.
2. Appendix F of the PDF describes random seed 0 and a general 300-epoch setup;
   the current repository uses five dataset split seeds, and B/D/wB fine-tuning
   configs use 100 epochs while pretraining/direct runs use 300.
3. The repository now reports five-split confidence intervals and expanded
   water statistics; verify that each paper figure/table was regenerated from
   the intended version before updating claims.

## Maintaining this file

Keep `CLAUDE.md` concise and stable. Add only non-obvious facts whose absence
would cause costly or scientifically invalid work. Put detailed protocols in
the relevant README and link them here instead of duplicating them.
