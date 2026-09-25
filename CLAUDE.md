# Project context

Research fork of MACE for the paper **"Multi-Fidelity Training Reshapes
Uncertainty Decomposition in Atomistic Models"**. The project studies how
aleatoric and epistemic uncertainty behave in mean-variance (MVE) MACE ensembles
across fidelity regimes (DFT pretraining, CC fine-tuning, CC-only), distribution
shifts (system-level and energy-based), and training dynamics, on ANI-1ccx and
periodic liquid water. It also contains an isolated ANI active-learning proof of
concept.

Two workspaces share this root:

- **Experiments** (this repository, `origin` = the experiments GitHub repository): code, data,
  training, evaluation and results.
- **Paper** (`ICLR27-UQ-MF/`): the ICLR 2027 manuscript, an independent Git
  repository synced with Overleaf through GitHub. The parent repository ignores
  it; its own local `CLAUDE.md` loads the paper rules.

Detailed, task-specific instructions live in `.claude/rules/`:
`reproducibility.md`, `statistics-and-plots.md`, `code-changes.md`,
`paper-writing.md`, `active-learning.md` (loaded for `active_learning/**`).
The full protocol and the pipeline audit are in [`README.md`](README.md).

## Terminology

- **LF / HF**: low fidelity = DFT `wb97x_tz.energy`; high fidelity = CC
  `ccsd(t)_cbs.energy`. Water uses `REF_energy` (BLYP as LF, CCSDT as HF).
- **Protocols**: LF-only, HF-only, and LF->HF (DFT pretraining, then CC
  fine-tuning of the same member).
- **Shifts / test sets**: System-ID/OOD (unseen molecular systems) and
  Energy-ID/OOD (higher within-system energy quantiles).
- **AU** = mean member-predicted variance; **EU** = population variance of
  member energy means (`unbiased=False`); **TU** = AU + EU. Per-atom by default
  (`variance / N_atoms^2`).
- **Ranking** metrics: Spearman, AUSE. **Calibration**: ENCE. Error: RMSE.
- **Split** = one statistical replicate (seeds `0..4`); **member** = one of the
  ten MVE models of a split's ensemble (seeds `0..9`).

## Non-negotiable invariants

- Five dataset splits; ten members per split form one ensemble. Members are not
  independent samples. Confidence intervals are over the five splits.
- Never silently use fewer than five splits or fewer than ten members.
- Fine-tuning pairs the same split and member seed with its parent:
  B <- A, D <- C, wB <- wA.
- Checkpoints are selected per member by validation Gaussian NLL only; ID/OOD
  tests are never used for selection.
- Do not overwrite, delete, rename or retrain datasets, checkpoints, caches or
  results unless the user explicitly requests that exact action (>1 TB).
- Never reset or revert changes you did not make. Do not commit or push unless
  asked.
- **Never use more than 2 GPUs at the same time**, counting every running job
  (training, inference, evaluation). The machine is shared. `bash eval.sh`
  without `--plots-only` uses six GPUs: do not run it as-is.
- **Never claim an experimental result unless it is recorded in
  `results/results.md`** (or read from an output file that a ledger entry lists).

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

Water is trained and evaluated on all five splits (ledger entry 2026-09-23) and is
outside the current paper.

## Where things live

| What | Path |
|---|---|
| MACE package incl. MVE head and Gaussian NLL | `mace/` (`modules/models.py`, `modules/loss.py`) |
| Evaluation, statistics, support filter | `eval/` |
| Dataset splitters and generated splits | `dataset/` (`ani1x_{system,energy}_split_<s>/`, `water_<s>/`) |
| Experiment configs and launchers | `experiment_X/config_X.yml`, `train_X.sh`, `eval_X.sh` |
| Run outputs (checkpoints, models, logs) | `experiment_X/{checkpoints,models,results,logs}_<s>/` |
| Per-split evaluation caches | `experiment_X/evaluation/cache/split_<s>/` |
| Five-split tables (CSV) and figures (SVG) | `experiment_X/evaluation/` (catalogue: `experiments/evaluation_figures.md`) |
| Rebuilt paper figures (PDF/SVG/PNG) | `new_figures/out/` (scripts in `new_figures/`) |
| Active-learning code and runs | `active_learning/ani_energy/` (`runs/` is Git-ignored) |
| Results ledger; curated tables/figures | `results/results.md`, `results/tables/`, `results/figures/` |
| Experiments index, ad-hoc launch scripts | `experiments/README.md`, `experiments/scripts/` |
| Notes and the previous NeurIPS manuscript | `notes/` |
| Manuscript (Overleaf) | `ICLR27-UQ-MF/iclr2027_conference.tex`, `ICLR27-UQ-MF/figures/` |
| W&B logs | `wandb/` (Git-ignored) |

`experiment_X/` and the code directories cannot be moved: launchers resolve the
repository root from their own location, and caches use fixed relative paths.
`dataset/ani1x-processed/` is legacy and unused.

**Generated (never hand-edit):** everything under `experiment_X/{checkpoints,
models,results,logs}_*`, `experiment_X/evaluation/`, `new_figures/out/`,
`new_figures/_cache/`, `eval/cache/`, `active_learning/ani_energy/runs/`,
`wandb/`, and the split files in `dataset/`.

## Setup

Use the existing conda environment `mace` (Python 3.11, torch 2.4.0,
e3nn 0.4.4), where this repository is installed in editable mode. Interpreter:
`$CONDA_PREFIX/bin/python` with `conda activate mace`. From scratch:

```bash
pip install -e ".[wandb,dev]"   # dependencies from setup.cfg
pip install scikit-learn        # isotonic recalibration in eval/reliability.py
```

Experiment configs set `wandb: True` (entity `uq-mf`), so training needs W&B.

## Tests and linting

```bash
# statistics / plotting / aggregation
cd eval && MPLCONFIGDIR=/tmp/mpl-cache \
  $CONDA_PREFIX/bin/python -B -m unittest -v test_plotting
# active learning
$CONDA_PREFIX/bin/python -B active_learning/ani_energy/test_workflow.py
$CONDA_PREFIX/bin/python -B active_learning/ani_energy/test_five_splits.py
$CONDA_PREFIX/bin/python -B active_learning/ani_energy/test_reporting.py
# shell syntax
bash -n eval.sh active_learning/ani_energy/run.sh active_learning/ani_energy/run_all.sh
# upstream MACE tests (slow; CI) and linting
python -m pytest tests --ignore=tests/test_cueq_oeq.py
pre-commit run --all-files      # black, isort, pylint, whitespace
```

## Running experiments

Run from the repository root.

```bash
bash experiment_A/train_A.sh <split:0-4> <member:0-9> <gpu>   # train one member
./check_experiments.sh                                         # all 450 members
experiment_A/eval_A.sh <gpu>          # GPU inference + five-split aggregation
MPLCONFIGDIR=/tmp/mpl-cache \
EVAL_PYTHON=$CONDA_PREFIX/bin/python \
bash eval.sh --plots-only             # rebuild aggregates from caches, CPU only
python -B eval/support_filter.py --report      # support-filter bounds per split
bash new_figures/make_all.sh                   # rebuild paper figures, CPU only
bash active_learning/ani_energy/run_all.sh <gpu> --epochs 50            # AL
bash active_learning/ani_energy/run_all.sh <gpu> --aggregate-only --epochs 50
```

`bash eval.sh` without `--plots-only` launches A-F evaluations concurrently on a
hard-coded GPU map; edit it only when asked. Dataset generation scripts can
overwrite split files: never run them against completed experiments.

## Reproducibility

- Record every citable run in `results/results.md` (skill: `record-result`):
  commit, exact command, configuration, dataset and split, seeds, metrics with
  their source files, output paths, and interpretation.
- Prefer cache-only recomputation over GPU inference over retraining. When
  reporting completion, state which one was used and name every output changed.
- Details: `.claude/rules/reproducibility.md`.

## Maintaining this file

Keep it below about 200 lines and limited to facts whose absence would cause
costly or scientifically invalid work. Put detail in `.claude/rules/` or
`README.md` and link it here.
