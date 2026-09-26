# Uncertainty quantification for multi-fidelity atomistic models

Research code, experiments and results behind *Multi-Fidelity Training Reshapes
Uncertainty Decomposition in Atomistic Models* (ICLR 2027 submission) and its
follow-ups. Ensembles of MACE models with a mean-variance (MVE) head are trained on
low-fidelity (DFT), high-fidelity (coupled-cluster) or both (pretrain + fine-tune)
labels; their uncertainty is split into aleatoric (AU) and epistemic (EU) parts and
evaluated for ranking (Spearman, AUSE) and calibration (ENCE) under distribution shift.

This repository is a fork of [MACE](https://github.com/ACEsuit/mace) (package in
`mace/`, upstream README in [docs/upstream-mace.md](docs/upstream-mace.md)); all
research code is in the `uq/` package and its `uq` command.

## Status

- 3 datasets, 9 experiments x 5 dataset splits x 10 ensemble members = **450 trained
  models, all complete and evaluated** (`uq status`).
- Active-learning proof of concept on ANI-1x Energy-OOD (400 more members).
- Citable numbers live only in the ledger [`results/results.md`](results/results.md).

## Quick start

```bash
conda env create -f environment.yml               # once: the exact environment of the results
conda activate mace
pip install --no-deps --no-build-isolation -e .   # once: registers `uq` (dependencies: setup.cfg)

uq list                     # experiments, their legacy letters and parents
uq status                   # trained members per split, evaluated splits
uq summary D --test ood     # headline metrics with 95% intervals over the five splits
uq show D                   # where its spec, config, data, runs and outputs are
uq evaluate D --gpu 6 --dry-run
uq --help
```

## Repository map

```text
uq/             research package: datasets, training, evaluation, active_learning, CLI
data/           datasets: <dataset>/dataset.yaml + split_<s>/ (large files git-ignored)
experiments/    <dataset>/<model>-<uq>/<protocol>/{experiment.yaml, config.yaml, evaluation/}
runs/           checkpoints, logs, W&B, active-learning runs (git-ignored, ~650 GB)
results/        results.md = ledger of every citable result; tables/, figures/
papers/iclr27/  figure scripts + outputs of the ICLR 2027 paper, drafts
notebooks/      exploratory analyses
docs/           documentation (start at docs/README.md)
mace/ tests/    upstream MACE (+ MVE head) and its tests
ICLR27-UQ-MF/   manuscript, separate Git repository synced with Overleaf (ignored here)
```

## Experiments

| Experiment id | Legacy | Dataset / shift | Protocol |
|---|---|---|---|
| `ani1x-system/mace-mve/lf-only` | A | ANI-1x, unseen molecular systems | DFT only |
| `ani1x-system/mace-mve/lf-to-hf` | B | same | DFT pretraining -> CC fine-tuning |
| `ani1x-system/mace-mve/hf-only` | E | same | CC only |
| `ani1x-energy/mace-mve/lf-only` | C | ANI-1x, high-energy conformers | DFT only |
| `ani1x-energy/mace-mve/lf-to-hf` | D | same | DFT pretraining -> CC fine-tuning |
| `ani1x-energy/mace-mve/hf-only` | F | same | CC only |
| `water-random/mace-mve/{lf-only,lf-to-hf,hf-only}` | wA, wB, wC | liquid water (BLYP / CCSD(T)) | as above |

## Workflow

1. **Data**: `uq data build <dataset>` from `data/<dataset>/dataset.yaml` ([datasets](docs/datasets.md)).
2. **Train**: `uq train <experiment> --gpus 6 7` trains missing members, parents first ([experiments](docs/experiments.md)).
3. **Evaluate**: `uq evaluate <experiment> --gpu 6` runs every split, then the five-split aggregate ([evaluation](docs/evaluation.md)).
4. **Record**: add the result to `results/results.md` with commit, command, data, seeds, metrics and output paths.
5. **Figures**: paper figures are rebuilt from tracked outputs on CPU (`uq figures iclr27`).

New datasets, models, UQ methods and studies: [docs/extending.md](docs/extending.md).

## Rules that keep results valid

- Five dataset splits are the statistical replicates; the ten members of a split are
  one ensemble. Intervals are 95% Student-t over splits ([concepts](docs/concepts.md)).
- Checkpoints are selected per member by validation Gaussian NLL, never by test data.
- Never use more than **two GPUs** at a time, counting every job (shared machine):
  `uq train` refuses more than two and `uq evaluate` uses one per invocation.
- Do not overwrite, delete or retrain datasets, checkpoints or evaluation outputs
  without a recorded reason; the per-split outputs cost GPU days.
- A number is citable only with a ledger entry.

## Tests

```bash
python -m unittest discover -s uq/tests -t .      # uq: statistics, CLI, command snapshots, AL (CPU)
uq check                                          # all experiment/dataset specs
python -m pytest tests --ignore=tests/test_cueq_oeq.py   # upstream MACE (slow; needs pytest)
```

## Working with agents

`CLAUDE.md` (also `AGENTS.md`) holds the rules and map in under a hundred lines;
`.claude/rules/` adds area-specific rules when those files are touched, and
`.claude/skills/` has step-by-step procedures (train, evaluate, record a result,
add an experiment, rebuild paper figures).

## History

The layout before 2026-09-26 (letters `experiment_A` ... and loose scripts) is the tag
`iclr27-submission`; old paths translate with `uq where <path>`
([migration map](docs/migration-2026-09.md)). MACE is MIT-licensed ([LICENSE.md](LICENSE.md)).
