# Agent guide — UQ for multi-fidelity atomistic models

Research fork of MACE: ensembles of MACE models with a mean-variance (MVE) head,
trained on low-fidelity (DFT), high-fidelity (coupled-cluster) or both
(pretrain -> fine-tune), studied through the aleatoric/epistemic decomposition under
distribution shift and during training; ICLR 2027 submission + follow-ups. Two
workspaces share this root: this repository (code, data, experiments, results) and
`ICLR27-UQ-MF/` (the manuscript: separate Git repo synced with Overleaf, own CLAUDE.md).

## Orient cheaply (prefer these to reading files)

```bash
uq list | uq status | uq show <exp> | uq summary <exp> --test ood | uq where <old path> | uq --help
```

Experiments are `<dataset>/<model>-<uq>/<protocol>` (e.g. `ani1x-energy/mace-mve/lf-to-hf`);
legacy letters A-F, wA-wC are aliases (`uq show D`). Docs: `docs/README.md` indexes one
page per topic — read only the page you need. Environment: `conda activate mace`
(Python 3.11, torch 2.4, e3nn 0.4.4; exact pins in `environment.yml`); `uq` is installed in
editable mode (`pip install --no-deps --no-build-isolation -e .` after adding a top-level package).

## Map

| Path | What |
|---|---|
| `uq/` | research package; `paths.py` is the only place that knows the layout; `registry.py` reads specs |
| `experiments/<id>/` | `experiment.yaml` (spec), `config.yaml` (MACE hyperparameters), `evaluation/` (tracked outputs; `split_<s>/` per split) |
| `data/<dataset>/` | `dataset.yaml` + `split_<s>/` (xyz/h5 ignored) |
| `runs/` | ignored, ~650 GB: `<id>/split_<s>/{checkpoints,results,logs,models}`, `active-learning/`, `wandb/` |
| `results/results.md` | ledger: the only source of citable numbers; `results/tables/` |
| `papers/iclr27/` | figure scripts (`figures/`, outputs `figures/out/`), `paper_numbers.py`, drafts |
| `mace/`, `tests/` | upstream MACE fork; MVE code only inside `### MVE ###` fences |

## Non-negotiable

- Five dataset splits (seeds 0-4) are the replicates; ten members (seeds 0-9) form one
  ensemble, never independent samples. Intervals are over splits. Never silently use
  fewer splits or members.
- LF->HF member (s, m) is fine-tuned from parent member (s, m). Checkpoints are selected
  per member by validation Gaussian NLL only; test sets never select anything.
- **Never use more than 2 GPUs at the same time**, counting every running job (shared
  machine; check `nvidia-smi`). `uq train` enforces it; `uq evaluate` uses one GPU.
- Never overwrite, delete, rename or retrain datasets, checkpoints, per-split evaluation
  outputs or results unless the user asks for that exact action.
- Never claim a result that is not in `results/results.md` (or read from a file an entry
  lists); record new ones with the `record-result` skill.
- Reportable reliability numbers use the support filter and `trim: 0.0`.
- Never reset or revert changes you did not make. Do not commit or push unless asked.

## Working rules

- Build paths with `uq.paths` / `uq.registry`; never hard-code `runs/...` or letters in new code.
- Do not open `*_raw.csv` (5k-50k rows), `.npz`, checkpoints or W&B files to "look":
  use `uq summary`, `uq status`, or a pandas one-liner that prints a few numbers.
- Verify with the narrowest check: `uq check`; `python -m unittest discover -s uq/tests -t .`
  (46 tests, CPU; they pin every recorded experiment's commands); `uq evaluate <exp> --gpu G
  --dry-run`; `uq train ... --dry-run`.
- A change that alters numbers must be followed by `uq aggregate` (CPU) and a ledger entry.
- GPU inference is not bit-reproducible (EU varies ~5% relative per configuration).
- Epoch checkpoints exist only for every 5th epoch plus each member's best (pruned
  2026-09-26, `checkpoints/PRUNED.json`): finer epoch analyses need retraining.

Path-specific rules load automatically from `.claude/rules/` (evaluation/statistics,
experiments/data, active learning, paper, MACE fork). Procedures: skills
`train-experiment`, `evaluate-experiment`, `new-experiment`, `add-dataset`,
`paper-figures`, `record-result`.
