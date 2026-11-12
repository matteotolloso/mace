# Repository layout and naming

Every path is built by `uq/paths.py`; if the layout changes, that module and this
page are the only places to update.

```text
<repo>/
├── README.md, CLAUDE.md, AGENTS.md   entry points for humans and agents
├── uq/                  research package and the `uq` command
│   ├── paths.py         directory layout (the only module that hard-codes it)
│   ├── registry.py      experiment/dataset specs (YAML), legacy-letter lookup, validation
│   ├── cli.py           `uq list|show|check|status|train|train-member|prune-checkpoints|evaluate|aggregate|summary|where|data|al|figures`
│   ├── legacy.py        old experiment letters and pre-2026-09 paths -> current paths
│   ├── datasets/        dataset builders (ANI-1x splitters, water n2p2 converter)
│   ├── training/        member launcher, <=2-GPU scheduler, completeness check, checkpoint pruning
│   ├── evaluation/      one module per evaluation step, pipeline, aggregation, statistics
│   ├── active_learning/ active-learning workflow
│   └── tests/           unit tests (python -m unittest discover -s uq/tests -t .), snapshots/
├── data/                datasets: <dataset>/dataset.yaml (tracked) + split_<s>/ (large files ignored)
├── experiments/         one folder per experiment (tracked): experiment.yaml, config.yaml, evaluation/
├── runs/                heavy, regenerable artifacts (git-ignored, ~650 GB): checkpoints, W&B, AL runs
├── results/             results.md (the ledger of citable results), tables/, figures/
├── papers/iclr27/       figure scripts and outputs of the ICLR 2027 submission, drafts, rebuttal
├── papers/strategy/     assessment of the work, review synthesis, development plan, findings, literature
├── notebooks/           exploratory notebooks and one-off analyses
├── docs/                documentation (this folder)
├── environment.yml      exact conda environment of the ICLR27 results
├── mace/ tests/ scripts/ setup.cfg pyproject.toml   upstream MACE fork (keep in place)
└── ICLR27-UQ-MF/        manuscript: separate Git repo synced with Overleaf (ignored here)
```

## Experiment ids

`<dataset>/<model>-<uq>/<protocol>`, which is also the folder under `experiments/`
and `runs/`:

- `<dataset>`: a folder of `data/` with a `dataset.yaml` — the data *and* its split
  design, e.g. `ani1x-system`, `ani1x-energy`, `water-random`;
- `<model>-<uq>`: model and uncertainty method: `mace-mve` (MACE, ensemble of ten
  mean-variance members) or `mace-mse` (MACE, ensemble of ten plain MSE members, the
  EU-only baseline);
- `<protocol>`: `lf-only`, `lf-to-hf`, `hf-only`.

The nine experiments of the ICLR27 paper keep their old letters as aliases
(`uq show B`); the table is in [`experiments/README.md`](../experiments/README.md).

## Per experiment

```text
experiments/<id>/
  experiment.yaml          spec: dataset, protocol, parent, splits, members, evaluation steps
  config.yaml              MACE hyperparameters (no paths; the launcher adds them)
  evaluation/              five-split aggregates *.csv / *.svg (tracked)
    split_<s>/             per-split outputs of every evaluation step (tracked, GPU-costly)
runs/<id>/split_<s>/       (ignored)
  checkpoints/             <name>_run-<m>_epoch-<e>.pt, final <name>_run-<m>.model; after
                           pruning only epochs % 5 == 0 + each member's best (PRUNED.json)
  results/                 MACE per-epoch logs <name>_run-<m>_train.txt (checkpoint selection)
  logs/, models/           MACE logs; last model written by MACE
```

`<name>` is the config's `name` (`mace`), `<m>` the member seed, `<s>` the split.

## Per dataset

```text
data/<dataset>/
  dataset.yaml             fidelities, file roles, energy keys, build arguments (tracked)
  split_<s>/               *.xyz (ignored), split_summary.json + system_assignments.csv (tracked)
    geometry/<file>.npz    support-filter geometry cache (tracked, ANI-1x only)
data/raw/                  ani1x-release.h5 (ignored), water-n2p2/<theory>/input.data (tracked)
data/legacy/               unused legacy exports (ignored)
```

## Other `runs/` content

```text
runs/active-learning/ani1x-energy/<run>/   active-learning runs and aggregates
runs/wandb/                                W&B local files (`--wandb_dir`)
runs/_logs/                                logs of `uq train` jobs
runs/_migration/                           records of the 2026-09 restructure (golden commands, manifest)
```

Set `UQ_RUNS=/other/disk/runs` to keep heavy artifacts elsewhere; nothing else changes.

## Tracked vs ignored

Tracked: code, specs, configs, evaluation outputs (per split and aggregates),
dataset metadata and geometry caches, paper figure outputs, the ledger and
`results/tables/`. Ignored: `runs/`, dataset `*.xyz`/`*.h5`, `ICLR27-UQ-MF/`,
logs and bytecode. The pre-2026-09 layout is at tag `iclr27-submission`;
`uq where <old path>` translates old paths ([migration-2026-09.md](migration-2026-09.md)).
