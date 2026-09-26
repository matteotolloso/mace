# The 2026-09 restructure: old -> new

On 2026-09-26 the repository moved from per-letter `experiment_X/` folders and loose
scripts to the `uq` package and the layout in [layout.md](layout.md). Nothing was
recomputed: files were renamed on the same filesystem (137,638 files), and every
aggregate, paper figure and paper number was regenerated and compared (CSVs and PNGs
byte-identical; SVG/PDF identical up to timestamps and random element ids).

- The old layout is the tag **`iclr27-submission`** (`c3bcc56`, the ICLR 2027
  submission). `git worktree add ../mace-old iclr27-submission` checks it out alongside.
- `uq where <old path>` prints the new location of any old path (relative or
  absolute), using `uq/legacy.py`.
- Old ledger entries, the paper's provenance comments and the active-learning run
  records keep their old paths; they are historical and resolve through `uq where`.
- Details of the process: `docs/restructure/PLAN.md` (plan, verification record) and
  `docs/restructure/migrate_layout.py` (the applied moves; manifest in
  `runs/_migration/manifest.json`).

## Experiments

| Legacy | Experiment id |
|---|---|
| A / B / E | `ani1x-system/mace-mve/{lf-only,lf-to-hf,hf-only}` |
| C / D / F | `ani1x-energy/mace-mve/{lf-only,lf-to-hf,hf-only}` |
| wA / wB / wC | `water-random/mace-mve/{lf-only,lf-to-hf,hf-only}` |

## Paths

| Old | New |
|---|---|
| `experiment_X/config_X.yml` | `experiments/<id>/config.yaml` (path fields removed) |
| `experiment_X/evaluation/*` | `experiments/<id>/evaluation/*` |
| `experiment_X/evaluation/cache/split_<s>/` | `experiments/<id>/evaluation/split_<s>/` |
| `experiment_X/{checkpoints,models,logs,results}_<s>/` | `runs/<id>/split_<s>/{checkpoints,models,logs,results}/` |
| `dataset/ani1x_{system,energy}_split_<s>/` | `data/ani1x-{system,energy}/split_<s>/` |
| `dataset/water_<s>/` | `data/water-random/split_<s>/` (sources also in `data/raw/water-n2p2/`) |
| `dataset/ani1x-release.h5`, `dataset/ani1x-processed/` | `data/raw/ani1x-release.h5`, `data/legacy/ani1x-processed/` |
| `eval/cache/geometry/<kind>_<s>_<file>.npz` | `data/ani1x-<kind>/split_<s>/geometry/<file>.npz` |
| `new_figures/_cache/epochs/<X>_<s>.npz` | `experiments/<id>/evaluation/split_<s>/epoch_predictions_cc_test_ood.npz` |
| `new_figures/` (scripts, `out/`, `paper_numbers.py`) | `papers/iclr27/figures/` |
| `new_figures/README.md`, `new_figures/results.tex` | `papers/iclr27/README.md`, `papers/iclr27/drafts/results.tex` |
| `active_learning/ani_energy/runs/` | `runs/active-learning/ani1x-energy/` |
| `wandb/`, `experiments/runs/` | `runs/wandb/`, `runs/_logs/` |
| `experiments/evaluation_figures.md` | `docs/evaluation-outputs.md` |
| README project sections / MACE README | `docs/*.md` / `docs/upstream-mace.md` |

## Code

| Old | New |
|---|---|
| `eval/<module>.py` | `uq/evaluation/<module>.py` (`aggregate_replicates` -> `aggregate`, `csv_cache_utils` -> `csv_cache`, `epoch_quality_finetune_{mixed,same}_dataset` -> `epoch_quality_{mixed,same}`, `finetune` -> `finetune_comparison`, `plt_*`/`plot_idealized_au_eu` -> `extra/`) |
| `eval/test_plotting.py` | `uq/tests/test_evaluation.py` |
| `new_figures/{run_,}epoch_predictions.py`, `build_geometry_cache.py` | `uq.evaluation.epoch_predictions`, `uq data support --build` |
| `dataset/ani1x_{system,energy}_splitter.py`, `convert_water_n2p2_to_extxyz.py` | `uq/datasets/ani1x_{system,energy}.py`, `water_n2p2.py` |
| `dataset/ani1x_analysis.ipynb`, coverage script/CSV | `notebooks/ani1x/` |
| `active_learning/ani_energy/*.py`, tests | `uq/active_learning/*.py`, `uq/tests/active_learning/` |
| `experiments/scripts/water_split_overlap.py` | `python -m uq.datasets.water_overlap` |

## Commands

| Old | New |
|---|---|
| `bash experiment_X/train_X.sh s m gpu` | `uq train-member X s m --gpu gpu` |
| `experiments/scripts/train_water_splits.py` | `uq train <experiments> --gpus G [G]` |
| `./check_experiments.sh` | `uq status` |
| `experiment_X/eval_X.sh gpu`, `eval_water.sh`, `eval.sh` | `uq evaluate X --gpu gpu` |
| `bash eval.sh --plots-only` | `uq aggregate` |
| `bash dataset/make_dataset.sh`, `make_dataset_w.sh` | `uq data build <dataset>` |
| `python eval/support_filter.py --build/--report` | `uq data support [--build]` |
| `bash active_learning/ani_energy/run.sh gpu ...` / `run_all.sh gpu ...` | `uq al run gpu ...` / `uq al run-all gpu ...` |
| `bash new_figures/make_all.sh` | `uq figures iclr27` |

The new launchers issue the same commands as the old scripts (verified token by
token: training 36/36 with `--wandb_dir runs/wandb` added, evaluation 574/574,
aggregation 9/9). One-time environment step after pulling:
`pip install --no-deps --no-build-isolation -e .` (registers `uq` and its command).
