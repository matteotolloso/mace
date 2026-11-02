# Repository restructure (2026-09) — plan and record

**Status: complete (2026-09-26): tag `restructure-2026-09`** (branch `restructure`, its
phase commits squashed into one at the user's request; the phase hashes cited below no
longer exist). This file is kept as
the record of what was decided, done and verified; current documentation is in
`docs/` (start at `docs/README.md`, old -> new map in `docs/migration-2026-09.md`).

## Decisions (user, 2026-09-26)

- Layout: **nested by meaning** — `<dataset>/<model>-<uq>/<protocol>`; tracked
  definitions and evaluation in `experiments/`, heavy artifacts in a git-ignored
  `runs/` tree with the same names; legacy letters stay as aliases.
- Research package and CLI: **`uq`** (`uq status`, `python -m uq.evaluation.reliability`).
- Git: tag `iclr27-submission` = `c3bcc56` (done); all work on local branch
  `restructure` (commits allowed there; nothing is pushed; main untouched).
- Old launchers (`experiment_X/train_X.sh`, `eval_X.sh`, `eval.sh`,
  `eval_water.sh`, `check_experiments.sh`, one-off water scripts): **delete** once
  the CLI is verified to issue identical commands. History + tag keep them.
- Paper manuscript clone `ICLR27-UQ-MF/` stays at the root during review/rebuttal.
  Its local CLAUDE.md imports `../.claude/rules/{paper-writing,reproducibility}.md`:
  keep those two file names.

## Invariants during the migration

- Never delete, rewrite or retrain datasets, checkpoints, caches or results; heavy
  directories are only **renamed** (same filesystem, `/raid`), recorded in
  `runs/_migration/manifest.json`, reversible with the revert command.
- Numbers must not change: aggregate CSVs byte-identical, SVGs identical after
  `runs/_migration/golden/svgnorm.py`, `paper_numbers` output byte-identical.
- ≤ 2 GPUs at any time (no GPU is needed for this migration).

## Target layout

```
mace/ tests/ scripts/ setup.cfg pyproject.toml   upstream MACE fork (unchanged places)
uq/                      research package + CLI (`uq ...`)
  paths.py               the only module that knows the directory layout
  registry.py            experiment/dataset specs (experiments/**/experiment.yaml, data/*/dataset.yaml)
  legacy.py              letter aliases, old->new path translation, migration executor
  cli.py, __main__.py    `uq list|status|train|train-member|evaluate|aggregate|summary|data|al|figures`
  datasets/              ani1x system/energy splitters, water n2p2 converter, builder
  training/              member launcher, <=2-GPU scheduler, completeness
  evaluation/            reliability, epoch quality, distributions, energy quantiles,
                         aggregation, statistics, support filter, pipeline (spec -> commands)
  active_learning/       AL workflow (ex active_learning/ani_energy)
  tests/
experiments/             tracked: README index, <id>/{experiment.yaml, config.yaml, evaluation/}
  ani1x-system/mace-mve/{lf-only,lf-to-hf,hf-only}     A B E
  ani1x-energy/mace-mve/{lf-only,lf-to-hf,hf-only}     C D F
  water-random/mace-mve/{lf-only,lf-to-hf,hf-only}     wA wB wC
  active-learning/ani1x-energy/                        AL study definition + aggregate results
data/                    datasets (big files ignored; dataset.yaml, split metadata, geometry cache tracked)
  raw/ani1x-release.h5, raw/water-n2p2/<theory>/input.data
  ani1x-system/split_<s>/  ani1x-energy/split_<s>/  water-random/split_<s>/<theory>/
runs/                    git-ignored heavy artifacts: <id>/split_<s>/{checkpoints,models,logs,results},
                         active-learning/ani1x-energy/<run>/, wandb/, _logs/, _migration/
results/                 results.md (ledger, name kept), tables/ (now tracked), figures/
papers/iclr27/           figures/ (ex new_figures, incl. paper_numbers.py) + figures/out/, drafts/, README
docs/                    topic docs, migration map, upstream MACE README
notebooks/               exploratory notebooks and one-off analyses
ICLR27-UQ-MF/            manuscript clone (independent repo, ignored) — unchanged
```

## Mappings

### Experiments (legacy letter -> id)

| Legacy | Experiment id | Parent |
|---|---|---|
| A | `ani1x-system/mace-mve/lf-only` | — |
| B | `ani1x-system/mace-mve/lf-to-hf` | A |
| E | `ani1x-system/mace-mve/hf-only` | — |
| C | `ani1x-energy/mace-mve/lf-only` | — |
| D | `ani1x-energy/mace-mve/lf-to-hf` | C |
| F | `ani1x-energy/mace-mve/hf-only` | — |
| wA | `water-random/mace-mve/lf-only` | — |
| wB | `water-random/mace-mve/lf-to-hf` | wA |
| wC | `water-random/mace-mve/hf-only` | — |

### Paths (old -> new)

| Old | New |
|---|---|
| `experiment_X/config_X.yml` | `experiments/<id>/config.yaml` (path fields removed; launcher passes them) |
| `experiment_X/evaluation/*` (aggregates) | `experiments/<id>/evaluation/*` |
| `experiment_X/evaluation/cache/split_<s>/` | `experiments/<id>/evaluation/split_<s>/` |
| `experiment_X/{checkpoints,models,logs,results}_<s>/` | `runs/<id>/split_<s>/{checkpoints,models,logs,results}/` |
| `new_figures/_cache/epochs/<X>_<s>.npz` | `experiments/<id>/evaluation/split_<s>/epoch_predictions_cc_test_ood.npz` |
| `dataset/ani1x-release.h5` | `data/raw/ani1x-release.h5` |
| `dataset/ani1x-processed/` (legacy, unused) | `data/legacy/ani1x-processed/` |
| `dataset/ani1x_{system,energy}_split_<s>/` | `data/ani1x-{system,energy}/split_<s>/` |
| `dataset/water_<s>/` | `data/water-random/split_<s>/` |
| `dataset/water_0/<theory>/input.data` | `data/raw/water-n2p2/<theory>/input.data` (split copies kept) |
| `eval/cache/geometry/<kind>_<s>_<file>.npz` | `data/ani1x-<kind>/split_<s>/geometry/<file>.npz` |
| `active_learning/ani_energy/runs/` | `runs/active-learning/ani1x-energy/` |
| `wandb/` | `runs/wandb/` |
| `experiments/runs/` (scheduler logs) | `runs/_logs/` |
| `eval/*.py` | `uq/evaluation/*.py` (see code map) |
| `dataset/*.py`, `make_dataset*.sh` | `uq/datasets/*.py`, `uq data build` |
| `utils/train_split_member.sh`, `train_X.sh` | `uq train-member` (`uq/training/launch.py`) |
| `experiments/scripts/train_water_splits.py` | `uq train` (`uq/training/scheduler.py`) |
| `check_experiments.sh` | `uq status` (`uq/training/status.py`) |
| `eval_X.sh`, `eval.sh`, `eval_water.sh`, `eval/split_cache.sh` | `uq evaluate`, `uq aggregate` (`uq/evaluation/pipeline.py`) |
| `active_learning/ani_energy/*.py` | `uq/active_learning/*.py` |
| `new_figures/` | `papers/iclr27/figures/` (`paper_numbers.py` stays in figures/; `results.tex` -> `papers/iclr27/drafts/`) |
| `experiments/evaluation_figures.md` | `docs/evaluation-outputs.md` |
| `README.md` project part / MACE part | `README.md` (short) + `docs/*.md` / `docs/upstream-mace.md` |

### Code map (eval -> uq.evaluation)

`reliability`, `epoch_quality`, `epoch_raw`, `train_curves`, `distribution`,
`energy_ood`, `support_filter`, `plot_style`, `replicate_statistics` keep their
names; `aggregate_replicates` -> `aggregate`; `csv_cache_utils` -> `csv_cache`;
`epoch_quality_finetune_mixed_dataset` -> `epoch_quality_mixed`;
`epoch_quality_finetune_same_dataset` -> `epoch_quality_same`; `finetune` ->
`finetune_comparison`; `new_figures/{epoch_predictions,run_epoch_predictions}` ->
`epoch_predictions`; exploratory `plt_*`, `plot_idealized_au_eu` -> `uq/evaluation/extra/`;
`test_plotting` -> `uq/tests/test_evaluation.py`. Output file names are unchanged.

## Verification protocol (golden references in `runs/_migration/golden/`)

- `train/<X>.txt`: commands of `train_X.sh s m 5` for s in {0,3}, m in {0,9}
  (fake `python` from `golden/shim/`). New: `uq train-member <X> s m --gpu 5 --dry-run`
  must match after path translation; known deliberate difference: `--wandb_dir runs/wandb`.
- `eval/<X>.txt`: all commands of `eval_X.sh 6`. New: `uq evaluate <X> --gpu 6 --dry-run`
  must match after translating paths and `eval/<m>.py` -> `-m uq.evaluation.<m'>`.
- `eval/plots_only.txt`: `eval.sh --plots-only` commands; `check_experiments.txt`: 450/450.
- Determinism baseline (done): old CPU aggregation reproduces every tracked CSV
  byte-for-byte and every SVG after `svgnorm.py` (177/177).
- After the move: `uq aggregate --all` -> `git status` shows no CSV change and
  SVGs identical after normalization; figures rebuilt with `SOURCE_DATE_EPOCH`
  compare equal (PNG pixels / normalized SVG); `papers/iclr27/figures/paper_numbers.py` output
  byte-identical to `golden/iclr27_main_text_numbers.txt`; AL tests pass; a tiny
  CPU inference with old code (worktree at the tag) and new code gives identical CSVs.

## Phases

- [x] 0. Inventory, design, user decisions; tag `iclr27-submission`; branch `restructure`;
      carry-over commit `f81c3f6`; golden references captured.
- [x] 1. Package skeleton: `uq/{__init__,__main__,paths,registry,legacy,cli}.py`,
      specs (`experiments/**/experiment.yaml`, `data/*/dataset.yaml`), configs moved,
      editable reinstall done (`pip install --no-deps --no-build-isolation -e .`; `uq` on PATH).
- [x] 2. Port code with `git mv` + import fixes (evaluation, datasets, training,
      active learning incl. provenance path aliases, figures -> papers/iclr27/figures).
- [x] 3. Pipeline + launcher + scheduler + status + summary; dry-run equivalence vs golden:
      training 36/36, evaluation 574/574, aggregate 9/9 (`runs/_migration/equivalence.py`).
- [x] 4. Layout migration applied (`docs/restructure/migrate_layout.py`, 425 steps,
      137,638 files, counts verified; manifest `runs/_migration/manifest.json`).
- [x] 5. Post-move verification: `uq status` 450/450; `uq aggregate` -> CSVs byte-identical,
      177/177 SVGs identical after normalization; 20 paper figures identical (PNG bytes,
      SVG/PDF normalized); paper_numbers identical except the printed AL path; AL reports,
      aggregate (100 inputs) and manifests verify through path aliases (sources: only the
      edited code/configs + mace/modules/models.py, which changed on 2026-09-22 before the
      restructure); moved inference modules differ from the tag only in imports/docstrings
      (GPU inference itself is not bit-reproducible: EU varies up to ~5% relative run to run).
- [x] 6. Docs: README (short), docs/ (concepts, layout, datasets, experiments, evaluation,
      evaluation-outputs, active-learning, known-issues, extending, migration map,
      upstream MACE README), per-folder READMEs, ledger note + entry, .gitignore.
- [x] 7. Agent layer: CLAUDE.md (63 lines), AGENTS.md pointer, path-scoped rules
      (always-on context ~108 lines, was ~409), skills train-experiment,
      evaluate-experiment, new-experiment, add-dataset, paper-figures, record-result.
- [x] 8. Final review: CLI fixes (`uq --help`, broken pipe) + 5 CLI tests (35 total),
      stale references fixed, memory updated.
- [x] 9. Footprint and safeguards (user request "reduce the memory footprint"): epoch
      checkpoints pruned with `uq prune-checkpoints` (83,618 files, 1245.9 GiB; ledger
      entry of 2026-09-26), `git gc` of unreachable objects, merged local branches
      deleted, `.codex` and build leftovers removed; `environment.yml`; fail-fast
      `registry.BACKENDS`; command snapshot tests, an index test and machine-independent
      old-path translation (46 tests); branch squashed and tagged `restructure-2026-09`.

## Follow-ups for the user

- Merge `restructure` into `main` when satisfied (nothing was pushed), and push the tags
  `iclr27-submission` and `restructure-2026-09` with it (the ledger cites them). The
  untracked artifacts are already in the new layout, so the old scripts on `main` no
  longer find them; stay on this branch. (`migrate_layout.py revert` still moves the
  files back, but the old evaluation code refuses the pruned checkpoint folders.)
- Keep the anonymous code mirror pinned to the submission commit during reviews.
- Other checkouts/environments: `conda env create -f environment.yml`, then
  `pip install --no-deps --no-build-isolation -e .`.
- The remote still has the merged branches `mve`, `mve_sp`, `revision`, `water`
  (deleted locally); delete them there if they are no longer needed.
