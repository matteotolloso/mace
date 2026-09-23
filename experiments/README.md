# Experiments index

The experiment configurations, launchers and run outputs are **not** stored in
this folder. They stay in the `experiment_X/` directories at the repository root,
because every launcher resolves the repository root from its own location
(`$(dirname "${BASH_SOURCE[0]}")/..`) and the evaluation caches, aggregates, paper
figures and the active-learning code all use those relative paths. Moving them
would break training, evaluation and the more than 1 TB of existing artifacts.

| What | Where |
|---|---|
| Configurations | `experiment_{A..F,wA..wC}/config_X.yml` (paths ending in `_0` are defaults that the launcher overrides per split) |
| Training launcher (one member) | `experiment_X/train_X.sh <split:0-4> <member:0-9> <gpu>`, via `utils/train_split_member.sh` |
| Evaluation launcher (five splits) | `experiment_X/eval_X.sh <gpu>`; all ANI families: `eval.sh`; water: `eval_water.sh` |
| Completeness check | `check_experiments.sh` |
| Run outputs | `experiment_X/{checkpoints,models,results,logs}_<split>/` |
| Per-split evaluation caches | `experiment_X/evaluation/cache/split_<split>/` |
| Five-split aggregates (CSV, SVG) | `experiment_X/evaluation/`; what each figure shows and whether it is paper-safe: `evaluation_figures.md` |
| Active-learning runs | `active_learning/ani_energy/runs/` (Git-ignored) |
| Rebuilt paper figures | `new_figures/out/` |
| Experiment matrix and protocol | `README.md` and `CLAUDE.md` |

## `scripts/`

Ad-hoc launch scripts that are not part of the pipeline, kept as a record of how
runs were scheduled. Run them from the repository root.

- `scripts/runner`: launched experiment F members for splits 4 and 1 on GPUs 4-7
  (moved here from the repository root; its content is unchanged).
