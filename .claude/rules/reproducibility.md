# Reproducibility and the results ledger

## Claims require a ledger entry

- Never state an experimental result (in chat, README, notes or the paper) unless
  it is recorded in `results/results.md`, or read directly from a generated
  artifact that a ledger entry points to. If it is not recorded, say so and
  offer to produce and record it.
- A number is reportable only with its provenance: code commit, exact command,
  configuration file, dataset split(s), random seed(s), metric definition, and
  output path.
- Do not round, re-derive or "clean up" numbers when copying them. Quote the
  artifact's value and say which file it came from.
- A result that was not reproduced from the recorded command is a draft: label
  it as such in the ledger's interpretation field.

## What every recorded run must capture

Use the template in `results/results.md` (or the `record-result` skill):

| Field | Where it comes from in this repository |
|---|---|
| Commit | `git rev-parse HEAD`, plus `git status --porcelain` if the tree is dirty |
| Command | the exact shell line, run from the repository root |
| Configuration | `experiment_X/config_X.yml`, plus any CLI overrides |
| Dataset and split | `dataset/ani1x_{system,energy}_split_<s>/` or `dataset/water_<s>/`, and the test file used |
| Seeds | dataset split seeds `0..4`; ensemble member seeds `0..9` (`--seed`) |
| Metrics | the metric names as computed by `eval/reliability.py` / `eval/epoch_quality.py`, with units |
| Outputs | `experiment_X/evaluation/...`, `new_figures/out/...`, or `active_learning/ani_energy/runs/...` |

## Statistical unit

- Five dataset splits (`0..4`) are the statistical replicates; confidence
  intervals are computed over the five split-level results.
- The ten members of a split form **one** ensemble. They are never independent
  samples, and never a substitute for splits.
- Never silently use fewer than five splits or fewer than ten members. If a
  split or member is missing, report the gap instead of aggregating what exists.

## Protecting artifacts

- Do not overwrite, delete, rename or retrain datasets, checkpoints, caches or
  results unless the user explicitly requests that exact action. Checkpoints and
  caches occupy more than 1 TB and are expensive to reproduce.
- Dataset generation (`dataset/make_dataset*.sh`) can replace split files that
  completed experiments depend on; never run it against an existing split.
- Prefer cache-only (CPU) recomputation over GPU inference, and GPU inference
  over retraining. When reporting completion, state which of the three was used
  and name the exact outputs that changed.
- The worktree often contains the user's own uncommitted plots and edits. Never
  reset, checkout or revert changes you did not make. Do not commit or push
  unless asked.
