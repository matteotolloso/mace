# Reproducibility and the results ledger

## Claims require a ledger entry

- State an experimental result (chat, README, docs, paper) only if it is recorded in
  `results/results.md` or read directly from an output file that an entry lists. If it
  is not recorded, say so and offer to produce and record it (`record-result` skill).
- A number is reportable only with its provenance: commit, exact command, experiment
  spec and config, dataset split(s), seeds, metric definition, output path.
- Quote values as stored in the output file; do not round or re-derive them silently.
- A result not reproduced from its recorded command is a draft; label it so.
- Entries written before 2026-09-26 use the old layout's paths and experiment letters;
  translate them with `uq where <path>` (`docs/migration-2026-09.md`), never edit them.

## What every recorded run captures

| Field | Source |
|---|---|
| Commit | `git rev-parse HEAD`, plus `git status --porcelain` if the tree is dirty |
| Command | the exact `uq ...` / `python -m ...` line |
| Configuration | `experiments/<id>/experiment.yaml` + `config.yaml`, plus overrides |
| Data | `data/<dataset>/split_<s>/` and the files/roles used |
| Seeds | dataset split seeds `0..4`; member seeds `0..9` |
| Metrics | names as computed in `uq/evaluation/`, with units |
| Outputs | `experiments/<id>/evaluation/...`, `papers/<venue>/figures/out/...`, `runs/active-learning/...` |

## Statistical unit

Five dataset splits are the replicates; the ten members of a split form one ensemble
and are never a substitute for splits. If a split or member is missing, report the gap
instead of aggregating what exists.

## Protecting artifacts

- Do not overwrite, delete, rename or retrain datasets, checkpoints, per-split
  evaluation outputs or results unless the user requests that exact action (`runs/` is
  ~2 TB and GPU-costly). `uq data build` never rebuilds an existing split; keep it so.
- Prefer cache-only (CPU) recomputation (`uq aggregate`, paper figures) over GPU
  inference, and GPU inference over retraining; when reporting completion, say which
  was used and name the outputs that changed.
- GPU inference is not bit-reproducible: re-running an evaluation changes per-split
  files slightly (EU up to ~5% relative per configuration); aggregates from existing
  per-split files are byte-reproducible.
- The worktree may hold the user's uncommitted edits: never reset, checkout or revert
  changes you did not make; commit or push only when asked.
