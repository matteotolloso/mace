---
name: record-result
description: Record an experiment or analysis in the results ledger (results/results.md) with its commit, command, spec/config, dataset split, seeds, metrics and output paths. Use after producing or regenerating any number that may be quoted in a paper, the README, the docs or chat.
---

# Record a result in the ledger

1. Collect provenance from the machine, not from memory:
   - `git rev-parse HEAD` and `git status --porcelain` (note a dirty tree).
   - The exact command (`uq ...` or `python -m uq....`), run from the repository root.
   - The experiment(s): `experiments/<id>/experiment.yaml` + `config.yaml` (`uq show <id>`)
     and any overrides.
   - Data: `data/<dataset>/split_<s>/` and the roles/files used; split seeds; member seeds.
   - Compute: cached data / CPU / GPU inference / retraining.
2. Read every value from its output file during this task (`uq summary <id>` prints the
   aggregate summaries with their file path). Quote stored values with units and name
   the file. Never type a number you did not read from a file now.
3. Check the statistical contract: five splits, ten members per split, intervals over
   splits (`docs/concepts.md`). If anything is missing, record the gap and set
   `Status: draft`.
4. Append a new entry at the end of `results/results.md` using its template. Never edit
   or delete earlier entries (their pre-2026-09-26 paths stay as written); mark a
   superseded entry in a new one.
5. Fill unknown fields with `TODO` and tell the user which ones are missing.
6. Do not commit unless asked; report the new entry to the user.
