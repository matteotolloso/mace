---
name: record-result
description: Record an experiment or analysis in the results ledger (results/results.md) with its commit, command, configuration, dataset split, seeds, metrics and output paths. Use after producing or regenerating any number that may be quoted in the paper, README or chat.
---

# Record a result in the ledger

1. Collect provenance from the machine, not from memory:
   - `git rev-parse HEAD` and `git status --porcelain` (note a dirty tree).
   - The exact command that produced the outputs, run from the repository root.
   - The configuration file(s) (`experiment_X/config_X.yml` or the script's
     arguments) and any CLI overrides.
   - Dataset directory, test file(s) and split seeds; member seeds (`0..9`).
   - Whether the work used cached data, CPU inference, GPU inference or retraining.
2. Read every metric value from its output file (CSV/JSON written by the script).
   Quote the stored value with units and name the file it came from. Never type a
   number you did not read from a file during this task.
3. Check the statistical contract before recording: five dataset splits, ten
   members per split, intervals over splits (see
   `.claude/rules/statistics-and-plots.md`). If anything is missing, record the
   gap and set `Status: draft`.
4. Append a new entry at the end of `results/results.md` using the template in
   that file. Do not edit or delete earlier entries; mark superseded ones.
5. Fill unknown fields with `TODO` and tell the user which ones are missing.
6. Do not commit; report the new entry to the user.
