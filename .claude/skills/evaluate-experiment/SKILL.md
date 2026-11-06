---
name: evaluate-experiment
description: Run the per-split GPU evaluation of an experiment and its five-split aggregate with `uq evaluate`, or rebuild aggregates on CPU with `uq aggregate`, then read the headline metrics with `uq summary`. Use when asked to evaluate a newly trained experiment, regenerate evaluation outputs, or report its metrics.
---

# Evaluate an experiment

1. Preconditions: `uq status <exp>` shows 10/10 members in every split (and for the
   parent of an `lf-to-hf` experiment).
2. Existing outputs of the ICLR27 experiments are tracked and cited. GPU re-evaluation
   changes them slightly (not bit-reproducible). Do not re-run them unless the user asks;
   to only redraw aggregates use `uq aggregate <exp>` (CPU, byte-reproducible CSVs).
3. `uq evaluate <exp> --gpu G --dry-run` and check the steps (`docs/evaluation.md`).
   One GPU per invocation; at most three GPUs across all running jobs.
4. Launch detached: `nohup uq evaluate <exp> --gpu G > runs/_logs/eval_<name>.log 2>&1 &`.
   It stops at the first failing step; rerun after fixing (steps reuse finished CSVs).
   `--splits`/`--steps`/`--no-aggregate` narrow a rerun.
5. Results: `uq summary <exp> [--test ood] [--cal]` prints mean [95% CI] per metric and
   signal with the source file. Interpret with `docs/evaluation-outputs.md` (which
   outputs can back a claim; energy-split OOD epoch curves are unfiltered).
6. Record reported numbers with the `record-result` skill; commit only if asked.
7. Disk: a newly trained experiment keeps a checkpoint for every epoch. Once it is
   evaluated and recorded, show the user `uq prune-checkpoints <exp>` (dry run) and run
   `--apply` only if they ask (`docs/experiments.md#pruning-epoch-checkpoints`).
