# Runs (git-ignored)

Heavy, regenerable artifacts; only this README is tracked. Laid out by `uq/paths.py`
(set `UQ_RUNS` to move the whole tree to another disk).

```text
<experiment id>/split_<s>/checkpoints/   epoch *.pt (every 5th + each member's best), final *.model, PRUNED.json
<experiment id>/split_<s>/results/       MACE per-epoch logs (checkpoint selection reads these)
<experiment id>/split_<s>/{logs,models}/ MACE logs, last written model
active-learning/ani1x-energy/<run>/      active-learning runs and aggregates (docs/active-learning.md)
wandb/                                   W&B local files
_logs/                                   logs of `uq train` jobs
_migration/                              record of the 2026-09 restructure (golden commands, manifest)
```

About 650 GB (2026-09-26): experiments ~330 GB, active learning ~330 GB. Do not delete or
overwrite anything here without an explicit decision: retraining costs GPU days, and
evaluation outputs reference these checkpoints. Epoch checkpoints that no analysis
reads are removed only with `uq prune-checkpoints` (docs/experiments.md).
