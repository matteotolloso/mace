"""Delete per-epoch checkpoints that no analysis reads: ``uq prune-checkpoints``.

    uq prune-checkpoints                       # dry run over all experiments: what would go
    uq prune-checkpoints B --apply             # delete for one experiment (irreversible)

MACE saves a checkpoint every epoch (`save_all_checkpoints: True`), but every epoch
analysis samples epochs with ``epoch % n == 0`` for n in {5, 10}, and final ensembles use
each member's best-validation checkpoint. Per split, this keeps

- every ``<name>_run-<m>_epoch-<e>.pt`` with ``e % stride == 0`` (stride 5 by default),
- each member's best-validation checkpoint (validation Gaussian NLL, as the pipeline selects),
- every final ``<name>_run-<m>.model`` (and every file that is not an epoch checkpoint),

and deletes the other epoch checkpoints. It refuses splits with incomplete members.
Before deleting, it runs the pipeline's own selection code (sampled epochs for n = 5
and 10, best checkpoints) on a simulated pruned folder (symlinks to the kept files plus
the pruning record) and refuses if anything would change; after deleting it checks the
real folder again. Each pruned split gets ``checkpoints/PRUNED.json`` (rule, date,
every deleted file and its size); checkpoint selection accepts exactly those gaps.
Afterwards, epoch analyses at a finer stride and checkpoint re-selection by another
criterion are no longer possible.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from uq import paths, registry

STRIDES_USED = (5, 10)   # every-n-epochs values used by the evaluation pipeline
RECORD = "PRUNED.json"


def _pipeline_view(exp, split, checkpoints=None):
    """What the evaluation pipeline selects from a split's checkpoint folder."""
    from uq.evaluation.epoch_quality import discover_checkpoint_map, select_common_epochs
    from uq.evaluation.reliability import select_best_checkpoints

    checkpoints = Path(checkpoints or paths.checkpoints_dir(exp.id, split))
    checkpoint_map = discover_checkpoint_map(checkpoints, exp.run_name)
    sampled = {n: select_common_epochs(checkpoint_map, n) for n in STRIDES_USED}
    best = sorted(p.name for p in select_best_checkpoints(checkpoints, paths.results_dir(exp.id, split),
                                                          exp.run_name, "loss", "min", None))
    return {"sampled_epochs": sampled, "best": best}


def plan(exp, split, stride=5):
    """(files to delete, bytes) for one split; raises if the split is not prunable."""
    from uq.evaluation.reliability import CHECKPOINT_PATTERN

    if 5 % stride:
        raise SystemExit(f"stride {stride} would drop epochs the pipeline samples (it must divide 5)")
    incomplete = [m for m in exp.members if not exp.member_complete(split, m)]
    if incomplete:
        raise SystemExit(f"{exp.label} split {split}: members {incomplete} are incomplete; not pruning")
    best = set(_pipeline_view(exp, split)["best"])
    doomed = []
    for path in sorted(paths.checkpoints_dir(exp.id, split).glob("*.pt")):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None or match.group("experiment") != exp.run_name:
            continue
        if int(match.group("epoch")) % stride == 0 or path.name in best:
            continue
        doomed.append(path)
    return doomed, sum(p.stat().st_size for p in doomed)


def _simulated_view(exp, split, doomed, record):
    """Pipeline view of the folder as it would be after pruning (nothing is deleted)."""
    source = paths.checkpoints_dir(exp.id, split)
    gone = {p.name for p in doomed}
    with tempfile.TemporaryDirectory(dir=paths.RUNS) as tmp:
        for path in source.iterdir():
            if path.name not in gone and path.name != RECORD:
                os.symlink(path, Path(tmp) / path.name)
        (Path(tmp) / RECORD).write_text(json.dumps(record))
        return _pipeline_view(exp, split, tmp)


def prune_split(exp, split, stride=5):
    before = _pipeline_view(exp, split)
    doomed, size = plan(exp, split, stride)
    if not doomed:
        return 0, 0
    record_path = paths.checkpoints_dir(exp.id, split) / RECORD
    record = json.loads(record_path.read_text()) if record_path.exists() else {"events": []}
    event = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "stride": stride,
             "rule": f"keep epoch % {stride} == 0, best-validation checkpoint (loss/min) per member, all .model",
             "freed_bytes": size, "deleted": [{"file": p.name, "bytes": p.stat().st_size} for p in doomed]}
    record["events"].append(event)
    if _simulated_view(exp, split, doomed, record) != before:
        raise SystemExit(f"{exp.label} split {split}: pruning would change what the pipeline selects; nothing deleted")
    record_path.write_text(json.dumps(record, indent=1))
    for path in doomed:
        os.remove(path)
    if _pipeline_view(exp, split) != before:
        raise SystemExit(f"{exp.label} split {split}: pipeline selection changed after pruning; stopping")
    return len(doomed), size


def main(names=(), stride=5, apply=False) -> int:
    logging.disable(logging.CRITICAL)
    gib = 1024 ** 3
    total_files = total_bytes = 0
    for exp in registry.resolve(names):
        files = size = 0
        for split in exp.splits:
            if apply:
                n, b = prune_split(exp, split, stride)
            else:
                doomed, b = plan(exp, split, stride)
                n = len(doomed)
            files, size = files + n, size + b
        total_files, total_bytes = total_files + files, total_bytes + size
        print(f"{exp.label:40s} {'deleted' if apply else 'would delete'} {files:6d} checkpoints, {size / gib:8.1f} GiB")
    print(f"total: {total_files} files, {total_bytes / gib:.1f} GiB" + ("" if apply else "  (dry run; add --apply)"))
    return 0
