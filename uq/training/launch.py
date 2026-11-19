"""Train one ensemble member: ``uq train-member <experiment> <split> <member> --gpu G``.

Builds the MACE command from the experiment spec: the config holds the
hyperparameters, this module adds the split's data files, the run directories
``runs/<id>/split_<s>/{models,logs,checkpoints,results}``, the member seed, the
W&B run name and directory and, for LF->HF experiments, the parent member
``runs/<parent>/split_<s>/checkpoints/<name>_run-<member>.model`` as foundation
model. Replaces ``experiment_X/train_X.sh`` + ``utils/train_split_member.sh``
(same arguments in the same order; ``--wandb_dir`` is new).

The interpreter is ``$PYTHON_BIN`` if set, else the current Python.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time

from uq import paths, registry


def command(experiment, split: int, member: int, resume: bool = False) -> list:
    exp = registry.experiment(experiment)
    registry.require_backend(exp)
    if split not in exp.splits:
        raise SystemExit(f"split {split} is not in {exp.label} (splits {list(exp.splits)})")
    if member not in exp.members:
        raise SystemExit(f"member {member} is not in {exp.label} (members {list(exp.members)})")
    python = os.environ.get("PYTHON_BIN", sys.executable)
    cmd = [
        python, str(paths.MACE_RUN_TRAIN),
        "--config", str(exp.config_path),
        "--train_file", str(exp.data_file("train", split)),
        "--valid_file", str(exp.data_file("val", split)),
        "--test_file", str(exp.data_file(exp.training.get("test_role", "id"), split)),
        "--model_dir", str(paths.models_dir(exp.id, split)),
        "--log_dir", str(paths.logs_dir(exp.id, split)),
        "--checkpoints_dir", str(paths.checkpoints_dir(exp.id, split)),
        "--results_dir", str(paths.results_dir(exp.id, split)),
        "--seed", str(member),
        "--wandb_name", f"split_{split}_{exp.run_name}_seed_{member}",
    ]
    if exp.parent:
        cmd += ["--foundation_model", str(exp.parent.final_model(split, member))]
    cmd += ["--wandb_dir", str(paths.WANDB)]
    if resume:   # continue after the latest epoch checkpoint (see prepare_resume)
        cmd += ["--restart_latest", "--resume_exact"]
    return cmd


def check_inputs(experiment, split: int, member: int) -> list:
    """Missing inputs (data files, parent model) for one member; empty if ready."""
    exp = registry.experiment(experiment)
    needed = [exp.data_file(role, split) for role in ("train", "val", exp.training.get("test_role", "id"))]
    if exp.parent:
        needed.append(exp.parent.final_model(split, member))
    return [path for path in needed if not path.is_file()]


def stale_outputs(experiment, split: int, member: int) -> list:
    """Outputs of an earlier, unfinished run of an incomplete member; empty if none.

    MACE appends to ``results/<name>_run-<m>_train.txt``, so a fresh run on top of an
    aborted one would mix two runs' validation rows and checkpoint selection could pick
    an epoch whose checkpoint was overwritten. Such members are refused until the user
    removes these files (or resumes the run by hand).
    """
    exp = registry.experiment(experiment)
    if exp.member_complete(split, member):
        return []
    found = [exp.train_log(split, member), exp.final_model(split, member)]
    found += sorted(paths.checkpoints_dir(exp.id, split).glob(f"{exp.run_name}_run-{member}_epoch-*.pt"))
    return [path for path in found if path.exists()]


def latest_epoch(experiment, split: int, member: int):
    """Highest epoch with a checkpoint ``<name>_run-<m>_epoch-<e>.pt``, or None."""
    exp = registry.experiment(experiment)
    pattern = re.compile(rf"{re.escape(exp.run_name)}_run-{member}_epoch-(\d+)\.pt$")
    directory = paths.checkpoints_dir(exp.id, split)
    if not directory.is_dir():
        return None
    epochs = [int(match.group(1)) for path in directory.iterdir() if (match := pattern.match(path.name))]
    return max(epochs, default=None)


def prepare_resume(experiment, split: int, member: int) -> int:
    """Make an interrupted member resumable after its latest epoch checkpoint N; returns N.

    MACE logs an epoch's rows before saving its checkpoint, so the results log of a killed run
    can hold rows of epoch N+1 (training rows, maybe its validation row) without a checkpoint.
    Those rows would be duplicated by the resumed run. The log is copied unchanged to
    ``<log>.pre-resume-<time>`` and rewritten with only the rows of epochs <= N (and the
    epoch-less rows); nothing else is touched. The resumed run then appends epochs N+1, ...
    """
    exp = registry.experiment(experiment)
    if exp.member_complete(split, member):
        raise SystemExit(f"{exp.label} s{split} m{member} is complete; nothing to resume")
    if exp.final_model(split, member).exists():
        raise SystemExit(f"{paths.rel(exp.final_model(split, member))} exists: not an interrupted run")
    epoch = latest_epoch(exp.id, split, member)
    if epoch is None:
        raise SystemExit(f"{exp.label} s{split} m{member}: no epoch checkpoint to resume from")
    log_path = exp.train_log(split, member)
    if log_path.is_file():
        lines = log_path.read_text().splitlines(keepends=True)
        kept = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:   # a line cut by the kill
                continue
            row_epoch = row.get("epoch") if isinstance(row, dict) else None
            if row_epoch is None or int(row_epoch) <= epoch:
                kept.append(line if line.endswith("\n") else line + "\n")
        if len(kept) != len(lines):
            backup = log_path.with_name(f"{log_path.name}.pre-resume-{time.strftime('%Y%m%d-%H%M%S')}")
            backup.write_bytes(log_path.read_bytes())
            log_path.write_text("".join(kept))
    return epoch


def prepare(experiment, split: int) -> None:
    """Create the run directories of one split (as the old shell launcher did)."""
    exp = registry.experiment(experiment)
    for directory in (paths.models_dir, paths.logs_dir, paths.checkpoints_dir, paths.results_dir):
        directory(exp.id, split).mkdir(parents=True, exist_ok=True)
    paths.WANDB.mkdir(parents=True, exist_ok=True)


def train_member(experiment, split: int, member: int, gpu=None, dry_run: bool = False,
                 resume: bool = False) -> int:
    """Train one member. ``gpu=None`` keeps the inherited ``CUDA_VISIBLE_DEVICES`` (e.g. set by SLURM).
    A complete member is skipped; with ``resume`` an interrupted one continues (see prepare_resume)."""
    exp = registry.experiment(experiment)
    if exp.member_complete(split, member):
        print(f"{exp.label} split {split} member {member} is already complete; skipped")
        return 0
    stale = stale_outputs(exp.id, split, member)
    resuming = bool(stale) and resume
    cmd = command(exp.id, split, member, resume=resuming)
    missing = check_inputs(exp.id, split, member)
    where = f"GPU: {gpu}" if gpu is not None else f"GPU: inherited ({os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
    print(f"Experiment: {exp.label}\nSplit: {split}  Member: {member}  {where}\n"
          f"Outputs: {paths.rel(paths.run_dir(exp.id, split))}/{{models,logs,checkpoints,results}}")
    prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
    print(prefix + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        for path in missing:
            print(f"warning: missing input {path}")
        return 0
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(map(str, missing)))
    if stale and not resume:
        raise SystemExit(f"{len(stale)} outputs of an unfinished earlier run exist (MACE would append to its "
                         f"result log); pass --resume or remove them first:\n  " + "\n  ".join(map(str, stale[:5])))
    if resuming:
        print(f"resuming after epoch {prepare_resume(exp.id, split, member)}")
    prepare(exp.id, split)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = subprocess.call(cmd, cwd=paths.ROOT, env=env)
    if code and exp.member_complete(split, member):
        # e.g. MACE's closing metrics on the whole training set ran out of GPU memory after the final
        # model was saved: the member is complete (final model + result log), so its children may start
        print(f"warning: training exited with status {code} after the member was complete; treated as done")
        return 0
    return code
