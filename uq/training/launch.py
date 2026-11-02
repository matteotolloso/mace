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

import os
import shlex
import subprocess
import sys

from uq import paths, registry


def command(experiment, split: int, member: int) -> list:
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


def prepare(experiment, split: int) -> None:
    """Create the run directories of one split (as the old shell launcher did)."""
    exp = registry.experiment(experiment)
    for directory in (paths.models_dir, paths.logs_dir, paths.checkpoints_dir, paths.results_dir):
        directory(exp.id, split).mkdir(parents=True, exist_ok=True)
    paths.WANDB.mkdir(parents=True, exist_ok=True)


def train_member(experiment, split: int, member: int, gpu: int, dry_run: bool = False) -> int:
    exp = registry.experiment(experiment)
    cmd = command(exp.id, split, member)
    missing = check_inputs(exp.id, split, member)
    print(f"Experiment: {exp.label}\nSplit: {split}  Member: {member}  GPU: {gpu}\n"
          f"Outputs: {paths.rel(paths.run_dir(exp.id, split))}/{{models,logs,checkpoints,results}}")
    print(f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        for path in missing:
            print(f"warning: missing input {path}")
        return 0
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(map(str, missing)))
    stale = stale_outputs(exp.id, split, member)
    if stale:
        raise SystemExit(f"{len(stale)} outputs of an unfinished earlier run exist (MACE would append to its "
                         f"result log); remove them first:\n  " + "\n  ".join(map(str, stale[:5])))
    prepare(exp.id, split)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.call(cmd, cwd=paths.ROOT, env=env)
