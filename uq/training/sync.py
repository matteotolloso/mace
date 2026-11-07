"""Copy an experiment's outputs back from a remote cluster (CINECA Leonardo): ``uq sync-back``.

    uq sync-back ani1x-system/mace-mse/lf-only --host leonardo            # dry run: what would come
    uq sync-back ani1x-system/mace-mse/lf-only --host leonardo --apply    # copy
    uq sync-back ... --checkpoints --apply                                # also the (pruned) checkpoints

Outputs flow remote -> local only (``papers/iclr27/rebuttal/RUNS.md``, protocol item 8). Per split it
copies ``results/``, ``logs/``, ``models/`` and, with ``--checkpoints``, ``checkpoints/`` of the run
directory, plus ``experiments/<id>/evaluation/split_<s>/``; member logs of ``runs/_logs`` go to
``runs/_logs/remote/``. rsync runs with ``--ignore-existing``: a local file is never overwritten (a
remote file that differs from an existing local one is reported, not copied). ``--checkpoints``
refuses splits whose remote checkpoint folder has no ``PRUNED.json`` (unpruned LF-scale splits hold
~50 GB each). The five-split aggregate (``uq aggregate``) and the ledger entry are done locally
afterwards. Remote locations: ``--remote-root`` / ``--remote-runs`` or ``$UQ_REMOTE_ROOT`` /
``$UQ_REMOTE_RUNS`` (repository clone and its ``UQ_RUNS`` on the remote machine).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath

from uq import paths, registry


def _remote(path, local_base, remote_base) -> str:
    return str(PurePosixPath(remote_base) / Path(path).relative_to(local_base).as_posix())


def transfers(exp, splits, remote_root, remote_runs, checkpoints=False):
    """(remote dir, local dir, what) pairs for one experiment."""
    pairs = []
    for split in splits:
        run = paths.run_dir(exp.id, split)
        subdirs = ["results", "logs", "models"] + (["checkpoints"] if checkpoints else [])
        for sub in subdirs:
            pairs.append((_remote(run / sub, paths.RUNS, remote_runs), run / sub, sub))
        ev = paths.split_evaluation_dir(exp.id, split)
        pairs.append((_remote(ev, paths.ROOT, remote_root), ev, "evaluation"))
    return pairs


def _ssh(host, command):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True)


def run(names, host, splits=None, remote_root=None, remote_runs=None, checkpoints=False, apply=False) -> int:
    remote_root = remote_root or os.environ.get("UQ_REMOTE_ROOT")
    remote_runs = remote_runs or os.environ.get("UQ_REMOTE_RUNS")
    if not (remote_root and remote_runs):
        raise SystemExit("remote locations unknown: pass --remote-root/--remote-runs or set "
                         "UQ_REMOTE_ROOT/UQ_REMOTE_RUNS")
    status = 0
    for name in names:
        exp = registry.experiment(name)
        exp_splits = list(splits if splits is not None else exp.splits)
        for remote, local, what in transfers(exp, exp_splits, remote_root, remote_runs, checkpoints):
            if what == "checkpoints":
                if _ssh(host, f"test -f {shlex.quote(remote)}/PRUNED.json").returncode != 0:
                    print(f"REFUSED {exp.id}: {remote} is not pruned (no PRUNED.json); not copied")
                    status = 1
                    continue
            if _ssh(host, f"test -d {shlex.quote(remote)}").returncode != 0:
                print(f"missing on {host}: {remote}")
                status = 1
                continue
            if apply:
                local.mkdir(parents=True, exist_ok=True)
            cmd = ["rsync", "-a", "--ignore-existing", "--itemize-changes"] + ([] if apply else ["--dry-run"])
            result = subprocess.run(cmd + [f"{host}:{remote}/", f"{local}/"], capture_output=True, text=True)
            copied = [line for line in result.stdout.splitlines() if line.startswith(">f")]
            print(f"{'copied' if apply else 'would copy'} {len(copied):5d} files  {paths.rel(local)}")
            if result.returncode:
                print(result.stderr.strip())
                status = 1
            # files present on both sides but different: never overwritten, reported
            check = subprocess.run(["rsync", "-a", "--dry-run", "--existing", "--checksum", "--itemize-changes",
                                    f"{host}:{remote}/", f"{local}/"], capture_output=True, text=True)
            differ = [line.split(maxsplit=1)[1] for line in check.stdout.splitlines() if line.startswith(">f")]
            if differ:
                print(f"  {len(differ)} files differ from the local copy and were kept local: {differ[:5]}")
        logs = paths.SCHEDULER_LOGS / "remote"
        pattern = f"{exp.id.replace('/', '__')}_s*_m*.log"
        if apply:
            logs.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-a", "--ignore-existing"] + ([] if apply else ["--dry-run"])
        remote_logs = _remote(paths.SCHEDULER_LOGS, paths.RUNS, remote_runs)
        subprocess.run(cmd + ["--include", pattern, "--exclude", "*", f"{host}:{remote_logs}/", f"{logs}/"])
    return status
