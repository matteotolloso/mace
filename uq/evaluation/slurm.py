"""Evaluate experiments on a SLURM cluster (CINECA Leonardo): ``uq evaluate-slurm``.

    uq evaluate-slurm ani1x-system/mace-mse/lf-only ani1x-system/mace-mse/lf-to-hf --dry-run
    uq evaluate-slurm ... --after 63716417,63716418 --prune --submit

One 1-GPU job per (experiment, split) runs exactly ``uq evaluate <exp> --splits <s> --no-aggregate``
(the five-split aggregate and the ledger entry are made on the DGX after the outputs are synced back,
``papers/iclr27/rebuttal/RUNS.md``). A job first checks that every member of its split, and of the
parent's split for an LF->HF experiment, is complete, and fails otherwise. ``--after`` makes the jobs
wait for the given training jobs (``afterany``: a training job can exit non-zero although its members
are complete, so each evaluation job decides by the completeness check). ``--prune``
adds one CPU job, after all evaluation jobs succeeded, that runs ``uq prune-checkpoints <exps> --apply``
(approved for Leonardo, decision D9; the parent's epoch checkpoints are read by the child's
evaluation, so pruning waits for every evaluation of the invocation).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from uq import paths, registry


def _header(tag, *, account, partition, qos, walltime, out_dir, after, gpu=True, kind="afterok"):
    lines = ["#!/bin/bash", f"#SBATCH --job-name={tag}", f"#SBATCH --account={account}",
             f"#SBATCH --partition={partition}"]
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    lines += ["#SBATCH --nodes=1", "#SBATCH --ntasks=1"]
    if gpu:
        lines += ["#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=120G"]
    else:
        lines += ["#SBATCH --cpus-per-task=4", "#SBATCH --mem=30G"]
    lines += [f"#SBATCH --time={walltime}", f"#SBATCH --output={out_dir}/%x_%j.out"]
    if after:
        lines.append(f"#SBATCH --dependency={kind}:{':'.join(after)}")
        if kind == "afterok":
            lines.append("#SBATCH --kill-on-invalid-dep=yes")
    return lines


def _env(python, threads):
    return [
        "set -u",
        f"cd {shlex.quote(str(paths.ROOT))}",
        f"export UQ_RUNS={shlex.quote(str(paths.RUNS))}",
        f"export PYTHON_BIN={shlex.quote(python)}",
        f"export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}",
        'export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$SLURM_JOB_ID"',
        'echo "job $SLURM_JOB_ID on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}, '
        'commit $(git rev-parse --short HEAD), start $(date)"',
    ]


def complete_check(exp_id: str, split: int) -> str:
    """Python snippet: exit 1 unless every member of the split (and of the parent's split) is complete."""
    return ("from uq import registry; import sys; e = registry.experiment(%r); "
            "es = [e] + ([e.parent] if e.parent else []); "
            "missing = [(x.id, m) for x in es for m in x.members if not x.member_complete(%d, m)]; "
            "print('incomplete members:', missing) if missing else None; "
            "sys.exit(1 if missing else 0)") % (exp_id, split)


def evaluation_script(exp_id, split, *, account, partition, qos, walltime, python, out_dir, after, tag):
    lines = _header(tag, account=account, partition=partition, qos=qos, walltime=walltime,
                    out_dir=out_dir, after=after, kind="afterany")
    lines += _env(python, 8)
    q = shlex.quote
    lines += [
        f"{q(python)} -c {q(complete_check(exp_id, split))} || exit 1",
        'gpu="${CUDA_VISIBLE_DEVICES:-0}"; gpu="${gpu%%,*}"',
        f"{q(python)} -m uq evaluate {exp_id} --gpu \"$gpu\" --splits {split} --no-aggregate",
        "status=$?",
        'echo "job $SLURM_JOB_ID finished with status $status at $(date)"',
        "exit $status",
        "",
    ]
    return "\n".join(lines)


def prune_script(exp_ids, *, account, partition, walltime, python, out_dir, after, tag):
    lines = _header(tag, account=account, partition=partition, qos=None, walltime=walltime,
                    out_dir=out_dir, after=after, gpu=False)
    lines += _env(python, 4)
    lines += [
        f"{shlex.quote(python)} -m uq prune-checkpoints {' '.join(exp_ids)} --apply",
        "status=$?",
        'echo "job $SLURM_JOB_ID finished with status $status at $(date)"',
        "exit $status",
        "",
    ]
    return "\n".join(lines)


def _sbatch(path: Path) -> str:
    result = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=True)
    return result.stdout.strip().split(";")[0]


def run(names, splits=None, after=None, prune=False, qos="normal", walltime="24:00:00",
        prune_partition="lrd_all_serial", prune_time="04:00:00", account=None, partition=None,
        python=None, out=None, submit=False, dry_run=False) -> int:
    account = account or os.environ.get("UQ_SLURM_ACCOUNT")
    partition = partition or os.environ.get("UQ_SLURM_PARTITION", "boost_usr_prod")
    python = python or os.environ.get("PYTHON_BIN", sys.executable)
    if not account:
        raise SystemExit("no SLURM account: pass --account or set UQ_SLURM_ACCOUNT")
    exps = [registry.experiment(name) for name in names]
    after = [str(a) for a in (after or [])]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out) if out else paths.RUNS / "_slurm" / f"eval-{stamp}"
    work = [(exp.id, s) for exp in exps for s in (splits if splits is not None else exp.splits)]
    print(f"{len(work)} evaluation jobs (1 GPU each, qos {qos}, time {walltime}, account {account})"
          + (f", after {len(after)} jobs" if after else "") + (" + 1 prune job" if prune else "")
          + f"; scripts in {out_dir}")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest, eval_ids = [], []
    for exp_id, split in work:
        tag = f"uqeval_{exp_id.replace('/', '__')}_s{split}"
        text = evaluation_script(exp_id, split, account=account, partition=partition, qos=qos,
                                 walltime=walltime, python=python, out_dir=out_dir, after=after, tag=tag)
        if dry_run:
            print(f"--- {tag}\n{text}")
            continue
        path = out_dir / f"{tag}.sbatch"
        path.write_text(text)
        job_id = _sbatch(path) if submit else ""
        eval_ids.append(job_id)
        manifest.append(f"{tag}\t{job_id}\t{exp_id} split {split}")
        print(f"{tag}\t{job_id or '(not submitted)'}")
    if prune:
        tag = f"uqprune_{stamp}"
        text = prune_script([exp.id for exp in exps], account=account, partition=prune_partition,
                            walltime=prune_time, python=python, out_dir=out_dir,
                            after=[i for i in eval_ids if i] or ["<evaluation job ids>"], tag=tag)
        if dry_run:
            print(f"--- {tag}\n{text}")
        else:
            path = out_dir / f"{tag}.sbatch"
            path.write_text(text)
            job_id = _sbatch(path) if submit else ""
            manifest.append(f"{tag}\t{job_id}\tprune {' '.join(exp.id for exp in exps)}")
            print(f"{tag}\t{job_id or '(not submitted)'}")
    if not dry_run:
        (out_dir / "jobs.tsv").write_text("tag\tjob_id\twhat\n" + "\n".join(manifest) + "\n")
    return 0
