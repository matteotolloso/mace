"""Train missing members on a SLURM cluster (CINECA Leonardo): ``uq train-slurm``.

    uq train-slurm ani1x-system/mace-mse/lf-only ani1x-system/mace-mse/lf-to-hf --pack 2 --dry-run
    uq train-slurm ... --pack 2 --submit          # write the scripts and sbatch them

Each SLURM job reserves **one GPU** (``--gres=gpu:1``, ``--cpus-per-task=8``, ``--mem=120G``: a quarter
of a Leonardo Booster node, so the job is charged as one GPU) and runs up to ``--pack`` *chains* in
parallel on it. A chain is one member of a pretraining/direct experiment followed, when the
invocation also names its LF->HF child experiment, by the child member with the same split and seed
(the child starts as soon as its parent is complete, inside the same job). Children whose parent is
already complete form chains of their own.

Every member runs ``uq train-member <experiment> <split> <member>``, i.e. exactly
``launch.command`` (the commands pinned by the tests), with the GPU given by SLURM; complete members
are skipped and, with ``--resume``, interrupted ones continue after their latest epoch checkpoint
(e.g. after a 24 h walltime). ``--after <job ids>`` makes the jobs wait for those jobs (``afterany``) and
keeps children whose parent those jobs are still training (a back-up for children that a chain did not
start, e.g. because the parent's MACE run exited non-zero after saving its model). Member logs go to ``runs/_logs/<job name>.log`` as with ``uq train``;
the scripts and SLURM's own output go to ``--out``. Account, partition and interpreter come from the
options or from ``$UQ_SLURM_ACCOUNT``, ``$UQ_SLURM_PARTITION`` and ``$PYTHON_BIN``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from uq import paths, registry
from uq.training import scheduler


def chains(names, splits=None, members=None, by_experiment=False, parents_in_flight=False):
    """Pending members grouped into chains [parent, child] / [member], in scheduler order.

    ``parents_in_flight``: a child whose parent is neither complete nor in this invocation is kept as a
    chain of its own (its parent is being trained by the jobs given to ``--after``)."""
    pending = scheduler.plan(names, splits, members, by_experiment)
    pending_set = set(pending)
    result, used = [], set()
    for job in pending:
        if job in used:
            continue
        exp = registry.experiment(job[0])
        if exp.parent_id:   # a child: chained after its parent if the parent is pending here
            parent = (exp.parent.id, job[1], job[2])
            if parent in pending_set:
                continue    # added together with its parent below
            if not exp.parent.member_complete(job[1], job[2]) and not parents_in_flight:
                print(f"warning: {exp.label} s{job[1]} m{job[2]} skipped: parent member neither complete "
                      f"nor in this invocation", file=sys.stderr)
                continue
            result.append([job])
            used.add(job)
            continue
        chain = [job]
        for other in pending:
            other_exp = registry.experiment(other[0])
            if other_exp.parent_id == exp.id and other[1:] == job[1:] and other not in used:
                chain.append(other)
                used.add(other)
        used.add(job)
        result.append(chain)
    return result


def script(chain_group, *, account, partition, qos, walltime, pack, python, resume, out_dir, tag,
           after=None, after_ok=None) -> str:
    """The sbatch script of one 1-GPU job running the given chains in parallel."""
    threads = max(1, 8 // pack)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={tag}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --qos={qos}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=120G",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={out_dir}/%x_%j.out",
    ]
    deps = []
    if after:   # afterany: every member skips itself if complete and fails if its inputs are missing
        deps.append(f"afterany:{':'.join(map(str, after))}")
    if after_ok:   # e.g. wait for a prune job that frees the space these members need
        deps.append(f"afterok:{':'.join(map(str, after_ok))}")
    if deps:
        lines.append(f"#SBATCH --dependency={','.join(deps)}")
    if after_ok:
        lines.append("#SBATCH --kill-on-invalid-dep=yes")
    lines += [
        "set -u",
        f"cd {shlex.quote(str(paths.ROOT))}",
        f"export UQ_RUNS={shlex.quote(str(paths.RUNS))}",
        f"export PYTHON_BIN={shlex.quote(python)}",
        f"export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}",
        "export WANDB_MODE=${WANDB_MODE:-offline}",
    ]
    if pack > 1:
        lines.append("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    lines.append(f"mkdir -p {shlex.quote(str(paths.SCHEDULER_LOGS))}")
    lines.append('echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, commit $(git rev-parse --short HEAD)"')
    lines.append("pids=()")
    flag = " --resume" if resume else ""
    for chain in chain_group:
        steps = []
        for exp_id, split, member in chain:
            log = paths.SCHEDULER_LOGS / f"{scheduler.job_name(exp_id, split, member)}.log"
            steps.append(f"{shlex.quote(python)} -m uq train-member {exp_id} {split} {member}{flag} "
                         f">> {shlex.quote(str(log))} 2>&1")
        lines.append("( " + " && ".join(steps) + " ) &")
        lines.append("pids+=($!)")
    lines += [
        "status=0",
        'for pid in "${pids[@]}"; do wait "$pid" || status=1; done',
        'echo "job $SLURM_JOB_ID finished with status $status at $(date)"',
        "exit $status",
        "",
    ]
    return "\n".join(lines)


def run(names, splits=None, members=None, pack=1, by_experiment=False, resume=False, qos="normal",
        walltime="24:00:00", account=None, partition=None, python=None, out=None, submit=False,
        dry_run=False, after=None, after_ok=None) -> int:
    account = account or os.environ.get("UQ_SLURM_ACCOUNT")
    partition = partition or os.environ.get("UQ_SLURM_PARTITION", "boost_usr_prod")
    python = python or os.environ.get("PYTHON_BIN", sys.executable)
    if not account:
        raise SystemExit("no SLURM account: pass --account or set UQ_SLURM_ACCOUNT")
    if pack < 1:
        raise SystemExit("--pack must be >= 1")
    groups = chains(names, splits, members, by_experiment, parents_in_flight=bool(after))
    jobs = [groups[i:i + pack] for i in range(0, len(groups), pack)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out) if out else paths.RUNS / "_slurm" / stamp
    n_members = sum(len(chain) for chain in groups)
    print(f"{n_members} members in {len(groups)} chains -> {len(jobs)} 1-GPU jobs "
          f"(pack {pack}, qos {qos}, time {walltime}, account {account}); scripts in {out_dir}")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, group in enumerate(jobs):
        tag = f"uq_{stamp}_{index:03d}"
        text = script(group, account=account, partition=partition, qos=qos, walltime=walltime, pack=pack,
                      python=python, resume=resume, out_dir=out_dir, tag=tag, after=after,
                      after_ok=after_ok)
        names_in_job = " ".join(scheduler.job_name(*job) for chain in group for job in chain)
        if dry_run:
            print(f"--- {tag}: {names_in_job}")
            print(text)
            continue
        path = out_dir / f"{tag}.sbatch"
        path.write_text(text)
        job_id = ""
        if submit:
            result = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
        manifest.append(f"{tag}\t{job_id}\t{names_in_job}")
        print(f"{tag}\t{job_id or '(not submitted)'}\t{names_in_job}")
    if not dry_run:
        (out_dir / "jobs.tsv").write_text("tag\tjob_id\tmembers\n" + "\n".join(manifest) + "\n")
    return 0
