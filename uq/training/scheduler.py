"""Train every missing member of one or more experiments on at most two GPUs.

    uq train B --gpus 6 7 --dry-run          # show the plan
    uq train wA wB wC --gpus 6 7 --jobs-per-gpu 3

Runs ``uq train-member`` for each (experiment, split, member) that is not complete
(see ``uq status``). LF->HF members start only after the matching parent member
(same split and member seed) is complete, and parents are trained first when they
are in the same invocation. Several small jobs may share a GPU (``--jobs-per-gpu``).
Hard limit: never more than two GPUs at the same time (project rule).

Safe to restart: complete members are skipped. Logs go to
``runs/_logs/<id with / -> __>_s<split>_m<member>.log``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from uq import paths, registry
from uq.training import launch

MAX_GPUS = 2


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def job_name(exp_id, split, member):
    return f"{exp_id.replace('/', '__')}_s{split}_m{member}"


def plan(names, splits=None, members=None):
    """Pending (experiment, split, member) jobs: parents first, then interleaved by member."""
    experiments = registry.resolve(names)
    depth = {exp.id: (1 if exp.parent_id else 0) for exp in experiments}
    jobs = [(exp.id, s, m) for exp in experiments
            for m in (members or exp.members) for s in (splits or exp.splits)
            if s in exp.splits and m in exp.members and not exp.member_complete(s, m)]
    order = {exp.id: i for i, exp in enumerate(experiments)}
    jobs.sort(key=lambda job: (depth[job[0]], job[2], order[job[0]], job[1]))
    return jobs


def _ready(job):
    exp = registry.experiment(job[0])
    return exp.parent is None or exp.parent.member_complete(job[1], job[2])


def run(names, gpus, splits=None, members=None, jobs_per_gpu=1, dry_run=False, poll_seconds=30) -> int:
    gpus = list(dict.fromkeys(gpus))
    if len(gpus) > MAX_GPUS:
        raise SystemExit(f"at most {MAX_GPUS} GPUs may be used at the same time (got {gpus})")
    pending = plan(names, splits, members)
    log(f"{len(pending)} members to train on GPUs {gpus} ({jobs_per_gpu} jobs per GPU)")
    if dry_run:
        for exp_id, s, m in pending:
            parent = registry.experiment(exp_id).parent
            wait = f"  (after {parent.label} s{s} m{m})" if parent and not parent.member_complete(s, m) else ""
            print(f"  {registry.experiment(exp_id).label} split {s} member {m}{wait}")
        return 0

    paths.SCHEDULER_LOGS.mkdir(parents=True, exist_ok=True)
    slots = {gpu: [] for gpu in gpus}   # gpu -> list of (job, Popen)
    failed = []
    while pending or any(slots.values()):
        for gpu, running in slots.items():   # reap finished jobs
            for item in list(running):
                job, proc = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                name = job_name(*job)
                if proc.returncode == 0 and registry.experiment(job[0]).member_complete(job[1], job[2]):
                    log(f"done    {name} (GPU {gpu})")
                else:
                    log(f"FAILED  {name} (GPU {gpu}, exit {proc.returncode}); "
                        f"see {paths.rel(paths.SCHEDULER_LOGS / (name + '.log'))}")
                    failed.append(job)
        for gpu, running in slots.items():   # launch runnable jobs into free slots
            while len(running) < jobs_per_gpu:
                runnable = next((job for job in pending if _ready(job)), None)
                if runnable is None:
                    break
                pending.remove(runnable)
                name = job_name(*runnable)
                launch.prepare(runnable[0], runnable[1])
                handle = (paths.SCHEDULER_LOGS / f"{name}.log").open("w")
                proc = subprocess.Popen(launch.command(*runnable), cwd=paths.ROOT, stdout=handle,
                                        stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
                running.append((runnable, proc))
                log(f"started {name} (GPU {gpu}, pid {proc.pid})")
        failed_parents = {(registry.experiment(j[0]).id, j[1], j[2]) for j in failed}
        for job in [j for j in pending if registry.experiment(j[0]).parent_id]:
            parent = registry.experiment(job[0]).parent
            if (parent.id, job[1], job[2]) in failed_parents:
                pending.remove(job)
                failed.append(job)
                log(f"SKIPPED {job_name(*job)}: parent member failed")
        if pending and not any(slots.values()):
            for job in pending:   # nothing running and nothing runnable: parents are missing
                log(f"SKIPPED {job_name(*job)}: parent member missing")
            failed.extend(pending)
            pending = []
        if pending or any(slots.values()):
            time.sleep(poll_seconds)

    log(f"finished; {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:], gpus=[0], dry_run=True))
