"""Train every missing member of one or more experiments on at most three GPUs.

    uq train B --gpus 6 7 --dry-run          # show the plan
    uq train wA wB wC --gpus 6 7 --jobs-per-gpu 3

Runs ``uq train-member`` for each (experiment, split, member) that is not complete
(see ``uq status``). LF->HF members start only after the matching parent member
(same split and member seed) is complete, and parents are trained first when they
are in the same invocation; a fine-tuning member whose parent is complete is launched
before any pretraining/direct member that has not started yet. By default members are
interleaved across experiments (all member-0 runs, then member 1, ...);
``--by-experiment`` finishes the experiments in the order given instead. Several small
jobs may share a GPU (``--jobs-per-gpu``). Each member gets ``--threads-per-job`` CPU
threads (``OMP_NUM_THREADS``/``MKL_NUM_THREADS``, default 8: torch otherwise starts one
thread per core and six packed jobs saturated the shared 256-core machine), and packed
jobs use ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (about 5 GB instead of
12-18 GB per ANI-1x member). Neither changes what is computed.
Hard limit: never more than three GPUs at the same time (project rule). A restarted scheduler
adopts members still being trained by the previous one and skips members completed meanwhile.

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

MAX_GPUS = 3  # project rule (user, 2026-11-03): never more than three GPUs at the same time


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def job_name(exp_id, split, member):
    return f"{exp_id.replace('/', '__')}_s{split}_m{member}"


def plan(names, splits=None, members=None, by_experiment=False):
    """Pending (experiment, split, member) jobs: parents first, then interleaved by member
    (or, with ``by_experiment``, experiment by experiment in the order given)."""
    experiments = registry.resolve(names)
    depth = {exp.id: (1 if exp.parent_id else 0) for exp in experiments}
    jobs = [(exp.id, s, m) for exp in experiments
            for m in (members or exp.members) for s in (splits or exp.splits)
            if s in exp.splits and m in exp.members and not exp.member_complete(s, m)]
    order = {exp.id: i for i, exp in enumerate(experiments)}
    if by_experiment:
        jobs.sort(key=lambda job: (depth[job[0]], order[job[0]], job[2], job[1]))
    else:
        jobs.sort(key=lambda job: (depth[job[0]], job[2], order[job[0]], job[1]))
    return jobs


def _ready(job):
    exp = registry.experiment(job[0])
    return exp.parent is None or exp.parent.member_complete(job[1], job[2])


def _next_runnable(pending):
    """A fine-tuning member whose parent is complete first, else the first ready job."""
    children = (job for job in pending if registry.experiment(job[0]).parent_id and _ready(job))
    return next(children, None) or next((job for job in pending if _ready(job)), None)


def job_env(gpu, jobs_per_gpu=1, threads_per_job=8) -> dict:
    """Environment of one member process (an explicit setting in the caller's env wins)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, str(threads_per_job))
    if jobs_per_gpu > 1:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


class _Adopted:
    """A member process started by an earlier scheduler that is still running (no exit code)."""

    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        if os.path.exists(f"/proc/{self.pid}"):
            return None
        self.returncode = 0   # unknown; success is judged by member_complete
        return 0


def running_members(pending):
    """{job: (pid, gpu)} for pending members already being trained by a live process of this user
    (matched on the exact ``launch.command``), e.g. after the scheduler itself was restarted."""
    wanted = {"\0".join(launch.command(*job)): job for job in pending}
    found = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/cmdline", "rb") as handle:
                cmdline = handle.read().decode(errors="replace").rstrip("\0")
            job = wanted.get(cmdline)
            if job is None:
                continue
            with open(f"/proc/{entry.name}/environ", "rb") as handle:
                env = dict(item.split("=", 1) for item in handle.read().decode(errors="replace").split("\0")
                           if "=" in item)
        except OSError:
            continue
        found[job] = (int(entry.name), int(env.get("CUDA_VISIBLE_DEVICES", "-1").split(",")[0]))
    return found


def run(names, gpus, splits=None, members=None, jobs_per_gpu=1, dry_run=False, poll_seconds=30,
        by_experiment=False, threads_per_job=8) -> int:
    gpus = list(dict.fromkeys(gpus))
    if len(gpus) > MAX_GPUS:
        raise SystemExit(f"at most {MAX_GPUS} GPUs may be used at the same time (got {gpus})")
    pending = plan(names, splits, members, by_experiment)
    adopted = running_members(pending)
    log(f"{len(pending)} members to train on GPUs {gpus} ({jobs_per_gpu} jobs per GPU, "
        f"{threads_per_job} CPU threads per job; {len(adopted)} already running)")
    if dry_run:
        for exp_id, s, m in pending:
            parent = registry.experiment(exp_id).parent
            wait = f"  (after {parent.label} s{s} m{m})" if parent and not parent.member_complete(s, m) else ""
            print(f"  {registry.experiment(exp_id).label} split {s} member {m}{wait}")
        return 0

    paths.SCHEDULER_LOGS.mkdir(parents=True, exist_ok=True)
    slots = {gpu: [] for gpu in gpus}   # gpu -> list of (job, Popen)
    for job, (pid, gpu) in adopted.items():   # wait for them instead of refusing their partial outputs
        if gpu not in slots:
            raise SystemExit(f"{job_name(*job)} is running on GPU {gpu} (pid {pid}), not in {gpus}")
        pending.remove(job)
        slots[gpu].append((job, _Adopted(pid)))
        log(f"adopted {job_name(*job)} (GPU {gpu}, pid {pid})")
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
                runnable = _next_runnable(pending)
                if runnable is None:
                    break
                pending.remove(runnable)
                name = job_name(*runnable)
                if registry.experiment(runnable[0]).member_complete(runnable[1], runnable[2]):
                    log(f"skipped {name}: completed meanwhile (another scheduler?)")
                    continue
                stale = launch.stale_outputs(*runnable)
                if stale:
                    log(f"REFUSED {name}: {len(stale)} outputs of an unfinished earlier run "
                        f"(e.g. {paths.rel(stale[0])}); remove them, then rerun")
                    failed.append(runnable)
                    continue
                launch.prepare(runnable[0], runnable[1])
                handle = (paths.SCHEDULER_LOGS / f"{name}.log").open("w")
                proc = subprocess.Popen(launch.command(*runnable), cwd=paths.ROOT, stdout=handle,
                                        stderr=subprocess.STDOUT,
                                        env=job_env(gpu, jobs_per_gpu, threads_per_job))
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
