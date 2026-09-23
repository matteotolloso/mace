#!/usr/bin/env python3
"""Train the missing water ensemble members (wA, wB, wC) on at most two GPUs.

Runs every (experiment, split, member) that is not complete, using the normal
per-member launchers (experiment_wX/train_wX.sh). A member counts as complete
exactly as in check_experiments.sh: both
    experiment_wX/checkpoints_<s>/mace_run-<m>.model
    experiment_wX/results_<s>/mace_run-<m>_train.txt
exist. wB (LF->HF fine-tuning) members start only after the matching wA member
(same split and member seed) is complete, which is what train_wB.sh requires.

Hard limit: never more than two GPUs at the same time (project rule). Several
small water jobs may share one GPU (--jobs-per-gpu).

Safe to restart: complete members are skipped. Run from the repository root:

    python -B experiments/scripts/train_water_splits.py --gpus 6 7 --splits 1 2 3 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "experiments" / "runs" / "water"
MEMBERS = range(10)
MAX_GPUS = 2


def complete(experiment: str, split: int, member: int) -> bool:
    exp_dir = ROOT / f"experiment_{experiment}"
    return (exp_dir / f"checkpoints_{split}" / f"mace_run-{member}.model").is_file() and (
        exp_dir / f"results_{split}" / f"mace_run-{member}_train.txt"
    ).is_file()


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--experiments", nargs="+", default=["wA", "wC", "wB"])
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()

    if len(set(args.gpus)) > MAX_GPUS:
        parser.error(f"at most {MAX_GPUS} GPUs may be used at the same time (got {args.gpus})")
    if any(s not in range(5) for s in args.splits):
        parser.error("splits must be in 0..4")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Longest jobs first (wA also unblocks wB), interleaved across splits.
    pending = [(e, s, m) for e in args.experiments for m in MEMBERS for s in args.splits
               if not complete(e, s, m)]
    order = {"wA": 0, "wC": 1, "wB": 2}
    pending.sort(key=lambda job: (order.get(job[0], 3), job[2], job[1]))
    log(f"{len(pending)} members to train on GPUs {args.gpus} "
        f"({args.jobs_per_gpu} jobs per GPU)")

    slots = {gpu: [] for gpu in args.gpus}   # gpu -> list of (job, Popen)
    failed = []
    while pending or any(slots.values()):
        # Reap finished jobs.
        for gpu, running in slots.items():
            for item in list(running):
                job, proc = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                name = "{}_s{}_m{}".format(*job)
                if proc.returncode == 0 and complete(*job):
                    log(f"done    {name} (GPU {gpu})")
                else:
                    log(f"FAILED  {name} (GPU {gpu}, exit {proc.returncode}); see {LOG_DIR / (name + '.log')}")
                    failed.append(job)
        # Launch runnable jobs into free slots.
        for gpu, running in slots.items():
            while len(running) < args.jobs_per_gpu:
                runnable = next((j for j in pending
                                 if j[0] != "wB" or complete("wA", j[1], j[2])), None)
                if runnable is None:
                    break
                pending.remove(runnable)
                experiment, split, member = runnable
                name = f"{experiment}_s{split}_m{member}"
                handle = (LOG_DIR / f"{name}.log").open("w")
                proc = subprocess.Popen(
                    ["bash", f"experiment_{experiment}/train_{experiment}.sh",
                     str(split), str(member), str(gpu)],
                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                running.append((runnable, proc))
                log(f"started {name} (GPU {gpu}, pid {proc.pid})")
        # A wB job whose parent failed can never run.
        blocked = [j for j in pending if j[0] == "wB" and ("wA", j[1], j[2]) in failed]
        for job in blocked:
            pending.remove(job)
            failed.append(job)
            log("SKIPPED {}_s{}_m{}: parent wA member failed".format(*job))
        if pending and not any(slots.values()):
            # Nothing running and nothing runnable: remaining wB jobs wait on wA
            # members that this invocation will never train.
            for job in pending:
                log("SKIPPED {}_s{}_m{}: parent wA member missing".format(*job))
            failed.extend(pending)
            pending = []
        if pending or any(slots.values()):
            time.sleep(args.poll_seconds)

    log(f"finished; {len(failed)} failed: {failed}")
    print("ALL_WATER_TRAINING_DONE", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
