#!/usr/bin/env python3
"""Run epoch_predictions.py for every experiment/split needed by Figs. 3 and 5.

Usage: python new_figures/run_epoch_predictions.py <gpu> [streams]

Only the energy split (C, D, F) is re-evaluated. On the system split the
training-support filter removes no configuration, so the existing per-epoch caches
(experiment_X/evaluation/cache/split_S/epoch_quality*_ood.csv) are already exact
and are used directly.

Each stream holds ~17 GB on an A100 (ten float64 members), so two streams fit on a
40 GB card. The pretraining stage (C) is scored on the same cc_test_ood geometries
as fine-tuning, against their DFT labels, so both stages see identical
configurations. Finished outputs are skipped and a job already running for the same
output is waited for, so the runner can be restarted at any time.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "_cache" / "epochs"
DFT, CC = "wb97x_tz.energy", "ccsd(t)_cbs.energy"
SPECS = [("C", "energy", DFT), ("D", "energy", CC), ("F", "energy", CC)]


def run(job, gpu):
    experiment, kind, key, split = job
    out = OUT / f"{experiment}_{split}.npz"
    if out.exists():
        return f"cached {out.name}"
    # A previous runner may have left this job running; wait for it instead of duplicating it.
    while subprocess.call(["pgrep", "-f", "--", f"--output {out}"], stdout=subprocess.DEVNULL) == 0:
        time.sleep(30)
    if out.exists():
        return f"joined {out.name}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = OUT / f"{experiment}_{split}.log"
    with log.open("w") as handle:
        code = subprocess.call([sys.executable, "-B", str(HERE / "epoch_predictions.py"),
                                "--experiment", experiment, "--split", str(split), "--kind", kind,
                                "--energy-key", key, "--output", str(out)],
                               cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return f"{'done  ' if code == 0 and out.exists() else 'FAILED'} {out.name} (exit {code}, log {log.name})"


if __name__ == "__main__":
    gpu = int(sys.argv[1])
    streams = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(e, k, key, s) for (e, k, key) in SPECS for s in range(5)]
    with ThreadPoolExecutor(streams) as pool:
        for message in pool.map(lambda job: run(job, gpu), jobs):
            print(message, flush=True)
    print("ALL_EPOCH_PREDICTIONS_DONE", flush=True)
