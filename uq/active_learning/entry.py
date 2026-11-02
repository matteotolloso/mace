"""``uq al`` entry points (replace ``run.sh`` and ``run_all.sh``).

    uq al run 6 --split-seed 0 --epochs 50              # one split  (workflow)
    uq al run-all 6 --epochs 50                         # five splits + aggregate (five_splits)
    uq al run-all 6 --aggregate-only --wait --epochs 50

``run`` sets ``CUDA_VISIBLE_DEVICES`` to the GPU and runs ``uq.active_learning.workflow``
with the remaining arguments; ``run-all`` runs ``uq.active_learning.five_splits --gpu``.
"""

from __future__ import annotations

import os
import subprocess
import sys

from uq import paths
from uq.active_learning.common import RUNS


def main(command, gpu, rest) -> int:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if command == "run":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        argv = [sys.executable, "-B", "-m", "uq.active_learning.workflow", *rest]
    else:
        argv = [sys.executable, "-B", "-m", "uq.active_learning.five_splits", "--gpu", str(gpu), *rest]
    return subprocess.call(argv, env=env, cwd=paths.ROOT)


def summaries():
    return sorted([*RUNS.glob("aggregate*/summary_ci95.json"), *RUNS.glob("aggregate*/common_evaluator/summary_ci95.json")])


def replot_all(dry_run=False) -> int:
    """Re-render the plots of every AL aggregate (part of ``uq aggregate``)."""
    status = 0
    for summary in summaries():
        argv = [sys.executable, "-B", "-m", "uq.active_learning.plot_results", "--summary", str(summary)]
        print(" ".join(argv[1:]) if dry_run else f"[al plots] {paths.rel(summary)}", flush=True)
        if not dry_run:
            status |= subprocess.call(argv, cwd=paths.ROOT)
    return status
