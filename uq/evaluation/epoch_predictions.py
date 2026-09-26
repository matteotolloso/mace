#!/usr/bin/env python3
"""Per-configuration ensemble predictions for every sampled training epoch.

The epoch-quality caches keep only per-epoch summaries, so they cannot be
restricted to the training-support region afterwards. This module reuses the same
checkpoint discovery, model loading and ensemble evaluation code as
``uq.evaluation.epoch_quality`` and stores, per sampled epoch, the per-configuration
squared error, AU and EU of one test file.

Output ``experiments/<id>/evaluation/split_<s>/epoch_predictions_<file>.npz``
(per-atom quantities, eV and eV^2, file order of the XYZ):
  epochs         (E,)
  sq_error       (E, N)
  aleatoric_var  (E, N)
  epistemic_var  (E, N)

By default the file is the dataset's high-fidelity OOD file (``cc_test_ood`` on
ANI-1x) scored against the experiment's own labels, so an LF-only parent (DFT
labels) and its LF->HF child (CC labels) are compared on identical geometries.

    python -m uq.evaluation.epoch_predictions one C --split 2 --gpu 6   # one split
    python -m uq.evaluation.epoch_predictions all C D F --gpu 6         # 5 splits each, 2 streams

Each stream holds ~17 GB on an A100 (ten float64 members), so two fit on a 40 GB
card. Finished outputs are skipped and a job already running for the same output
is waited for, so ``all`` can be restarted at any time.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from uq import paths, registry

LOGGER = logging.getLogger("epoch_predictions")


def default_output(experiment, split, role="ood", fidelity="hf") -> Path:
    exp = registry.experiment(experiment)
    stem = Path(exp.data_file(role, split, fidelity)).stem
    return paths.split_evaluation_dir(exp.id, split) / f"epoch_predictions_{stem}.npz"


def predict(experiment, split, role="ood", fidelity="hf", every_n_epochs=5, batch_size=256,
            output=None, max_epochs=None, device_name="cuda"):
    """Evaluate every sampled epoch of one split and write the npz (see module doc)."""
    import torch

    from mace.tools import torch_tools
    from uq.evaluation.epoch_quality import (
        _get_model_dtype, build_dataloader, discover_checkpoint_map, evaluate_split, load_models,
        release_models, select_common_epochs,
    )

    exp = registry.experiment(experiment)
    output = Path(output or default_output(exp.id, split, role, fidelity))
    if output.exists() and max_epochs is None:
        print(f"cached {paths.rel(output)}")
        return output
    torch_tools.set_default_dtype("float64")
    device = torch_tools.init_device(device_name)
    checkpoints = discover_checkpoint_map(paths.checkpoints_dir(exp.id, split), exp.run_name)
    epochs = select_common_epochs(checkpoints, every_n_epochs)
    if max_epochs:
        epochs = epochs[:max_epochs]
    seeds = sorted(checkpoints)
    if seeds != list(exp.members):
        raise RuntimeError(f"Expected members {list(exp.members)}, found {seeds}")

    sample = load_models([checkpoints[seeds[0]][epochs[0]]], device=device)[0]
    torch.set_default_dtype(_get_model_dtype(sample))
    loader = build_dataloader(exp.data_file(role, split, fidelity), exp.fidelity.energy_key, sample,
                              batch_size=batch_size, head_name=None)
    release_models([sample], device)

    sq, au, eu = [], [], []
    for epoch in epochs:
        start = time.perf_counter()
        models = load_models([checkpoints[s][epoch] for s in seeds], device=device)
        rows = evaluate_split(models, loader, device, per_atom=True, head_name=None, split_name=f"epoch {epoch}")
        release_models(models, device)
        if [row["config_index"] for row in rows] != list(range(len(rows))):
            raise RuntimeError("Rows are not in file order.")
        sq.append([row["sq_error"] for row in rows])
        au.append([row["aleatoric_var"] for row in rows])
        eu.append([row["epistemic_var"] for row in rows])
        print(f"{exp.label} split {split} epoch {epoch}: {time.perf_counter() - start:.1f}s", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, epochs=np.asarray(epochs), sq_error=np.asarray(sq),
                        aleatoric_var=np.asarray(au), epistemic_var=np.asarray(eu))
    if max_epochs is None:
        tmp.replace(output)
    else:
        tmp.unlink()
    return output


def _run_one(job, gpu):
    experiment, split = job
    out = default_output(experiment, split)
    if out.exists():
        return f"cached {paths.rel(out)}"
    # A previous runner may have left this job running; wait for it instead of duplicating it.
    while subprocess.call(["pgrep", "-f", "--", f"--output {out}"], stdout=subprocess.DEVNULL) == 0:
        time.sleep(30)
    if out.exists():
        return f"joined {paths.rel(out)}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = out.with_suffix(".log")
    out.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        code = subprocess.call([sys.executable, "-B", "-m", "uq.evaluation.epoch_predictions", "one",
                                experiment, "--split", str(split), "--output", str(out)],
                               cwd=paths.ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    status = "done  " if code == 0 and out.exists() else "FAILED"
    return f"{status} {paths.rel(out)} (exit {code}, log {log.name})"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("one", help="one experiment and split (uses the current CUDA device)")
    one.add_argument("experiment")
    one.add_argument("--split", type=int, required=True)
    one.add_argument("--role", default="ood", help="test role of the file (default ood)")
    one.add_argument("--fidelity", default="hf", help="whose file: lf or hf (default hf)")
    one.add_argument("--every-n-epochs", type=int, default=5)
    one.add_argument("--max-epochs", type=int, help="evaluate only the first N sampled epochs (timing)")
    one.add_argument("--batch-size", type=int, default=256)
    one.add_argument("--output", type=Path)
    run_all = sub.add_parser("all", help="every split of the given experiments on one GPU")
    run_all.add_argument("experiments", nargs="+")
    run_all.add_argument("--gpu", type=int, required=True)
    run_all.add_argument("--streams", type=int, default=2, help="concurrent jobs on the GPU")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    if args.command == "one":
        predict(args.experiment, args.split, role=args.role, fidelity=args.fidelity,
                every_n_epochs=args.every_n_epochs, batch_size=args.batch_size, output=args.output,
                max_epochs=args.max_epochs)
        return 0
    jobs = [(registry.experiment(e).id, s) for e in args.experiments for s in registry.experiment(e).splits]
    with ThreadPoolExecutor(args.streams) as pool:
        for message in pool.map(lambda job: _run_one(job, args.gpu), jobs):
            print(message, flush=True)
    print("ALL_EPOCH_PREDICTIONS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
