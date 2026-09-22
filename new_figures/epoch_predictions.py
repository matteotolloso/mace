#!/usr/bin/env python3
"""Per-configuration ensemble predictions for every sampled training epoch.

The existing epoch-quality caches keep only per-epoch summaries, so they cannot be
restricted to the training-support region afterwards. This program reuses the same
checkpoint discovery, model loading and ensemble evaluation code as
eval/epoch_quality.py, and stores the per-configuration squared error, AU and EU
for each sampled epoch. Results are cached per (experiment, split, test, key).

Output npz (per-atom quantities, eV and eV^2, file order of the XYZ):
  epochs         (E,)
  sq_error       (E, N)
  aleatoric_var  (E, N)
  epistemic_var  (E, N)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import torch  # noqa: E402
from epoch_quality import (  # noqa: E402
    _get_model_dtype, build_dataloader, discover_checkpoint_map, evaluate_split,
    load_models, release_models, select_common_epochs,
)
from mace.tools import torch_tools  # noqa: E402

LOGGER = logging.getLogger("epoch_predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--kind", choices=("energy", "system"), required=True)
    parser.add_argument("--test", default="cc_test_ood")
    parser.add_argument("--energy-key", required=True)
    parser.add_argument("--every-n-epochs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, help="Evaluate only the first N sampled epochs (timing)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    if args.output.exists() and args.max_epochs is None:
        print(f"cached {args.output}")
        return
    torch_tools.set_default_dtype("float64")
    device = torch_tools.init_device("cuda")
    checkpoints = discover_checkpoint_map(ROOT / f"experiment_{args.experiment}" / f"checkpoints_{args.split}", "mace")
    epochs = select_common_epochs(checkpoints, args.every_n_epochs)
    if args.max_epochs:
        epochs = epochs[: args.max_epochs]
    seeds = sorted(checkpoints)
    if seeds != list(range(10)):
        raise RuntimeError(f"Expected members 0..9, found {seeds}")

    sample = load_models([checkpoints[seeds[0]][epochs[0]]], device=device)[0]
    torch.set_default_dtype(_get_model_dtype(sample))
    loader = build_dataloader(ROOT / "dataset" / f"ani1x_{args.kind}_split_{args.split}" / f"{args.test}.xyz",
                              args.energy_key, sample, batch_size=args.batch_size, head_name=None)
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
        print(f"{args.experiment} split {args.split} epoch {epoch}: {time.perf_counter() - start:.1f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, epochs=np.asarray(epochs), sq_error=np.asarray(sq),
                        aleatoric_var=np.asarray(au), epistemic_var=np.asarray(eu))
    if args.max_epochs is None:
        tmp.replace(args.output)
    else:
        tmp.unlink()


if __name__ == "__main__":
    main()
