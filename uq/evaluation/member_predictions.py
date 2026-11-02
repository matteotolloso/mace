#!/usr/bin/env python3
"""Per-member predictions of the final ensemble on one test file (GPU).

Selects each member's checkpoint exactly as ``reliability`` does (lowest validation
loss in its MACE results log), predicts every configuration of ``--split-path`` with
every member and saves the raw model outputs, so that single-member and
ensemble-size analyses (``uq.evaluation.ensemble_size``) run on CPU afterwards.

Output ``members_<role>.npz`` (one per split and test role):

    member_seeds   [M]      member seed of each row (from the checkpoint names)
    checkpoints    [M]      selected checkpoint file names
    num_atoms      [N]
    ref_energy     [N]      reference total energy (eV), file order
    pred_energy    [M, N]   member total energy (eV)
    pred_var       [M, N]   member predicted variance of the total energy (eV^2); 0 without MVE head

Per-atom quantities are ``pred_energy / num_atoms`` and ``pred_var / num_atoms**2``,
as in ``reliability.evaluate_split``. The ensemble mean, mean variance (AU) and
population variance of the means (EU) of all M members reproduce the
``reliability_<role>_nocal_raw.csv`` columns up to GPU non-determinism.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch

from uq.evaluation.reliability import (
    _cast_batch_dict_dtype,
    _get_model_dtype,
    _select_head_tensor,
    build_dataloader,
    load_models,
    select_best_checkpoints,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--energy-key", required=True)
    parser.add_argument("--selection-key", default="loss")
    parser.add_argument("--selection-mode", default="min")
    parser.add_argument("--head-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def member_seed(path: Path) -> int:
    match = re.search(r"_run-(\d+)", path.name)
    if match is None:
        raise RuntimeError(f"cannot read the member seed from {path.name}")
    return int(match.group(1))


def predict(models, loader, device, head_name):
    """[M, N] total energies and variances, [N] reference energies and atom counts."""
    model_heads = getattr(models[0], "heads", None)
    dtype = _get_model_dtype(models[0])
    energies, variances, refs, atoms = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_dict = _cast_batch_dict_dtype(batch.to_dict(), dtype)
            batch_heads = batch["head"] if "head" in batch.keys else None
            preds, vars_ = [], []
            for model in models:
                output = model(batch_dict, training=False, compute_force=False, compute_virials=False,
                               compute_stress=False, compute_hessian=False, compute_edge_forces=False)
                energy = _select_head_tensor(output.get("energy_mean", output.get("energy")),
                                             batch_heads, head_name, model_heads)
                var = output.get("energy_var", None)
                if var is None and output.get("energy_logvar", None) is not None:
                    var = torch.exp(output["energy_logvar"])
                var = _select_head_tensor(var, batch_heads, head_name, model_heads)
                if var is None:
                    var = torch.zeros_like(energy)
                preds.append(energy)
                vars_.append(var)
            energies.append(torch.stack(preds).cpu().numpy())
            variances.append(torch.stack(vars_).cpu().numpy())
            refs.append(batch.energy.cpu().numpy())
            atoms.append((batch.ptr[1:] - batch.ptr[:-1]).cpu().numpy())
    return (np.concatenate(energies, axis=1), np.concatenate(variances, axis=1),
            np.concatenate(refs), np.concatenate(atoms))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    start = time.perf_counter()
    device = torch.device(args.device)
    checkpoints = select_best_checkpoints(args.checkpoints_dir, args.results_dir, args.experiment_name,
                                          args.selection_key, args.selection_mode, args.head_name)
    seeds = [member_seed(path) for path in checkpoints]
    LOGGER.info("Selected checkpoints: %s", [path.name for path in checkpoints])
    models = load_models(checkpoints, device)
    loader, count = build_dataloader(args.split_path, args.energy_key, models[0], args.batch_size, args.head_name)
    pred_energy, pred_var, ref_energy, num_atoms = predict(models, loader, device, args.head_name)
    if pred_energy.shape != (len(models), count):
        raise RuntimeError(f"unexpected prediction shape {pred_energy.shape} for {len(models)} members x {count}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, member_seeds=np.asarray(seeds), checkpoints=np.asarray([p.name for p in checkpoints]),
                        num_atoms=num_atoms, ref_energy=ref_energy, pred_energy=pred_energy, pred_var=pred_var)
    LOGGER.info("Wrote %s (%d members x %d configurations) in %.1fs", args.output, len(models), count,
                time.perf_counter() - start)


if __name__ == "__main__":
    main()
