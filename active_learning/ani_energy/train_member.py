#!/usr/bin/env python3
"""Exact full-model warm start using the existing MACE training loop."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common import HERE, inventory, save_json
from runtime import load_ensemble, make_loader, reliability

import torch
from ase.io import read
from mace import tools
from mace.tools.scripts_utils import LRScheduler, get_loss_fn, get_optimizer, get_params_options


def continue_member(config_path, initial_path):
    args, _ = tools.check_args(tools.build_default_arg_parser().parse_args(["--config", str(config_path)]))
    for key in ("work_dir", "model_dir", "log_dir", "results_dir", "checkpoints_dir", "downloads_dir"):
        if not Path(getattr(args, key)).resolve().is_relative_to(config_path.parent.resolve()):
            raise ValueError(f"{key} must stay inside this AL training attempt.")
    if args.swa or args.ema or args.distributed or args.compute_forces or not args.predict_mve:
        raise ValueError("This adapter expects the current single-head, energy-only MVE protocol (no SWA/EMA/DDP).")
    tools.set_seeds(args.seed)
    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    model = load_ensemble([initial_path], device)[0]
    # Preserve E0s, scale/shift buffers, mean and variance heads; do not rebuild a foundation model.
    model.requires_grad_(True)
    model.train()
    train_loader = make_loader(
        read(args.train_file, index=":"), model, args.batch_size, labeled=True,
        training=True, seed=args.seed, num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    valid_loader = make_loader(
        read(args.valid_file, index=":"), model, args.valid_batch_size, labeled=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    optimizer = get_optimizer(args, get_params_options(args, model))
    scheduler = LRScheduler(optimizer, args)
    tag = tools.get_tag(args.name, args.seed)
    resolved = {**vars(args), "hidden_irreps": str(args.hidden_irreps)}
    save_json(config_path.parent / "resolved_args.json", resolved)
    tools.train(
        model=model, loss_fn=get_loss_fn(args, dipole_only=False, compute_dipole=False),
        train_loader=train_loader, valid_loaders={getattr(model, "heads", ["Default"])[0]: valid_loader},
        optimizer=optimizer, lr_scheduler=scheduler, start_epoch=0,
        max_num_epochs=args.max_num_epochs, patience=args.patience,
        checkpoint_handler=tools.CheckpointHandler(args.checkpoints_dir, tag, keep=True),
        logger=tools.MetricsLogger(args.results_dir, tag + "_train"),
        eval_interval=args.eval_interval,
        output_args={"energy": True, "forces": False, "virials": False, "stress": False,
                     "dipoles": False, "polarizabilities": False},
        device=device, log_errors=args.error_table, max_grad_norm=args.clip_grad,
        save_all_checkpoints=True, log_wandb=False, log_epoch_outputs=False,
    )
    companion = Path(args.checkpoints_dir) / f"{tag}.model"
    torch.save(model.cpu(), companion)
    best = reliability.select_best_checkpoints(
        Path(args.checkpoints_dir), Path(args.results_dir), args.name, "loss", "min", None
    )
    if len(best) != 1:
        raise RuntimeError("Expected exactly one trained member.")
    selected = load_ensemble(best, device)[0]
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / "best.model"
    torch.save(selected.cpu(), destination)
    save_json(config_path.parent / "finished.json", {
        "model": str(destination), "selected_checkpoint": str(best[0]),
        "selection_key": "loss", "selection_mode": "min",
        "artifacts": inventory([destination, best[0], companion, config_path,
                                config_path.parent / "resolved_args.json",
                                Path(args.results_dir) / f"{tag}_train.txt"]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    options = parser.parse_args()
    if not options.config.resolve().is_relative_to(HERE / "runs"):
        parser.error("Training configs must live inside active_learning/ani_energy/runs/.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    continue_member(options.config.resolve(), options.initial.resolve())
