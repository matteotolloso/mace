#!/usr/bin/env python3
"""Plot exact ensemble train/validation curves from checkpoints.

For each common epoch across ensemble members, this script:
- loads the full ensemble checkpoint set for that epoch
- evaluates the train split with the ensemble prediction
- evaluates the validation split with the ensemble prediction

It then writes exact ensemble metrics per epoch:
- ``train_loss`` / ``val_loss`` as the mean ensemble Gaussian NLL energy
- ``train_rmse_e_atom`` / ``val_rmse_e_atom`` as the exact ensemble RMSE per atom

This is more expensive than reading the ``*_train.txt`` logs, but it matches
the actual ensemble prediction instead of averaging member-wise metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int
from epoch_quality import (
    build_dataloader,
    discover_checkpoint_map,
    evaluate_split,
    load_models,
    release_models,
    select_common_epochs,
    setup_logging,
)
from mace.tools import torch_tools


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=False)
    parser.add_argument("--train-split", type=str, required=True)
    parser.add_argument("--validation-split", type=str, required=True)
    parser.add_argument("--energy-key-train", type=str, required=True)
    parser.add_argument("--energy-key-val", type=str, required=True)
    parser.add_argument("--head", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu"],
        default="cpu",
    )
    parser.add_argument(
        "--default-dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--every-n-epochs",
        type=int,
        default=1,
        help="Evaluate only epochs divisible by this value.",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system quantities instead of per-atom ones.",
    )
    parser.add_argument("--output-csv", type=str, default="train_curves.csv")
    parser.add_argument("--output-plot", type=str, default="train_curves.png")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.set_defaults(per_atom=True)
    return parser.parse_args()


def summarize_raw_rows(raw_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not raw_rows:
        return {
            "loss": float("nan"),
            "rmse_e_atom": float("nan"),
        }

    nll = np.array([row["nll_energy"] for row in raw_rows], dtype=float)
    sq_errors = np.array([row["sq_error"] for row in raw_rows], dtype=float)
    finite_nll = nll[np.isfinite(nll)]
    finite_sq_errors = sq_errors[np.isfinite(sq_errors)]
    return {
        "loss": float(np.mean(finite_nll)) if finite_nll.size > 0 else float("nan"),
        "rmse_e_atom": (
            float(np.sqrt(np.mean(finite_sq_errors)))
            if finite_sq_errors.size > 0
            else float("nan")
        ),
    }


def compute_rows(args: argparse.Namespace) -> List[Dict[str, float]]:
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoint_map = discover_checkpoint_map(checkpoints_dir, args.experiment_name)
    epochs = select_common_epochs(checkpoint_map, args.every_n_epochs)
    seeds = sorted(checkpoint_map)
    LOGGER.info(
        "Evaluating exact ensemble train curves on %d epochs with %d members",
        len(epochs),
        len(seeds),
    )

    device = torch_tools.init_device(args.device)

    sample_epoch = epochs[0]
    sample_model = load_models([checkpoint_map[seeds[0]][sample_epoch]], device=device)[0]
    train_loader = build_dataloader(
        Path(args.train_split),
        args.energy_key_train,
        sample_model,
        batch_size=args.batch_size,
        head_name=args.head,
    )
    val_loader = build_dataloader(
        Path(args.validation_split),
        args.energy_key_val,
        sample_model,
        batch_size=args.batch_size,
        head_name=args.head,
    )
    release_models([sample_model], device)

    rows: List[Dict[str, float]] = []
    for epoch in epochs:
        LOGGER.info("Evaluating ensemble epoch %d", epoch)
        checkpoint_paths = [checkpoint_map[seed][epoch] for seed in seeds]
        models = load_models(checkpoint_paths, device=device)
        try:
            train_raw_rows = evaluate_split(
                models=models,
                data_loader=train_loader,
                device=device,
                per_atom=args.per_atom,
                head_name=args.head,
                split_name=f"train epoch {epoch}",
            )
            val_raw_rows = evaluate_split(
                models=models,
                data_loader=val_loader,
                device=device,
                per_atom=args.per_atom,
                head_name=args.head,
                split_name=f"val epoch {epoch}",
            )
        finally:
            release_models(models, device)

        train_summary = summarize_raw_rows(train_raw_rows)
        val_summary = summarize_raw_rows(val_raw_rows)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_summary["loss"],
                "val_loss": val_summary["loss"],
                "train_rmse_e_atom": train_summary["rmse_e_atom"],
                "val_rmse_e_atom": val_summary["rmse_e_atom"],
            }
        )

    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_rmse_e_atom",
        "val_rmse_e_atom",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.array([int(row["epoch"]) for row in rows], dtype=int)
    train_loss = np.array([row["train_loss"] for row in rows], dtype=float)
    val_loss = np.array([row["val_loss"] for row in rows], dtype=float)
    train_rmse = np.array([row["train_rmse_e_atom"] for row in rows], dtype=float)
    val_rmse = np.array([row["val_rmse_e_atom"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    loss_ax, rmse_ax = axes

    loss_ax.plot(
        epochs,
        train_loss,
        color="tab:blue",
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label="Train Loss",
    )
    loss_ax.plot(
        epochs,
        val_loss,
        color="tab:orange",
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label="Val Loss",
    )
    loss_ax.set_title("Exact Ensemble Loss Curves", fontsize=18)
    loss_ax.set_ylabel("Loss", fontsize=15)
    loss_ax.grid(alpha=0.3)
    loss_ax.legend(fontsize=13)
    loss_ax.tick_params(axis="both", labelsize=13)

    rmse_ax.plot(
        epochs,
        train_rmse,
        color="tab:green",
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label="Train RMSE_E_per_atom",
    )
    rmse_ax.plot(
        epochs,
        val_rmse,
        color="tab:red",
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label="Val RMSE_E_per_atom",
    )
    rmse_ax.set_title("Exact Ensemble RMSE_E_per_atom", fontsize=18)
    rmse_ax.set_xlabel("Epoch", fontsize=15)
    rmse_ax.set_ylabel("RMSE_E_per_atom", fontsize=15)
    rmse_ax.grid(alpha=0.3)
    rmse_ax.legend(fontsize=13)
    rmse_ax.tick_params(axis="both", labelsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    torch_tools.set_default_dtype(args.default_dtype)

    output_csv = Path(args.output_csv)
    cached_rows = load_cached_csv_rows(
        output_csv,
        required_fields=[
            "epoch",
            "train_loss",
            "val_loss",
            "train_rmse_e_atom",
            "val_rmse_e_atom",
        ],
        field_parsers={
            "epoch": parse_int,
            "train_loss": parse_float,
            "val_loss": parse_float,
            "train_rmse_e_atom": parse_float,
            "val_rmse_e_atom": parse_float,
        },
        key_fields=["epoch"],
        compute_missing_rows=lambda: compute_rows(args),
        logger=LOGGER,
        label="train_curves",
    )
    if cached_rows is not None:
        write_plot(Path(args.output_plot), cached_rows)
        return

    rows = compute_rows(args)
    write_csv(output_csv, rows)
    write_plot(Path(args.output_plot), rows)


if __name__ == "__main__":
    main()
