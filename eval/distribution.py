#!/usr/bin/env python3
"""Plot AU/EU/error distributions for the best ensemble of an experiment.

Workflow:
1. Read one ``<experiment>_run-<seed>_train.txt`` file per ensemble member from
   ``--results-dir``.
2. Select the best validation epoch independently for each seed using
   ``--selection-key`` and ``--selection-mode``.
3. Load the selected checkpoints and build the ensemble.
4. Evaluate the ensemble on a requested split.
5. Plot four histograms, one per row:
   - aleatoric uncertainty (AU)
   - epistemic uncertainty (EU)
   - signed energy error
   - RMSE_E_per_atom (equal to absolute per-config error in per-atom mode)

Per-atom mode is the default:
- energies are normalized as ``E / N``
- variances are normalized as ``var / N^2``
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int, read_typed_csv
from reliability import (
    build_dataloader,
    evaluate_split,
    load_models,
    select_best_checkpoints,
    trim_rows_by_key,
)
from mace.tools import torch_tools


LOGGER = logging.getLogger(__name__)


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--split-path", type=str, required=True)
    parser.add_argument("--energy-key", type=str, required=True)
    parser.add_argument("--head", type=str, default=None)
    parser.add_argument("--selection-key", type=str, default="loss")
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["auto", "min", "max"],
        default="min",
    )
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
    parser.add_argument("--num-bins", type=int, default=50)
    parser.add_argument(
        "--trim",
        type=float,
        default=0.0,
        help="Symmetric fraction to trim from the lowest and highest total uncertainty values before plotting. Example: 0.005 trims 0.5%% on each side.",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system quantities instead of per-atom ones.",
    )
    parser.add_argument("--output-plot", type=str, default="distribution.png")
    parser.add_argument("--output-csv", type=str, default="distribution.csv")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=10,
    )
    parser.set_defaults(per_atom=True)
    return parser.parse_args()


def release_models(models: List[torch.nn.Module], device: torch.device) -> None:
    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_index",
        "num_atoms",
        "ref_energy",
        "pred_energy",
        "error",
        "sq_error",
        "rmse_e_atom",
        "aleatoric_var",
        "epistemic_var",
        "total_var",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            enriched_row = dict(row)
            enriched_row["rmse_e_atom"] = float(
                np.sqrt(max(float(row.get("sq_error", float("nan"))), 0.0))
            )
            writer.writerow({key: enriched_row.get(key, "") for key in fieldnames})


def _finite_array(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.array(
        [row[key] for row in rows if key in row and np.isfinite(row[key])],
        dtype=float,
    )


def write_plot(path: Path, rows: List[Dict[str, float]], per_atom: bool, num_bins: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uniform_fontsize = 26

    au = _finite_array(rows, "aleatoric_var")
    eu = _finite_array(rows, "epistemic_var")
    err = _finite_array(rows, "error")
    rmse_e_atom = np.array(
        [
            float(np.sqrt(max(float(row.get("sq_error", float("nan"))), 0.0)))
            for row in rows
            if np.isfinite(float(row.get("sq_error", float("nan"))))
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(4, 1, figsize=(8, 13))
    specs = [
        (axes[0], au, "Aleatoric Uncertainty", "tab:blue"),
        (axes[1], eu, "Epistemic Uncertainty", "tab:orange"),
        (axes[2], err, "Signed Energy Error", "tab:green"),
        (axes[3], rmse_e_atom, "RMSE_E_per_atom", "tab:red"),
    ]

    for ax, values, title, color in specs:
        if values.size == 0:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=uniform_fontsize,
            )
            ax.set_title(title, fontsize=uniform_fontsize)
            ax.grid(alpha=0.3)
            continue
        ax.hist(values, bins=num_bins, color=color, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_title(f"{title} (n={values.size})", fontsize=uniform_fontsize)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    axes[0].set_xlabel("AU per atom^2" if per_atom else "AU", fontsize=uniform_fontsize)
    axes[1].set_xlabel("EU per atom^2" if per_atom else "EU", fontsize=uniform_fontsize)
    axes[2].set_xlabel("Error per atom" if per_atom else "Error", fontsize=uniform_fontsize)
    axes[3].set_xlabel("RMSE_E_per_atom" if per_atom else "RMSE_E", fontsize=uniform_fontsize)
    for ax in axes:
        ax.set_ylabel("Count", fontsize=uniform_fontsize)
        ax.tick_params(axis="both", labelsize=uniform_fontsize)
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

    # fig.suptitle("Uncertainty and Error Distributions", y=0.98, fontsize=uniform_fontsize)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    total_start = time.perf_counter()

    output_csv_path = Path(args.output_csv)

    def compute_distribution_rows_from_existing_csv() -> List[Dict[str, float]]:
        legacy_rows = read_typed_csv(
            output_csv_path,
            required_fields=[
                "config_index",
                "num_atoms",
                "ref_energy",
                "pred_energy",
                "error",
                "sq_error",
                "aleatoric_var",
                "epistemic_var",
                "total_var",
            ],
            field_parsers={
                "config_index": parse_int,
                "num_atoms": parse_float,
                "ref_energy": parse_float,
                "pred_energy": parse_float,
                "error": parse_float,
                "sq_error": parse_float,
                "aleatoric_var": parse_float,
                "epistemic_var": parse_float,
                "total_var": parse_float,
            },
        )
        for row in legacy_rows:
            row["rmse_e_atom"] = float(np.sqrt(max(float(row["sq_error"]), 0.0)))
        return legacy_rows

    cached_rows = load_cached_csv_rows(
        output_csv_path,
        required_fields=[
            "config_index",
            "num_atoms",
            "ref_energy",
            "pred_energy",
            "error",
            "sq_error",
            "rmse_e_atom",
            "aleatoric_var",
            "epistemic_var",
            "total_var",
        ],
        field_parsers={
            "config_index": parse_int,
            "num_atoms": parse_float,
            "ref_energy": parse_float,
            "pred_energy": parse_float,
            "error": parse_float,
            "sq_error": parse_float,
            "rmse_e_atom": parse_float,
            "aleatoric_var": parse_float,
            "epistemic_var": parse_float,
            "total_var": parse_float,
        },
        logger=LOGGER,
        label="distribution",
        key_fields=["config_index"],
        compute_missing_rows=compute_distribution_rows_from_existing_csv,
    )
    if cached_rows is not None:
        write_plot(Path(args.output_plot), cached_rows, per_atom=args.per_atom, num_bins=args.num_bins)
        LOGGER.info("Saved plot: %s", args.output_plot)
        LOGGER.info("Total runtime: %.2fs", time.perf_counter() - total_start)
        return

    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)
    LOGGER.info("Using device=%s default_dtype=%s", device, args.default_dtype)

    checkpoint_paths = select_best_checkpoints(
        checkpoints_dir=Path(args.checkpoints_dir),
        results_dir=Path(args.results_dir),
        experiment_name=args.experiment_name,
        selection_key=args.selection_key,
        selection_mode=args.selection_mode,
        head_name=args.head,
    )
    LOGGER.info("Selected %d ensemble checkpoints", len(checkpoint_paths))

    models = load_models(checkpoint_paths, device=device)
    try:
        data_loader, num_configs = build_dataloader(
            Path(args.split_path),
            args.energy_key,
            models[0],
            batch_size=args.batch_size,
            head_name=args.head,
        )
        LOGGER.info(
            "Evaluating distribution on %s (%d configs)",
            args.split_path,
            num_configs,
        )
        raw_rows = evaluate_split(
            models=models,
            data_loader=data_loader,
            device=device,
            per_atom=args.per_atom,
            head_name=args.head,
            split_name=Path(args.split_path).stem,
            log_every_batches=args.log_every_batches,
        )
    finally:
        release_models(models, device)

    raw_rows = trim_rows_by_key(raw_rows, args.trim, key="total_var")
    write_csv(Path(args.output_csv), raw_rows)
    write_plot(Path(args.output_plot), raw_rows, per_atom=args.per_atom, num_bins=args.num_bins)

    LOGGER.info("Saved CSV: %s", args.output_csv)
    LOGGER.info("Saved plot: %s", args.output_plot)
    LOGGER.info("Total runtime: %.2fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
