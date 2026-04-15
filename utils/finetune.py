#!/usr/bin/env python3
"""Plot pretrain-vs-finetune AU/EU summaries on ID and OOD test splits.

For a fine-tuning experiment, this script:
1. Selects the best pretrain checkpoint independently for each seed from the
   pretrain ``*_train.txt`` logs.
2. Selects the best fine-tuned checkpoint independently for each seed from the
   fine-tune ``*_train.txt`` logs.
3. Evaluates both ensembles on the same ID and OOD test splits.
4. Computes the mean aleatoric and epistemic variance over each split.
5. Produces a bar plot with 8 bars covering all combinations of:
   - pretrain / finetune
   - ID / OOD
   - AU / EU

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

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_str
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
    parser.add_argument("--pretrain-checkpoints-dir", type=str, required=True)
    parser.add_argument("--pretrain-results-dir", type=str, required=True)
    parser.add_argument("--pretrain-experiment-name", type=str, required=True)
    parser.add_argument("--finetune-checkpoints-dir", type=str, required=True)
    parser.add_argument("--finetune-results-dir", type=str, required=True)
    parser.add_argument("--finetune-experiment-name", type=str, required=True)
    parser.add_argument("--test-id-split", type=str, required=True)
    parser.add_argument("--test-ood-split", type=str, required=True)
    parser.add_argument("--energy-key-test", type=str, required=True)
    parser.add_argument("--head", type=str, default=None)
    parser.add_argument(
        "--selection-key-pretrain",
        type=str,
        default="loss",
        help="Metric used to select the best pretrain checkpoint per seed.",
    )
    parser.add_argument(
        "--selection-key-finetune",
        type=str,
        default="loss",
        help="Metric used to select the best fine-tuned checkpoint per seed.",
    )
    parser.add_argument(
        "--selection-mode-pretrain",
        type=str,
        choices=["auto", "min", "max"],
        default="min",
    )
    parser.add_argument(
        "--selection-mode-finetune",
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
    parser.add_argument(
        "--trim",
        type=float,
        default=0.0,
        help="Symmetric fraction to trim from the lowest and highest total uncertainty values before summarizing. Example: 0.005 trims 0.5%% on each side.",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system quantities instead of per-atom ones.",
    )
    parser.add_argument("--output-csv", type=str, default="finetune.csv")
    parser.add_argument("--output-plot", type=str, default="finetune.png")
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


def summarize_rows(raw_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not raw_rows:
        return {
            "mean_au": float("nan"),
            "mean_eu": float("nan"),
            "mean_tu": float("nan"),
            "rmse_energy": float("nan"),
        }

    au = np.array([row["aleatoric_var"] for row in raw_rows], dtype=float)
    eu = np.array([row["epistemic_var"] for row in raw_rows], dtype=float)
    tu = np.array([row["total_var"] for row in raw_rows], dtype=float)
    sq_err = np.array([row["sq_error"] for row in raw_rows], dtype=float)
    return {
        "mean_au": float(np.nanmean(au)),
        "mean_eu": float(np.nanmean(eu)),
        "mean_tu": float(np.nanmean(tu)),
        "rmse_energy": float(np.sqrt(np.nanmean(sq_err))),
    }


def evaluate_ensemble_on_split(
    checkpoint_paths: List[Path],
    split_path: Path,
    energy_key: str,
    device: torch.device,
    batch_size: int,
    per_atom: bool,
    head_name: str | None,
    split_name: str,
    log_every_batches: int,
    trim: float,
) -> Dict[str, float]:
    models = load_models(checkpoint_paths, device=device)
    try:
        data_loader, _ = build_dataloader(
            split_path,
            energy_key,
            models[0],
            batch_size=batch_size,
            head_name=head_name,
        )
        raw_rows = evaluate_split(
            models=models,
            data_loader=data_loader,
            device=device,
            per_atom=per_atom,
            head_name=head_name,
            split_name=split_name,
            log_every_batches=log_every_batches,
        )
        raw_rows = trim_rows_by_key(raw_rows, trim, key="total_var")
        return summarize_rows(raw_rows)
    finally:
        release_models(models, device)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "split",
        "mean_au",
        "mean_eu",
        "mean_tu",
        "rmse_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: List[Dict[str, object]], per_atom: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

    row_map = {(str(row["phase"]), str(row["split"])): row for row in rows}
    phase_order = ["pretrain", "finetune"]
    split_order = ["id", "ood"]
    metric_specs = [("mean_au", "AU"), ("mean_eu", "EU")]
    phase_colors = {
        "pretrain": "tab:blue",
        "finetune": "tab:green",
    }
    bar_width = 0.34
    group_x = np.arange(len(metric_specs))

    for ax, split in zip(axes, split_order):
        max_value = 0.0
        bar_groups = []
        for phase_idx, phase in enumerate(phase_order):
            offset = (phase_idx - 0.5) * bar_width
            x = group_x + offset
            values = []
            for metric_key, _ in metric_specs:
                row = row_map.get((phase, split))
                values.append(float("nan") if row is None else float(row[metric_key]))
            bars = ax.bar(
                x,
                values,
                width=bar_width,
                color=phase_colors[phase],
                label=phase.capitalize(),
            )
            bar_groups.append((bars, values))
            finite_values = [value for value in values if np.isfinite(value)]
            if finite_values:
                max_value = max(max_value, max(finite_values))

        ax.set_xticks(group_x)
        ax.set_xticklabels([label for _, label in metric_specs])
        ax.set_title(split.upper())
        ax.grid(axis="y", alpha=0.3)
        if max_value > 0.0:
            ax.set_ylim(0.0, max_value * 1.18)

        for bars, values in bar_groups:
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        value + max(max_value * 0.02, 1e-12),
                        f"{value:.2e}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0,
                    )

    axes[0].set_ylabel("Mean uncertainty per atom^2" if per_atom else "Mean uncertainty")
    fig.suptitle("AU/EU Change After Finetuning", y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    total_start = time.perf_counter()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)
    LOGGER.info("Using device=%s default_dtype=%s", device, args.default_dtype)

    def compute_rows() -> List[Dict[str, object]]:
        pretrain_checkpoints = select_best_checkpoints(
            checkpoints_dir=Path(args.pretrain_checkpoints_dir),
            results_dir=Path(args.pretrain_results_dir),
            experiment_name=args.pretrain_experiment_name,
            selection_key=args.selection_key_pretrain,
            selection_mode=args.selection_mode_pretrain,
            head_name=args.head,
        )
        finetune_checkpoints = select_best_checkpoints(
            checkpoints_dir=Path(args.finetune_checkpoints_dir),
            results_dir=Path(args.finetune_results_dir),
            experiment_name=args.finetune_experiment_name,
            selection_key=args.selection_key_finetune,
            selection_mode=args.selection_mode_finetune,
            head_name=args.head,
        )

        rows: List[Dict[str, object]] = []
        eval_specs = [
            ("pretrain", pretrain_checkpoints, "id", Path(args.test_id_split)),
            ("pretrain", pretrain_checkpoints, "ood", Path(args.test_ood_split)),
            ("finetune", finetune_checkpoints, "id", Path(args.test_id_split)),
            ("finetune", finetune_checkpoints, "ood", Path(args.test_ood_split)),
        ]

        for phase, checkpoint_paths, split_name, split_path in eval_specs:
            LOGGER.info(
                "Evaluating %s ensemble on %s split (%s) with %d members",
                phase,
                split_name,
                split_path,
                len(checkpoint_paths),
            )
            summary = evaluate_ensemble_on_split(
                checkpoint_paths=checkpoint_paths,
                split_path=split_path,
                energy_key=args.energy_key_test,
                device=device,
                batch_size=args.batch_size,
                per_atom=args.per_atom,
                head_name=args.head,
                split_name=f"{phase}_{split_name}",
                log_every_batches=args.log_every_batches,
                trim=args.trim,
            )
            row = {"phase": phase, "split": split_name, **summary}
            rows.append(row)
            LOGGER.info(
                "%s %s summary: mean_au=%.6e mean_eu=%.6e mean_tu=%.6e rmse=%-.6e",
                phase,
                split_name,
                row["mean_au"],
                row["mean_eu"],
                row["mean_tu"],
                row["rmse_energy"],
            )
        return rows

    cached_rows = load_cached_csv_rows(
        Path(args.output_csv),
        required_fields=[
            "phase",
            "split",
            "mean_au",
            "mean_eu",
            "mean_tu",
            "rmse_energy",
        ],
        field_parsers={
            "phase": parse_str,
            "split": parse_str,
            "mean_au": parse_float,
            "mean_eu": parse_float,
            "mean_tu": parse_float,
            "rmse_energy": parse_float,
        },
        logger=LOGGER,
        label="finetune",
        key_fields=["phase", "split"],
        compute_missing_rows=compute_rows,
    )
    if cached_rows is not None:
        write_plot(Path(args.output_plot), cached_rows, per_atom=args.per_atom)
        LOGGER.info("Saved plot: %s", args.output_plot)
        LOGGER.info("Total runtime: %.2fs", time.perf_counter() - total_start)
        return

    rows = compute_rows()

    write_csv(Path(args.output_csv), rows)
    write_plot(Path(args.output_plot), rows, per_atom=args.per_atom)

    LOGGER.info("Saved CSV: %s", args.output_csv)
    LOGGER.info("Saved plot: %s", args.output_plot)
    LOGGER.info("Total runtime: %.2fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
