#!/usr/bin/env python3
"""Concatenate pretrain and fine-tune epoch-quality curves on one timeline.

The script evaluates two ensembles directly from checkpoint directories on the
same dataset:
- a pretrain ensemble
- a fine-tuned ensemble

For each stage it computes the same metrics as ``utils/epoch_quality.py`` over
all common epochs, then concatenates the two timelines and draws a vertical line
at the fine-tuning start.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int, parse_str
from epoch_quality import (
    _get_model_dtype,
    build_dataloader,
    discover_checkpoint_map,
    evaluate_split,
    load_models,
    release_models,
    select_common_epochs,
    setup_logging,
    summarize_epoch,
    with_free_scale_suffix,
)
from reliability import trim_rows_by_key
from mace.tools import torch_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrain-checkpoints-dir", type=str, required=True)
    parser.add_argument("--pretrain-experiment-name", type=str, required=True)
    parser.add_argument("--finetune-checkpoints-dir", type=str, required=True)
    parser.add_argument("--finetune-experiment-name", type=str, required=True)
    parser.add_argument("--split-path", type=str, required=True)
    parser.add_argument("--energy-key", type=str, required=True)
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
        "--num-bins",
        type=int,
        default=15,
        help="Number of uncertainty bins used for ENCE.",
    )
    parser.add_argument(
        "--every-n-epochs-pretrain",
        type=int,
        default=5,
        help="Evaluate only common pretrain epochs divisible by this value.",
    )
    parser.add_argument(
        "--every-n-epochs-finetune",
        type=int,
        default=5,
        help="Evaluate only common fine-tune epochs divisible by this value.",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.0,
        help="Symmetric fraction to trim from the lowest and highest total uncertainty values before summarizing each epoch.",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system quantities instead of per-atom ones.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="epoch_quality_finetune_same.csv",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="epoch_quality_finetune_same.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Epoch Quality: Same Dataset Pretrain + Finetune",
    )
    parser.add_argument(
        "--free-scale",
        action="store_true",
        help="Also save a second autoscaled plot with '-free-scale' appended to the output filename.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.set_defaults(per_atom=True)
    return parser.parse_args()


def evaluate_stage(
    checkpoints_dir: Path,
    experiment_name: str,
    split_path: Path,
    energy_key: str,
    device,
    batch_size: int,
    per_atom: bool,
    head_name: str | None,
    every_n_epochs: int,
    trim: float,
    num_bins: int,
) -> List[Dict[str, float]]:
    checkpoint_map = discover_checkpoint_map(checkpoints_dir, experiment_name)
    epochs = select_common_epochs(checkpoint_map, every_n_epochs)
    seeds = sorted(checkpoint_map)

    sample_epoch = epochs[0]
    sample_model = load_models([checkpoint_map[seeds[0]][sample_epoch]], device=device)[0]
    _ = _get_model_dtype(sample_model)
    data_loader = build_dataloader(
        split_path,
        energy_key,
        sample_model,
        batch_size=batch_size,
        head_name=head_name,
    )
    release_models([sample_model], device)

    rows: List[Dict[str, float]] = []
    for epoch in epochs:
        checkpoint_paths = [checkpoint_map[seed][epoch] for seed in seeds]
        models = load_models(checkpoint_paths, device=device)
        raw_rows = evaluate_split(
            models=models,
            data_loader=data_loader,
            device=device,
            per_atom=per_atom,
            head_name=head_name,
            split_name=f"{experiment_name} epoch {epoch}",
        )
        raw_rows = trim_rows_by_key(raw_rows, trim, key="total_var")
        summary = summarize_epoch(raw_rows, num_bins=num_bins)
        summary["epoch"] = epoch
        rows.append(summary)
        release_models(models, device)

    return rows


def build_combined_rows(
    pretrain_rows: List[Dict[str, float]],
    finetune_rows: List[Dict[str, float]],
) -> Tuple[List[Dict[str, float]], int]:
    if not pretrain_rows:
        raise RuntimeError("Pretrain evaluation produced no rows.")
    if not finetune_rows:
        raise RuntimeError("Finetune evaluation produced no rows.")

    max_pretrain_epoch = max(int(row["epoch"]) for row in pretrain_rows)
    finetune_epochs = np.array(sorted({int(row["epoch"]) for row in finetune_rows}), dtype=int)
    finetune_step = int(np.min(np.diff(finetune_epochs))) if finetune_epochs.size > 1 else 1
    finetune_start = max_pretrain_epoch + finetune_step
    min_finetune_epoch = min(int(row["epoch"]) for row in finetune_rows)
    finetune_shift = finetune_start - min_finetune_epoch

    combined_rows: List[Dict[str, float]] = []
    for row in pretrain_rows:
        combined_rows.append(
            {
                "phase": "pretrain",
                "source_epoch": int(row["epoch"]),
                "epoch": int(row["epoch"]),
                **{key: value for key, value in row.items() if key != "epoch"},
            }
        )
    for row in finetune_rows:
        combined_rows.append(
            {
                "phase": "finetune",
                "source_epoch": int(row["epoch"]),
                "epoch": int(row["epoch"]) + finetune_shift,
                **{key: value for key, value in row.items() if key != "epoch"},
            }
        )
    return combined_rows, finetune_start


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "source_epoch",
        "epoch",
        "pearson_aleatoric",
        "pearson_epistemic",
        "pearson_total",
        "spearman_aleatoric",
        "spearman_epistemic",
        "spearman_total",
        "ause_aleatoric",
        "ause_epistemic",
        "ause_total",
        "ence_aleatoric",
        "ence_epistemic",
        "ence_total",
        "magnitude_aleatoric",
        "magnitude_epistemic",
        "magnitude_total",
        "rmse_e_atom",
        "nll_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, float]] | None:
    metric_fields = [
        "pearson_aleatoric",
        "pearson_epistemic",
        "pearson_total",
        "spearman_aleatoric",
        "spearman_epistemic",
        "spearman_total",
        "ause_aleatoric",
        "ause_epistemic",
        "ause_total",
        "ence_aleatoric",
        "ence_epistemic",
        "ence_total",
        "magnitude_aleatoric",
        "magnitude_epistemic",
        "magnitude_total",
        "rmse_e_atom",
        "nll_energy",
    ]
    field_parsers = {
        "phase": parse_str,
        "source_epoch": parse_int,
        "epoch": parse_int,
    }
    for field in metric_fields:
        field_parsers[field] = parse_float
    cached_rows = load_cached_csv_rows(
        path,
        required_fields=["phase", "source_epoch", "epoch", *metric_fields],
        field_parsers=field_parsers,
        key_fields=["phase", "source_epoch", "epoch"],
        label="epoch_quality_finetune_same",
    )
    if cached_rows is None:
        return None
    return cached_rows


def infer_finetune_start(rows: List[Dict[str, float]]) -> int:
    finetune_epochs = sorted(
        int(row["epoch"]) for row in rows if str(row["phase"]) == "finetune"
    )
    if not finetune_epochs:
        raise RuntimeError("Cached epoch-quality CSV contains no finetune rows.")
    return finetune_epochs[0]


def write_plot(
    path: Path,
    rows: List[Dict[str, float]],
    finetune_start: int,
    title: str,
    *,
    fixed_scales: bool = True,
    title_suffix: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (int(row["epoch"]), 0 if row["phase"] == "pretrain" else 1))
    epochs = np.array([int(row["epoch"]) for row in rows], dtype=int)
    style_map = {
        "aleatoric": {"color": "tab:blue", "label": "AU"},
        "epistemic": {"color": "tab:orange", "label": "EU"},
        "total": {"color": "tab:green", "label": "Total"},
    }
    metric_specs = [
        ("spearman", "Spearman ↑"),
        ("ause", "AUSE ↓"),
        ("ence", "ENCE ↓"),
        ("magnitude", "|Uncertainty|"),
    ]

    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14
    suptitle_fontsize = 20

    def plot_series_with_clipped_markers(
        ax,
        x_values: np.ndarray,
        y_values: np.ndarray,
        *,
        color: str,
        label: str,
        y_limits: tuple[float, float],
        log_scale: bool = False,
    ) -> None:
        values = np.array(y_values, dtype=float)
        finite_mask = np.isfinite(values)
        low, high = y_limits
        low_mask = finite_mask & (values < low)
        high_mask = finite_mask & (values > high)
        in_range_mask = finite_mask & ~(low_mask | high_mask)

        clipped = values.copy()
        clipped[low_mask] = low
        clipped[high_mask] = high

        if np.any(finite_mask):
            ax.plot(
                x_values[finite_mask],
                clipped[finite_mask],
                color=color,
                linewidth=2.3,
                label=label,
            )
        if np.any(in_range_mask):
            ax.plot(
                x_values[in_range_mask],
                clipped[in_range_mask],
                linestyle="None",
                marker="o",
                color=color,
                markersize=6.0,
            )
        out_of_range_mask = low_mask | high_mask
        if np.any(out_of_range_mask):
            ax.plot(
                x_values[out_of_range_mask],
                clipped[out_of_range_mask],
                linestyle="None",
                marker="*",
                color=color,
                markersize=10.0,
            )

    fig, axes = plt.subplots(6, 1, figsize=(11, 20), sharex=True)
    for ax, (metric_prefix, panel_title) in zip(axes[:4], metric_specs):
        plotted_uncertainties = (
            ["aleatoric", "epistemic"]
            if metric_prefix == "magnitude"
            else list(style_map.keys())
        )
        y_limits = {
            "spearman": (0.0, 1.0),
            "ause": (0.0, 0.5),
            "ence": (0.0, 2.0),
            "magnitude": (1e-6, 1e2),
        }[metric_prefix]
        for unc_name in plotted_uncertainties:
            style = style_map[unc_name]
            values = np.array([row[f"{metric_prefix}_{unc_name}"] for row in rows], dtype=float)
            if metric_prefix == "magnitude":
                values = np.where(values > 0.0, values, np.nan)
            if fixed_scales:
                plot_series_with_clipped_markers(
                    ax,
                    epochs,
                    values,
                    color=style["color"],
                    label=style["label"],
                    y_limits=y_limits,
                    log_scale=(metric_prefix == "magnitude"),
                )
            else:
                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    color=style["color"],
                    label=style["label"],
                    linewidth=2.3,
                    markersize=6.0,
                )
        ax.axvline(finetune_start, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(panel_title, fontsize=title_fontsize)
        ax.set_ylabel(panel_title, fontsize=label_fontsize)
        if fixed_scales:
            if metric_prefix == "spearman":
                ax.set_ylim(0.0, 1.0)
            elif metric_prefix == "ause":
                ax.set_ylim(0.0, 0.5)
            elif metric_prefix == "ence":
                ax.set_ylim(0.0, 2.0)
        if metric_prefix == "magnitude":
            ax.set_yscale("log", base=10)
            if fixed_scales:
                ax.set_ylim(1e-6, 1e2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=legend_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    rmse_values = np.array([row["rmse_e_atom"] for row in rows], dtype=float)
    rmse_values = np.where(rmse_values > 0.0, rmse_values, np.nan)
    if fixed_scales:
        plot_series_with_clipped_markers(
            axes[4],
            epochs,
            rmse_values,
            color="black",
            label="RMSE",
            y_limits=(1e-6, 1e2),
            log_scale=True,
        )
    else:
        axes[4].plot(epochs, rmse_values, marker="o", color="black", label="RMSE", linewidth=2.3, markersize=6.0)
    axes[4].axvline(finetune_start, color="black", linestyle="--", linewidth=1.2)
    axes[4].set_title("RMSE_E_per_atom ↓", fontsize=title_fontsize)
    axes[4].set_ylabel("RMSE", fontsize=label_fontsize)
    axes[4].set_yscale("log", base=10)
    if fixed_scales:
        axes[4].set_ylim(1e-6, 1e2)
    axes[4].grid(alpha=0.3)
    axes[4].legend(fontsize=legend_fontsize)
    axes[4].tick_params(axis="both", labelsize=tick_fontsize)

    nll_values = np.array([row["nll_energy"] for row in rows], dtype=float)
    if fixed_scales:
        plot_series_with_clipped_markers(
            axes[5],
            epochs,
            nll_values,
            color="black",
            label="NLL",
            y_limits=(-0.5, 0.5),
        )
    else:
        axes[5].plot(epochs, nll_values, marker="o", color="black", label="NLL", linewidth=2.3, markersize=6.0)
    axes[5].axvline(finetune_start, color="black", linestyle="--", linewidth=1.2)
    axes[5].set_title("NLL ↓", fontsize=title_fontsize)
    axes[5].set_ylabel("NLL", fontsize=label_fontsize)
    axes[5].set_xlabel("Epoch", fontsize=label_fontsize)
    if fixed_scales:
        axes[5].set_ylim(-0.5, 0.5)
    axes[5].grid(alpha=0.3)
    axes[5].legend(fontsize=legend_fontsize)
    axes[5].tick_params(axis="both", labelsize=tick_fontsize)

    fig.suptitle(f"{title}{title_suffix}", fontsize=suptitle_fontsize)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    def compute_combined_rows() -> List[Dict[str, float]]:
        pretrain_rows = evaluate_stage(
            checkpoints_dir=Path(args.pretrain_checkpoints_dir),
            experiment_name=args.pretrain_experiment_name,
            split_path=Path(args.split_path),
            energy_key=args.energy_key,
            device=device,
            batch_size=args.batch_size,
            per_atom=args.per_atom,
            head_name=args.head,
            every_n_epochs=args.every_n_epochs_pretrain,
            trim=args.trim,
            num_bins=args.num_bins,
        )
        finetune_rows = evaluate_stage(
            checkpoints_dir=Path(args.finetune_checkpoints_dir),
            experiment_name=args.finetune_experiment_name,
            split_path=Path(args.split_path),
            energy_key=args.energy_key,
            device=device,
            batch_size=args.batch_size,
            per_atom=args.per_atom,
            head_name=args.head,
            every_n_epochs=args.every_n_epochs_finetune,
            trim=args.trim,
            num_bins=args.num_bins,
        )
        combined_rows, _ = build_combined_rows(pretrain_rows, finetune_rows)
        return combined_rows

    cached_rows = load_cached_csv_rows(
        Path(args.output_csv),
        required_fields=[
            "phase",
            "source_epoch",
            "epoch",
            "pearson_aleatoric",
            "pearson_epistemic",
            "pearson_total",
            "spearman_aleatoric",
            "spearman_epistemic",
            "spearman_total",
            "ause_aleatoric",
            "ause_epistemic",
            "ause_total",
            "ence_aleatoric",
            "ence_epistemic",
            "ence_total",
            "magnitude_aleatoric",
            "magnitude_epistemic",
            "magnitude_total",
            "rmse_e_atom",
            "nll_energy",
        ],
        field_parsers={
            "phase": parse_str,
            "source_epoch": parse_int,
            "epoch": parse_int,
            "pearson_aleatoric": parse_float,
            "pearson_epistemic": parse_float,
            "pearson_total": parse_float,
            "spearman_aleatoric": parse_float,
            "spearman_epistemic": parse_float,
            "spearman_total": parse_float,
            "ause_aleatoric": parse_float,
            "ause_epistemic": parse_float,
            "ause_total": parse_float,
            "ence_aleatoric": parse_float,
            "ence_epistemic": parse_float,
            "ence_total": parse_float,
            "magnitude_aleatoric": parse_float,
            "magnitude_epistemic": parse_float,
            "magnitude_total": parse_float,
            "rmse_e_atom": parse_float,
            "nll_energy": parse_float,
        },
        key_fields=["phase", "source_epoch", "epoch"],
        compute_missing_rows=compute_combined_rows,
        label="epoch_quality_finetune_same",
    )
    if cached_rows is not None:
        output_plot = Path(args.output_plot)
        write_plot(
            output_plot,
            cached_rows,
            finetune_start=infer_finetune_start(cached_rows),
            title=args.title,
        )
        if args.free_scale:
            write_plot(
                with_free_scale_suffix(output_plot),
                cached_rows,
                finetune_start=infer_finetune_start(cached_rows),
                title=args.title,
                fixed_scales=False,
                title_suffix=" - free-scale",
            )
        return

    combined_rows = compute_combined_rows()
    finetune_start = infer_finetune_start(combined_rows)
    write_csv(Path(args.output_csv), combined_rows)
    output_plot = Path(args.output_plot)
    write_plot(
        output_plot,
        combined_rows,
        finetune_start=finetune_start,
        title=args.title,
    )
    if args.free_scale:
        write_plot(
            with_free_scale_suffix(output_plot),
            combined_rows,
            finetune_start=finetune_start,
            title=args.title,
            fixed_scales=False,
            title_suffix=" - free-scale",
        )


if __name__ == "__main__":
    main()
