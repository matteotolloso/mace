#!/usr/bin/env python3
"""Concatenate pretrain and fine-tune epoch-quality curves across two datasets.

The script evaluates:
- a pretrain ensemble on a pretrain dataset
- a fine-tuned ensemble on a fine-tune dataset

It then concatenates the two timelines and draws a vertical line at the
fine-tuning start. This is useful for visual comparisons, but the two stages
should be interpreted carefully because they are computed on different targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int, parse_str
from epoch_quality_finetune_same_dataset import (
    build_combined_rows,
    evaluate_stage,
    infer_finetune_start,
    read_csv,
    with_free_scale_suffix,
    write_csv,
    write_plot,
)
from epoch_quality import setup_logging
from mace.tools import torch_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrain-checkpoints-dir", type=str, required=True)
    parser.add_argument("--pretrain-experiment-name", type=str, required=True)
    parser.add_argument("--pretrain-split-path", type=str, required=True)
    parser.add_argument("--pretrain-energy-key", type=str, required=True)
    parser.add_argument("--finetune-checkpoints-dir", type=str, required=True)
    parser.add_argument("--finetune-experiment-name", type=str, required=True)
    parser.add_argument("--finetune-split-path", type=str, required=True)
    parser.add_argument("--finetune-energy-key", type=str, required=True)
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
    parser.add_argument("--every-n-epochs-pretrain", type=int, default=5)
    parser.add_argument("--every-n-epochs-finetune", type=int, default=5)
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
    parser.add_argument("--output-csv", type=str, default="epoch_quality_finetune_mixed.csv")
    parser.add_argument("--output-plot", type=str, default="epoch_quality_finetune_mixed.png")
    parser.add_argument(
        "--title",
        type=str,
        default="Epoch Quality: Pretrain (DFT) + Finetune (CC)",
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


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    def compute_combined_rows():
        pretrain_rows = evaluate_stage(
            checkpoints_dir=Path(args.pretrain_checkpoints_dir),
            experiment_name=args.pretrain_experiment_name,
            split_path=Path(args.pretrain_split_path),
            energy_key=args.pretrain_energy_key,
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
            split_path=Path(args.finetune_split_path),
            energy_key=args.finetune_energy_key,
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
        label="epoch_quality_finetune_mixed",
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
