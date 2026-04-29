#!/usr/bin/env python3
"""Summarize uncertainty growth across within-system energy quartiles.

This script is intended for energy-OOD experiments. It:
1. selects the best checkpoint independently for each ensemble member using the
   same logic as ``utils/reliability.py``
2. evaluates the resulting ensemble on the existing ``test_id`` and
   ``test_ood`` splits
3. recovers the system identity and within-system energy metadata from the XYZs
4. computes:
   - one ID summary over the full ID split
   - ten OOD summaries over within-system energy deciles
5. plots only AU and EU in a single grouped-bar chart

The OOD deciles are system-balanced:
- first average over configurations within each system and quartile
- then average those means across systems
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
from pathlib import Path
from typing import Dict, List

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int, parse_str
from reliability import (
    build_dataloader,
    evaluate_split,
    load_models,
    select_best_checkpoints,
    setup_logging,
)
from mace.tools import torch_tools


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--test-id-split", type=str, required=True)
    parser.add_argument("--test-ood-split", type=str, required=True)
    parser.add_argument("--energy-key-test", type=str, required=True)
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
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--output-csv-raw", type=str, default="energy_ood_raw.csv")
    parser.add_argument("--output-csv-bins", type=str, default="energy_ood_summary.csv")
    parser.add_argument("--output-plot", type=str, default="energy_ood.png")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def release_models(models: List[torch.nn.Module], device: torch.device) -> None:
    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_split_metadata(split_path: Path, split_label: str) -> List[Dict[str, object]]:
    atoms_list = ase.io.read(str(split_path), index=":")
    metadata: List[Dict[str, object]] = []
    for local_index, atoms in enumerate(atoms_list):
        system = atoms.info.get("system", atoms.info.get("compound", f"system_{local_index}"))
        metadata.append(
            {
                "split": split_label,
                "config_index": local_index,
                "system": str(system),
                "compound": str(atoms.info.get("compound", system)),
                "conf_idx": int(atoms.info.get("conf_idx", local_index)),
                "relative_energy_eV": float(atoms.info.get("relative_energy_eV", np.nan)),
                "energy_quantile": float(atoms.info.get("energy_quantile", np.nan)),
            }
        )
    return metadata


def build_raw_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    device = torch_tools.init_device(args.device)
    checkpoint_paths = select_best_checkpoints(
        checkpoints_dir=Path(args.checkpoints_dir),
        results_dir=Path(args.results_dir),
        experiment_name=args.experiment_name,
        selection_key=args.selection_key,
        selection_mode=args.selection_mode,
        head_name=args.head,
    )
    models = load_models(checkpoint_paths, device=device)
    try:
        combined_rows: List[Dict[str, object]] = []
        for split_label, split_path in [
            ("id", Path(args.test_id_split)),
            ("ood", Path(args.test_ood_split)),
        ]:
            metadata_rows = read_split_metadata(split_path, split_label)
            metadata_map = {
                (str(row["split"]), int(row["config_index"])): row for row in metadata_rows
            }
            loader, _ = build_dataloader(
                split_path,
                args.energy_key_test,
                models[0],
                batch_size=args.batch_size,
                head_name=args.head,
            )
            raw_rows = evaluate_split(
                models=models,
                data_loader=loader,
                device=device,
                per_atom=True,
                head_name=args.head,
                split_name=f"{args.experiment_name} {split_label}",
                log_every_batches=args.log_every_batches,
            )
            for row in raw_rows:
                metadata = metadata_map[(split_label, int(row["config_index"]))]
                combined_rows.append(
                    {
                        "split": split_label,
                        "config_index": int(row["config_index"]),
                        "system": metadata["system"],
                        "compound": metadata["compound"],
                        "conf_idx": metadata["conf_idx"],
                        "ref_energy": float(row["ref_energy"]),
                        "pred_energy": float(row["pred_energy"]),
                        "relative_energy_eV": metadata["relative_energy_eV"],
                        "energy_quantile": metadata["energy_quantile"],
                        "aleatoric_var": float(row["aleatoric_var"]),
                        "epistemic_var": float(row["epistemic_var"]),
                    }
                )
        return combined_rows
    finally:
        release_models(models, device)


def _system_balanced_mean(rows: List[Dict[str, object]], key: str) -> tuple[float, int]:
    by_system: Dict[str, List[float]] = {}
    for row in rows:
        value = float(row[key])
        if not np.isfinite(value):
            continue
        by_system.setdefault(str(row["system"]), []).append(value)
    if not by_system:
        return float("nan"), 0
    per_system_means = np.array([np.mean(values) for values in by_system.values()], dtype=float)
    return float(np.mean(per_system_means)), len(by_system)


def build_summary_rows(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    valid_rows = [
        row
        for row in raw_rows
        if np.isfinite(float(row["aleatoric_var"]))
        and np.isfinite(float(row["epistemic_var"]))
    ]
    if not valid_rows:
        raise RuntimeError("No finite rows available to build the energy-OOD plot.")

    rows_out: List[Dict[str, object]] = []

    id_rows = [row for row in valid_rows if str(row["split"]) == "id"]
    if id_rows:
        au_mean, n_systems = _system_balanced_mean(id_rows, "aleatoric_var")
        eu_mean, _ = _system_balanced_mean(id_rows, "epistemic_var")
        rows_out.append(
            {
                "group": "id",
                "x_index": 0,
                "label": "ID mean",
                "split": "id",
                "quantile_lo": 0.0,
                "quantile_hi": 1.0,
                "num_systems": n_systems,
                "num_configs": len(id_rows),
                "au_mean": au_mean,
                "eu_mean": eu_mean,
            }
        )

    quantile_edges = np.linspace(0.0, 1.0, 11)
    ood_rows = [
        row
        for row in valid_rows
        if str(row["split"]) == "ood" and np.isfinite(float(row["energy_quantile"]))
    ]
    next_x_index = 1
    for group_idx, (q_lo, q_hi) in enumerate(
        zip(quantile_edges[:-1], quantile_edges[1:]),
        start=1,
    ):
        group_name = f"q{group_idx}"
        q_hi_cmp = 1.0000001 if group_idx == 10 else q_hi
        label = f"{q_lo:.2f}-{q_hi:.2f}"
        group_rows = [
            row
            for row in ood_rows
            if q_lo <= float(row["energy_quantile"]) < q_hi_cmp
        ]
        if not group_rows:
            continue
        au_mean, n_systems = _system_balanced_mean(group_rows, "aleatoric_var")
        eu_mean, _ = _system_balanced_mean(group_rows, "epistemic_var")
        rows_out.append(
            {
                "group": group_name,
                "x_index": next_x_index,
                "label": label,
                "split": "ood",
                "quantile_lo": q_lo,
                "quantile_hi": min(q_hi, 1.0),
                "num_systems": n_systems,
                "num_configs": len(group_rows),
                "au_mean": au_mean,
                "eu_mean": eu_mean,
            }
        )
        next_x_index += 1

    return rows_out


def write_plot(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = sorted(summary_rows, key=lambda row: int(row["x_index"]))
    x = np.array([int(row["x_index"]) for row in rows], dtype=float)
    labels = [str(row["label"]) for row in rows]
    au = np.array([float(row["au_mean"]) for row in rows], dtype=float)
    eu = np.array([float(row["eu_mean"]) for row in rows], dtype=float)
    au = np.where(au > 0.0, au, np.nan)
    eu = np.where(eu > 0.0, eu, np.nan)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    width = 0.34
    ax.bar(x - width / 2, au, width=width, color="tab:blue", label="AU")
    ax.bar(x + width / 2, eu, width=width, color="tab:orange", label="EU")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Variance", fontsize=13)
    ax.set_yscale("log", base=10)
    ax.set_title("Uncertainty vs within-system energy quantile bin", fontsize=18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    torch_tools.set_default_dtype(args.default_dtype)

    raw_fields = [
        "split",
        "config_index",
        "system",
        "compound",
        "conf_idx",
        "ref_energy",
        "pred_energy",
        "relative_energy_eV",
        "energy_quantile",
        "aleatoric_var",
        "epistemic_var",
    ]
    raw_rows = load_cached_csv_rows(
        Path(args.output_csv_raw),
        required_fields=raw_fields,
        field_parsers={
            "split": parse_str,
            "config_index": parse_int,
            "system": parse_str,
            "compound": parse_str,
            "conf_idx": parse_int,
            "ref_energy": parse_float,
            "pred_energy": parse_float,
            "relative_energy_eV": parse_float,
            "energy_quantile": parse_float,
            "aleatoric_var": parse_float,
            "epistemic_var": parse_float,
        },
        key_fields=["split", "config_index"],
        compute_missing_rows=lambda: build_raw_rows(args),
        logger=LOGGER,
        label="energy_ood_raw",
    )
    if raw_rows is None:
        raw_rows = build_raw_rows(args)
        write_csv(Path(args.output_csv_raw), raw_fields, raw_rows)

    summary_fields = [
        "summary_version",
        "group",
        "x_index",
        "label",
        "split",
        "quantile_lo",
        "quantile_hi",
        "num_systems",
        "num_configs",
        "au_mean",
        "eu_mean",
    ]
    summary_rows = load_cached_csv_rows(
        Path(args.output_csv_bins),
        required_fields=summary_fields,
        field_parsers={
            "summary_version": parse_str,
            "group": parse_str,
            "x_index": parse_int,
            "label": parse_str,
            "split": parse_str,
            "quantile_lo": parse_float,
            "quantile_hi": parse_float,
            "num_systems": parse_int,
            "num_configs": parse_int,
            "au_mean": parse_float,
            "eu_mean": parse_float,
        },
        key_fields=["group"],
        compute_missing_rows=lambda: [
            {"summary_version": "occupied_ood_bins_v2", **row}
            for row in build_summary_rows(raw_rows)
        ],
        logger=LOGGER,
        label="energy_ood_summary",
    )
    if summary_rows is None:
        summary_rows = [
            {"summary_version": "occupied_ood_bins_v2", **row}
            for row in build_summary_rows(raw_rows)
        ]
        write_csv(Path(args.output_csv_bins), summary_fields, summary_rows)

    write_plot(Path(args.output_plot), summary_rows)


if __name__ == "__main__":
    main()
