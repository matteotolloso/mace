#!/usr/bin/env python3
"""Plot epoch-wise aleatoric uncertainty for each ensemble member.

Input files are expected to be the per-model logs produced by training with
`--log_epoch_outputs=true`, i.e. lines with:
    mode == "epoch_outputs_config"
and fields including:
    epoch, split, loader, config_index, pred_energy_var
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ConfigKey = Tuple[str, int]  # (loader, config_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to *_epoch_outputs.txt files (one file per ensemble member).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default="valid",
        help="Which split to use from the epoch output file.",
    )
    parser.add_argument(
        "--loader",
        type=str,
        default=None,
        help="Optional loader name filter. If omitted, use all loaders.",
    )
    parser.add_argument(
        "--per_atom",
        action="store_true",
        help="Normalize variance by N^2 before aggregation.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="au_members_vs_epoch.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="au_members_vs_epoch.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--plot_log_variance",
        action="store_true",
        help="Plot log(variance) values on the y-axis.",
    )
    parser.add_argument(
        "--log_eps",
        type=float,
        default=1e-30,
        help="Small epsilon added before log when --plot_log_variance is used.",
    )
    return parser.parse_args()


def load_member_file(
    path: Path, split: str, loader: Optional[str]
) -> Dict[int, Dict[ConfigKey, Dict[str, float]]]:
    data: Dict[int, Dict[ConfigKey, Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("mode") != "epoch_outputs_config":
                continue
            if row.get("split") != split:
                continue
            if loader is not None and row.get("loader") != loader:
                continue
            if "pred_energy_var" not in row:
                continue

            epoch = int(row["epoch"])
            row_loader = str(row.get("loader", ""))
            config_index = int(row["config_index"])
            key: ConfigKey = (row_loader, config_index)

            if epoch not in data:
                data[epoch] = {}
            data[epoch][key] = {
                "pred_energy_var": float(row["pred_energy_var"]),
                "num_atoms": int(row["num_atoms"]) if "num_atoms" in row else np.nan,
                "ref_energy": float(row["ref_energy"]) if "ref_energy" in row else np.nan,
            }
    return data


def validate_alignment(
    epoch: int,
    key: ConfigKey,
    ref_energies: np.ndarray,
    num_atoms_vals: np.ndarray,
    align_tol: float = 1e-8,
) -> None:
    finite_ref = np.isfinite(ref_energies)
    if np.any(finite_ref) and np.ptp(ref_energies[finite_ref]) > align_tol:
        raise RuntimeError(
            f"Detected misaligned members at epoch={epoch}, key={key}: "
            "ref_energy differs across input files."
        )
    finite_n = np.isfinite(num_atoms_vals)
    if np.any(finite_n) and np.ptp(num_atoms_vals[finite_n]) > 0:
        raise RuntimeError(
            f"Detected misaligned members at epoch={epoch}, key={key}: "
            "num_atoms differs across input files."
        )


def top_share(values: np.ndarray, k: int) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    total = float(np.sum(finite_values))
    if total <= 0.0:
        return float("nan")
    sorted_values = np.sort(finite_values)[::-1]
    return float(np.sum(sorted_values[: min(k, len(sorted_values))]) / total)


def compute_member_epoch_au(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]],
    member_names: List[str],
    per_atom: bool,
) -> List[Dict[str, float]]:
    common_epochs = sorted(set.intersection(*(set(member.keys()) for member in members)))
    rows: List[Dict[str, float]] = []

    for epoch in common_epochs:
        common_configs = sorted(set.intersection(*(set(member[epoch].keys()) for member in members)))
        if not common_configs:
            continue

        per_member_values = [[] for _ in member_names]
        per_member_epoch_vars = [[] for _ in member_names]
        for key in common_configs:
            pred_vars = np.array(
                [member[epoch][key]["pred_energy_var"] for member in members], dtype=float
            )
            ref_energies = np.array(
                [member[epoch][key]["ref_energy"] for member in members], dtype=float
            )
            num_atoms_vals = np.array(
                [member[epoch][key]["num_atoms"] for member in members], dtype=float
            )
            validate_alignment(epoch, key, ref_energies, num_atoms_vals)

            if per_atom:
                n_atoms = float(num_atoms_vals[0]) if np.isfinite(num_atoms_vals[0]) else np.nan
                if (not np.isfinite(n_atoms)) or n_atoms <= 0:
                    raise RuntimeError(
                        "Missing/invalid 'num_atoms' in epoch outputs. "
                        "Per-atom normalization requires num_atoms."
                    )
                pred_vars = pred_vars / (n_atoms**2)

            for member_idx, pred_var in enumerate(pred_vars):
                if np.isfinite(pred_var):
                    per_member_values[member_idx].append(float(pred_var))
                    per_member_epoch_vars[member_idx].append(float(pred_var))

        row: Dict[str, float] = {"epoch": epoch}
        for member_name, member_vals, member_epoch_vars in zip(
            member_names, per_member_values, per_member_epoch_vars
        ):
            row[member_name] = (
                float(np.mean(member_vals)) if len(member_vals) > 0 else float("nan")
            )
            member_epoch_vars_np = np.array(member_epoch_vars, dtype=float)
            row[f"{member_name}_top1_share"] = top_share(member_epoch_vars_np, 1)
            row[f"{member_name}_top5_share"] = top_share(member_epoch_vars_np, 5)
            row[f"{member_name}_top20_share"] = top_share(member_epoch_vars_np, 20)
        rows.append(row)

    return rows


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_plot(
    path: Path,
    rows: List[Dict[str, str]],
    member_names: List[str],
    plot_log_variance: bool,
    log_eps: float,
) -> None:
    if not rows:
        raise RuntimeError("Cannot create AU-members plot from empty CSV.")

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.array([int(row["epoch"]) for row in rows], dtype=int)

    fig, axes = plt.subplots(
        len(member_names) + 1,
        1,
        figsize=(10, 4 * (len(member_names) + 1)),
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    for member_name in member_names:
        values = np.array([float(row[member_name]) for row in rows], dtype=float)
        if plot_log_variance:
            values = np.log(np.maximum(values, log_eps))
        ax.plot(epochs, values, marker="o", label=member_name)

    ax.set_xlabel("Epoch")
    if plot_log_variance:
        ax.set_ylabel("log(AU eV^2/atom^2)")
        ax.set_title("Aleatoric Uncertainty per Epoch by Ensemble Member (log scale)")
    else:
        ax.set_ylabel("AU (eV^2/atom^2)")
        ax.set_title("Aleatoric Uncertainty per Epoch by Ensemble Member")
    ax.grid(alpha=0.3)
    ax.legend()

    for plot_idx, member_name in enumerate(member_names, start=1):
        member_ax = axes[plot_idx]
        top1 = np.array(
            [float(row[f"{member_name}_top1_share"]) for row in rows], dtype=float
        )
        top5 = np.array(
            [float(row[f"{member_name}_top5_share"]) for row in rows], dtype=float
        )
        top20 = np.array(
            [float(row[f"{member_name}_top20_share"]) for row in rows], dtype=float
        )
        member_ax.plot(epochs, top1, marker="o", label="top1")
        member_ax.plot(epochs, top5, marker="o", label="top5")
        member_ax.plot(epochs, top20, marker="o", label="top20")
        member_ax.set_ylabel("Fraction of AU")
        member_ax.set_title(f"{member_name}: AU concentration")
        member_ax.set_ylim(0.0, 1.0)
        member_ax.grid(alpha=0.3)
        member_ax.legend()

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    member_names = [path.stem for path in input_paths]
    members = [
        load_member_file(path, split=args.split, loader=args.loader)
        for path in input_paths
    ]
    if any(len(member) == 0 for member in members):
        raise RuntimeError(
            "At least one input file has no matching 'epoch_outputs_config' rows "
            f"for split='{args.split}' and loader='{args.loader}'."
        )

    rows = compute_member_epoch_au(
        members=members,
        member_names=member_names,
        per_atom=args.per_atom,
    )
    if not rows:
        raise RuntimeError("No common epochs/configurations found across ensemble files.")

    fieldnames = ["epoch"]
    for member_name in member_names:
        fieldnames.append(member_name)
        fieldnames.append(f"{member_name}_top1_share")
        fieldnames.append(f"{member_name}_top5_share")
        fieldnames.append(f"{member_name}_top20_share")

    write_csv(Path(args.output_csv), rows, fieldnames)
    plot_rows = read_csv(Path(args.output_csv))
    write_plot(
        Path(args.output_plot),
        plot_rows,
        member_names=member_names,
        plot_log_variance=args.plot_log_variance,
        log_eps=args.log_eps,
    )
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
