#!/usr/bin/env python3
"""Plot epoch-wise aleatoric uncertainty for each ensemble member.

Input files are expected to be the per-model logs produced by training with
`--log_epoch_outputs=true`, i.e. lines with:
    mode == "epoch_outputs_config"
and fields including:
    epoch, split, loader, config_index, var_e_per_atom_2

Legacy files with `pred_energy_var` are also supported.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int

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
        dest="per_atom",
        action="store_true",
        help="Use per-atom variance (default).",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Convert per-atom variance back to total-system variance.",
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
        "--trim",
        type=float,
        default=0.0,
        help="Symmetric fraction to trim from the lowest and highest AU values before computing per-epoch summaries. Example: 0.005 trims 0.5%% on each side.",
    )
    parser.add_argument(
        "--log_eps",
        type=float,
        default=1e-30,
        help="Small epsilon added before log when --plot_log_variance is used.",
    )
    parser.set_defaults(per_atom=True)
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
            if "var_e_per_atom_2" not in row and "pred_energy_var" not in row:
                continue

            epoch = int(row["epoch"])
            row_loader = str(row.get("loader", ""))
            config_index = int(row["config_index"])
            key: ConfigKey = (row_loader, config_index)

            if epoch not in data:
                data[epoch] = {}
            data[epoch][key] = {
                "var_e_per_atom_2": float(row["var_e_per_atom_2"])
                if "var_e_per_atom_2" in row
                else np.nan,
                "pred_energy_var_legacy_total": float(row["pred_energy_var"])
                if "pred_energy_var" in row
                else np.nan,
                "num_atoms": int(row["num_atoms"]) if "num_atoms" in row else np.nan,
                "ref_energy": float(row["ref_energy"]) if "ref_energy" in row else np.nan,
            }
    return data


def resolve_logged_variance(row: Dict[str, float], per_atom: bool) -> float:
    per_atom_var = float(row.get("var_e_per_atom_2", np.nan))
    legacy_total_var = float(row.get("pred_energy_var_legacy_total", np.nan))
    num_atoms = float(row.get("num_atoms", np.nan))

    if np.isfinite(per_atom_var):
        if per_atom:
            return per_atom_var
        if (not np.isfinite(num_atoms)) or num_atoms <= 0:
            raise RuntimeError(
                "Missing/invalid 'num_atoms' in epoch outputs. Total-variance mode "
                "requires num_atoms when using per-atom logged variance."
            )
        return per_atom_var * (num_atoms**2)

    if np.isfinite(legacy_total_var):
        if not per_atom:
            return legacy_total_var
        if (not np.isfinite(num_atoms)) or num_atoms <= 0:
            raise RuntimeError(
                "Missing/invalid 'num_atoms' in epoch outputs. Per-atom "
                "normalization requires num_atoms."
            )
        return legacy_total_var / (num_atoms**2)

    return float("nan")


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


def trim_values(values: np.ndarray, trim: float) -> np.ndarray:
    if trim <= 0.0:
        return values
    if trim >= 0.5:
        raise ValueError("--trim must be < 0.5")
    finite_values = values[np.isfinite(values)]
    n = len(finite_values)
    count_each_side = int(np.floor(trim * n))
    if n == 0 or count_each_side == 0 or (2 * count_each_side) >= n:
        return finite_values
    return np.sort(finite_values)[count_each_side : n - count_each_side]


def compute_member_epoch_au(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]],
    member_names: List[str],
    per_atom: bool,
    trim: float,
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
                [resolve_logged_variance(member[epoch][key], per_atom) for member in members],
                dtype=float,
            )
            ref_energies = np.array(
                [member[epoch][key]["ref_energy"] for member in members], dtype=float
            )
            num_atoms_vals = np.array(
                [member[epoch][key]["num_atoms"] for member in members], dtype=float
            )
            validate_alignment(epoch, key, ref_energies, num_atoms_vals)

            for member_idx, pred_var in enumerate(pred_vars):
                if np.isfinite(pred_var):
                    per_member_values[member_idx].append(float(pred_var))
                    per_member_epoch_vars[member_idx].append(float(pred_var))

        row: Dict[str, float] = {"epoch": epoch}
        for member_name, member_vals, member_epoch_vars in zip(
            member_names, per_member_values, per_member_epoch_vars
        ):
            row[member_name] = (
                float(np.mean(trim_values(np.array(member_vals, dtype=float), trim)))
                if len(member_vals) > 0
                else float("nan")
            )
            member_epoch_vars_np = trim_values(np.array(member_epoch_vars, dtype=float), trim)
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

    fieldnames = ["epoch"]
    for member_name in member_names:
        fieldnames.append(member_name)
        fieldnames.append(f"{member_name}_top1_share")
        fieldnames.append(f"{member_name}_top5_share")
        fieldnames.append(f"{member_name}_top20_share")

    output_csv_path = Path(args.output_csv)

    def compute_rows() -> List[Dict[str, float]]:
        computed_rows = compute_member_epoch_au(
            members=members,
            member_names=member_names,
            per_atom=args.per_atom,
            trim=args.trim,
        )
        if not computed_rows:
            raise RuntimeError("No common epochs/configurations found across ensemble files.")
        return computed_rows

    field_parsers = {"epoch": parse_int}
    for member_name in member_names:
        field_parsers[member_name] = parse_float
        field_parsers[f"{member_name}_top1_share"] = parse_float
        field_parsers[f"{member_name}_top5_share"] = parse_float
        field_parsers[f"{member_name}_top20_share"] = parse_float

    plot_rows = load_cached_csv_rows(
        output_csv_path,
        required_fields=fieldnames,
        field_parsers=field_parsers,
        key_fields=["epoch"],
        compute_missing_rows=compute_rows,
        label="epoch_au_members_configs",
    )
    if plot_rows is None:
        rows = compute_rows()
        write_csv(output_csv_path, rows, fieldnames)
        plot_rows = read_csv(output_csv_path)

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
