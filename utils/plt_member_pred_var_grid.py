#!/usr/bin/env python3
"""Plot a 6x6 grid of member prediction/variance scatters for selected systems.

The script:
1. loads per-configuration rows from ensemble ``*_epoch_outputs.txt`` files
2. selects the best common epoch on the validation split using ensemble RMSE_E
   (or RMSE_E_per_atom in per-atom mode)
3. evaluates one target split at that epoch
4. ranks systems by total uncertainty = aleatoric + epistemic
5. plots:
   - first two rows: 12 systems with lowest total uncertainty
   - middle two rows: 12 systems around the median total uncertainty
   - last two rows: 12 systems with highest total uncertainty

Each panel shows one system. The x-axis is the member energy prediction and the
y-axis is the corresponding member predicted variance.

New epoch-output logs store per-atom variance as `var_e_per_atom_2`. Legacy
logs with `pred_energy_var` are also supported.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int, parse_str

ConfigKey = Tuple[str, int]  # (loader, config_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to *_epoch_outputs.txt files, one per ensemble member.",
    )
    parser.add_argument(
        "--selection_split",
        type=str,
        choices=["train", "valid", "test"],
        default="valid",
        help="Split used to select the best common epoch.",
    )
    parser.add_argument(
        "--plot_split",
        type=str,
        choices=["train", "valid", "test"],
        default="test",
        help="Split used to build the 6x6 panel plot.",
    )
    parser.add_argument(
        "--selection_loader",
        type=str,
        default=None,
        help="Optional loader filter for best-epoch selection.",
    )
    parser.add_argument(
        "--plot_loader",
        type=str,
        default=None,
        help="Optional loader filter for plotting.",
    )
    parser.add_argument(
        "--per_atom",
        dest="per_atom",
        action="store_true",
        help="Use per-atom energies and per-atom variances (default).",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system energies and variances.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="member_pred_var_grid.csv",
        help="Output CSV path containing the plotted panel data.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="member_pred_var_grid.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.0,
        help="Symmetric fraction to trim from the lowest and highest total uncertainty systems before selecting panels. Example: 0.005 trims 0.5%% on each side.",
    )
    parser.set_defaults(per_atom=True)
    return parser.parse_args()


def load_member_file(path: Path) -> Dict[str, Dict[int, Dict[ConfigKey, Dict[str, float]]]]:
    data: Dict[str, Dict[int, Dict[ConfigKey, Dict[str, float]]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("mode") != "epoch_outputs_config":
                continue
            if "pred_energy" not in row:
                continue

            split = str(row["split"])
            epoch = int(row["epoch"])
            loader = str(row.get("loader", ""))
            config_index = int(row["config_index"])
            key: ConfigKey = (loader, config_index)

            if split not in data:
                data[split] = {}
            if epoch not in data[split]:
                data[split][epoch] = {}
            data[split][epoch][key] = {
                "pred_energy": float(row["pred_energy"]),
                "var_e_per_atom_2": float(row["var_e_per_atom_2"])
                if "var_e_per_atom_2" in row
                else np.nan,
                "pred_energy_var_legacy_total": float(row["pred_energy_var"])
                if "pred_energy_var" in row
                else np.nan,
                "ref_energy": float(row["ref_energy"]) if "ref_energy" in row else np.nan,
                "num_atoms": int(row["num_atoms"]) if "num_atoms" in row else np.nan,
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
                "Missing/invalid num_atoms in epoch outputs. Total variance mode "
                "requires num_atoms when using per-atom logged variance."
            )
        return per_atom_var * (num_atoms**2)

    if np.isfinite(legacy_total_var):
        if not per_atom:
            return legacy_total_var
        if (not np.isfinite(num_atoms)) or num_atoms <= 0:
            raise RuntimeError(
                "Missing/invalid num_atoms in epoch outputs. Per-atom mode requires num_atoms."
            )
        return legacy_total_var / (num_atoms**2)

    return float("nan")


def filter_loader(
    member_data: Dict[int, Dict[ConfigKey, Dict[str, float]]],
    loader: Optional[str],
) -> Dict[int, Dict[ConfigKey, Dict[str, float]]]:
    if loader is None:
        return member_data
    filtered: Dict[int, Dict[ConfigKey, Dict[str, float]]] = {}
    for epoch, config_map in member_data.items():
        selected = {
            key: value for key, value in config_map.items() if key[0] == loader
        }
        if selected:
            filtered[epoch] = selected
    return filtered


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
            f"Misaligned ensemble at epoch={epoch}, key={key}: ref_energy differs."
        )
    finite_n = np.isfinite(num_atoms_vals)
    if np.any(finite_n) and np.ptp(num_atoms_vals[finite_n]) > 0:
        raise RuntimeError(
            f"Misaligned ensemble at epoch={epoch}, key={key}: num_atoms differs."
        )


def common_epoch_keys(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]]
) -> List[int]:
    if not members:
        return []
    return sorted(set.intersection(*(set(member.keys()) for member in members)))


def compute_epoch_rmse_e(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]],
    epoch: int,
    per_atom: bool,
) -> float:
    common_configs = sorted(set.intersection(*(set(member[epoch].keys()) for member in members)))
    if not common_configs:
        raise RuntimeError(f"No common configurations found at epoch {epoch}.")

    sq_errors: List[float] = []
    for key in common_configs:
        pred_energies = np.array(
            [member[epoch][key]["pred_energy"] for member in members], dtype=float
        )
        ref_energies = np.array(
            [member[epoch][key]["ref_energy"] for member in members], dtype=float
        )
        num_atoms_vals = np.array(
            [member[epoch][key]["num_atoms"] for member in members], dtype=float
        )
        validate_alignment(epoch, key, ref_energies, num_atoms_vals)
        ref_energy = float(ref_energies[0])
        ensemble_pred_energy = float(np.mean(pred_energies))
        if per_atom:
            num_atoms = float(num_atoms_vals[0]) if np.isfinite(num_atoms_vals[0]) else np.nan
            if (not np.isfinite(num_atoms)) or num_atoms <= 0:
                raise RuntimeError(
                    "Missing/invalid num_atoms in epoch outputs. Per-atom mode requires num_atoms."
                )
            ref_energy /= num_atoms
            ensemble_pred_energy /= num_atoms
        sq_errors.append((ensemble_pred_energy - ref_energy) ** 2)

    return float(np.sqrt(np.mean(sq_errors)))


def select_best_epoch(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]], per_atom: bool
) -> int:
    epochs = common_epoch_keys(members)
    if not epochs:
        raise RuntimeError("No common epochs found across ensemble members.")

    best_epoch = epochs[0]
    best_rmse = compute_epoch_rmse_e(members, best_epoch, per_atom)
    for epoch in epochs[1:]:
        rmse = compute_epoch_rmse_e(members, epoch, per_atom)
        if rmse < best_rmse:
            best_epoch = epoch
            best_rmse = rmse
    return best_epoch


def build_system_rows(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]],
    member_names: List[str],
    epoch: int,
    per_atom: bool,
) -> List[Dict[str, object]]:
    common_configs = sorted(set.intersection(*(set(member[epoch].keys()) for member in members)))
    if not common_configs:
        raise RuntimeError(f"No common configurations found at epoch {epoch}.")

    rows: List[Dict[str, object]] = []
    for key in common_configs:
        loader, config_index = key
        pred_energies = np.array(
            [member[epoch][key]["pred_energy"] for member in members], dtype=float
        )
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

        ref_energy = float(ref_energies[0])
        num_atoms = float(num_atoms_vals[0]) if np.isfinite(num_atoms_vals[0]) else np.nan
        if per_atom:
            if (not np.isfinite(num_atoms)) or num_atoms <= 0:
                raise RuntimeError(
                    "Missing/invalid num_atoms in epoch outputs. Per-atom mode requires it."
                )
            pred_energies = pred_energies / num_atoms
            ref_energy = ref_energy / num_atoms
        aleatoric_var = (
            float(np.nanmean(pred_vars)) if np.any(np.isfinite(pred_vars)) else float("nan")
        )
        epistemic_var = float(np.var(pred_energies, ddof=0)) if len(pred_energies) > 1 else 0.0
        total_var = aleatoric_var + epistemic_var if np.isfinite(aleatoric_var) else float("nan")

        rows.append(
            {
                "epoch": epoch,
                "loader": loader,
                "config_index": int(config_index),
                "num_atoms": int(num_atoms) if np.isfinite(num_atoms) else "",
                "ref_energy": ref_energy,
                "member_names": list(member_names),
                "member_pred_energy": pred_energies.tolist(),
                "member_pred_var": pred_vars.tolist(),
                "aleatoric_var": aleatoric_var,
                "epistemic_var": epistemic_var,
                "total_var": total_var,
            }
        )
    return rows


def select_panel_rows(system_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not system_rows:
        raise RuntimeError("Cannot select panels from empty system rows.")

    valid_rows = [row for row in system_rows if np.isfinite(row["total_var"])]
    if not valid_rows:
        raise RuntimeError("No systems with finite total uncertainty were found.")

    ordered = sorted(valid_rows, key=lambda row: row["total_var"])
    n = len(ordered)

    low_rows = ordered[: min(12, n)]
    high_rows = ordered[max(0, n - 12) :]

    remaining = ordered[len(low_rows) : max(0, n - len(high_rows))]
    if len(remaining) <= 12:
        mid_rows = remaining
    else:
        mid_start = max(0, (len(remaining) // 2) - 6)
        mid_rows = remaining[mid_start : mid_start + 12]

    selected = low_rows + mid_rows + high_rows
    return selected[:36]


def trim_system_rows(
    system_rows: List[Dict[str, object]],
    trim: float,
) -> List[Dict[str, object]]:
    if trim <= 0.0:
        return system_rows
    if trim >= 0.5:
        raise ValueError("--trim must be < 0.5")

    finite_rows = [
        (idx, float(row["total_var"]))
        for idx, row in enumerate(system_rows)
        if np.isfinite(row["total_var"])
    ]
    n = len(finite_rows)
    count_each_side = int(np.floor(trim * n))
    if n == 0 or count_each_side == 0 or (2 * count_each_side) >= n:
        return system_rows

    ordered = sorted(finite_rows, key=lambda item: item[1])
    keep_idx = {idx for idx, _ in ordered[count_each_side : n - count_each_side]}
    return [row for idx, row in enumerate(system_rows) if idx in keep_idx]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "panel_index",
        "group",
        "epoch",
        "loader",
        "config_index",
        "num_atoms",
        "ref_energy",
        "member_index",
        "member_name",
        "pred_energy",
        "pred_var",
        "aleatoric_var",
        "epistemic_var",
        "total_var",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for panel_index, row in enumerate(rows):
            if panel_index < 12:
                group = "low"
            elif panel_index < 24:
                group = "middle"
            else:
                group = "high"
            for member_index, (member_name, pred_energy, pred_var) in enumerate(
                zip(
                    row["member_names"],
                    row["member_pred_energy"],
                    row["member_pred_var"],
                )
            ):
                writer.writerow(
                    {
                        "panel_index": panel_index,
                        "group": group,
                        "epoch": row["epoch"],
                        "loader": row["loader"],
                        "config_index": row["config_index"],
                        "num_atoms": row["num_atoms"],
                        "ref_energy": row["ref_energy"],
                        "member_index": member_index,
                        "member_name": member_name,
                        "pred_energy": pred_energy,
                        "pred_var": pred_var,
                        "aleatoric_var": row["aleatoric_var"],
                        "epistemic_var": row["epistemic_var"],
                        "total_var": row["total_var"],
                    }
                )


def read_panel_rows(path: Path) -> Optional[List[Dict[str, object]]]:
    cached_rows = load_cached_csv_rows(
        path,
        required_fields=[
            "panel_index",
            "group",
            "epoch",
            "loader",
            "config_index",
            "num_atoms",
            "ref_energy",
            "member_index",
            "member_name",
            "pred_energy",
            "pred_var",
            "aleatoric_var",
            "epistemic_var",
            "total_var",
        ],
        field_parsers={
            "panel_index": parse_int,
            "group": parse_str,
            "epoch": parse_int,
            "loader": parse_str,
            "config_index": parse_int,
            "num_atoms": parse_float,
            "ref_energy": parse_float,
            "member_index": parse_int,
            "member_name": parse_str,
            "pred_energy": parse_float,
            "pred_var": parse_float,
            "aleatoric_var": parse_float,
            "epistemic_var": parse_float,
            "total_var": parse_float,
        },
        label="member_pred_var_grid",
    )
    if not cached_rows:
        return None

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in cached_rows:
        grouped.setdefault(int(row["panel_index"]), []).append(row)

    panel_rows: List[Dict[str, object]] = []
    for panel_index in sorted(grouped):
        entries = sorted(grouped[panel_index], key=lambda row: int(row["member_index"]))
        first = entries[0]
        num_atoms = float(first["num_atoms"])
        panel_rows.append(
            {
                "epoch": int(first["epoch"]),
                "loader": str(first["loader"]),
                "config_index": int(first["config_index"]),
                "num_atoms": int(num_atoms) if np.isfinite(num_atoms) else "",
                "ref_energy": float(first["ref_energy"]),
                "member_names": [str(entry["member_name"]) for entry in entries],
                "member_pred_energy": [float(entry["pred_energy"]) for entry in entries],
                "member_pred_var": [float(entry["pred_var"]) for entry in entries],
                "aleatoric_var": float(first["aleatoric_var"]),
                "epistemic_var": float(first["epistemic_var"]),
                "total_var": float(first["total_var"]),
            }
        )
    return panel_rows


def flatten_panel_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    flat_rows: List[Dict[str, object]] = []
    for panel_index, row in enumerate(rows):
        if panel_index < 12:
            group = "low"
        elif panel_index < 24:
            group = "middle"
        else:
            group = "high"
        for member_index, (member_name, pred_energy, pred_var) in enumerate(
            zip(
                row["member_names"],
                row["member_pred_energy"],
                row["member_pred_var"],
            )
        ):
            flat_rows.append(
                {
                    "panel_index": panel_index,
                    "group": group,
                    "epoch": row["epoch"],
                    "loader": row["loader"],
                    "config_index": row["config_index"],
                    "num_atoms": row["num_atoms"],
                    "ref_energy": row["ref_energy"],
                    "member_index": member_index,
                    "member_name": member_name,
                    "pred_energy": pred_energy,
                    "pred_var": pred_var,
                    "aleatoric_var": row["aleatoric_var"],
                    "epistemic_var": row["epistemic_var"],
                    "total_var": row["total_var"],
                }
            )
    return flat_rows


def write_plot(path: Path, rows: List[Dict[str, object]], per_atom: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 6, figsize=(18, 18))
    axes_flat = axes.flatten()

    for ax in axes_flat:
        ax.set_visible(False)

    for panel_index, row in enumerate(rows[:36]):
        ax = axes_flat[panel_index]
        ax.set_visible(True)
        x = np.array(row["member_pred_energy"], dtype=float)
        y = np.array(row["member_pred_var"], dtype=float)
        ax.scatter(x, y, s=28)
        ax.axvline(float(row["ref_energy"]), color="black", linestyle="--", linewidth=0.8)
        ax.set_box_aspect(1)
        ax.grid(alpha=0.25)

        loader = row["loader"]
        config_index = row["config_index"]
        total_var = row["total_var"]
        ale_var = row["aleatoric_var"]
        epi_var = row["epistemic_var"]
        title = (
            f"{loader}:{config_index}\n"
            f"T={total_var:.2e} A={ale_var:.2e} E={epi_var:.2e}"
        )
        ax.set_title(title, fontsize=9)

        if per_atom:
            ax.set_xlabel("Pred. energy / atom")
            ax.set_ylabel("Pred. variance / atom^2")
        else:
            ax.set_xlabel("Pred. energy")
            ax.set_ylabel("Pred. variance")

    fig.suptitle(
        "Member predictions vs predicted variances per system\n"
        "Top rows: lowest total uncertainty | middle rows: median total uncertainty | "
        "bottom rows: highest total uncertainty",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    member_names = [path.stem for path in input_paths]
    members_all = [load_member_file(path) for path in input_paths]

    selection_members = [
        filter_loader(member.get(args.selection_split, {}), args.selection_loader)
        for member in members_all
    ]
    if any(len(member) == 0 for member in selection_members):
        raise RuntimeError(
            "At least one input file has no matching rows for best-epoch selection."
        )
    best_epoch = select_best_epoch(selection_members, per_atom=args.per_atom)

    plot_loader = args.plot_loader if args.plot_loader is not None else args.selection_loader
    plot_members = [
        filter_loader(member.get(args.plot_split, {}), plot_loader)
        for member in members_all
    ]
    if any(best_epoch not in member for member in plot_members):
        raise RuntimeError(
            f"Selected epoch {best_epoch} is not available for split '{args.plot_split}'."
        )

    def compute_panel_rows() -> List[Dict[str, object]]:
        system_rows = build_system_rows(
            members=plot_members,
            member_names=member_names,
            epoch=best_epoch,
            per_atom=args.per_atom,
        )
        system_rows_trimmed = trim_system_rows(system_rows, args.trim)
        return select_panel_rows(system_rows_trimmed)

    output_csv_path = Path(args.output_csv)
    cached_flat_rows = load_cached_csv_rows(
        output_csv_path,
        required_fields=[
            "panel_index",
            "group",
            "epoch",
            "loader",
            "config_index",
            "num_atoms",
            "ref_energy",
            "member_index",
            "member_name",
            "pred_energy",
            "pred_var",
            "aleatoric_var",
            "epistemic_var",
            "total_var",
        ],
        field_parsers={
            "panel_index": parse_int,
            "group": parse_str,
            "epoch": parse_int,
            "loader": parse_str,
            "config_index": parse_int,
            "num_atoms": parse_float,
            "ref_energy": parse_float,
            "member_index": parse_int,
            "member_name": parse_str,
            "pred_energy": parse_float,
            "pred_var": parse_float,
            "aleatoric_var": parse_float,
            "epistemic_var": parse_float,
            "total_var": parse_float,
        },
        key_fields=["panel_index", "member_index"],
        compute_missing_rows=lambda: flatten_panel_rows(compute_panel_rows()),
        label="member_pred_var_grid",
    )
    if cached_flat_rows is not None:
        cached_panel_rows = read_panel_rows(output_csv_path)
        if cached_panel_rows is None:
            raise RuntimeError(f"Failed to rebuild panel rows from cached CSV {output_csv_path}")
        write_plot(Path(args.output_plot), cached_panel_rows, per_atom=args.per_atom)
        print(f"Selected epoch: {int(cached_panel_rows[0]['epoch'])}")
        print(f"Saved CSV: {args.output_csv}")
        print(f"Saved plot: {args.output_plot}")
        return

    panel_rows = compute_panel_rows()
    write_csv(output_csv_path, panel_rows)
    write_plot(Path(args.output_plot), panel_rows, per_atom=args.per_atom)

    print(f"Selected epoch: {best_epoch}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
