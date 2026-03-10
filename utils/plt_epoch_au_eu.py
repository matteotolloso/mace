#!/usr/bin/env python3
"""Plot aleatoric/epistemic uncertainty evolution from epoch output logs.

Input files are expected to be the per-model logs produced by training with
`--log_epoch_outputs=true`, i.e. lines with:
    mode == "epoch_outputs_config"
and fields including:
    epoch, split, loader, config_index, pred_energy, pred_energy_var

Per-sample decomposition:
    EU(x) = (1/(M-1)) * sum_m (y_m(x) - y_bar(x))^2
    AU(x) = (1/M) * sum_m sigma_m^2(x)

Per-epoch values are the mean over all common samples in that epoch.
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
        help="Optional loader name filter (e.g. 'Default'). If omitted, use all loaders.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="ae_uncertainty_vs_epoch.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ae_uncertainty_vs_epoch.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Percentile for visualization clipping (applied only to plot y-values).",
    )
    parser.add_argument(
        "--drop_first_k_epochs",
        type=int,
        default=0,
        help="Exclude the first k epochs from the plot only.",
    )
    parser.add_argument(
        "--per_atom",
        action="store_true",
        help="Normalize uncertainties per atom: energy -> E/N, variance -> var/N^2.",
    )
    parser.add_argument(
        "--plot_log_variance",
        action="store_true",
        help="Plot log(variance) values (applied only to plotted y-values).",
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
            if "pred_energy" not in row:
                continue

            epoch = int(row["epoch"])
            row_loader = str(row.get("loader", ""))
            config_index = int(row["config_index"])
            key: ConfigKey = (row_loader, config_index)

            if epoch not in data:
                data[epoch] = {}
            data[epoch][key] = {
                "pred_energy": float(row["pred_energy"]),
                "pred_energy_var": float(row["pred_energy_var"])
                if "pred_energy_var" in row
                else np.nan,
                "num_atoms": int(row["num_atoms"]) if "num_atoms" in row else np.nan,
                "ref_energy": float(row["ref_energy"]) if "ref_energy" in row else np.nan,
            }
    return data


def compute_epoch_uncertainty(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]], per_atom: bool = False
) -> List[Tuple[int, float, float, float]]:
    if len(members) < 2:
        raise RuntimeError(
            "At least 2 ensemble members are required to estimate epistemic uncertainty."
        )

    common_epochs = sorted(set.intersection(*(set(m.keys()) for m in members)))
    results: List[Tuple[int, float, float, float]] = []
    align_tol = 1e-8

    for epoch in common_epochs:
        common_configs = set.intersection(*(set(m[epoch].keys()) for m in members))
        if len(common_configs) == 0:
            continue

        aleatoric_per_config = []
        epistemic_per_config = []
        err2_e_per_atom_per_config = []

        for key in common_configs:
            pred_energies = np.array([m[epoch][key]["pred_energy"] for m in members], dtype=float)
            pred_vars = np.array([m[epoch][key]["pred_energy_var"] for m in members], dtype=float)
            ref_energies = np.array([m[epoch][key]["ref_energy"] for m in members], dtype=float)
            num_atoms_vals = np.array([m[epoch][key]["num_atoms"] for m in members], dtype=float)

            # Guard against mismatched config ordering across ensemble members.
            # For the same (loader, config_index), ref energy and num_atoms should match.
            finite_ref = np.isfinite(ref_energies)
            if np.any(finite_ref) and np.ptp(ref_energies[finite_ref]) > align_tol:
                raise RuntimeError(
                    f"Detected misaligned members at epoch={epoch}, key={key}: "
                    "ref_energy differs across input files. "
                    "Ensure deterministic ordering and identical dataset split across members."
                )
            finite_n = np.isfinite(num_atoms_vals)
            if np.any(finite_n) and np.ptp(num_atoms_vals[finite_n]) > 0:
                raise RuntimeError(
                    f"Detected misaligned members at epoch={epoch}, key={key}: "
                    "num_atoms differs across input files. "
                    "Ensure deterministic ordering and identical dataset split across members."
                )

            # RMSE_E_per_atom uses ensemble mean prediction against reference energy.
            # This follows the standard MACE error metric scale (later converted to meV).
            if np.any(finite_ref):
                ref_e = float(ref_energies[finite_ref][0])
                n_atoms_rmse = float(num_atoms_vals[finite_n][0]) if np.any(finite_n) else np.nan
                if np.isfinite(n_atoms_rmse) and n_atoms_rmse > 0:
                    y_bar_total = float(np.mean(pred_energies))
                    err_e_per_atom = (y_bar_total - ref_e) / n_atoms_rmse
                    err2_e_per_atom_per_config.append(err_e_per_atom ** 2)

            if per_atom:
                n_atoms = members[0][epoch][key]["num_atoms"]
                if (not np.isfinite(n_atoms)) or n_atoms <= 0:
                    raise RuntimeError(
                        "Missing/invalid 'num_atoms' in epoch outputs. "
                        "Per-atom normalization requires --log_epoch_outputs rows with num_atoms."
                    )
                pred_energies = pred_energies / n_atoms
                pred_vars = pred_vars / (n_atoms**2)

            # EU(x) = (1/(M-1)) * sum_m (y_m(x) - y_bar(x))^2
            # AU(x) = (1/M) * sum_m sigma_m^2(x)
            y_bar = float(np.mean(pred_energies))
            epistemic = float(
                np.sum((pred_energies - y_bar) ** 2) / (len(pred_energies) - 1)
            )
            epistemic_per_config.append(epistemic)
            if np.any(np.isfinite(pred_vars)):
                aleatoric_per_config.append(float(np.nanmean(pred_vars)))

        if len(epistemic_per_config) == 0:
            continue

        aleatoric_mean = (
            float(np.mean(aleatoric_per_config))
            if len(aleatoric_per_config) > 0
            else float("nan")
        )
        epistemic_mean = float(np.mean(epistemic_per_config))
        rmse_e_per_atom_meV = (
            float(np.sqrt(np.mean(err2_e_per_atom_per_config)) * 1e3)
            if len(err2_e_per_atom_per_config) > 0
            else float("nan")
        )
        results.append(
            (
                epoch,
                aleatoric_mean,
                epistemic_mean,
                rmse_e_per_atom_meV,
            )
        )

    return results


def write_csv(path: Path, rows: List[Tuple[int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "au_e_per_atom2",
                "eu_e_per_atom2",
                "rmse_e_per_atom_meV",
            ]
        )
        writer.writerows(rows)


def read_csv(path: Path) -> List[Tuple[int, float, float, float]]:
    rows: List[Tuple[int, float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            au = float(row["au_e_per_atom2"])
            eu = float(row["eu_e_per_atom2"])
            rmse_meV = float(row["rmse_e_per_atom_meV"])
            rows.append((epoch, au, eu, rmse_meV))
    rows.sort(key=lambda x: x[0])
    return rows


def write_plot(
    path: Path,
    rows: List[Tuple[int, float, float, float]],
    clip_percentile: float,
    drop_first_k_epochs: int,
    per_atom: bool,
    plot_log_variance: bool,
    log_eps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_plot = rows
    if drop_first_k_epochs > 0:
        rows_plot = rows[drop_first_k_epochs:]
    if len(rows_plot) == 0:
        raise RuntimeError(
            "No epochs left to plot after applying --drop_first_k_epochs."
        )

    epochs = [r[0] for r in rows_plot]
    ale = np.array([r[1] for r in rows_plot], dtype=float)
    epi = np.array([r[2] for r in rows_plot], dtype=float)
    rmse_meV = np.array([r[3] for r in rows_plot], dtype=float)

    finite_ale = ale[np.isfinite(ale)]
    finite_epi = epi[np.isfinite(epi)]
    if finite_ale.size == 0 or finite_epi.size == 0:
        raise RuntimeError("Cannot plot: uncertainties contain no finite values.")

    ale_clip = float(np.percentile(finite_ale, clip_percentile))
    epi_clip = float(np.percentile(finite_epi, clip_percentile))

    ale_plot = np.minimum(ale, ale_clip)
    epi_plot = np.minimum(epi, epi_clip)

    ale_is_clipped = ale > ale_clip
    epi_is_clipped = epi > epi_clip

    if plot_log_variance:
        ale_plot = np.log(np.maximum(ale_plot, log_eps))
        epi_plot = np.log(np.maximum(epi_plot, log_eps))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(
        epochs,
        ale_plot,
        marker="o",
        label=f"Aleatoric (clipped at p{clip_percentile:g})",
    )
    ax1.plot(
        epochs,
        epi_plot,
        marker="o",
        label=f"Epistemic (clipped at p{clip_percentile:g})",
    )
    if np.any(ale_is_clipped):
        ax1.scatter(
            np.array(epochs)[ale_is_clipped],
            ale_plot[ale_is_clipped],
            marker="^",
            s=60,
            label="Aleatoric outlier (clipped)",
        )
    if np.any(epi_is_clipped):
        ax1.scatter(
            np.array(epochs)[epi_is_clipped],
            epi_plot[epi_is_clipped],
            marker="^",
            s=60,
            label="Epistemic outlier (clipped)",
        )
    ax1.set_xlabel("Epoch")
    if per_atom:
        ylabel = "Uncertainty per atom^2"
    else:
        ylabel = "Uncertainty"
    if plot_log_variance:
        ylabel = f"log({ylabel})"

    title = (
        "Aleatoric and Epistemic Uncertainty per Atom vs Epoch"
        if per_atom
        else "Aleatoric and Epistemic Uncertainty vs Epoch"
    )
    if plot_log_variance:
        title = "Log " + title
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    finite_rmse = np.isfinite(rmse_meV)
    if np.any(finite_rmse):
        ax2 = ax1.twinx()
        (rmse_line,) = ax2.plot(
            np.array(epochs)[finite_rmse],
            rmse_meV[finite_rmse],
            color="black",
            linestyle="--",
            marker="s",
            label="RMSE_E_per_atom (meV)",
        )
        ax2.set_ylabel("RMSE_E_per_atom (meV)")
        handles.append(rmse_line)
        labels.append("RMSE_E_per_atom (meV)")

    ax1.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    if output_csv.exists():
        rows = read_csv(output_csv)
        if len(rows) == 0:
            raise RuntimeError(f"Existing CSV is empty: {output_csv}")
        print(f"Using existing CSV: {output_csv}")
    else:
        member_maps = [
            load_member_file(Path(p), split=args.split, loader=args.loader)
            for p in args.inputs
        ]
        if any(len(m) == 0 for m in member_maps):
            raise RuntimeError(
                "At least one input file has no matching 'epoch_outputs_config' rows "
                f"for split='{args.split}' and loader='{args.loader}'."
            )

        rows = compute_epoch_uncertainty(member_maps, per_atom=args.per_atom)
        if len(rows) == 0:
            raise RuntimeError("No common epochs/configurations found across ensemble files.")
        write_csv(output_csv, rows)
        print(f"Saved CSV: {args.output_csv}")

    write_plot(
        Path(args.output_plot),
        rows,
        clip_percentile=args.clip_percentile,
        drop_first_k_epochs=args.drop_first_k_epochs,
        per_atom=args.per_atom,
        plot_log_variance=args.plot_log_variance,
        log_eps=args.log_eps,
    )
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
