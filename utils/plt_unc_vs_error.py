#!/usr/bin/env python3
"""Plot uncertainty-vs-error reliability diagrams from ensemble epoch logs.

Inputs are the ``*_epoch_outputs.txt`` files produced with
``--log_epoch_outputs=true`` during training. The script reads the
``mode == "epoch_outputs_config"`` rows, aligns the ensemble members by
``(split, epoch, loader, config_index)``, and then:

1. Selects the best common epoch on ``--selection_split`` using the RMSE of the
   ensemble mean prediction.
2. Builds per-configuration uncertainties on the selected epoch:
   - aleatoric variance = mean predicted variance across ensemble members
   - epistemic variance = variance of ensemble member predictions
   - total variance = aleatoric variance + epistemic variance
3. Optionally fits isotonic regressors on ``--selection_split`` and applies the
   calibrated mapping to the uncertainties on ``--plot_split``.
4. Sorts systems by uncertainty, bins them into equal-count bins, and plots
   RMSE vs RMV (root mean variance) for aleatoric, epistemic, and total
   uncertainty on the same parity-style reliability plot.

Reported summary metrics:
- ENCE is computed from the bins only.
- Pearson correlation is computed on all systems, without binning.

Outputs:
- ``--output_csv_raw`` stores per-configuration data for the selected epoch.
  This is the file to reuse with ``--input_csv`` when only the plot/bins need
  to be recomputed.
- ``--output_csv_bins`` stores the binned reliability data, including ENCE
  terms.
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
        default=None,
        help="Paths to *_epoch_outputs.txt files, one per ensemble member.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help=(
            "Optional raw per-config CSV produced by this script. If set, skip "
            "log parsing and rebuild bins/plot from the saved per-config data."
        ),
    )
    parser.add_argument(
        "--selection_split",
        type=str,
        choices=["train", "valid", "test"],
        default="valid",
        help=(
            "Split used to select the best epoch. If --isotonic_calibration is "
            "enabled, the calibrator is also fit on this split."
        ),
    )
    parser.add_argument(
        "--plot_split",
        type=str,
        choices=["train", "valid", "test"],
        default="test",
        help="Split used for the final reliability diagram.",
    )
    parser.add_argument(
        "--selection_loader",
        type=str,
        default=None,
        help="Optional loader filter for best-epoch selection/calibration.",
    )
    parser.add_argument(
        "--plot_loader",
        type=str,
        default=None,
        help="Optional loader filter for reliability plotting.",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=10,
        help="Number of equal-count uncertainty bins used in the reliability diagram.",
    )
    parser.add_argument(
        "--per_atom",
        dest="per_atom",
        action="store_true",
        help="Use per-atom energy errors and variances: E/N and var/N^2 (default).",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system energy errors and variances instead of per-atom values.",
    )
    parser.add_argument(
        "--isotonic_calibration",
        action="store_true",
        help=(
            "Fit isotonic regressors on the selected epoch of --selection_split "
            "and apply them to the uncertainties on --plot_split before binning. "
            "Calibration is fit against squared error."
        ),
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="unc_vs_error.png",
        help="Output reliability plot path.",
    )
    parser.add_argument(
        "--output_csv_raw",
        type=str,
        default="unc_vs_error_raw.csv",
        help="Output raw per-configuration CSV path for the selected epoch.",
    )
    parser.add_argument(
        "--output_csv_bins",
        type=str,
        default="unc_vs_error_bins.csv",
        help="Output binned reliability CSV path.",
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
                "pred_energy_var": float(row["pred_energy_var"])
                if "pred_energy_var" in row
                else np.nan,
                "ref_energy": float(row["ref_energy"]) if "ref_energy" in row else np.nan,
                "num_atoms": int(row["num_atoms"]) if "num_atoms" in row else np.nan,
            }
    return data


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


def compute_epoch_rmse(
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
        pred_energy = float(np.mean(pred_energies))

        if per_atom:
            n_atoms = float(num_atoms_vals[0]) if np.isfinite(num_atoms_vals[0]) else np.nan
            if (not np.isfinite(n_atoms)) or n_atoms <= 0:
                raise RuntimeError(
                    "Missing/invalid num_atoms in epoch outputs. Per-atom mode requires it."
                )
            ref_energy = ref_energy / n_atoms
            pred_energy = pred_energy / n_atoms

        sq_errors.append((pred_energy - ref_energy) ** 2)

    return float(np.sqrt(np.mean(sq_errors)))


def select_best_epoch(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]],
    per_atom: bool,
) -> int:
    epochs = common_epoch_keys(members)
    if not epochs:
        raise RuntimeError("No common epochs found across ensemble members.")

    best_epoch = epochs[0]
    best_rmse = compute_epoch_rmse(members, best_epoch, per_atom=per_atom)
    for epoch in epochs[1:]:
        rmse = compute_epoch_rmse(members, epoch, per_atom=per_atom)
        if rmse < best_rmse:
            best_epoch = epoch
            best_rmse = rmse
    return best_epoch


def compute_selected_epoch_rows(
    members: List[Dict[int, Dict[ConfigKey, Dict[str, float]]]],
    epoch: int,
    per_atom: bool,
) -> List[Dict[str, float]]:
    common_configs = sorted(set.intersection(*(set(member[epoch].keys()) for member in members)))
    if not common_configs:
        raise RuntimeError(f"No common configurations found at epoch {epoch}.")

    rows: List[Dict[str, float]] = []
    for key in common_configs:
        loader, config_index = key
        pred_energies = np.array(
            [member[epoch][key]["pred_energy"] for member in members], dtype=float
        )
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

        ref_energy = float(ref_energies[0])
        ensemble_pred_energy = float(np.mean(pred_energies))
        aleatoric_var = (
            float(np.nanmean(pred_vars)) if np.any(np.isfinite(pred_vars)) else float("nan")
        )
        epistemic_var = float(np.var(pred_energies, ddof=0))
        num_atoms = float(num_atoms_vals[0]) if np.isfinite(num_atoms_vals[0]) else np.nan

        if per_atom:
            if (not np.isfinite(num_atoms)) or num_atoms <= 0:
                raise RuntimeError(
                    "Missing/invalid num_atoms in epoch outputs. Per-atom mode requires it."
                )
            ref_energy = ref_energy / num_atoms
            ensemble_pred_energy = ensemble_pred_energy / num_atoms
            if np.isfinite(aleatoric_var):
                aleatoric_var = aleatoric_var / (num_atoms**2)
            epistemic_var = epistemic_var / (num_atoms**2)
        total_var = (
            aleatoric_var + epistemic_var
            if np.isfinite(aleatoric_var) and np.isfinite(epistemic_var)
            else float("nan")
        )

        error = ensemble_pred_energy - ref_energy
        rows.append(
            {
                "epoch": epoch,
                "loader": loader,
                "config_index": int(config_index),
                "num_atoms": int(num_atoms) if np.isfinite(num_atoms) else "",
                "ref_energy": ref_energy,
                "pred_energy": ensemble_pred_energy,
                "error": error,
                "sq_error": error**2,
                "aleatoric_var": aleatoric_var,
                "epistemic_var": epistemic_var,
                "total_var": total_var,
                "aleatoric_var_raw": aleatoric_var,
                "epistemic_var_raw": epistemic_var,
                "total_var_raw": total_var,
            }
        )
    return rows


def fit_isotonic_calibrators(
    selection_rows: List[Dict[str, float]],
) -> Dict[str, object]:
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:
        raise ImportError(
            "Isotonic calibration requires scikit-learn to be installed."
        ) from exc

    calibrators: Dict[str, object] = {}
    for unc_name, unc_key in (
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ):
        valid_rows = [
            row
            for row in selection_rows
            if np.isfinite(row[unc_key]) and np.isfinite(row["sq_error"])
        ]
        if len(valid_rows) < 2:
            continue

        x = np.array([row[unc_key] for row in valid_rows], dtype=float)
        y = np.array([row["sq_error"] for row in valid_rows], dtype=float)
        if np.unique(x).size < 2:
            continue

        regressor = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds="clip")
        regressor.fit(x, y)
        calibrators[unc_name] = regressor

    return calibrators


def apply_isotonic_calibration(
    rows: List[Dict[str, float]],
    calibrators: Dict[str, object],
) -> List[Dict[str, float]]:
    calibrated_rows: List[Dict[str, float]] = []
    for row in rows:
        updated_row = dict(row)
        for unc_name, unc_key in (
            ("aleatoric", "aleatoric_var"),
            ("epistemic", "epistemic_var"),
            ("total", "total_var"),
        ):
            regressor = calibrators.get(unc_name)
            if regressor is not None and np.isfinite(row[unc_key]):
                calibrated_value = float(regressor.predict([row[unc_key]])[0])
                updated_row[unc_key] = max(calibrated_value, 0.0)

        calibrated_rows.append(updated_row)
    return calibrated_rows


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_raw_csv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed: Dict[str, float] = {
                "epoch": int(row["epoch"]),
                "loader": row["loader"],
                "config_index": int(row["config_index"]),
                "num_atoms": float(row["num_atoms"]) if row["num_atoms"] else np.nan,
                "ref_energy": float(row["ref_energy"]),
                "pred_energy": float(row["pred_energy"]),
                "error": float(row["error"]),
                "sq_error": float(row["sq_error"]),
                "aleatoric_var": float(row["aleatoric_var"])
                if row["aleatoric_var"]
                else np.nan,
                "epistemic_var": float(row["epistemic_var"]),
                "total_var": (
                    float(row["total_var"])
                    if row.get("total_var")
                    else float(row["aleatoric_var"]) + float(row["epistemic_var"])
                ),
                "aleatoric_var_raw": (
                    float(row["aleatoric_var_raw"])
                    if row.get("aleatoric_var_raw")
                    else np.nan
                ),
                "epistemic_var_raw": (
                    float(row["epistemic_var_raw"])
                    if row.get("epistemic_var_raw")
                    else np.nan
                ),
                "total_var_raw": (
                    float(row["total_var_raw"])
                    if row.get("total_var_raw")
                    else (
                        float(row["aleatoric_var_raw"]) + float(row["epistemic_var_raw"])
                        if row.get("aleatoric_var_raw") and row.get("epistemic_var_raw")
                        else np.nan
                    )
                ),
            }
            rows.append(parsed)
    return rows


def build_binned_rows(
    raw_rows: List[Dict[str, float]], num_bins: int
) -> List[Dict[str, float]]:
    if not raw_rows:
        raise RuntimeError("Cannot build reliability bins from empty raw rows.")

    binned_rows: List[Dict[str, float]] = []
    uncertainty_specs = [
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ]
    selected_epoch = int(raw_rows[0]["epoch"])

    for unc_name, unc_key in uncertainty_specs:
        valid_rows = [row for row in raw_rows if np.isfinite(row[unc_key])]
        if not valid_rows:
            continue
        valid_rows = sorted(valid_rows, key=lambda row: row[unc_key])
        effective_bins = min(num_bins, len(valid_rows))
        for bin_idx, bin_rows in enumerate(np.array_split(np.array(valid_rows, dtype=object), effective_bins)):
            bin_list = list(bin_rows)
            if not bin_list:
                continue
            sq_errors = np.array([row["sq_error"] for row in bin_list], dtype=float)
            variances = np.array([row[unc_key] for row in bin_list], dtype=float)
            mean_sq_error = float(np.mean(sq_errors))
            mean_variance = float(np.mean(variances))
            ence_term = (
                float(abs(mean_variance - mean_sq_error) / mean_variance)
                if mean_variance > 0.0
                else float("nan")
            )
            binned_rows.append(
                {
                    "epoch": selected_epoch,
                    "uncertainty_type": unc_name,
                    "bin_index": bin_idx,
                    "count": len(bin_list),
                    "rmse": float(np.sqrt(mean_sq_error)),
                    "rmv": float(np.sqrt(mean_variance)),
                    "mse": mean_sq_error,
                    "mean_variance": mean_variance,
                    "ence_term": ence_term,
                    "uncertainty_min": float(np.min(variances)),
                    "uncertainty_max": float(np.max(variances)),
                }
            )
    return binned_rows


def compute_ence_summary(binned_rows: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for unc_name in ["aleatoric", "epistemic", "total"]:
        rows = [row for row in binned_rows if row["uncertainty_type"] == unc_name]
        finite_terms = np.array(
            [row["ence_term"] for row in rows if np.isfinite(row["ence_term"])],
            dtype=float,
        )
        summary[unc_name] = (
            float(np.mean(finite_terms)) if finite_terms.size > 0 else float("nan")
        )
    return summary


def compute_pearson_summary(raw_rows: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    sq_errors = np.array([row["sq_error"] for row in raw_rows], dtype=float)

    for unc_name, unc_key in (
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ):
        variances = np.array([row[unc_key] for row in raw_rows], dtype=float)
        valid = np.isfinite(variances) & np.isfinite(sq_errors)
        if np.count_nonzero(valid) < 2:
            summary[unc_name] = float("nan")
            continue

        x = variances[valid]
        y = sq_errors[valid]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            summary[unc_name] = float("nan")
            continue

        summary[unc_name] = float(np.corrcoef(x, y)[0, 1])

    return summary


def write_plot(
    path: Path,
    binned_rows: List[Dict[str, float]],
    per_atom: bool,
    isotonic_calibration: bool,
    raw_rows: List[Dict[str, float]],
) -> None:
    if not binned_rows:
        raise RuntimeError("Cannot create reliability plot from empty binned rows.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))

    style_map = {
        "aleatoric": {"marker": "o", "label": "Aleatoric"},
        "epistemic": {"marker": "s", "label": "Epistemic"},
        "total": {"marker": "^", "label": "Total"},
    }
    ence_summary = compute_ence_summary(binned_rows)
    pearson_summary = compute_pearson_summary(raw_rows)
    max_xy = 0.0
    for unc_name in ["aleatoric", "epistemic", "total"]:
        rows = [row for row in binned_rows if row["uncertainty_type"] == unc_name]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["bin_index"])
        rmv = np.array([row["rmv"] for row in rows], dtype=float)
        rmse = np.array([row["rmse"] for row in rows], dtype=float)
        max_xy = max(max_xy, float(np.max(rmv)), float(np.max(rmse)))
        ax.plot(
            rmv,
            rmse,
            marker=style_map[unc_name]["marker"],
            label=style_map[unc_name]["label"],
        )

    parity_max = max_xy * 1.05 if max_xy > 0 else 1.0
    ax.plot([0.0, parity_max], [0.0, parity_max], linestyle="--", color="black", label="Ideal")
    ax.set_xlim(0.0, parity_max)
    ax.set_ylim(0.0, parity_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    if per_atom:
        ax.set_title("Reliability Diagram (per-atom energies)")
    else:
        ax.set_title("Reliability Diagram")
    if isotonic_calibration:
        ax.set_title(ax.get_title() + " with Isotonic Calibration")
    ax.grid(alpha=0.3)
    ax.legend()
    summary_lines = []
    if np.isfinite(ence_summary["aleatoric"]):
        summary_lines.append(
            f"Aleatoric ENCE (binned) = {ence_summary['aleatoric']:.4f}"
        )
    if np.isfinite(ence_summary["epistemic"]):
        summary_lines.append(
            f"Epistemic ENCE (binned) = {ence_summary['epistemic']:.4f}"
        )
    if np.isfinite(ence_summary["total"]):
        summary_lines.append(
            f"Total ENCE (binned) = {ence_summary['total']:.4f}"
        )
    if np.isfinite(pearson_summary["aleatoric"]):
        summary_lines.append(
            f"Aleatoric Pearson (all systems) = {pearson_summary['aleatoric']:.4f}"
        )
    if np.isfinite(pearson_summary["epistemic"]):
        summary_lines.append(
            f"Epistemic Pearson (all systems) = {pearson_summary['epistemic']:.4f}"
        )
    if np.isfinite(pearson_summary["total"]):
        summary_lines.append(
            f"Total Pearson (all systems) = {pearson_summary['total']:.4f}"
        )
    if summary_lines:
        ax.text(
            0.04,
            0.96,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    if args.input_csv is None and not args.inputs:
        raise RuntimeError("Provide either --inputs or --input_csv.")

    if args.input_csv is not None:
        raw_rows = read_raw_csv(Path(args.input_csv))
    else:
        members_all = [load_member_file(Path(path)) for path in args.inputs]
        selection_members = [
            filter_loader(member.get(args.selection_split, {}), args.selection_loader)
            for member in members_all
        ]
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

        raw_rows = compute_selected_epoch_rows(
            plot_members,
            epoch=best_epoch,
            per_atom=args.per_atom,
        )
        if args.isotonic_calibration:
            selection_rows = compute_selected_epoch_rows(
                selection_members,
                epoch=best_epoch,
                per_atom=args.per_atom,
            )
            calibrators = fit_isotonic_calibrators(selection_rows)
            raw_rows = apply_isotonic_calibration(raw_rows, calibrators)
        raw_fieldnames = [
            "epoch",
            "loader",
            "config_index",
            "num_atoms",
            "ref_energy",
            "pred_energy",
            "error",
            "sq_error",
            "aleatoric_var",
            "epistemic_var",
            "total_var",
            "aleatoric_var_raw",
            "epistemic_var_raw",
            "total_var_raw",
        ]
        write_csv(Path(args.output_csv_raw), raw_rows, raw_fieldnames)

    binned_rows = build_binned_rows(raw_rows, num_bins=args.num_bins)
    bins_fieldnames = [
        "epoch",
        "uncertainty_type",
        "bin_index",
        "count",
        "rmse",
        "rmv",
        "mse",
        "mean_variance",
        "ence_term",
        "uncertainty_min",
        "uncertainty_max",
    ]
    write_csv(Path(args.output_csv_bins), binned_rows, bins_fieldnames)
    write_plot(
        Path(args.output_plot),
        binned_rows,
        per_atom=args.per_atom,
        isotonic_calibration=args.isotonic_calibration,
        raw_rows=raw_rows,
    )

    print(f"Saved raw CSV: {args.output_csv_raw}")
    print(f"Saved binned CSV: {args.output_csv_bins}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
