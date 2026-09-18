#!/usr/bin/env python3
"""Plot five-split means, approximate 95% CIs and individual split values."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from common import RUN_TAG, HERE, inventory, load_json, save_json, verify_inventory
from five_splits import confidence_interval

COLORS = {"hf_only": "#167b83", "lf_hf": "#bb4c47", "cross_regime": "#694d86",
          "common_evaluator": "#694d86"}
LABELS = {"hf_only": "HF-only", "lf_hf": "LF -> HF"}
TESTS = {"energy_id": "Energy-ID", "energy_ood": "Held-out Energy-OOD"}


def log_interval(row):
    """Geometric mean and back-transformed t interval across all five splits."""
    values = row["split_values"]
    if len(values) != 5 or any(value is None or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Log-space intervals require five finite, strictly positive values.")
    stats = confidence_interval([math.log(value) for value in values])
    try:
        bounds = {key: math.exp(stats[key]) for key in ("mean", "ci95_lower", "ci95_upper")}
    except OverflowError as exc:
        raise ValueError("Log-space interval exceeds representable range.") from exc
    if any(not math.isfinite(value) or value <= 0 for value in bounds.values()):
        raise ValueError("Log-space interval exceeds representable range.")
    return {**row, **bounds}


def point(axis, x, row, color):
    """Five faint points retain the paired split identity through fixed horizontal offsets."""
    values = row["split_values"]
    if len(values) != 5:
        raise ValueError("Plots require five split-level values.")
    for split, value in enumerate(values):
        if value is not None:
            axis.scatter(x + (split - 2) * 0.065, value, s=23, color="#797979",
                         alpha=0.65, edgecolors="white", linewidths=0.4, zorder=3)
    if row["mean"] is None:
        axis.text(x, 0.5, "undefined", transform=axis.get_xaxis_transform(),
                  ha="center", fontsize=14)
        return
    axis.errorbar(x, row["mean"], yerr=[[row["mean"] - row["ci95_lower"]],
                                      [row["ci95_upper"] - row["mean"]]],
                  fmt="D", color=color, markersize=6, capsize=5,
                  linewidth=1.7, markeredgecolor="white", markeredgewidth=0.6, zorder=4)


def decorate(axis, labels, ylabel, title, zero=False):
    axis.set_xticks(range(len(labels)), labels)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=18, pad=12)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    if zero:
        axis.axhline(0, color="#555555", linewidth=0.8)


def render(summary_path, rmse_scale="log"):
    if rmse_scale not in ("linear", "log", "symlog"):
        raise ValueError(f"Unsupported RMSE scale: {rmse_scale}")
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / "runs" / "_plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = Path(summary_path)
    summary = load_json(summary_path)
    methods = summary.get("settings", {}).get("acquisition_metrics", ["tu"])
    destination = summary_path.parent / "plots"
    inputs = {"files": inventory([summary_path, Path(__file__), HERE / "five_splits.py"]),
              "matplotlib": matplotlib.__version__, "ood_rmse_scale": rmse_scale}
    stamp = destination / "plots.json"
    if stamp.exists():
        cached = load_json(stamp)
        if cached["inputs"] == inputs:
            try:
                verify_inventory(cached["artifacts"])
            except RuntimeError:
                print(f"Regenerating missing/changed plots: {destination}", flush=True)
            else:
                print(f"Using cached plots: {destination}", flush=True)
                return
    rows = {(row["regime"], row["test"], row["metric"]): row for row in summary["rows"]}
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    rmse_metrics = ["rmse_before_meV_per_atom", "rmse_random500_meV_per_atom",
                    *(f"rmse_{m}500_meV_per_atom" for m in methods)]
    display_rows = {key: row for key, row in rows.items() if key[2] in rmse_metrics}
    effective_scale = rmse_scale
    if rmse_scale == "log":
        try:
            transformed = {key: log_interval(row) for key, row in display_rows.items() if key[1] == "energy_ood"}
        except ValueError as exc:
            print(f"OOD RMSE falls back to linear arithmetic intervals: {exc}", flush=True)
            effective_scale = "linear"
        else:
            display_rows.update(transformed)

    def finish(figure, name, title, note, mean_label="arithmetic mean"):
        figure.suptitle(title, fontsize=22, y=0.98)
        figure.text(0.04, 0.045, f"Diamonds: {mean_label} and approximate 95% CI. Grey points: five dataset splits.", fontsize=14)
        figure.text(0.04, 0.018, note, fontsize=13)
        figure.tight_layout(rect=(0, 0.09, 1, 0.93))
        try:
            for suffix in ("svg",):
                path = destination / f"{name}.{suffix}"
                figure.savefig(path, dpi=180, facecolor="white")
                artifacts.append(path)
        finally:
            plt.close(figure)

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(18, 11), sharey="col")
    for r, regime in enumerate(LABELS):
        for c, (test, label) in enumerate(TESTS.items()):
            axis = axes[r, c]
            for x, metric in enumerate(rmse_metrics):
                point(axis, x, display_rows[(regime, test, metric)], COLORS[regime])
            geometric = test == "energy_ood" and effective_scale == "log"
            if geometric:
                axis.set_yscale("log")
            elif test == "energy_ood" and effective_scale == "symlog":
                axis.set_yscale("symlog", linthresh=10)
            decorate(axis, ["Before", "Random-500", *(f"{m.upper()}-500" for m in methods)],
                     "RMSE (meV/atom)\n" + ("Geometric mean; log-space t CI" if geometric else "Arithmetic mean; t CI"),
                     f"{LABELS[regime]} | {label}")
    finish(figure, "rmse", "Prediction error before and after acquisition",
           "OOD: geometric mean and log-space t CI; ID: arithmetic mean and t CI. No split values are removed."
           if effective_scale == "log" else "Arithmetic means and t intervals, without clipping; lower RMSE is better.",
           mean_label="mean as labeled")
    display_path = destination / "rmse_display.json"
    save_json(display_path, {"ood_scale": effective_scale, "rows": [
        {"regime": key[0], "test": key[1], "metric": key[2],
         "estimator": "geometric_mean_log_t" if key[1] == "energy_ood" and effective_scale == "log" else "arithmetic_mean_t",
         **{field: row[field] for field in ("mean", "ci95_lower", "ci95_upper", "split_values")}}
        for key, row in display_rows.items()
    ]})
    artifacts.append(display_path)

    figure, axes = plt.subplots(1, 2, figsize=(22 if len(methods) > 1 else 18, 7))
    for axis, (test, label) in zip(axes, TESTS.items()):
        labels = []
        for regime in LABELS:
            for method in ("random", *methods):
                x = len(labels)
                labels.append(f"{LABELS[regime]}\n{method.upper() if method != 'random' else 'Random'}")
                point(axis, x, rows[(regime, test, f"relative_improvement_{method}_percent")], COLORS[regime])
        decorate(axis, labels, "Improvement from baseline (%)", label, zero=True)
    finish(figure, "improvement", "Relative improvement after 500 CC labels",
           "Positive values mean lower RMSE than before acquisition; negative values mean degradation.")

    for contrast in (False, True):
        figure, axes = plt.subplots(2, 2, figsize=(20, 11))
        for r, (unit, ylabel) in enumerate((("meV_per_atom", "RMSE gain (meV/atom)"),
                                          ("percentage_points", "Improvement gain (percentage points)"))):
            for c, (test, label) in enumerate(TESTS.items()):
                axis = axes[r, c]
                if contrast:
                    for x, method in enumerate(methods):
                        key = ("cross_regime", test, f"lf_hf_minus_hf_only_{method}_gain_{unit}")
                        point(axis, x, rows[key], COLORS["cross_regime"])
                    labels = [m.upper() for m in methods]
                else:
                    labels = []
                    for regime in LABELS:
                        for method in methods:
                            point(axis, len(labels), rows[(regime, test, f"{method}_gain_over_random_{unit}")], COLORS[regime])
                            labels.append(f"{LABELS[regime]}\n{method.upper()}")
                decorate(axis, labels, ylabel, label, zero=True)
        finish(figure, "regime_contrast" if contrast else ("tu_gain" if methods == ["tu"] else "acquisition_gain"),
               "Does LF -> HF obtain a larger acquisition benefit?" if contrast else "Uncertainty acquisition gain over random",
               "Paired within each split. Positive values favor LF -> HF; an interval crossing zero is inconclusive."
               if contrast else "Paired within each split. Positive values favor uncertainty acquisition; negative values favor random.")

    if any(key[0] == "common_evaluator" for key in rows):
        figure, axes = plt.subplots(1, 2, figsize=(18, 7))
        for axis, (test, label) in zip(axes, TESTS.items()):
            point(axis, 0, rows[("common_evaluator", test, "lf_hf_acquisition_gain_meV_per_atom")], COLORS["common_evaluator"])
            decorate(axis, ["LF -> HF selection gain"], "RMSE gain (meV/atom)", label, zero=True)
        finish(figure, "common_evaluator", "Acquisition quality with the same LF -> HF evaluator",
               "Positive values favor the batch selected by LF -> HF TU over the batch selected by HF-only TU.")
    save_json(stamp, {"inputs": inputs, "artifacts": inventory(artifacts),
                     "note": "Log OOD RMSE uses geometric means and log-space t intervals; other plots use arithmetic intervals. Original summary unchanged."})
    print(f"Plots (SVG): {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "runs" / ("aggregate" + (f"_{RUN_TAG}" if RUN_TAG else "")) / "summary_ci95.json")
    parser.add_argument("--ood-rmse-scale", choices=("linear", "log", "symlog"), default="log",
                        help="log: geometric mean/log-space t CI; linear or symlog: arithmetic mean/t CI")
    args = parser.parse_args()
    if not args.summary.resolve().is_relative_to(HERE / "runs"):
        parser.error("Summary must belong to the isolated AL runs directory.")
    render(args.summary, args.ood_rmse_scale)
