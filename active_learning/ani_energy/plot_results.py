#!/usr/bin/env python3
"""Plot five-split means, approximate 95% CIs and individual split values."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import HERE, inventory, load_json, save_json, verify_inventory

COLORS = {"hf_only": "#167b83", "lf_hf": "#bb4c47", "cross_regime": "#694d86",
          "common_evaluator": "#694d86"}
LABELS = {"hf_only": "HF-only", "lf_hf": "LF -> HF"}
TESTS = {"energy_id": "Energy-ID", "energy_ood": "Held-out Energy-OOD"}


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
                  ha="center", fontsize=9)
        return
    axis.errorbar(x, row["mean"], yerr=[[row["mean"] - row["ci95_lower"]],
                                      [row["ci95_upper"] - row["mean"]]],
                  fmt="D", color=color, markersize=6, capsize=5,
                  linewidth=1.7, markeredgecolor="white", markeredgewidth=0.6, zorder=4)


def decorate(axis, labels, ylabel, title, zero=False):
    axis.set_xticks(range(len(labels)), labels)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=12, pad=12)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    if zero:
        axis.axhline(0, color="#555555", linewidth=0.8)


def render(summary_path, rmse_scale="symlog"):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / "runs" / "_plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = Path(summary_path)
    summary = load_json(summary_path)
    destination = summary_path.parent / "plots"
    inputs = {"files": inventory([summary_path, Path(__file__)]),
              "matplotlib": matplotlib.__version__, "ood_rmse_scale": rmse_scale}
    stamp = destination / "plots.json"
    if stamp.exists():
        cached = load_json(stamp)
        if cached["inputs"] == inputs:
            verify_inventory(cached["artifacts"])
            print(f"Using cached plots: {destination}", flush=True)
            return
    rows = {(row["regime"], row["test"], row["metric"]): row for row in summary["rows"]}
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def finish(figure, name, title, note):
        figure.suptitle(title, fontsize=16, y=0.98)
        figure.text(0.04, 0.045, "Diamonds: mean and approximate 95% CI. Grey points: five dataset splits.", fontsize=9)
        figure.text(0.04, 0.018, note, fontsize=9)
        figure.tight_layout(rect=(0, 0.09, 1, 0.93))
        try:
            for suffix in ("png", "pdf"):
                path = destination / f"{name}.{suffix}"
                figure.savefig(path, dpi=180, facecolor="white")
                artifacts.append(path)
        finally:
            plt.close(figure)

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="col")
    for r, regime in enumerate(LABELS):
        for c, (test, label) in enumerate(TESTS.items()):
            axis = axes[r, c]
            for x, metric in enumerate(("rmse_before_meV_per_atom", "rmse_random500_meV_per_atom", "rmse_tu500_meV_per_atom")):
                point(axis, x, rows[(regime, test, metric)], COLORS[regime])
            if test == "energy_ood" and rmse_scale == "symlog":
                axis.set_yscale("symlog", linthresh=10)
            decorate(axis, ["Before", "Random-500", "TU-500"],
                     "RMSE (meV/atom)" + ("; symlog axis" if test == "energy_ood" and rmse_scale == "symlog" else ""),
                     f"{LABELS[regime]} | {label}")
    finish(figure, "rmse", "Prediction error before and after acquisition",
           "Shared scales within each test. Student-t bounds are not clipped at zero; lower RMSE is better.")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (test, label) in zip(axes, TESTS.items()):
        labels = []
        for regime in LABELS:
            for method in ("random", "tu"):
                x = len(labels)
                labels.append(f"{LABELS[regime]}\n{method.upper() if method == 'tu' else 'Random'}")
                point(axis, x, rows[(regime, test, f"relative_improvement_{method}_percent")], COLORS[regime])
        decorate(axis, labels, "Improvement from baseline (%)", label, zero=True)
    finish(figure, "improvement", "Relative improvement after 500 CC labels",
           "Positive values mean lower RMSE than before acquisition; negative values mean degradation.")

    for contrast in (False, True):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for r, (unit, ylabel) in enumerate((("meV_per_atom", "RMSE gain (meV/atom)"),
                                          ("percentage_points", "Improvement gain (percentage points)"))):
            for c, (test, label) in enumerate(TESTS.items()):
                axis = axes[r, c]
                if contrast:
                    key = ("cross_regime", test, f"lf_hf_minus_hf_only_tu_gain_{unit}")
                    point(axis, 0, rows[key], COLORS["cross_regime"])
                    labels = ["LF -> HF gain minus HF-only gain"]
                else:
                    for x, regime in enumerate(LABELS):
                        point(axis, x, rows[(regime, test, f"tu_gain_over_random_{unit}")], COLORS[regime])
                    labels = list(LABELS.values())
                decorate(axis, labels, ylabel, label, zero=True)
        finish(figure, "regime_contrast" if contrast else "tu_gain",
               "Does LF -> HF obtain a larger benefit from TU?" if contrast else "TU acquisition gain over random",
               "Paired within each split. Positive values favor LF -> HF; an interval crossing zero is inconclusive."
               if contrast else "Paired within each split. Positive values favor TU; negative values favor random acquisition.")

    if any(key[0] == "common_evaluator" for key in rows):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, (test, label) in zip(axes, TESTS.items()):
            point(axis, 0, rows[("common_evaluator", test, "lf_hf_acquisition_gain_meV_per_atom")], COLORS["common_evaluator"])
            decorate(axis, ["LF -> HF selection gain"], "RMSE gain (meV/atom)", label, zero=True)
        finish(figure, "common_evaluator", "Acquisition quality with the same LF -> HF evaluator",
               "Positive values favor the batch selected by LF -> HF TU over the batch selected by HF-only TU.")
    save_json(stamp, {"inputs": inputs, "artifacts": inventory(artifacts),
                     "note": "Student-t intervals are unchanged; no clipping, trimming or averaging over members."})
    print(f"Plots (PNG and PDF): {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "runs/aggregate/summary_ci95.json")
    parser.add_argument("--ood-rmse-scale", choices=("linear", "symlog"), default="symlog")
    args = parser.parse_args()
    if not args.summary.resolve().is_relative_to(HERE / "runs"):
        parser.error("Summary must belong to the isolated AL runs directory.")
    render(args.summary, args.ood_rmse_scale)
