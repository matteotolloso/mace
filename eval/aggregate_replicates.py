#!/usr/bin/env python3
"""Aggregate cached evaluation CSVs across independent dataset splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from reliability import (
    compute_ause_summary,
    compute_ence_summary,
    compute_energy_rmse,
    compute_spearman_summary,
)
from replicate_statistics import aggregate_rows, confidence_arrays, read_csv, summarize, write_csv
from plot_style import save_svg


LOGGER = logging.getLogger(__name__)
UNCERTAINTIES = ("aleatoric", "epistemic", "total")
COLORS = {"aleatoric": "tab:blue", "epistemic": "tab:orange", "total": "tab:green"}
LABELS = {"aleatoric": "AU", "epistemic": "EU", "total": "Total"}
LOG_EPOCH_QUALITY_LIMITS = {
    "spearman": (-0.2, 0.8),
    "ause": (1.0e-2, 1.0e0),
    "ence": (1.0e-1, 1.0e2),
    "magnitude": (1.0e-7, 1.0e5),
    "rmse_e_atom": (1.0e-4, 1.0e2),
    "nll_energy": (-8.0, 4.0),
}
STANDARD_EPOCH_QUALITY_LIMITS = {
    "spearman": (0.0, 1.0),
    "ause": (0.0, 0.5),
    "ence": (0.0, 2.0),
    "magnitude": (1.0e-6, 1.0e2),
    "rmse_e_atom": (1.0e-3, 1.0e0),
    "nll_energy": (-0.5, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Read evaluation/cache/split_<seed> CSVs, treat each split as one "
            "independent replicate, and write mean/95% Student-t CI outputs."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--distribution-bins", type=int, default=50)
    parser.add_argument("--log-log-reliability", action="store_true")
    parser.add_argument("--reliability-axis-min", type=float, default=None,
                        help="Shared RMV/RMSE lower limit; defaults to 0 for linear plots, automatic for log plots.")
    parser.add_argument("--reliability-axis-max", type=float, default=None,
                        help="Shared RMV/RMSE upper limit; defaults to 0.10 for linear plots, automatic for log plots.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def load_replicates(cache_dir: Path, seeds: Sequence[int], filename: str) -> List[List[Dict[str, str]]]:
    paths = [cache_dir / f"split_{seed}" / filename for seed in seeds]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Cannot aggregate {filename}; missing {len(missing)} split cache(s): "
            + ", ".join(missing)
        )
    return [read_csv(path) for path in paths]


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_svg(fig, path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_band(
    ax, x, rows, metric: str, *, color: str, label: str, marker: str = "o",
    y_limits=None,
) -> None:
    mean, low, high = confidence_arrays(rows, metric, geometric=ax.get_yscale() == "log")
    finite = np.isfinite(x) & np.isfinite(mean)
    plotted_mean = np.clip(mean, *y_limits) if y_limits is not None else mean
    ax.plot(x[finite], plotted_mean[finite], color=color, linewidth=2.3, label=label)
    in_range = finite
    if y_limits is not None:
        in_range = finite & (mean >= y_limits[0]) & (mean <= y_limits[1])
    if np.any(in_range):
        ax.plot(x[in_range], plotted_mean[in_range], linestyle="None", marker=marker,
                color=color, markersize=6.0)
    outside = finite & ~in_range
    if np.any(outside):
        ax.plot(x[outside], plotted_mean[outside], linestyle="None", marker="x",
                color=color, markersize=9.0, markeredgewidth=2.0, clip_on=False)
    band = finite & np.isfinite(low) & np.isfinite(high)
    if np.any(band):
        plotted_low = np.clip(low, *y_limits) if y_limits is not None else low
        plotted_high = np.clip(high, *y_limits) if y_limits is not None else high
        ax.fill_between(x, plotted_low, plotted_high, where=band, color=color, alpha=0.2, linewidth=0)


def use_log_intervals(ax, rows, metrics):
    """Use geometric summaries only if every plotted replicate group is positive."""
    valid = all(np.isfinite(float(row.get(f"{metric}_geometric_{key}", np.nan)))
                and float(row[f"{metric}_geometric_{key}"]) > 0
                for row in rows for metric in metrics for key in ("mean", "ci95_low", "ci95_high"))
    if valid:
        ax.set_yscale("log")
        return True
    LOGGER.warning("Nonpositive/missing replicate values: using arithmetic intervals on a linear axis.")
    return False


def draw_epoch_quality(
    rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    *,
    fixed_scales: bool,
    log_positive_metrics: bool,
):
    """Draw one epoch-quality view with intervals matched to each axis scale."""
    rows = sorted(rows, key=lambda row: float(row["epoch"]))
    x = np.asarray([float(row["epoch"]) for row in rows])
    panel_specs = [
        ("spearman", "Spearman ↑"),
        ("ause", "AUSE ↓"),
        ("ence", "ENCE ↓"),
        ("magnitude", "Uncertainty"),
    ]
    fig, axes = plt.subplots(6, 1, figsize=(14, 26), sharex=True)
    for ax, (prefix, ylabel) in zip(axes[:4], panel_specs):
        plotted = [f"{prefix}_{u}" for u in UNCERTAINTIES
                   if f"{prefix}_{u}" in metrics and not (prefix == "magnitude" and u == "total")]
        # Correlation is signed, so its panel remains linear. Positive metrics use
        # a geometric center and a t interval in log space on logarithmic views.
        if prefix == "magnitude" or (log_positive_metrics and prefix in ("ause", "ence")):
            use_log_intervals(ax, rows, plotted)
        y_limits = (LOG_EPOCH_QUALITY_LIMITS[prefix] if log_positive_metrics else
                    STANDARD_EPOCH_QUALITY_LIMITS[prefix] if fixed_scales else None)
        for uncertainty in UNCERTAINTIES:
            metric = f"{prefix}_{uncertainty}"
            if metric in metrics and not (prefix == "magnitude" and uncertainty == "total"):
                plot_band(ax, x, rows, metric, color=COLORS[uncertainty],
                          label=LABELS[uncertainty], y_limits=y_limits)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both" if ax.get_yscale() == "log" else "major")
        ax.legend()
        if log_positive_metrics:
            ax.set_ylim(*LOG_EPOCH_QUALITY_LIMITS[prefix])
    if "rmse_e_atom" in metrics:
        use_log_intervals(axes[4], rows, ["rmse_e_atom"])
        rmse_limits = (LOG_EPOCH_QUALITY_LIMITS["rmse_e_atom"] if log_positive_metrics else
                       STANDARD_EPOCH_QUALITY_LIMITS["rmse_e_atom"] if fixed_scales else None)
        plot_band(axes[4], x, rows, "rmse_e_atom", color="black", label="RMSE",
                  y_limits=rmse_limits)
    axes[4].set_ylabel("RMSE ↓")
    axes[4].grid(alpha=0.3, which="both" if axes[4].get_yscale() == "log" else "major")
    axes[4].legend()
    if log_positive_metrics:
        axes[4].set_ylim(*LOG_EPOCH_QUALITY_LIMITS["rmse_e_atom"])
    if "nll_energy" in metrics:
        nll_limits = (LOG_EPOCH_QUALITY_LIMITS["nll_energy"] if log_positive_metrics else
                      STANDARD_EPOCH_QUALITY_LIMITS["nll_energy"] if fixed_scales else None)
        plot_band(axes[5], x, rows, "nll_energy", color="black", label="GNLL",
                  y_limits=nll_limits)
    axes[5].set_ylabel("GNLL ↓")
    axes[5].set_xlabel("Epoch")
    axes[5].grid(alpha=0.3)
    axes[5].legend()
    if log_positive_metrics:
        axes[5].set_ylim(*LOG_EPOCH_QUALITY_LIMITS["nll_energy"])
    if "phase" in rows[0]:
        finetune_epochs = [float(row["epoch"]) for row in rows if str(row["phase"]) == "finetune"]
        if finetune_epochs:
            for ax in axes:
                ax.axvline(min(finetune_epochs), color="0.2", linestyle="--", linewidth=1.2)
    if fixed_scales:
        for ax, key in zip(axes, ("spearman", "ause", "ence", "magnitude",
                                  "rmse_e_atom", "nll_energy")):
            ax.set_ylim(*STANDARD_EPOCH_QUALITY_LIMITS[key])
    return fig, axes


def aggregate_epoch_quality(
    replicate_rows: Sequence[Sequence[Dict[str, str]]], output_csv: Path, output_plot: Path
) -> None:
    first = replicate_rows[0][0]
    key_fields = [field for field in ("phase", "source_epoch", "epoch") if field in first]
    metrics = [
        f"{prefix}_{uncertainty}"
        for prefix in ("pearson", "spearman", "ause", "ence", "magnitude")
        for uncertainty in UNCERTAINTIES
        if f"{prefix}_{uncertainty}" in first
    ]
    metrics += [metric for metric in ("rmse_e_atom", "nll_energy") if metric in first]
    rows = aggregate_rows(replicate_rows, key_fields=key_fields, metric_fields=metrics)
    write_csv(output_csv, rows)

    free_scale_path = output_plot.with_name(f"{output_plot.stem}-free-scale{output_plot.suffix}")
    fig, _ = draw_epoch_quality(rows, metrics, fixed_scales=False, log_positive_metrics=False)
    save_figure(fig, free_scale_path)
    fig, _ = draw_epoch_quality(rows, metrics, fixed_scales=True, log_positive_metrics=False)
    save_figure(fig, output_plot)
    log_scale_path = output_plot.with_name(f"{output_plot.stem}-log-scale{output_plot.suffix}")
    fig, _ = draw_epoch_quality(rows, metrics, fixed_scales=False, log_positive_metrics=True)
    save_figure(fig, log_scale_path)


def aggregate_line_csv(
    name: str,
    replicate_rows: Sequence[Sequence[Dict[str, str]]],
    output_csv: Path,
    output_plot: Path,
) -> None:
    first = replicate_rows[0][0]
    if name == "train_curves":
        metrics = ["train_loss", "val_loss", "train_rmse_e_atom", "val_rmse_e_atom"]
        panel_metrics = [metrics[:2], metrics[2:]]
        panel_labels = ["Loss", "RMSE per atom"]
    elif name == "epoch_raw_train" or name.startswith("epoch_raw_"):
        metrics = ["au_var", "eu_var", "tu_var", "rmse_energy"]
        panel_metrics = [metrics[:3], metrics[3:]]
        panel_labels = ["Variance", "RMSE"]
    else:
        raise RuntimeError(f"Unsupported line CSV: {name}")
    metrics = [metric for metric in metrics if metric in first]
    rows = aggregate_rows(replicate_rows, key_fields=["epoch"], metric_fields=metrics)
    write_csv(output_csv, rows)
    rows = sorted(rows, key=lambda row: float(row["epoch"]))
    x = np.asarray([float(row["epoch"]) for row in rows])
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for ax, panel, ylabel in zip(axes, panel_metrics, panel_labels):
        for index, metric in enumerate(panel):
            if metric in metrics:
                plot_band(ax, x, rows, metric, color=color_cycle[index], label=metric.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Epoch")
    save_figure(fig, output_plot)


def reliability_summary_rows(
    raw_replicates: Sequence[Sequence[Dict[str, str]]],
    bin_replicates: Sequence[Sequence[Dict[str, str]]],
) -> List[Dict[str, object]]:
    replicate_summaries: List[List[Dict[str, object]]] = []
    for raw_text, bins_text in zip(raw_replicates, bin_replicates):
        raw = [{key: _float_or_text(value) for key, value in row.items()} for row in raw_text]
        bins = [{key: _float_or_text(value) for key, value in row.items()} for row in bins_text]
        summaries = {
            "ence": compute_ence_summary(bins),
            "spearman": compute_spearman_summary(raw),
            "ause": compute_ause_summary(raw),
        }
        rows: List[Dict[str, object]] = [
            {"metric": "rmse", "uncertainty_type": "ensemble", "value": compute_energy_rmse(raw)}
        ]
        for metric, values in summaries.items():
            rows.extend(
                {"metric": metric, "uncertainty_type": uncertainty, "value": values[uncertainty]}
                for uncertainty in UNCERTAINTIES
            )
        replicate_summaries.append(rows)
    return aggregate_rows(
        replicate_summaries,
        key_fields=["metric", "uncertainty_type"],
        metric_fields=["value"],
    )


def aggregate_reliability(
    filename: str,
    bin_replicates: Sequence[Sequence[Dict[str, str]]],
    raw_replicates: Sequence[Sequence[Dict[str, str]]],
    output_dir: Path,
    *,
    log_log: bool,
    axis_min: float | None,
    axis_max: float | None,
) -> None:
    metrics = ["count", "rmse", "rmv", "mse", "mean_variance", "ence_term", "uncertainty_min", "uncertainty_max"]
    rows = aggregate_rows(
        bin_replicates,
        key_fields=["uncertainty_type", "bin_index"],
        metric_fields=metrics,
    )
    output_csv = output_dir / filename
    write_csv(output_csv, rows)
    summary_rows = reliability_summary_rows(raw_replicates, bin_replicates)
    summary_name = filename.replace("_bins.csv", "_summary.csv")
    write_csv(output_dir / summary_name, summary_rows)

    fig, (ax, text_ax) = plt.subplots(1, 2, figsize=(21, 12), gridspec_kw={"width_ratios": [1.2, 0.7]})
    text_ax.axis("off")
    if log_log:
        log_log = use_log_intervals(ax, rows, ["rmse", "rmv"])
        if log_log:
            ax.set_xscale("log")
        else:
            axis_min = 0.0
    for uncertainty in UNCERTAINTIES:
        subset = sorted(
            [row for row in rows if row["uncertainty_type"] == uncertainty],
            key=lambda row: float(row["bin_index"]),
        )
        x, _, _ = confidence_arrays(subset, "rmv", geometric=log_log)
        plot_band(ax, x, subset, "rmse", color=COLORS[uncertainty], label=LABELS[uncertainty])
    all_values = [
        float(row[f"{field}_geometric_mean"] if log_log else row[field])
        for row in rows
        for field in ("rmv", "rmse")
        if np.isfinite(float(row[field])) and (not log_log or float(row[field]) > 0)
    ]
    low = axis_min if axis_min is not None else (10 ** np.floor(np.log10(min(all_values))) if log_log else 0.0)
    high = axis_max if axis_max is not None else (10 ** np.ceil(np.log10(max(all_values))) if log_log else 0.1)
    ax.plot([low, high], [low, high], "--", color="black", alpha=0.5, label="Ideal")
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RMV" + (" (geometric mean)" if log_log else ""))
    ax.set_ylabel("RMSE" + (" (geometric mean)" if log_log else ""))
    ax.grid(alpha=0.3, which="both" if log_log else "major")
    ax.legend()
    summary_lines = ["Arithmetic mean [95% CI], n=5"]
    for row in summary_rows:
        mean = float(row["value"])
        lo = float(row["value_ci95_low"])
        hi = float(row["value_ci95_high"])
        summary_lines.append(
            f"{row['metric']} {row['uncertainty_type']}:\n{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        )
    text_ax.text(0.02, 0.98, "\n\n".join(summary_lines), va="top", family="monospace", fontsize=16)
    plot_name = filename.replace("_bins.csv", ".svg")
    save_figure(fig, output_dir / plot_name)


def aggregate_distribution(
    filename: str,
    replicate_rows: Sequence[Sequence[Dict[str, str]]],
    output_dir: Path,
    num_bins: int,
) -> None:
    specs = [
        ("aleatoric_var", "Aleatoric uncertainty", "tab:blue"),
        ("epistemic_var", "Epistemic uncertainty", "tab:orange"),
        ("error", "Signed energy error", "tab:green"),
        ("rmse_e_atom", "Absolute energy error", "tab:red"),
    ]
    output_rows: List[Dict[str, object]] = []
    fig, axes = plt.subplots(4, 1, figsize=(9, 13))
    for ax, (metric, label, color) in zip(axes, specs):
        arrays = [
            np.asarray([float(row[metric]) for row in rows if _is_finite(row.get(metric))])
            for rows in replicate_rows
        ]
        finite_all = np.concatenate([values for values in arrays if values.size])
        edges = np.histogram_bin_edges(finite_all, bins=num_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        densities = [np.histogram(values, bins=edges, density=True)[0] for values in arrays]
        for index, center in enumerate(centers):
            stats = summarize(density[index] for density in densities)
            output_rows.append(
                {
                    "metric": metric,
                    "bin_index": index,
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "bin_center": center,
                    **{f"density_{key}": value for key, value in stats.items()},
                }
            )
        mean = np.asarray([np.mean([density[index] for density in densities]) for index in range(len(centers))])
        low = np.asarray([summarize(density[index] for density in densities)["ci95_low"] for index in range(len(centers))])
        high = np.asarray([summarize(density[index] for density in densities)["ci95_high"] for index in range(len(centers))])
        ax.plot(centers, mean, color=color, linewidth=2.2)
        ax.fill_between(centers, low, high, color=color, alpha=0.2)
        ax.set_title(label)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Value")
    write_csv(output_dir / filename, output_rows)
    save_figure(fig, output_dir / Path(filename).with_suffix(".svg"))


def aggregate_bar_csv(
    filename: str,
    replicate_rows: Sequence[Sequence[Dict[str, str]]],
    output_dir: Path,
) -> None:
    first = replicate_rows[0][0]
    if filename == "finetune.csv":
        keys = ["phase", "split"]
        metrics = ["mean_au", "mean_eu", "mean_tu", "rmse_energy"]
        label_field = "split"
    else:
        keys = [field for field in ("group", "x_index", "label", "split", "quantile_lo", "quantile_hi") if field in first]
        metrics = [metric for metric in ("au_mean", "eu_mean") if metric in first]
        label_field = "label"
    rows = aggregate_rows(replicate_rows, key_fields=keys, metric_fields=metrics)
    write_csv(output_dir / filename, rows)
    labels = [" / ".join(str(row[field]) for field in keys if field in ("phase", label_field)) for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6.5))
    width = 0.8 / max(1, len(metrics))
    for index, metric in enumerate(metrics):
        mean, low, high = confidence_arrays(rows, metric)
        yerr = np.vstack((mean - low, high - mean))
        ax.bar(x + (index - (len(metrics) - 1) / 2) * width, mean, width, yerr=yerr, capsize=4, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 6 else 0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plot_stem = Path(filename).stem
    if plot_stem.startswith("energy_ood_") and plot_stem.endswith("_bins"):
        plot_stem = plot_stem[: -len("_bins")]
    save_figure(fig, output_dir / f"{plot_stem}.svg")


def _float_or_text(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _is_finite(value: str | None) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    split_zero = args.cache_dir / f"split_{args.split_seeds[0]}"
    if not split_zero.is_dir():
        raise RuntimeError(f"Missing first split cache directory: {split_zero}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filenames = sorted(path.name for path in split_zero.glob("*.csv"))
    processed = set()
    for filename in filenames:
        if filename in processed:
            continue
        if filename.startswith("epoch_quality_finetune_same"):
            continue  # Preserve legacy CSV caches, but never regenerate these plots.
        replicates = load_replicates(args.cache_dir, args.split_seeds, filename)
        stem = Path(filename).stem
        LOGGER.info("Aggregating %s", filename)
        if stem.startswith("reliability_") and stem.endswith("_raw"):
            continue
        if stem.startswith("reliability_") and stem.endswith("_bins"):
            raw_filename = filename.replace("_bins.csv", "_raw.csv")
            raw_replicates = load_replicates(args.cache_dir, args.split_seeds, raw_filename)
            aggregate_reliability(
                filename,
                replicates,
                raw_replicates,
                args.output_dir,
                log_log=args.log_log_reliability,
                axis_min=args.reliability_axis_min,
                axis_max=args.reliability_axis_max,
            )
            processed.add(raw_filename)
        elif stem.startswith("epoch_quality"):
            aggregate_epoch_quality(replicates, args.output_dir / filename, args.output_dir / f"{stem}.svg")
        elif stem == "train_curves" or stem.startswith("epoch_raw_"):
            aggregate_line_csv(stem, replicates, args.output_dir / filename, args.output_dir / f"{stem}.svg")
        elif stem.startswith("distribution_"):
            aggregate_distribution(filename, replicates, args.output_dir, args.distribution_bins)
        elif stem == "finetune" or stem.startswith("energy_ood_") and stem.endswith("_bins"):
            aggregate_bar_csv(filename, replicates, args.output_dir)
        elif stem.startswith("energy_ood_") and stem.endswith("_raw"):
            continue
        else:
            LOGGER.warning("No aggregate plot schema for %s; leaving its split caches untouched.", filename)


if __name__ == "__main__":
    main()
