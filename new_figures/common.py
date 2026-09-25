"""Shared data access, metrics, statistics and style for the paper figures.

Final-model metrics are recomputed from the per-configuration reliability caches
(experiment_X/evaluation/cache/split_S/reliability_{id,ood}_{nocal,cal}_raw.csv)
for all five dataset splits. Every metric is computed within a split, then
summarized across the five splits as mean +/- 95% Student-t interval (df=4).

Training-support filter
-----------------------
Held-out Energy-OOD contains a handful of configurations compressed below the
smallest interatomic distance present anywhere in that split's training data
(cc_train, cc_val, dft_train, dft_val). Models extrapolate catastrophically there,
and a single one can dominate a mean-of-squares metric. Every test configuration
below its split's bound is excluded, for every model and signal alike. The rule
reads geometry only, never predictions or uncertainties. It removes 13-16 of 5000
Energy-OOD configurations per split and essentially nothing elsewhere (the system
split's training data already covers compressed geometries).

The rule is implemented once, in eval/support_filter.py, and is shared with the
main aggregation step. It replaces the 0.5% total-variance trim that the paper's
current tables used, which removed the highest-*uncertainty* configurations and
therefore acted on the very quantity being evaluated.
"""

from __future__ import annotations

import csv
import logging
import math
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
EPOCHS = HERE / "_cache" / "epochs"
sys.path.insert(0, str(ROOT / "eval"))
logging.disable(logging.CRITICAL)

from reliability import (  # noqa: E402
    build_binned_rows, compute_ause, compute_ence_summary, compute_spearman_summary,
)
from support_filter import (  # noqa: E402  (shared with eval/aggregate_replicates.py)
    GEOMETRY_CACHE as GEOMETRY, geometry, support_bound, supported, test_file,
)

SPLITS = range(5)
T95_DF4 = 2.7764451051977987
SIGNALS = ("AU", "EU", "TU")
SIGNAL_KEY = {"AU": "aleatoric_var", "EU": "epistemic_var", "TU": "total_var"}
SIGNAL_NAME = {"AU": "aleatoric", "EU": "epistemic", "TU": "total"}

# Experiment families (README): system split A/B/E, energy split C/D/F.
PROTOCOLS = ("HF-only", "LF→HF")
EXPERIMENT = {("system", "LF-only"): "A", ("system", "LF→HF"): "B", ("system", "HF-only"): "E",
              ("energy", "LF-only"): "C", ("energy", "LF→HF"): "D", ("energy", "HF-only"): "F"}
TESTS = (("system", "id", "System-ID"), ("system", "ood", "System-OOD"),
         ("energy", "id", "Energy-ID"), ("energy", "ood", "Energy-OOD"))

# Palettes validated with the dataviz skill's validator (light surface, all pairs).
PROTOCOL_COLOR = {"HF-only": "#009099", "LF→HF": "#c0443d", "LF-only": "#9a9a9a"}
SIGNAL_COLOR = {"AU": "#2f6fc0", "EU": "#e8952f", "TU": "#1d6b3a"}
SHIFT_COLOR = {"system": "#6f4e9c", "energy": "#a6761d"}
SIGNAL_MARKER = {"AU": "o", "EU": "s", "TU": "D"}
SIGNAL_LINESTYLE = {"AU": (0, (5, 2)), "EU": (0, (1, 1.5)), "TU": "-"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
# Calibration panels: only TU is expected to match the error (the predictive
# variance is AU + EU), so TU is drawn at full strength and AU/EU are faded.
CAL_ALPHA = {"AU": 0.35, "EU": 0.35, "TU": 1.0}


# --------------------------------------------------------------------------- style

def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
        "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8, "axes.edgecolor": "#444444", "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
        "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "lines.linewidth": 1.2,
    })
    return plt


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.02,
                    dpi=300 if suffix == "png" else None)
    print(f"wrote {OUT / name}.{{pdf,svg,png}}")


# ----------------------------------------------------------------------- statistics

def ci(values):
    """Mean and 95% Student-t half-width across the five dataset splits."""
    values = [v for v in values if v is not None and math.isfinite(v)]
    if len(values) != 5:
        raise ValueError(f"expected five finite split values, got {values}")
    return statistics.mean(values), T95_DF4 * statistics.stdev(values) / math.sqrt(5)


def interval(values, log=False):
    """(center, low, high) over the five splits.

    Linear: arithmetic mean with a 95% Student-t interval. Log (for strictly
    positive quantities on log axes): geometric mean with the interval computed in
    log space and back-transformed, so both bounds stay positive.
    """
    if log:
        m, h = ci([math.log(v) for v in values])
        return math.exp(m), math.exp(m - h), math.exp(m + h)
    m, h = ci(values)
    return m, m - h, m + h


def legend_handles(protocol_marker=True):
    """Legend entries for the protocol colours and the signal marker shapes."""
    from matplotlib.lines import Line2D

    if protocol_marker:
        handles = [Line2D([], [], color=PROTOCOL_COLOR[p], marker="o", ls="", ms=4.5, label=p)
                   for p in PROTOCOLS]
    else:
        handles = [Line2D([], [], color=PROTOCOL_COLOR[p], lw=1.5, label=p) for p in PROTOCOLS]
    handles += [Line2D([], [], color=MUTED, marker=SIGNAL_MARKER[s], ls="", ms=4, label=s)
                for s in SIGNALS]
    return handles


# The support filter, the geometry cache and test_file() live in
# eval/support_filter.py, which eval/aggregate_replicates.py uses as well.

# ------------------------------------------------------------- final-model metrics

@lru_cache(None)
def final_rows(experiment, split, test, calibrated=False):
    path = (ROOT / f"experiment_{experiment}" / "evaluation" / "cache" / f"split_{split}"
            / f"reliability_{test}_{'cal' if calibrated else 'nocal'}_raw.csv")
    with path.open() as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    if [int(r["config_index"]) for r in rows] != list(range(len(rows))):
        raise RuntimeError(f"{path} is not in file order")
    return rows


def filtered_rows(experiment, split, test, calibrated=False):
    kind = "system" if experiment in "ABE" else "energy"
    rows = final_rows(experiment, split, test, calibrated)
    keep = supported(kind, split, test_file(experiment, test))
    if len(keep) != len(rows):
        raise RuntimeError(f"{experiment}/{split}/{test}: {len(rows)} rows vs {len(keep)} geometries")
    return [row for row, k in zip(rows, keep) if k]


def metrics(rows, num_bins=15):
    """RMSE (meV/atom), and Spearman / AUSE / ENCE per signal, as in eval/reliability.py."""
    sq = np.array([r["sq_error"] for r in rows])
    abs_err = np.sqrt(sq)
    spearman = compute_spearman_summary(rows)
    ence = compute_ence_summary(build_binned_rows(rows, num_bins=num_bins))
    out = {"RMSE": 1000 * float(np.sqrt(sq.mean()))}
    for s in SIGNALS:
        name = SIGNAL_NAME[s]
        scores = np.array([r[SIGNAL_KEY[s]] for r in rows])
        out[("Spearman", s)] = spearman[name]
        out[("AUSE", s)] = compute_ause(scores, abs_err)
        out[("ENCE", s)] = ence[name]
    return out


@lru_cache(None)
def final_metrics(experiment, test, calibrated=False):
    """Per-split metric dicts for one experiment/test (five entries)."""
    return tuple(metrics(filtered_rows(experiment, s, test, calibrated)) for s in SPLITS)
