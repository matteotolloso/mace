#!/usr/bin/env python3
"""Appendix training-dynamics figures (replace epoch_quality*-log-scale.svg).

Writes four figures:
  figA_dynamics_system_hf    HF-only E, 300 epochs on CC
  figA_dynamics_system_lfhf  LF->HF: A (DFT pretraining, epochs 0-299) then B (CC
                             fine-tuning, 300-399, shaded)
  figA_dynamics_energy_hf    HF-only F
  figA_dynamics_energy_lfhf  LF->HF: C then D
Columns: training set, ID test set, OOD test set (for LF->HF the DFT files and
labels while pretraining and the CC files afterwards, so RMSE and GNLL jump at
epoch 300 partly because the target changes). Rows: Spearman (up), AUSE, ENCE
(log), mean per-atom AU and EU variance (log), ensemble RMSE (log), mean Gaussian
NLL of the ensemble with TU variance, as in eval/epoch_quality.py.

Data: the cached per-epoch curves (epoch_data.py), except Energy-OOD, which is
recomputed from per-configuration predictions restricted to the training-support
region (_cache/epochs/{C,D,F}_<split>.npz; pretraining is scored on the
cc_test_ood geometries against DFT labels). Train and ID are unfiltered (training
data is inside the support; the filter would remove at most 2 of 5000 CC ID
configurations). Lines: five-split means (geometric on log axes), smoothed by an
exponential moving average restarted at epoch 300; bands: 95% Student-t
intervals over the five splits (log space on log axes), smoothed the same way.
Colour = signal. In the ENCE row TU is drawn at full strength and AU/EU faded
(CAL_ALPHA): only TU is expected to be calibrated.
"""

from common import (
    CAL_ALPHA, INK, MUTED, SIGNAL_COLOR, SIGNALS, panel_label, save, style,
)
from epoch_data import hf_only, lf_to_hf_any
from fig5_decomposition import PRETRAIN, curve, stage_background

COLUMNS = (("train", "Training set"), ("id", "ID test set"), ("ood", "OOD test set"))
# Y labels are rotated 90° counter-clockwise, which rotates the arrow glyphs too:
# "←" renders as a downward arrow (lower is better) and "→" as an upward one.
ROWS = (
    ("Spearman", "Spearman  →", False),
    ("AUSE", "AUSE  ←", False),
    ("ENCE", "ENCE  ←", True),
    ("mag", "mean variance\n(eV²/atom²)", True),
    ("RMSE", "RMSE (meV/atom)  ←", True),
    ("NLL", "Gaussian NLL  ←", False),
)
FIGURES = (
    ("system", "hf", "System split, HF-only (E)"),
    ("system", "lfhf", "System split, LF→HF (A → B)"),
    ("energy", "hf", "Energy split, HF-only (F)"),
    ("energy", "lfhf", "Energy split, LF→HF (C → D)"),
)
LOG_TICKS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def series(metric):
    """(label, key function, line style) for every line of one row."""
    if metric in ("RMSE", "NLL"):
        scale = 1000.0 if metric == "RMSE" else 1.0
        return [(None, lambda e, m=metric, k=scale: k * e[m], dict(color=INK, lw=1.2))]
    if metric == "mag":
        return [(s, lambda e, s=s: e[f"mag_{s}"], dict(color=SIGNAL_COLOR[s], lw=1.1))
                for s in ("AU", "EU")]
    out = []
    for s in SIGNALS:
        line = dict(color=SIGNAL_COLOR[s], lw=1.1)
        if metric == "ENCE":
            line.update(alpha=CAL_ALPHA[s], lw=1.5 if s == "TU" else 0.9)
        out.append((s, lambda e, s=s, m=metric: e[(m, s)], line))
    return out


def draw(kind, protocol, title):
    plt = style()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fetch_any = hf_only if protocol == "hf" else lf_to_hf_any
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(5.5, 7.6), sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.18, "wspace": 0.1})
    missing = []
    for c, (test, col_title) in enumerate(COLUMNS):
        fetch = lambda _kind, split, t=test: fetch_any(kind, split, t)  # noqa: E731
        for r, (metric, ylabel, log) in enumerate(ROWS):
            ax = axes[r, c]
            if protocol == "lfhf":
                stage_background(ax)
            else:
                ax.grid(axis="x", visible=False)
            for _, key, line in series(metric):
                if not curve(ax, kind, key, log, fetch=fetch, **line):
                    missing.append(f"{test} {metric}")
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(FixedLocator(LOG_TICKS))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            if r == 0:
                ax.set_title(col_title, loc="left", pad=3)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(ROWS) - 1:
                ax.set_xlabel("epoch")
    axes[0, 0].set_xlim(0, 400 if protocol == "lfhf" else 300)
    axes[0, 0].xaxis.set_major_locator(FixedLocator([0, 100, 200, 300, 400]))
    for ax, letter in zip(axes.flat, "abcdefghijklmnopqr"):
        panel_label(ax, letter, x=-0.02, y=0.98)
    if missing:
        fig.text(0.5, 0.5, "missing per-epoch data: " + ", ".join(missing), ha="center",
                 color="#b00020", fontsize=8)
    handles = [Line2D([], [], color=SIGNAL_COLOR[s], lw=1.3, label=s) for s in SIGNALS]
    handles += [Line2D([], [], color=INK, lw=1.2, label="ensemble (RMSE, GNLL)"),
                Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI, 5 splits")]
    if protocol == "lfhf":
        handles.append(Patch(color="#f1f1ef", ec=MUTED, lw=0.4, label="CC fine-tuning"))
    fig.legend(handles=handles, loc="upper center", ncol=6 if protocol == "lfhf" else 5, bbox_to_anchor=(0.52, 0.935),
               handlelength=1.6, columnspacing=1.0)
    fig.suptitle(title, x=0.12, y=0.955, ha="left", fontsize=8, fontweight="bold")
    assert PRETRAIN == 300
    save(fig, f"figA_dynamics_{kind}_{protocol}")
    plt.close(fig)


def main():
    for kind, protocol, title in FIGURES:
        draw(kind, protocol, title)


if __name__ == "__main__":
    main()
