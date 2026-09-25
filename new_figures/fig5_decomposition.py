#!/usr/bin/env python3
"""Fig. 5 - The decomposition as an extrapolation detector (Finding 3).

LF->HF over all training epochs, DFT pretraining (0-299) then CC fine-tuning
(300-399, shaded). Color = system vs energy split, line style = AU dashed,
EU dotted, TU solid.
Row 1, training data (DFT train set while pretraining, CC train set while
fine-tuning): (a) AUSE, (b) ENCE, log y (6 lines each).
Row 2, ID test set (cached curves, see lf_to_hf_id): (c) AUSE, (d) ENCE.
Row 3, OOD test set: (e) AUSE, (f) ENCE, log y.

Lines are five-split means (geometric for ENCE), smoothed by an exponential moving
average restarted at the fine-tuning boundary so the transition is not smeared;
shaded bands are the 95% Student-t intervals over the five splits, smoothed the
same way (log space for ENCE). In the ENCE panels TU is drawn at full strength and
AU/EU faded (CAL_ALPHA).

The EU/AU panels that used to be (g) and (h) are now the appendix figure
figA_eu_au_ratio.py, which imports the helpers below (as does paper_numbers.py).
"""

import csv
import math

import numpy as np

from common import (
    CAL_ALPHA, MUTED, PROTOCOL_COLOR, ROOT, SHIFT_COLOR, SIGNAL_LINESTYLE, SIGNALS, SPLITS,
    ci, panel_label, save, style, supported,
)
from epoch_data import across_splits, band, ema, lf_to_hf, lf_to_hf_id, lf_to_hf_train

PRETRAIN = 300
SHIFTS = (("system", "System split"), ("energy", "Energy split"))
BINS = [("ID", None, None)] + [(f"{lo:.1f}–{lo + 0.1:.1f}", lo, lo + 0.1)
                               for lo in (0.5, 0.6, 0.7, 0.8, 0.9)]
FINAL = (("LF-only", "C", "dft"), ("HF-only", "F", "cc"), ("LF→HF", "D", "cc"))
GRID_SEP = "#bdbdbd"


def phased_ema(epochs, values):
    out = np.empty_like(values)
    for mask in (epochs < PRETRAIN, epochs >= PRETRAIN):
        if mask.any():
            out[mask] = ema(values[mask])
    return out


def curve(ax, kind, key, log, fetch=lf_to_hf, **line):
    epochs, values = across_splits(lambda s: fetch(kind, s), key, log=log)
    if epochs is None:
        return False
    mean, half = band(values)
    back = (lambda v: 10 ** v) if log else (lambda v: v)
    smooth, lo, hi = (phased_ema(epochs, v) for v in (mean, mean - half, mean + half))
    for mask in (epochs < PRETRAIN, epochs >= PRETRAIN):
        if not mask.any():
            continue
        ax.fill_between(epochs[mask], back(lo[mask]), back(hi[mask]), color=line["color"],
                        alpha=0.13 * line.get("alpha", 1.0), lw=0, zorder=1)
        ax.plot(epochs[mask], back(smooth[mask]), **line)
    return epochs, mean


def stage_background(ax):
    ax.axvspan(PRETRAIN - 2.5, 400, color="#f1f1ef", zorder=0, lw=0)
    ax.axvline(PRETRAIN - 2.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.grid(axis="x", visible=False)


def bin_ratios(experiment, prefix, split):
    """Median per-configuration log10(EU/AU) per bin, one final model and split."""
    path = ROOT / f"experiment_{experiment}" / "evaluation" / "cache" / f"split_{split}" / "energy_ood_raw.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    keep = {t: supported("energy", split, f"{prefix}_test_{t}") for t in ("id", "ood")}
    out = []
    for _, lo, hi in BINS:
        values = []
        for row in rows:
            test = row["split"]
            if not keep[test][int(row["config_index"])]:
                continue
            if lo is None:
                if test != "id":
                    continue
            else:
                q = float(row["energy_quantile"])
                if test != "ood" or not (lo <= q < hi or (hi >= 1.0 and q == 1.0)):
                    continue
            values.append(math.log10(float(row["epistemic_var"]) / float(row["aleatoric_var"])))
        out.append(float(np.median(values)))
    return out


LOG_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]


def epoch_axes(ax, title, ylabel, log):
    """Two-stage (pretraining / fine-tuning) epoch axis, shared by Fig. 5 and figA."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    stage_background(ax)
    ax.set_title(title, loc="left", pad=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.set_xlim(0, 400)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(LOG_TICKS))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    trans = ax.get_xaxis_transform()
    ax.text(150, 0.97, "DFT pretraining", transform=trans, ha="center", va="top",
            fontsize=6, color=MUTED)
    ax.text(350, 0.97, "CC fine-tuning", transform=trans, ha="center", va="top",
            fontsize=6, color=MUTED)


def shift_legend_handles(signals=True):
    """Split colours, optionally signal line styles, and the 95% band."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color=SHIFT_COLOR[k], lw=1.6, label=l) for k, l in SHIFTS]
    if signals:
        handles += [Line2D([], [], color=MUTED, lw=1.2, ls=SIGNAL_LINESTYLE[s], label=s)
                    for s in SIGNALS]
    handles.append(Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI, 5 splits"))
    return handles


def main():
    plt = style()
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.1), gridspec_kw={"hspace": 0.5, "wspace": 0.22})
    ax_tr_ause, ax_tr_ence, ax_id_ause, ax_id_ence, ax_ause, ax_ence = axes.flat
    missing = []

    for kind, label in SHIFTS:
        color = SHIFT_COLOR[kind]
        if across_splits(lambda s: lf_to_hf(kind, s), lambda e: e[("AUSE", "TU")])[0] is None:
            missing.append(label)
            continue
        for s in SIGNALS:
            line = dict(color=color, lw=1.2, ls=SIGNAL_LINESTYLE[s])
            cal_line = dict(line, alpha=CAL_ALPHA[s], lw=1.6 if s == "TU" else 1.0)
            for ax_a, ax_e, fetch in ((ax_tr_ause, ax_tr_ence, lf_to_hf_train),
                                      (ax_id_ause, ax_id_ence, lf_to_hf_id),
                                      (ax_ause, ax_ence, lf_to_hf)):
                curve(ax_a, kind, lambda e, s=s: e[("AUSE", s)], False, fetch=fetch, **line)
                curve(ax_e, kind, lambda e, s=s: e[("ENCE", s)], True, fetch=fetch, **cal_line)

    ause_label, ence_label = "AUSE  ←", "ENCE  ←"  # rotated label: "←" renders as a downward arrow
    for ax, title, ylabel, log in ((ax_tr_ause, "Train set: ranking", ause_label, False),
                                   (ax_tr_ence, "Train set: calibration", ence_label, True),
                                   (ax_id_ause, "ID test set: ranking", ause_label, False),
                                   (ax_id_ence, "ID test set: calibration", ence_label, True),
                                   (ax_ause, "OOD test set: ranking", ause_label, False),
                                   (ax_ence, "OOD test set: calibration", ence_label, True)):
        epoch_axes(ax, title, ylabel, log)

    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter, x=-0.01, y=1.0)
    if missing:
        fig.text(0.5, 0.5, "missing per-epoch data: " + ", ".join(missing), ha="center",
                 color="#b00020", fontsize=8)
    fig.legend(handles=shift_legend_handles(), loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 0.955), handlelength=1.8, columnspacing=1.2)
    save(fig, "fig5_decomposition")


if __name__ == "__main__":
    main()
