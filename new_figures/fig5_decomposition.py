#!/usr/bin/env python3
"""Fig. 5 - The decomposition as an extrapolation detector (Finding 3).

LF->HF over all training epochs, DFT pretraining (0-299) then CC fine-tuning
(300-399, shaded). Color = system vs energy split, line style = AU dashed,
EU dotted, TU solid.
Row 1, training data (DFT train set while pretraining, CC train set while
fine-tuning): (a) AUSE, (b) ENCE, log y (6 lines each).
Row 2, ID test set (cached curves, see lf_to_hf_id): (c) AUSE, (d) ENCE.
Row 3, OOD test set: (e) AUSE, (f) ENCE, log y.
Row 4: (g) log10(mean EU / mean AU) on the OOD test set (2 lines).
(h) Final models on the energy split: median per-configuration EU/AU in Energy-ID
    and in each within-system energy-quantile bin of Energy-OOD, one line per
    protocol; five-split means with 95% intervals. The median is used because a
    mean of variances is carried by a few configurations: the current appendix
    Fig. 9 shows EU exploding in the top bin for that reason alone.

Lines in (a)-(g) are five-split means (geometric for ENCE and the ratio),
smoothed by an exponential moving average restarted at the fine-tuning boundary so
the transition is not smeared; shaded bands are the 95% Student-t intervals over the
five splits, smoothed the same way (log space for ENCE and the ratio). The ratio uses means rather than the per-
configuration median proposed in new_figures.md: the System-OOD per-epoch caches
store only mean magnitudes, and with the support filter applied the means are no
longer dominated by a few extrapolation configurations.
"""

import csv
import math

import numpy as np

from common import (
    MUTED, PROTOCOL_COLOR, ROOT, SHIFT_COLOR, SIGNAL_LINESTYLE, SIGNALS, SPLITS,
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
        ax.fill_between(epochs[mask], back(lo[mask]), back(hi[mask]), color=line["color"],
                        alpha=0.13, lw=0, zorder=1)
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


def main():
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(4, 2, figsize=(7.0, 8.0), gridspec_kw={"hspace": 0.55, "wspace": 0.22})
    ax_tr_ause, ax_tr_ence, ax_id_ause, ax_id_ence, ax_ause, ax_ence, ax_ratio, ax_q = axes.flat
    missing = []

    for kind, label in SHIFTS:
        color = SHIFT_COLOR[kind]
        if not curve(ax_ratio, kind, lambda e: e["mag_EU"] / e["mag_AU"], True, color=color, lw=1.6):
            missing.append(label)
            continue
        for s in SIGNALS:
            curve(ax_tr_ause, kind, lambda e, s=s: e[("AUSE", s)], False, fetch=lf_to_hf_train,
                  color=color, lw=1.2, ls=SIGNAL_LINESTYLE[s])
            curve(ax_tr_ence, kind, lambda e, s=s: e[("ENCE", s)], True, fetch=lf_to_hf_train,
                  color=color, lw=1.2, ls=SIGNAL_LINESTYLE[s])
            curve(ax_id_ause, kind, lambda e, s=s: e[("AUSE", s)], False, fetch=lf_to_hf_id,
                  color=color, lw=1.2, ls=SIGNAL_LINESTYLE[s])
            curve(ax_id_ence, kind, lambda e, s=s: e[("ENCE", s)], True, fetch=lf_to_hf_id,
                  color=color, lw=1.2, ls=SIGNAL_LINESTYLE[s])
            curve(ax_ause, kind, lambda e, s=s: e[("AUSE", s)], False, color=color, lw=1.2,
                  ls=SIGNAL_LINESTYLE[s])
            curve(ax_ence, kind, lambda e, s=s: e[("ENCE", s)], True, color=color, lw=1.2,
                  ls=SIGNAL_LINESTYLE[s])

    ause_label, ence_label = "AUSE  (lower is better)", "ENCE  (lower is better)"
    for ax, title, ylabel in ((ax_tr_ause, "Train set: ranking", ause_label),
                              (ax_tr_ence, "Train set: calibration", ence_label),
                              (ax_id_ause, "ID test set: ranking", ause_label),
                              (ax_id_ence, "ID test set: calibration", ence_label),
                              (ax_ause, "OOD test set: ranking", ause_label),
                              (ax_ence, "OOD test set: calibration", ence_label),
                              (ax_ratio, "OOD test set: EU / AU", "EU / AU")):
        stage_background(ax)
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 400)
    for ax in (ax_ratio, ax_ence, ax_tr_ence, ax_id_ence):
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_ratio.axhline(1.0, color="#555555", lw=0.7, zorder=1)
    for ax in (ax_tr_ause, ax_tr_ence, ax_id_ause, ax_id_ence, ax_ause, ax_ence, ax_ratio):
        trans = ax.get_xaxis_transform()
        ax.text(150, 0.97, "DFT pretraining", transform=trans, ha="center", va="top",
                fontsize=6, color=MUTED)
        ax.text(350, 0.97, "CC fine-tuning", transform=trans, ha="center", va="top",
                fontsize=6, color=MUTED)
    log_ticks = FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    for ax in (ax_ratio, ax_ence, ax_tr_ence, ax_id_ence):
        ax.yaxis.set_major_locator(log_ticks)

    # (d) final models: EU/AU against depth into the energy tail
    x = np.array([0.0] + [1.4 + i for i in range(len(BINS) - 1)])
    for name, experiment, prefix in FINAL:
        per_split = np.array([bin_ratios(experiment, prefix, s) for s in SPLITS])
        means, halves = zip(*(ci(list(col)) for col in per_split.T))
        means, halves = np.array(means), np.array(halves)
        faint = name == "LF-only"
        color = PROTOCOL_COLOR[name]
        ax_q.plot(x[1:], 10 ** means[1:], color=color, lw=1.0 if faint else 1.5,
                  alpha=0.7 if faint else 1.0, marker="o", ms=3, mec="white", mew=0.4)
        ax_q.errorbar(x, 10 ** means, yerr=[10 ** means - 10 ** (means - halves),
                                            10 ** (means + halves) - 10 ** means],
                      fmt="o", ms=3, color=color, mec="white", mew=0.4, elinewidth=0.9,
                      capsize=0, alpha=0.7 if faint else 1.0)
    ax_q.axhline(1.0, color="#555555", lw=0.7, zorder=1)
    ax_q.axvline(0.7, color=GRID_SEP, lw=0.6)
    ax_q.set_yscale("log")
    ax_q.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2]))
    ax_q.yaxis.set_minor_locator(NullLocator())
    ax_q.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_q.set_xticks(x, [b[0] for b in BINS], fontsize=6.2)
    ax_q.set_xlabel("Energy-ID  |  Energy-OOD by energy quantile $q_s(x)$")
    ax_q.set_ylabel("median EU / AU")
    ax_q.set_title("Final models: EU / AU vs energy depth", loc="left", pad=3)
    ax_q.grid(axis="x", visible=False)

    for ax, letter in zip(axes.flat, "abcdefgh"):
        panel_label(ax, letter, x=-0.01, y=1.0)
    if missing:
        fig.text(0.5, 0.5, "missing per-epoch data: " + ", ".join(missing), ha="center",
                 color="#b00020", fontsize=8)

    handles = [Line2D([], [], color=SHIFT_COLOR[k], lw=1.6, label=l) for k, l in SHIFTS]
    handles += [Line2D([], [], color=MUTED, lw=1.2, ls=SIGNAL_LINESTYLE[s], label=s) for s in SIGNALS]
    handles.append(Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.945),
               handlelength=1.8, columnspacing=1.2)
    ax_q.legend(handles=[Line2D([], [], color=PROTOCOL_COLOR[p], lw=1.5, marker="o", ms=3, label=p)
                         for p, _, _ in FINAL], loc="lower right", fontsize=6, ncol=3,
                columnspacing=0.9, handlelength=1.5)
    save(fig, "fig5_decomposition")


if __name__ == "__main__":
    main()
