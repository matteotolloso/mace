#!/usr/bin/env python3
"""Fig. 4 - Ranking vs. calibration under shift (Finding 2).

(a) ENCE (log x) vs AUSE plane, one sub-panel per shift type. For each protocol an
    arrow moves TU from ID to OOD; AU and EU are small hollow markers joined by a
    thin line from their ID to their OOD position. Lower-left is better on both.
(b) ENCE before and after isotonic recalibration (fitted on the validation set),
    per protocol and signal, on System-OOD and Energy-OOD. Drawn as dumbbells
    rather than bars: bars on a log axis have no meaningful baseline.

All values are five-split geometric means (ENCE) or arithmetic means (AUSE) of
metrics recomputed from per-configuration predictions with the training-support
filter (see common.py). The calibrated Energy-OOD values replace the corrupted
ones of the current appendix figure (RMSE 1.70 eV/atom, ENCE ~4700).
"""

import math

import numpy as np

from common import (
    EXPERIMENT, INK, MUTED, PROTOCOL_COLOR, PROTOCOLS, SIGNAL_MARKER, SIGNALS,
    ci, final_metrics, panel_label, save, style,
)


def gmean(values):
    return math.exp(float(np.mean(np.log(values))))


def gci(values):
    m, h = ci([math.log(v) for v in values])
    return math.exp(m), math.exp(m - h), math.exp(m + h)


def point(kind, protocol, test, signal, calibrated=False):
    per_split = final_metrics(EXPERIMENT[(kind, protocol)], test, calibrated)
    return (gmean([m[("ENCE", signal)] for m in per_split]),
            float(np.mean([m[("AUSE", signal)] for m in per_split])))


def main():
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig = plt.figure(figsize=(7.0, 2.55))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.62, 0.78, 0.78], wspace=0.08)
    plane = [fig.add_subplot(grid[0, 0])]
    plane.append(fig.add_subplot(grid[0, 1], sharex=plane[0], sharey=plane[0]))
    bells = [fig.add_subplot(grid[0, 3])]
    bells.append(fig.add_subplot(grid[0, 4], sharey=bells[0]))

    # (a) ID -> OOD movement in the ranking/calibration plane
    for ax, (kind, title) in zip(plane, (("system", "System shift"), ("energy", "Energy shift"))):
        for p in PROTOCOLS:
            color = PROTOCOL_COLOR[p]
            for s in ("AU", "EU"):
                (x0, y0), (x1, y1) = point(kind, p, "id", s), point(kind, p, "ood", s)
                ax.plot([x0, x1], [y0, y1], color=color, lw=0.6, alpha=0.45, zorder=2)
                ax.plot([x0], [y0], SIGNAL_MARKER[s], ms=3.4, mfc="white", mec=color, mew=0.7,
                        alpha=0.55, zorder=3)
                ax.plot([x1], [y1], SIGNAL_MARKER[s], ms=3.8, mfc="white", mec=color, mew=0.9,
                        zorder=3)
            (x0, y0), (x1, y1) = point(kind, p, "id", "TU"), point(kind, p, "ood", "TU")
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                        color=color, lw=1.5, shrinkA=0, shrinkB=1.5))
            ax.plot([x0], [y0], "D", ms=3.6, color=color, mec="white", mew=0.5, zorder=5)
            ax.text(x0, y0, " ID", fontsize=5.8, color=color, ha="right", va="bottom", zorder=5)
            ax.text(x1, y1, " OOD", fontsize=5.8, color=color, ha="left", va="top", zorder=5)
        ax.set_title(title, loc="left", pad=11)
        ax.set_xscale("log")
        ax.set_xlabel("ENCE  ↓ (calibration)")
    plane[0].set_ylabel("AUSE, ranking (lower is better)")
    plt.setp(plane[1].get_yticklabels(), visible=False)
    plane[0].set_xlim(0.15, 25)
    plane[0].set_ylim(0.1, 0.42)
    for ax in plane:
        ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.text(0.03, 0.03, "better", transform=ax.transAxes, fontsize=6, color=MUTED,
                ha="left", va="bottom", style="italic")
    panel_label(plane[0], "a", x=-0.02, y=1.06)

    # (b) uncalibrated -> isotonic-recalibrated ENCE
    rows = [(p, s) for p in PROTOCOLS for s in SIGNALS]
    ys = {}
    y = 0.0
    for p in PROTOCOLS:
        for s in SIGNALS:
            ys[(p, s)] = y
            y -= 1.0
        y -= 0.6
    for ax, (kind, title) in zip(bells, (("system", "System-OOD"), ("energy", "Energy-OOD"))):
        for p, s in rows:
            before = gci([m[("ENCE", s)] for m in final_metrics(EXPERIMENT[(kind, p)], "ood")])
            after = gci([m[("ENCE", s)] for m in final_metrics(EXPERIMENT[(kind, p)], "ood", True)])
            yy = ys[(p, s)]
            color = PROTOCOL_COLOR[p]
            ax.annotate("", xy=(after[0], yy), xytext=(before[0], yy), zorder=2,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                        color=color, lw=0.9, alpha=0.8, shrinkA=2.5, shrinkB=2.5))
            ax.plot([before[0]], [yy], SIGNAL_MARKER[s], ms=3.8, mfc="white", mec=color, mew=0.9,
                    zorder=3)
            ax.plot([after[0]], [yy], SIGNAL_MARKER[s], ms=3.8, color=color, mec="white", mew=0.5,
                    zorder=3)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", pad=11)
        ax.set_xlabel("ENCE  ↓")
        if kind == "system":
            ax.text(0.0, 1.0, "hollow: uncalibrated  →  filled: isotonic", transform=ax.transAxes,
                    fontsize=5.6, color=MUTED, va="bottom", ha="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bells[0].xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2]))
    bells[0].set_xlim(0.15, 3.5)
    bells[1].xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    bells[1].set_xlim(1.0, 25)
    bells[0].set_yticks([ys[r] for r in rows], [f"{p}  {s}" for p, s in rows], fontsize=6.4)
    for tick, (p, _) in zip(bells[0].get_yticklabels(), rows):
        tick.set_color(PROTOCOL_COLOR[p])
    plt.setp(bells[1].get_yticklabels(), visible=False)
    bells[0].set_ylim(min(ys.values()) - 0.7, 0.7)
    panel_label(bells[0], "b", x=-0.02, y=1.06)

    handles = [Line2D([], [], color=PROTOCOL_COLOR[p], lw=1.5, label=p) for p in PROTOCOLS]
    handles += [Line2D([], [], color=MUTED, marker=SIGNAL_MARKER[s], ls="", ms=3.8, label=s)
                for s in SIGNALS]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.09),
               handletextpad=0.35, columnspacing=1.1)
    save(fig, "fig4_rank_vs_calibration")


if __name__ == "__main__":
    main()
