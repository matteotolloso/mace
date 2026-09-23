#!/usr/bin/env python3
"""Fig. 4 - Ranking vs. calibration under shift (Finding 2).

ENCE (log x) vs AUSE plane: (a) system shift, (b) energy shift. Every signal
(AU, EU, TU) is drawn the same way: a hollow marker at its ID position, a filled
marker at its OOD position, joined by a thin line. Lower-left is better on both.
The isotonic-recalibration dumbbells that used to be panel (b) are now the
appendix figure figA_recalibration.py.

All values are five-split geometric means (ENCE) or arithmetic means (AUSE) of
metrics recomputed from per-configuration predictions with the training-support
filter (see common.py). """

import numpy as np

from common import (
    EXPERIMENT, MUTED, PROTOCOL_COLOR, PROTOCOLS, SIGNAL_MARKER, SIGNALS, final_metrics,
    interval, legend_handles, panel_label, save, style,
)


def point(kind, protocol, test, signal, calibrated=False):
    per_split = final_metrics(EXPERIMENT[(kind, protocol)], test, calibrated)
    return (interval([m[("ENCE", signal)] for m in per_split], log=True)[0],
            float(np.mean([m[("AUSE", signal)] for m in per_split])))


def main():
    plt = style()
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig = plt.figure(figsize=(5.0, 2.65))
    grid = fig.add_gridspec(1, 2, wspace=0.08)
    plane = [fig.add_subplot(grid[0, 0])]
    plane.append(fig.add_subplot(grid[0, 1], sharex=plane[0], sharey=plane[0]))

    # ID -> OOD movement in the ranking/calibration plane
    for ax, (kind, title) in zip(plane, (("system", "System shift"), ("energy", "Energy shift"))):
        for p in PROTOCOLS:
            color = PROTOCOL_COLOR[p]
            for s in SIGNALS:
                (x0, y0), (x1, y1) = point(kind, p, "id", s), point(kind, p, "ood", s)
                ax.plot([x0, x1], [y0, y1], color=color, lw=0.8, alpha=0.5, zorder=2)
                ax.plot([x0], [y0], SIGNAL_MARKER[s], ms=4.0, mfc="white", mec=color, mew=0.9,
                        zorder=3)
                ax.plot([x1], [y1], SIGNAL_MARKER[s], ms=4.0, color=color, mec=color, mew=0.9,
                        zorder=3)
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
    for ax, letter in zip(plane, "ab"):
        panel_label(ax, letter, x=-0.02, y=1.06)

    from matplotlib.lines import Line2D
    handles = legend_handles(protocol_marker=False) + [
        Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=MUTED, mew=0.9, label="ID"),
        Line2D([], [], marker="o", ls="", ms=4, color=MUTED, mew=0.9, label="OOD"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.09),
               handletextpad=0.35, columnspacing=1.1)
    save(fig, "fig4_rank_vs_calibration")


if __name__ == "__main__":
    main()
