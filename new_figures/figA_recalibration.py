#!/usr/bin/env python3
"""Appendix figure - ENCE before and after isotonic recalibration (formerly Fig. 4b).

ENCE on System-OOD (a) and Energy-OOD (b) for each protocol and signal: hollow
marker = uncalibrated, filled marker = after isotonic recalibration fitted on the
validation set, arrow from the first to the second. Drawn as dumbbells rather
than bars because bars on a log axis have no meaningful baseline. Values are
five-split geometric means of ENCE recomputed from per-configuration predictions
with the training-support filter (reliability_ood_{nocal,cal}_raw.csv, see
common.py).
"""

from common import (
    EXPERIMENT, MUTED, PROTOCOL_COLOR, PROTOCOLS, SIGNAL_MARKER, SIGNALS, final_metrics,
    interval, legend_handles, panel_label, save, style,
)

SHIFTS = (("system", "System-OOD"), ("energy", "Energy-OOD"))


def main():
    plt = style()
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, bells = plt.subplots(1, 2, figsize=(4.6, 2.45), sharey=True,
                              gridspec_kw={"wspace": 0.08})
    rows = [(p, s) for p in PROTOCOLS for s in SIGNALS]
    ys = {}
    y = 0.0
    for p in PROTOCOLS:
        for s in SIGNALS:
            ys[(p, s)] = y
            y -= 1.0
        y -= 0.6

    for ax, (kind, title) in zip(bells, SHIFTS):
        for p, s in rows:
            experiment = EXPERIMENT[(kind, p)]
            before = interval([m[("ENCE", s)] for m in final_metrics(experiment, "ood")], log=True)
            after = interval([m[("ENCE", s)] for m in final_metrics(experiment, "ood", True)],
                             log=True)
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
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bells[0].text(0.0, 1.0, "hollow: uncalibrated  →  filled: isotonic", transform=bells[0].transAxes,
                  fontsize=5.6, color=MUTED, va="bottom", ha="left")
    bells[0].xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2]))
    bells[0].set_xlim(0.15, 3.5)
    bells[1].xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    bells[1].set_xlim(1.0, 25)
    bells[0].set_yticks([ys[r] for r in rows], [f"{p}  {s}" for p, s in rows], fontsize=6.4)
    for tick, (p, _) in zip(bells[0].get_yticklabels(), rows):
        tick.set_color(PROTOCOL_COLOR[p])
    bells[0].set_ylim(min(ys.values()) - 0.7, 0.7)
    for ax, letter in zip(bells, "ab"):
        panel_label(ax, letter, x=-0.02, y=1.06)

    fig.legend(handles=legend_handles(protocol_marker=False)[len(PROTOCOLS):], loc="upper center",
               ncol=3, bbox_to_anchor=(0.6, 1.1), handletextpad=0.35, columnspacing=1.1)
    save(fig, "figA_recalibration")


if __name__ == "__main__":
    main()
