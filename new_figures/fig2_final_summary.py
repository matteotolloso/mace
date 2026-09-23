#!/usr/bin/env python3
"""Fig. 2 - Final-model summary (Finding 1).

Horizontal dot plot. Rows: System-ID, System-OOD, Energy-ID, Energy-OOD.
Columns (panels a-c): RMSE (down, log x), AUSE (down), ENCE (down, log x).
Spearman is not drawn: it tells the same ranking story as AUSE; the paper quotes
it in the text (see paper_numbers.py).
Color = training protocol, marker = uncertainty signal. Points are five-split
means with 95% Student-t intervals; log-scaled columns use geometric means with
log-space intervals so bounds stay positive. LF-only is omitted: it is evaluated
on DFT labels and is not comparable with the two CC-trained protocols.
"""

import numpy as np

from common import (
    EXPERIMENT, INK, PROTOCOL_COLOR, PROTOCOLS, SIGNAL_MARKER, SIGNALS, TESTS,
    final_metrics, interval, legend_handles, panel_label, save, style,
)

COLUMNS = (("RMSE", "RMSE (meV/atom)  ↓", True), ("AUSE", "AUSE  ↓", False),
           ("ENCE", "ENCE  ↓", True))


def main():
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [0.95, 1, 1.2], "wspace": 0.12})
    # y layout: each test set is a band; inside it, one slot per protocol x signal.
    slot = {}
    band_centers = []
    y = 0.0
    for t_index, (_, _, label) in enumerate(TESTS):
        ys = []
        for p in PROTOCOLS:
            for s in SIGNALS:
                slot[(label, p, s)] = y
                ys.append(y)
                y -= 0.42
            y -= 0.25
        band_centers.append((label, np.mean(ys), ys[0] + 0.3, ys[-1] - 0.3))
        y -= 0.45

    for ax, (metric, title, log) in zip(axes, COLUMNS):
        for i, (_, top, bottom) in enumerate((b[0], b[2], b[3]) for b in band_centers):
            if i % 2 == 0:
                ax.axhspan(bottom, top, color="#f4f4f2", zorder=0, lw=0)
        for kind, test, label in TESTS:
            for p in PROTOCOLS:
                per_split = final_metrics(EXPERIMENT[(kind, p)], test)
                if metric == "RMSE":
                    ys = [slot[(label, p, s)] for s in SIGNALS]
                    y0 = float(np.mean(ys))
                    c, lo, hi = interval([m["RMSE"] for m in per_split], log)
                    ax.errorbar(c, y0, xerr=[[c - lo], [hi - c]], fmt="o", ms=4.2,
                                color=PROTOCOL_COLOR[p], mec="white", mew=0.5,
                                elinewidth=1.0, capsize=0, zorder=3)
                    continue
                for s in SIGNALS:
                    c, lo, hi = interval([m[(metric, s)] for m in per_split], log)
                    ax.errorbar(c, slot[(label, p, s)], xerr=[[c - lo], [hi - c]],
                                fmt=SIGNAL_MARKER[s], ms=3.6, color=PROTOCOL_COLOR[p],
                                mec="white", mew=0.5, elinewidth=1.0, capsize=0, zorder=3)
        ax.set_title(title, loc="left", fontsize=7.8, color=INK, pad=4)
        if log:
            ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks([c for _, c, _, _ in band_centers], [l for l, _, _, _ in band_centers],
                       fontsize=7.4)
    axes[0].set_ylim(band_centers[-1][3] - 0.1, band_centers[0][2] + 0.1)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    for ax, ticks in ((axes[2], [0.2, 0.5, 1, 2, 5, 10]), (axes[0], [5, 10, 20, 50, 100])):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_xlim(4.5, 110)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, x=0.0, y=1.075)

    fig.legend(handles=legend_handles(), loc="upper center", ncol=5, bbox_to_anchor=(0.52, 1.07),
               handletextpad=0.3, columnspacing=1.3)
    save(fig, "fig2_final_summary")


if __name__ == "__main__":
    main()
