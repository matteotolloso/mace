#!/usr/bin/env python3
"""Water Fig. W1 - Final-model summary on the CCSDT test set.

Panels (a-c): RMSE (log x), AUSE, ENCE (log x), as in Fig. 2. Rows: HF-only (wC)
and LF->HF (wB); within each row one point per signal (marker). Points are
five-split means with 95% Student-t intervals; log-scaled columns use geometric
means with log-space intervals. In the ENCE panel TU is at full strength, AU/EU
faded (CAL_ALPHA). LF-only (wA) is omitted: it is scored on BLYP
labels and a different test file, so it is not comparable. No support filter.
See water_data.py for the caveats (split misalignment favours LF->HF).
"""

import numpy as np

from common import (
    CAL_ALPHA,
    INK, PROTOCOL_COLOR, PROTOCOLS, SIGNAL_MARKER, SIGNALS, interval, legend_handles,
    panel_label, save, style,
)
from water_data import final_metrics_water

COLUMNS = (("RMSE", "RMSE (meV/atom)  ←", True), ("AUSE", "AUSE  ←", False),
           ("ENCE", "ENCE  ←", True))


def main():
    plt = style()
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.75), sharey=True,
                             gridspec_kw={"width_ratios": [0.95, 1, 1.2], "wspace": 0.12})
    slot, centers = {}, []
    y = 0.0
    for p in PROTOCOLS:
        ys = []
        for s in SIGNALS:
            slot[(p, s)] = y
            ys.append(y)
            y -= 0.42
        centers.append((p, float(np.mean(ys))))
        y -= 0.45

    for ax, (metric, title, log) in zip(axes, COLUMNS):
        for p in PROTOCOLS:
            per_split = final_metrics_water(p)
            if metric == "RMSE":
                c, lo, hi = interval([m["RMSE"] for m in per_split], log)
                ax.errorbar(c, dict(centers)[p], xerr=[[c - lo], [hi - c]], fmt="o", ms=4.2,
                            color=PROTOCOL_COLOR[p], mec="white", mew=0.5, elinewidth=1.0,
                            capsize=0, zorder=3)
                continue
            for s in SIGNALS:
                c, lo, hi = interval([m[(metric, s)] for m in per_split], log)
                ax.errorbar(c, slot[(p, s)], xerr=[[c - lo], [hi - c]], fmt=SIGNAL_MARKER[s], alpha=CAL_ALPHA[s] if metric == "ENCE" else 1.0,
                            ms=4.3 if metric == "ENCE" and s == "TU" else 3.6, color=PROTOCOL_COLOR[p], mec="white", mew=0.5,
                            elinewidth=1.0, capsize=0, zorder=3)
        ax.set_title(title, loc="left", fontsize=7.8, color=INK, pad=4)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks([c for _, c in centers], [p for p, _ in centers], fontsize=7.4)
    axes[0].set_ylim(y + 0.2, 0.35)
    axes[0].xaxis.set_major_locator(FixedLocator([5, 7, 10, 15]))
    axes[0].set_xlim(5, 16)
    axes[2].xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20, 50]))
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, x=0.0, y=1.1)

    fig.legend(handles=legend_handles(), loc="upper center", ncol=5, bbox_to_anchor=(0.52, 1.2),
               handletextpad=0.3, columnspacing=1.3)
    save(fig, "figW_final_summary")


if __name__ == "__main__":
    main()
