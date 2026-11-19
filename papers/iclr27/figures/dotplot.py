"""Fig. 2-style horizontal dot plot, shared by the appendix figures of the revision.

Rows: the four test sets (bands); inside a band, one slot per (series, signal).
Columns: RMSE (meV/atom, log x), AUSE (linear), ENCE (log x); lower is better.
Colour = series (e.g. training protocol), marker = uncertainty signal; in the ENCE
column TU is drawn at full strength and AU/EU faded (CAL_ALPHA), as in Fig. 2.
The caller supplies five-split (center, low, high) triples; a missing value (None)
is drawn as "pending" text in its slot, never filled in; False leaves the slot empty.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from common import CAL_ALPHA, INK, MUTED, SIGNAL_MARKER, TESTS, panel_label

Triple = Tuple[float, float, float]
COLUMNS = (("RMSE", "RMSE (meV/atom)  ←", True), ("AUSE", "AUSE  ← (ranking)", False),
           ("ENCE", "ENCE  ← (calibration)", True))


def draw(plt, series: Sequence[Tuple[str, str]], signals: Callable[[str], Sequence[str]],
         value: Callable[[str, str, Optional[str], str], Optional[Triple]], hollow=lambda key, signal: False,
         marker=lambda signal: SIGNAL_MARKER[signal], fade=lambda signal: CAL_ALPHA[signal], rmse_per_signal=False,
         figsize=(7.0, 2.9), rmse_ticks=(5, 10, 20, 50, 100), ence_ticks=(0.2, 0.5, 1, 2, 5, 10),
         highlight=lambda key, signal: False, highlight_color="#fbecc4"):
    """series: (key, colour); signals(key): signals drawn for that series; value(test, key, signal, metric),
    signal None for RMSE unless rmse_per_signal (then one RMSE slot per signal); slots with
    highlight(key, signal) get a tinted strip behind them in every column. Returns (fig, axes)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True,
                             gridspec_kw={"width_ratios": [0.95, 1, 1.2], "wspace": 0.12})
    slot, bands = {}, []
    y = 0.0
    for _, _, label in TESTS:
        ys = []
        for key, _ in series:
            for s in signals(key):
                slot[(label, key, s)] = y
                ys.append(y)
                y -= 0.42
            y -= 0.25
        bands.append((label, float(np.mean(ys)), ys[0] + 0.3, ys[-1] - 0.3))
        y -= 0.45

    for ax, (metric, title, log) in zip(axes, COLUMNS):
        for i, (_, _, top, bottom) in enumerate(bands):
            if i % 2 == 0:
                ax.axhspan(bottom, top, color="#f4f4f2", zorder=0, lw=0)
        for (_, key, s), yy in slot.items():
            if highlight(key, s):
                ax.axhspan(yy - 0.21, yy + 0.21, color=highlight_color, zorder=0.5, lw=0)
        pending = set()
        for _, _, label in TESTS:
            for key, colour in series:
                if metric == "RMSE" and not rmse_per_signal:
                    points = [(float(np.mean([slot[(label, key, s)] for s in signals(key)])), None)]
                else:
                    points = [(slot[(label, key, s)], s) for s in signals(key)]
                for yy, s in points:
                    v = value(label, key, s, metric)
                    if v is False:
                        continue
                    if v is None:  # one label per series and test, at the centre of its slots
                        if (label, key) not in pending:
                            pending.add((label, key))
                            yc = float(np.mean([slot[(label, key, t)] for t in signals(key)]))
                            ax.text(0.5, yc, "pending", transform=ax.get_yaxis_transform(), ha="center",
                                    va="center", fontsize=6.5, color=MUTED, style="italic")
                        continue
                    c, lo, hi = v
                    open_marker = s is not None and hollow(key, s)
                    alpha = fade(s) if (metric == "ENCE" and s is not None) else 1.0
                    ax.errorbar(c, yy, xerr=[[c - lo], [hi - c]], fmt="o" if s is None else marker(s), alpha=alpha,
                                ms=4.2 if s is None else (4.3 if metric == "ENCE" and alpha == 1.0 else 3.6),
                                color=colour, mfc="white" if open_marker else colour,
                                mec=colour if open_marker else "white", mew=0.9 if open_marker else 0.5,
                                elinewidth=1.0, capsize=0, zorder=3)
        ax.set_title(title, loc="left", fontsize=7.8, color=INK, pad=4)
        if log:
            ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks([c for _, c, _, _ in bands], [l for l, _, _, _ in bands], fontsize=7.4)
    axes[0].set_ylim(bands[-1][3] - 0.1, bands[0][2] + 0.1)
    for ax, ticks in ((axes[2], ence_ticks), (axes[0], rmse_ticks)):
        ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, x=0.0, y=1.075)
    return fig, axes
