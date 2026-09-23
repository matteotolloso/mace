#!/usr/bin/env python3
"""Fig. 3 - Pretraining stabilizes HF uncertainty (Finding 1, dynamics).

2x2 grid. Columns: System-OOD, Energy-OOD. Rows: AUSE and ENCE (log y). Six
lines per panel: colour = protocol, HF-only (trained on CC from scratch) or
LF->HF (CC fine-tuning of the DFT-pretrained model); line style = signal, AU
dashed, EU dotted, TU solid (as in Fig. 5). x = HF-stage epoch starting at 0,
evaluated every 5th epoch. Lines: exponential moving average of the five-split
mean (AUSE arithmetic, ENCE geometric). Shaded bands: 95% Student-t interval over
the five splits (log space for ENCE), smoothed the same way. The unsmoothed means
are no longer drawn, to keep six lines readable.

Energy-OOD uses per-configuration predictions restricted to the training-support
region; System-OOD uses the existing per-epoch caches, which the filter does not
change (see epoch_data.py).
"""

import numpy as np

from common import (
    MUTED, PROTOCOL_COLOR, PROTOCOLS, SIGNAL_LINESTYLE, SIGNALS, panel_label, save, style,
)
from epoch_data import across_splits, band, ema, hf_stage

ROWS = (("AUSE", "AUSE  (lower is better)", False), ("ENCE", "ENCE  (lower is better)", True))
COLUMNS = (("system", "System-OOD"), ("energy", "Energy-OOD"))


def main():
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    missing = []
    for c, (kind, title) in enumerate(COLUMNS):
        for r, (metric, ylabel, log) in enumerate(ROWS):
            ax = axes[r, c]
            for p in PROTOCOLS:
                for s in SIGNALS:
                    epochs, values = across_splits(lambda x: hf_stage(kind, p, x),
                                                   lambda e, s=s: e[(metric, s)], log=log)
                    if epochs is None:
                        missing.append(f"{title} {p}")
                        break
                    mean, half = band(values)
                    transform = (lambda v: 10 ** v) if log else (lambda v: v)
                    ax.fill_between(epochs, transform(ema(mean - half)),
                                    transform(ema(mean + half)), color=PROTOCOL_COLOR[p],
                                    alpha=0.10, lw=0)
                    ax.plot(epochs, transform(ema(mean)), color=PROTOCOL_COLOR[p],
                            lw=1.6 if s == "TU" else 1.2, ls=SIGNAL_LINESTYLE[s])
            if log:
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(title, loc="left", pad=3)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel("HF-stage epoch")
            ax.grid(axis="x", visible=False)
    axes[1, 0].yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]))
    axes[1, 0].yaxis.set_minor_locator(NullLocator())
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0, 0].set_xlim(-5, 300)
    for ax, letter in zip(axes.flat, "abcd"):
        panel_label(ax, letter, x=-0.01, y=1.0)
    if missing:
        fig.text(0.5, 0.5, "missing per-epoch data:\n" + "\n".join(missing), ha="center",
                 va="center", color="#b00020", fontsize=8)
    handles = [Line2D([], [], color=PROTOCOL_COLOR[p], lw=1.6, label=p) for p in PROTOCOLS]
    handles += [Line2D([], [], color=MUTED, lw=1.2, ls=SIGNAL_LINESTYLE[s], label=s)
                for s in SIGNALS]
    handles.append(Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI, 5 splits"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.8, columnspacing=1.2)
    save(fig, "fig3_hf_dynamics")


if __name__ == "__main__":
    main()
