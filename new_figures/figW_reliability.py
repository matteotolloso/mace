#!/usr/bin/env python3
"""Water Fig. W2 - Reliability diagrams on the CCSDT test set.

(a) HF-only (wC), (b) LF->HF (wB). Per split, the 50 test configurations are
sorted by each signal and cut into the 15 equal-count bins that ENCE uses
(eval/reliability.py::build_binned_rows); each bin gives RMV = sqrt(mean
variance) and RMSE = sqrt(mean squared error). Points are five-split geometric
means of RMV and RMSE per bin index, with 95% log-space Student-t intervals on
RMSE (vertical bars; the RMV intervals are narrower and are not drawn, to
reduce clutter). Colour = signal. Diagonal = perfect
calibration; points above it are overconfident. Per-atom, meV/atom; log-log.
With ~3 configurations per bin, individual bins are noisy. TU is drawn at full
strength and AU/EU faded (CAL_ALPHA).
"""

import numpy as np

from common import (
    CAL_ALPHA, INK, MUTED, SIGNAL_COLOR, SIGNAL_MARKER, SIGNAL_NAME, SIGNALS, SPLITS, build_binned_rows,
    interval, panel_label, save, style,
)
from water_data import test_rows

PANELS = (("HF-only", "HF-only (wC)"), ("LF→HF", "LF→HF (wB)"))
NUM_BINS = 15


def binned(protocol):
    """{signal: (rmv[3 x B], rmse[3 x B])} with rows = center, low, high (meV/atom)."""
    per_split = [build_binned_rows(test_rows(protocol, s), num_bins=NUM_BINS) for s in SPLITS]
    out = {}
    for sig in SIGNALS:
        name = SIGNAL_NAME[sig]
        rmv, rmse = [], []
        for b in range(NUM_BINS):
            rows = [next(r for r in bins if r["uncertainty_type"] == name and r["bin_index"] == b)
                    for bins in per_split]
            rmv.append(interval([1000 * r["rmv"] for r in rows], log=True))
            rmse.append(interval([1000 * r["rmse"] for r in rows], log=True))
        out[sig] = (np.array(rmv).T, np.array(rmse).T)
    return out


def main():
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.6), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.14})
    lims = (0.3, 100)
    for ax, (protocol, title) in zip(axes, PANELS):
        ax.plot(lims, lims, color=MUTED, lw=0.8, ls="--", zorder=1)
        for sig, (rmv, rmse) in binned(protocol).items():
            ax.errorbar(rmv[0], rmse[0], yerr=[rmse[0] - rmse[1], rmse[2] - rmse[0]], fmt=SIGNAL_MARKER[sig],
                        ms=3.8 if sig == "TU" else 3.0, color=SIGNAL_COLOR[sig], mec="white",
                        mew=0.4, elinewidth=0.6, alpha=CAL_ALPHA[sig], capsize=0,
                        zorder=4 if sig == "TU" else 3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", pad=4, color=INK)
        ax.set_xlabel("RMV (meV/atom)")
        ax.text(0.97, 0.04, "underconfident", transform=ax.transAxes, fontsize=6, color=MUTED,
                ha="right", va="bottom", style="italic")
        ax.text(0.04, 0.96, "overconfident", transform=ax.transAxes, fontsize=6, color=MUTED,
                ha="left", va="top", style="italic")
    axes[0].set_ylabel("RMSE (meV/atom)")
    axes[0].set_xlim(*lims)
    axes[0].set_ylim(*lims)
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([1, 3, 10, 30]))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter, x=-0.02, y=1.02)
    handles = [Line2D([], [], color=SIGNAL_COLOR[s], marker=SIGNAL_MARKER[s], ls="", ms=4,
                      label=s) for s in SIGNALS]
    handles.append(Line2D([], [], color=MUTED, lw=0.8, ls="--", label="RMSE = RMV"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.3)
    save(fig, "figW_reliability")


if __name__ == "__main__":
    main()
