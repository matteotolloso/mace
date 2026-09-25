#!/usr/bin/env python3
"""Appendix reliability diagrams, one figure per split kind and calibration state.

Writes figA_reliability_{system,energy}[_cal]. Rows: LF-only (A/C, scored on DFT
labels), HF-only (E/F), LF->HF (B/D); columns: ID and OOD test set.

Per split, the support-filtered test configurations (common.filtered_rows, the
same rows as the reliability_* aggregates) are sorted by each signal and cut into
the 15 equal-count ENCE bins (eval/reliability.py::build_binned_rows); each bin
gives RMV = sqrt(mean variance) and RMSE = sqrt(mean squared error). For every
bin index, the point is the five-split geometric mean of RMV (x) and RMSE (y); the
band is the 95% log-space Student-t interval of RMSE over the five splits (df=4).
`_cal` uses the isotonic-recalibrated variances (fitted on validation). Log-log,
one common range per figure; dashed diagonal = perfect calibration. The corner
annotation is the five-split geometric-mean ENCE per signal (as in Fig. 2). TU is
drawn at full strength and AU/EU faded (CAL_ALPHA).
"""

import numpy as np

from common import (
    CAL_ALPHA, INK, MUTED, SIGNAL_COLOR, SIGNAL_MARKER, SIGNAL_NAME, SIGNALS, SPLITS, build_binned_rows,
    filtered_rows, final_metrics, interval, panel_label, save, style,
)

NUM_BINS = 15
ROWS = (("LF-only", "LF-only (DFT labels)"), ("HF-only", "HF-only"), ("LF→HF", "LF→HF"))
KINDS = {"system": ({"LF-only": "A", "HF-only": "E", "LF→HF": "B"}, "System"),
         "energy": ({"LF-only": "C", "HF-only": "F", "LF→HF": "D"}, "Energy")}


def binned(experiment, test, calibrated):
    """{signal: (rmv[3 x B], rmse[3 x B])}, rows = center, low, high, in meV/atom."""
    per_split = [build_binned_rows(filtered_rows(experiment, s, test, calibrated), num_bins=NUM_BINS)
                 for s in SPLITS]
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


def top_bin_widths(data):
    """hi/lo ratio of the RMSE interval in the highest-variance bin, per signal."""
    return {s: float(rmse[2, -1] / rmse[1, -1]) for s, (_, rmse) in data.items()}


def draw(kind, calibrated):
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    experiments, label = KINDS[kind]
    tests = (("id", f"{label}-ID"), ("ood", f"{label}-OOD"))
    data = {(p, t): binned(experiments[p], t, calibrated) for p, _ in ROWS for t, _ in tests}
    values = np.concatenate([np.concatenate([rmv[0], rmse[1], rmse[2]])
                             for d in data.values() for rmv, rmse in d.values()])
    lo = 10 ** (np.floor(np.log10(values.min()) * 2) / 2)
    hi = 10 ** (np.ceil(np.log10(values.max()) * 2) / 2)
    ticks = [v for v in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if lo <= v <= hi]

    fig, axes = plt.subplots(3, 2, figsize=(5.5, 7.2), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.16, "wspace": 0.1})
    widths = {}
    for r, (protocol, row_label) in enumerate(ROWS):
        for c, (test, col_label) in enumerate(tests):
            ax = axes[r, c]
            ax.plot((lo, hi), (lo, hi), color=MUTED, lw=0.8, ls="--", zorder=1)
            d = data[(protocol, test)]
            widths[(protocol, test)] = top_bin_widths(d)
            for sig, (rmv, rmse) in d.items():
                tu = sig == "TU"
                ax.fill_between(rmv[0], rmse[1], rmse[2], color=SIGNAL_COLOR[sig],
                                alpha=0.18 * CAL_ALPHA[sig], lw=0, zorder=2)
                ax.plot(rmv[0], rmse[0], color=SIGNAL_COLOR[sig], lw=1.4 if tu else 0.8,
                        marker=SIGNAL_MARKER[sig], ms=3.0 if tu else 2.4, mec="white", mew=0.3,
                        alpha=CAL_ALPHA[sig], zorder=4 if tu else 3)
            ence = final_metrics(experiments[protocol], test, calibrated)
            text = "\n".join(f"ENCE {s} {interval([m[('ENCE', s)] for m in ence], log=True)[0]:.2f}"
                             for s in SIGNALS)
            ax.text(0.97, 0.04, text, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=INK, linespacing=1.25)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(col_label, loc="left", pad=4)
            if c == 0:
                ax.set_ylabel(f"{row_label}\nRMSE (meV/atom)")
            if r == len(ROWS) - 1:
                ax.set_xlabel("RMV (meV/atom)")
    axes[0, 0].set_xlim(lo, hi)
    axes[0, 0].set_ylim(lo, hi)
    for ax in axes.flat:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_aspect("equal")
    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter, x=-0.02, y=1.02)
    handles = [Line2D([], [], color=SIGNAL_COLOR[s], marker=SIGNAL_MARKER[s], ms=3.5, lw=1.0,
                      label=s) for s in SIGNALS]
    handles += [Line2D([], [], color=MUTED, lw=0.8, ls="--", label="RMSE = RMV"),
                Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI, 5 splits")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.95),
               columnspacing=1.2, fontsize=7)
    name = f"figA_reliability_{kind}{'_cal' if calibrated else ''}"
    save(fig, name)
    plt.close(fig)
    return name, widths


def main():
    for kind in KINDS:
        for calibrated in (False, True):
            name, widths = draw(kind, calibrated)
            for (protocol, test), w in widths.items():
                print(f"  {name} {protocol} {test}: top-bin RMSE CI hi/lo "
                      + ", ".join(f"{s} {v:.2f}" for s, v in w.items()))


if __name__ == "__main__":
    main()
