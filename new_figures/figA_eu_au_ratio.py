#!/usr/bin/env python3
"""Appendix figure - how large EU is relative to AU (formerly Fig. 5g and 5h).

(a) LF->HF over all training epochs (DFT pretraining 0-299, CC fine-tuning
    300-399, shaded): mean EU / mean AU on the OOD test set, one line per split
    type. Five-split geometric mean smoothed as in Fig. 5, with the 95% interval
    computed in log space. The ratio uses means rather than per-configuration
    medians because the System-OOD per-epoch caches store only mean magnitudes;
    with the support filter applied, the means are no longer dominated by a few
    extrapolation configurations.
(b) Final models on the energy split: median per-configuration EU/AU in Energy-ID
    and in each within-system energy-quantile bin of Energy-OOD, one line per
    protocol (LF-only faint). Five-split means of the per-split log10 median with
    95% Student-t intervals, back-transformed. The median is used because a mean of
    variances is carried by a few configurations.
"""

import numpy as np

from common import PROTOCOL_COLOR, SHIFT_COLOR, SPLITS, ci, panel_label, save, style
from fig5_decomposition import (
    BINS, FINAL, GRID_SEP, SHIFTS, bin_ratios, curve, epoch_axes, shift_legend_handles,
)


def draw_ratio_over_training(ax):
    for kind, _ in SHIFTS:
        curve(ax, kind, lambda e: e["mag_EU"] / e["mag_AU"], True, color=SHIFT_COLOR[kind], lw=1.6)
    epoch_axes(ax, "OOD test set: EU / AU during LF→HF training", "mean EU / mean AU", log=True)
    ax.axhline(1.0, color="#555555", lw=0.7, zorder=1)


def draw_ratio_by_energy(ax):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    x = np.array([0.0] + [1.4 + i for i in range(len(BINS) - 1)])
    for name, experiment, prefix in FINAL:
        per_split = np.array([bin_ratios(experiment, prefix, s) for s in SPLITS])
        means, halves = zip(*(ci(list(col)) for col in per_split.T))
        means, halves = np.array(means), np.array(halves)
        faint = name == "LF-only"
        color = PROTOCOL_COLOR[name]
        ax.plot(x[1:], 10 ** means[1:], color=color, lw=1.0 if faint else 1.5,
                alpha=0.7 if faint else 1.0, marker="o", ms=3, mec="white", mew=0.4)
        ax.errorbar(x, 10 ** means, yerr=[10 ** means - 10 ** (means - halves),
                                          10 ** (means + halves) - 10 ** means],
                    fmt="o", ms=3, color=color, mec="white", mew=0.4, elinewidth=0.9,
                    capsize=0, alpha=0.7 if faint else 1.0)
    ax.axhline(1.0, color="#555555", lw=0.7, zorder=1)
    ax.axvline(0.7, color=GRID_SEP, lw=0.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x, [b[0] for b in BINS], fontsize=6.2)
    ax.set_xlabel("Energy-ID  |  Energy-OOD by quantile $q_s(x)$")
    ax.set_ylabel("median EU / AU")
    ax.set_title("Final models: EU / AU vs energy depth", loc="left", pad=3)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], color=PROTOCOL_COLOR[p], lw=1.5, marker="o", ms=3, label=p)
                       for p, _, _ in FINAL], loc="lower right", fontsize=6, ncol=3,
              columnspacing=0.9, handlelength=1.5)


def main():
    plt = style()
    fig, (ax_ratio, ax_q) = plt.subplots(1, 2, figsize=(7.0, 2.35),
                                         gridspec_kw={"wspace": 0.24})
    draw_ratio_over_training(ax_ratio)
    draw_ratio_by_energy(ax_q)
    for ax, letter in zip((ax_ratio, ax_q), "ab"):
        panel_label(ax, letter, x=-0.01, y=1.0)
    fig.legend(handles=shift_legend_handles(signals=False), loc="upper center", ncol=3,
               bbox_to_anchor=(0.3, 1.1), handlelength=1.8, columnspacing=1.2)
    save(fig, "figA_eu_au_ratio")


if __name__ == "__main__":
    main()
