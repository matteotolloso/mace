#!/usr/bin/env python3
"""Appendix figure - acquisition gains on Energy-ID (formerly Fig. 6b).

RMSE gain over random acquisition (meV/atom) on the unchanged Energy-ID test set,
for each protocol and signal: five-split mean with 95% Student-t interval, grey dots
for the individual splits (paired within each split). Negative values favour random
selection. Same data and drawing code as Fig. 6a (fig6_acquisition.py): the
support-filtered runs/aggregate_epochs_50/summary_ci95.json.
"""

from common import MUTED, legend_handles, save, style
from fig6_acquisition import gain_rows, gains


def main():
    plt = style()
    fig, ax = plt.subplots(figsize=(3.2, 2.35))
    gains(ax, gain_rows(), "energy_id")
    ax.set_ylabel("RMSE gain over random (meV/atom)")
    ax.set_title("Energy-ID", loc="left", pad=3)
    ax.set_ylim(-0.9, 0.3)
    ax.text(0.03, 0.03, "favours random ↓", transform=ax.transAxes, fontsize=6,
            color=MUTED, va="bottom", style="italic")
    fig.legend(handles=legend_handles()[2:], loc="upper center", ncol=3,
               bbox_to_anchor=(0.55, 1.08), handletextpad=0.3, columnspacing=1.0)
    save(fig, "figA_acquisition_id")


if __name__ == "__main__":
    main()
