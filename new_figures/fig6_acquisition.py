#!/usr/bin/env python3
"""Fig. 6 - Uncertainty-guided acquisition pays off, more with pretraining (Finding 4).

(a) RMSE gain over random acquisition (meV/atom) on held-out Energy-OOD, for each
    protocol and signal: five-split mean with 95% Student-t interval, grey dots
    for the individual splits (paired within each split).
(b) Within-system energy quantile q_s(x) of the 500 configurations each strategy
    selected, pooled over the five splits; one row per protocol. Random is shared
    between protocols.

The Energy-ID gains that used to be panel (b) are now the appendix figure
figA_acquisition_id.py, which reuses gain_rows() and gains() from this file.

Data: runs/aggregate_epochs_50/summary_ci95.json, which is support-filtered by
active_learning/ani_energy/support_filter.py, and the per-split selection.json
files. The highest-DFT-energy baseline described in notes/new_figures.md needs new
training runs and is not included.
"""

import json

import numpy as np

from common import (
    MUTED, PROTOCOL_COLOR, PROTOCOLS, ROOT, SIGNAL_COLOR, SIGNAL_MARKER, SIGNALS,
    geometry, panel_label, save, style,
)

RUNS = ROOT / "active_learning" / "ani_energy" / "runs"
REGIME = {"HF-only": "hf_only", "LF→HF": "lf_hf"}
RANDOM_COLOR = "#8c8c8c"


def gain_rows():
    summary = json.loads((RUNS / "aggregate_epochs_50" / "summary_ci95.json").read_text())
    if not summary.get("inputs", {}).get("support_filter"):
        raise RuntimeError("aggregate_epochs_50 is not the support-filtered aggregate; "
                           "run active_learning/ani_energy/support_filter.py --output aggregate_epochs_50")
    return {(r["regime"], r["test"], r["metric"]): r for r in summary["rows"]}


def selected_quantiles(strategy):
    values = []
    for s in range(5):
        run = RUNS / f"split_{s}_epochs_50"
        split = json.loads((run / "data" / "split.json").read_text())
        source = dict(zip(split["pool_ids"], split["pool_source_indices"]))
        ids = json.loads((run / "acquisition" / strategy / "selection.json").read_text())["data"]["ids"]
        quantile = geometry("energy", s, "cc_test_ood")["quantile"]
        values.append(quantile[[source[i] for i in ids]])
    return np.concatenate(values)


def gains(ax, rows, test):
    x = 0.0
    ticks = []
    for p in PROTOCOLS:
        for s in SIGNALS:
            row = rows[(REGIME[p], test, f"{s.lower()}_gain_over_random_meV_per_atom")]
            ax.scatter(x + (np.arange(5) - 2) * 0.07, row["split_values"], s=7, color="#9a9a9a",
                       edgecolors="white", linewidths=0.3, zorder=2)
            ax.errorbar(x, row["mean"], yerr=[[row["mean"] - row["ci95_lower"]],
                                              [row["ci95_upper"] - row["mean"]]],
                        fmt=SIGNAL_MARKER[s], ms=4.4, color=PROTOCOL_COLOR[p], mec="white",
                        mew=0.5, elinewidth=1.1, capsize=0, zorder=3)
            ticks.append((x, s))
            x += 1.0
        x += 0.45
    ax.axhline(0, color="#555555", lw=0.7, zorder=1)
    ax.set_xticks([t for t, _ in ticks], [s for _, s in ticks])
    ax.set_xlim(-0.6, x - 0.85)
    ax.grid(axis="x", visible=False)
    for i, p in enumerate(PROTOCOLS):
        center = np.mean([t for t, _ in ticks[3 * i:3 * i + 3]])
        ax.text(center, -0.13, p, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.8, color=PROTOCOL_COLOR[p], fontweight="bold")


def main():
    plt = style()
    rows = gain_rows()
    fig = plt.figure(figsize=(6.2, 2.45))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.2], wspace=0.3, hspace=0.18)
    ax_ood = fig.add_subplot(grid[:, 0])
    ax_q = [fig.add_subplot(grid[0, 1])]
    ax_q.append(fig.add_subplot(grid[1, 1], sharex=ax_q[0], sharey=ax_q[0]))

    gains(ax_ood, rows, "energy_ood")
    ax_ood.set_ylabel("RMSE gain over random (meV/atom)  →")  # rotated label: "→" renders as an upward arrow
    ax_ood.set_title("Held-out Energy-OOD", loc="left", pad=3)
    ax_ood.set_ylim(-0.3, 3.4)
    ax_ood.text(0.02, 0.97, "favours uncertainty ↑", transform=ax_ood.transAxes, fontsize=6,
                color=MUTED, va="top", style="italic")
    panel_label(ax_ood, "a", x=-0.02, y=1.03)

    bins = np.linspace(0.55, 1.0, 10)
    random_q = selected_quantiles("random")
    for ax, p in zip(ax_q, PROTOCOLS):
        weights = np.full(len(random_q), 1 / len(random_q))
        ax.hist(random_q, bins=bins, histtype="stepfilled", color="#e9e9e7", zorder=1,
                weights=weights, label="random")
        ax.hist(random_q, bins=bins, histtype="step", color=RANDOM_COLOR, lw=0.9, zorder=2,
                weights=weights)
        for s in SIGNALS:
            q = selected_quantiles(f"{REGIME[p]}_{s.lower()}")
            ax.hist(q, bins=bins, histtype="step", color=SIGNAL_COLOR[s], lw=1.3, zorder=3,
                    weights=np.full(len(q), 1 / len(q)), label=s,
                    linestyle=(0, (3, 1)) if s == "EU" else "-")
        ax.text(0.02, 0.93, p, transform=ax.transAxes, fontsize=6.8, color=PROTOCOL_COLOR[p],
                fontweight="bold", va="top")
        ax.set_ylabel("fraction", fontsize=6.6)
        ax.grid(axis="x", visible=False)
    plt.setp(ax_q[0].get_xticklabels(), visible=False)
    ax_q[1].set_xlabel("within-system energy quantile $q_s(x)$")
    ax_q[0].set_title("Selected configurations", loc="left", pad=3)
    ax_q[0].set_xlim(0.55, 1.0)
    handles, labels = ax_q[0].get_legend_handles_labels()
    ax_q[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                   fontsize=6, handlelength=1.4, borderaxespad=0.2, columnspacing=1.0)
    panel_label(ax_q[0], "b", x=-0.02, y=1.06)
    save(fig, "fig6_acquisition")


if __name__ == "__main__":
    main()
