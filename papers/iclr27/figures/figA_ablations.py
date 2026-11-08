#!/usr/bin/env python3
"""Appendix figure (rebuttal item C) - fine-tuning ablations of the AU collapse.

Forest plot of paired contrasts between each LF->HF ablation and its reference
(B on the system split, D on the energy split): one row group per ablation
(variance-head reset, beta-NLL, FT lr 1e-4, FT lr 1e-2), one row per test set.
Columns: RMSE ratio, AU AUSE difference, AU ENCE ratio, TU ENCE ratio of the
final, uncalibrated, support-filtered ensembles. Centers and 95% Student-t
intervals (df = 4) over the five paired splits, as computed by
``python -m uq.evaluation.compare_ablations`` (ratios in log space); this script
only reads ``results/tables/ablations_vs_reference.csv``. A pair that is missing
there is drawn as "pending", never filled in.
"""

import csv
from collections import namedtuple

from common import MUTED, ROOT, SHIFT_COLOR, TESTS, panel_label, save, style

TABLE = ROOT / "results" / "tables" / "ablations_vs_reference.csv"
ABLATIONS = (("lf-to-hf-reset-var", "variance-head reset"), ("lf-to-hf-beta-nll", "β-NLL (β = 0.5)"),
             ("lf-to-hf-lr1e-4", "FT lr 1e-4"), ("lf-to-hf-lr1e-2", "FT lr 1e-2"))
# (metric, uncertainty_type, column title, ticks of the log axis or None for a linear axis)
COLUMNS = (("rmse", "ensemble", "RMSE ratio ↓", (0.9, 1, 1.1, 1.2)),
           ("ause", "aleatoric", "AU AUSE diff. ↓", None),
           ("ence", "aleatoric", "AU ENCE ratio ↓", (0.25, 0.5, 1, 2)),
           ("ence", "total", "TU ENCE ratio ↓", (0.25, 0.5, 1, 2)))
Contrast = namedtuple("Contrast", "mean low high")


def load():
    out = {}
    with TABLE.open() as handle:
        for row in csv.DictReader(handle):
            if not row["ablation"] or row["missing"]:
                continue
            key = (row["ablation"], row["test"], row["metric"], row["uncertainty_type"])
            out[key] = Contrast(*(float(row[f"contrast_{k}"]) for k in ("mean", "ci95_low", "ci95_high")))
    return out


def main():
    plt = style()
    data = load()
    rows = [(a, label, kind, test, name) for a, label in ABLATIONS for kind, test, name in TESTS]
    gap = 0.9
    y = {}
    pos = 0.0
    for i, (a, _, kind, test, _) in enumerate(rows):
        if i and i % len(TESTS) == 0:
            pos += gap
        y[(a, kind, test)] = -pos
        pos += 1
    fig, axes = plt.subplots(1, len(COLUMNS), figsize=(7.0, 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.2})
    pending = []
    for ax, (metric, signal, title, log) in zip(axes, COLUMNS):
        ref = 1.0 if log else 0.0
        ax.axvline(ref, color="#555555", lw=0.7, zorder=1)
        for a, _, kind, test, name in rows:
            c = data.get((f"ani1x-{kind}/mace-mve/{a}", test, metric, signal))
            yy = y[(a, kind, test)]
            if c is None:
                pending.append((a, kind, test, metric, signal))
                ax.text(0.5, yy, "pending", transform=ax.get_yaxis_transform(), ha="center",
                        va="center", fontsize=6, color=MUTED, style="italic")
                continue
            color = SHIFT_COLOR[kind]
            ax.plot([c.low, c.high], [yy, yy], color=color, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(c.mean, yy, ls="", marker="o", ms=4.2, mew=1.0, mec=color,
                    mfc=color if test == "id" else "white", zorder=3)
        if log:
            ax.set_xscale("log")
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
            ax.xaxis.set_major_locator(FixedLocator(log))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(title, loc="left", pad=3)
        ax.grid(axis="y", visible=False)
    ticks = [(y[(a, kind, test)], name) for a, _, kind, test, name in rows]
    axes[0].set_yticks([t for t, _ in ticks], [n for _, n in ticks], fontsize=6.2)
    axes[0].tick_params(axis="y", length=0)
    for k, (_, label) in enumerate(ABLATIONS):
        top = y[(ABLATIONS[k][0], TESTS[0][0], TESTS[0][1])]
        axes[0].text(-0.62, top + 0.75, label, transform=axes[0].get_yaxis_transform(),
                     ha="left", va="bottom", fontsize=7, fontweight="bold")
        if k:
            for ax in axes:
                ax.axhline(top + 0.5 + gap / 2, color="#d0d0d0", lw=0.5, zorder=0)
    axes[0].set_ylim(min(y.values()) - 0.6, 1.4)
    for ax, letter in zip(axes, "abcd"):
        panel_label(ax, letter, x=-0.02, y=1.0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=SHIFT_COLOR[k], marker="o", ms=4.2, mfc=fill, ls="", label=f"{k} split, {t}")
               for k in ("system", "energy") for t, fill in (("ID", SHIFT_COLOR[k]), ("OOD", "white"))]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 1.03),
               columnspacing=1.2, handletextpad=0.3)
    save(fig, "figA_ablations")
    if pending:
        print(f"PENDING ({len(pending)} contrasts missing in {TABLE.name}):",
              sorted({(a, k) for a, k, *_ in pending}))


if __name__ == "__main__":
    main()
