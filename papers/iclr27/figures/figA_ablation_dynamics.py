#!/usr/bin/env python3
"""Appendix figure (rebuttal item C) - AU along fine-tuning for the LF->HF ablations.

Fig. 5 layout (fig3_hf_dynamics.py): rows AUSE and ENCE (log y) of AU; columns System-ID,
System-OOD, Energy-ID, Energy-OOD; x = fine-tuning epoch (every 5th, 0-95). One line per fine-tuning
variant, all starting from the same pretrained members: the paper's LF->HF (B / D), variance
head re-initialized at the start of fine-tuning, beta-NLL (beta = 0.5), FT learning rate 1e-4
and 1e-2 (paper: 1e-3). Lines: exponential moving average of the five-split mean (AUSE
arithmetic, ENCE geometric); bands: 95% Student-t interval over the five splits (log space for
ENCE), smoothed the same way. Data: per-split ``epoch_quality_<test>.csv`` of each experiment
(cached curves; the support filter removes no System configuration and at most 2 of 5000
Energy-ID ones, see epoch_data.py); Energy-OOD from the per-configuration predictions
``epoch_predictions_cc_test_ood.npz`` restricted to the training-support region, as in Fig. 5.
"""

import numpy as np

from common import MUTED, PROTOCOL_COLOR, SIGNAL_COLOR, panel_label, save, split_dir, style
from epoch_data import _from_csv, _from_npz, across_splits, band, ema

VARIANTS = (("lf-to-hf", "LF→HF (paper)", PROTOCOL_COLOR["LF→HF"]),
            ("lf-to-hf-reset-var", "variance-head reset", "#6f4e9c"),
            ("lf-to-hf-beta-nll", "β-NLL (β = 0.5)", SIGNAL_COLOR["TU"]),
            ("lf-to-hf-lr1e-4", "FT lr 1e-4", SIGNAL_COLOR["AU"]),
            ("lf-to-hf-lr1e-2", "FT lr 1e-2", SIGNAL_COLOR["EU"]))
COLUMNS = (("system", "id", "System-ID"), ("system", "ood", "System-OOD"), ("energy", "id", "Energy-ID"),
           ("energy", "ood", "Energy-OOD"))
ROWS = (("AUSE", "AU AUSE  ← (ranking)", False), ("ENCE", "AU ENCE  ← (calibration)", True))


def series(kind, variant, test, split):
    if kind == "energy" and test == "ood":  # support-filtered per-configuration predictions
        return _from_npz(f"ani1x-{kind}/mace-mve/{variant}", split)
    return _from_csv(split_dir(f"ani1x-{kind}/mace-mve/{variant}", split) / f"epoch_quality_{test}.csv")


def main():
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.8), sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    for c, (kind, test, title) in enumerate(COLUMNS):
        for r, (metric, ylabel, log) in enumerate(ROWS):
            ax = axes[r, c]
            for variant, _, colour in VARIANTS:
                epochs, values = across_splits(lambda s: series(kind, variant, test, s),
                                               lambda e: e[(metric, "AU")], log=log)
                if epochs is None or len(epochs) < 20:
                    raise RuntimeError(f"incomplete per-epoch data: {kind} {variant} {test}")
                mean, half = band(values)
                transform = (lambda v: 10 ** v) if log else (lambda v: v)
                ax.fill_between(epochs, transform(ema(mean - half)), transform(ema(mean + half)),
                                color=colour, alpha=0.10, lw=0)
                ax.plot(epochs, transform(ema(mean)), color=colour,
                        lw=1.8 if variant == "lf-to-hf" else 1.3)
            if log:
                ax.set_yscale("log")
                ax.axhline(1.0, color="#777777", lw=0.6, ls=(0, (2, 2)), zorder=1)
            if r == 0:
                ax.set_title(title, loc="left", pad=3)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel("fine-tuning epoch")
            ax.grid(axis="x", visible=False)
    axes[1, 0].yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]))
    axes[1, 0].yaxis.set_minor_locator(NullLocator())
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0, 0].set_xlim(-3, 98)
    for ax, letter in zip(axes.flat, "abcdefgh"):
        panel_label(ax, letter, x=-0.01, y=1.0)
    handles = [Line2D([], [], color=colour, lw=1.8 if v == "lf-to-hf" else 1.3, label=label)
               for v, label, colour in VARIANTS]
    handles.append(Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI, 5 splits"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.8, columnspacing=1.1)
    save(fig, "figA_ablation_dynamics")


if __name__ == "__main__":
    main()
