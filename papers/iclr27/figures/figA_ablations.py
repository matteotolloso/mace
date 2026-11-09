#!/usr/bin/env python3
"""Appendix figure (rebuttal item C) - fine-tuning ablations of the AU collapse, raw values.

Fig. 2 layout (dotplot.py): rows = four test sets, columns = RMSE, AUSE, ENCE of the final
ensembles (each member at its lowest validation-NLL checkpoint), uncalibrated, per atom,
support-filtered. Colour = fine-tuning variant: the LF->HF ensembles of the main experiments
(B system / D energy) and the ablations variance-head reset, beta-NLL, FT lr 1e-4, FT lr 1e-2,
all fine-tuned from the same pretrained members (same colours as figA_ablation_dynamics);
marker = signal (AU, TU). AU is the signal under test, so in the ENCE column AU is drawn at full
strength as well. Five-split means with 95% Student-t intervals (geometric, log-space for RMSE
and ENCE), recomputed from the per-configuration rows as in Fig. 2. The paired contrasts behind
the text (ledger 2026-11-07 "Fine-tuning ablations of the AU collapse" and its system-split
completion) are drawn by figA_ablations_contrasts.py.
"""

from common import EXPERIMENT, MUTED, SIGNAL_MARKER, TESTS, final_metrics, interval, save, style
from dotplot import draw
from figA_ablation_dynamics import VARIANTS

SIGNALS = ("AU", "TU")


def experiment(kind, variant):
    return EXPERIMENT[(kind, "LF→HF")] if variant == "lf-to-hf" else f"ani1x-{kind}/mace-mve/{variant}"


def main():
    plt = style()
    kind_of = {label: (kind, test) for kind, test, label in TESTS}

    def value(label, variant, signal, metric):
        kind, test = kind_of[label]
        values = final_metrics(experiment(kind, variant), test)
        log = metric in ("RMSE", "ENCE")
        return interval([m["RMSE"] if signal is None else m[(metric, signal)] for m in values], log)

    fig, axes = draw(plt, [(v, colour) for v, _, colour in VARIANTS], lambda key: SIGNALS, value,
                     fade=lambda signal: 1.0, figsize=(7.0, 5.6), rmse_ticks=(5, 10, 20, 50),
                     ence_ticks=(0.2, 0.5, 1, 2, 5, 10))
    for ax in axes:  # the tall layout needs the panel letters closer to the axes than dotplot's default
        for text in ax.texts:
            if text.get_text() in ("a", "b", "c"):
                text.set_y(1.04)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=colour, marker="o", ls="", ms=4.5, label=label)
               for _, label, colour in VARIANTS]
    handles += [Line2D([], [], color=MUTED, marker=SIGNAL_MARKER[s], ls="", ms=4, label=s) for s in SIGNALS]
    fig.legend(handles=handles, loc="upper center", ncol=7, bbox_to_anchor=(0.52, 1.005),
               handletextpad=0.3, columnspacing=1.0, fontsize=6.6)
    save(fig, "figA_ablations")


if __name__ == "__main__":
    main()
