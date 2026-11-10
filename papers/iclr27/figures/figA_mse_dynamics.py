#!/usr/bin/env python3
"""Appendix figure (rebuttal item D, dynamics) - MSE ensemble vs MVE ensemble along LF->HF training.

Fig. 5 layout: rows AUSE and ENCE (log y); columns System-ID, System-OOD, Energy-ID, Energy-OOD;
x = epoch over DFT pretraining (0-295, parent LF-only members) and CC fine-tuning (300-395,
shaded). Lines: EU of the ten-member MSE ensemble (``ani1x-<split>/mace-mse/{lf-only,lf-to-hf}``,
its only signal; dashed) and EU and TU of the paper's ten-member MVE ensemble (A -> B, C -> D;
solid). While pretraining, the curves are scored against the DFT labels (ID: ``dft_test_id``;
Energy-OOD: the ``cc_test_ood`` geometries), afterwards against the CC labels, as in Fig. 5.
Five-split means (AUSE arithmetic, ENCE geometric), smoothed by an exponential moving average
restarted at epoch 300; bands: 95% Student-t intervals over the five splits (log space for
ENCE), smoothed the same way. Data (epoch_data.py): the cached ``epoch_quality_finetune_mixed_*``
curves, except Energy-OOD, recomputed from ``epoch_predictions_cc_test_ood.npz`` restricted to
the training-support region. In the ENCE row the MVE EU is faded: only TU (and the MSE EU, the
MSE ensemble's predictive variance) is expected to be calibrated.
"""

from common import CAL_ALPHA, MUTED, SIGNAL_COLOR, panel_label, save, style
from epoch_data import lf_to_hf_any
from fig5_decomposition import PRETRAIN, curve, stage_background

MVE = None  # default LF->HF pair of epoch_data (A -> B, C -> D)
MSE = {kind: (f"ani1x-{kind}/mace-mse/lf-only", f"ani1x-{kind}/mace-mse/lf-to-hf") for kind in ("system", "energy")}
LINES = (  # (label, pair, signal, line style)
    ("MSE ens. EU", "mse", "EU", dict(color=SIGNAL_COLOR["EU"], lw=1.3, ls=(0, (3, 1.5)))),
    ("MVE ens. EU (ours)", "mve", "EU", dict(color=SIGNAL_COLOR["EU"], lw=1.3)),
    ("MVE ens. TU (ours)", "mve", "TU", dict(color=SIGNAL_COLOR["TU"], lw=1.5)),
)
COLUMNS = (("system", "id", "System-ID"), ("system", "ood", "System-OOD"), ("energy", "id", "Energy-ID"),
           ("energy", "ood", "Energy-OOD"))
ROWS = (("AUSE", "AUSE  ← (ranking)", False), ("ENCE", "ENCE  ← (calibration)", True))


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
            stage_background(ax)
            for label, which, signal, line in LINES:
                pair = MSE[kind] if which == "mse" else MVE
                style_ = dict(line)
                if metric == "ENCE" and which == "mve" and signal == "EU":
                    style_["alpha"] = CAL_ALPHA["EU"]
                fetch = lambda k, s, t=test, p=pair: lf_to_hf_any(k, s, t, pair=p)  # noqa: E731
                if not curve(ax, kind, lambda e, m=metric, g=signal: e[(m, g)], log, fetch=fetch, **style_):
                    raise RuntimeError(f"incomplete per-epoch data: {kind} {test} {label}")
            if log:
                ax.set_yscale("log")
                ax.axhline(1.0, color="#777777", lw=0.6, ls=(0, (2, 2)), zorder=1)
            if r == 0:
                ax.set_title(title, loc="left", pad=3)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel("epoch")
    axes[1, 0].yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
    axes[1, 0].yaxis.set_minor_locator(NullLocator())
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0, 0].set_xlim(0, 400)
    axes[0, 0].xaxis.set_major_locator(FixedLocator([0, 100, 200, 300]))
    for ax, letter in zip(axes.flat, "abcdefgh"):
        panel_label(ax, letter, x=-0.01, y=1.0)
    handles = [Line2D([], [], label=label, **{k: v for k, v in line.items()}) for label, _, _, line in LINES]
    handles.append(Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI, 5 splits"))
    handles.append(Patch(color="#f1f1ef", ec=MUTED, lw=0.4, label="CC fine-tuning"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04),
               handlelength=2.2, columnspacing=1.1)
    assert PRETRAIN == 300
    save(fig, "figA_mse_dynamics")


if __name__ == "__main__":
    main()
