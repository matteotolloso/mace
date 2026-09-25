#!/usr/bin/env python3
"""Water Fig. W3 - Uncertainty quality during training.

Row 1, CCSDT stage on the CCSDT test set (as Fig. 3): (a) AUSE, (b) ENCE (log y)
against CCSDT-training epoch; HF-only (wC, 300 epochs from scratch) vs LF->HF
(wB, 100 fine-tuning epochs). Colour = protocol.
Rows 2-3, LF->HF over the whole trajectory (as Fig. 5): BLYP pretraining (wA
checkpoints, epochs 0-299) then CCSDT fine-tuning (wB, 300-399, shaded).
Row 2 on the training data (BLYP train set, then CCSDT train set): (c) AUSE,
(d) ENCE. Row 3 on the test set (BLYP test set and labels, then CCSDT test set):
(e) AUSE, (f) ENCE. The target changes at epoch 300.

Line style = signal (AU dashed, EU dotted, TU solid). Lines: exponential moving
average of the five-split mean (AUSE arithmetic, ENCE geometric), restarted at
the fine-tuning boundary; bands: 95% Student-t interval over the five splits
(log space for ENCE), smoothed the same way. Epochs every 5. No support filter,
no trim; see water_data.py for the split-misalignment caveat. In the ENCE panels
TU is drawn at full strength and AU/EU faded (CAL_ALPHA).
"""

from common import (
    CAL_ALPHA, MUTED, PROTOCOL_COLOR, PROTOCOLS, SIGNAL_LINESTYLE, SIGNALS, panel_label, save, style,
)
from epoch_data import across_splits, band, ema
from fig5_decomposition import LOG_TICKS, curve, stage_background
from water_data import PRETRAIN, hf_stage, lf_to_hf

AUSE_LABEL, ENCE_LABEL = "AUSE  (lower is better)", "ENCE  (lower is better)"


def log_axis(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(LOG_TICKS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def trajectory_axes(ax, title, ylabel, log):
    stage_background(ax)
    ax.set_title(title, loc="left", pad=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.set_xlim(0, 400)
    if log:
        log_axis(ax)
    trans = ax.get_xaxis_transform()
    ax.text(150, 0.97, "BLYP pretraining", transform=trans, ha="center", va="top",
            fontsize=6, color=MUTED)
    ax.text(350, 0.97, "CCSDT\nfine-tuning", transform=trans, ha="center", va="top",
            fontsize=6, color=MUTED)


def main():
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.1), gridspec_kw={"hspace": 0.5, "wspace": 0.22})
    (ax_hf_ause, ax_hf_ence), (ax_tr_ause, ax_tr_ence), (ax_te_ause, ax_te_ence) = axes

    # Row 1: CCSDT stage, both protocols
    for ax, metric, log in ((ax_hf_ause, "AUSE", False), (ax_hf_ence, "ENCE", True)):
        for p in PROTOCOLS:
            for s in SIGNALS:
                epochs, values = across_splits(lambda x, p=p: hf_stage(p, x),
                                               lambda e, s=s: e[(metric, s)], log=log)
                mean, half = band(values)
                back = (lambda v: 10 ** v) if log else (lambda v: v)
                fade = CAL_ALPHA[s] if metric == "ENCE" else 1.0
                ax.fill_between(epochs, back(ema(mean - half)), back(ema(mean + half)),
                                color=PROTOCOL_COLOR[p], alpha=0.10 * fade, lw=0)
                ax.plot(epochs, back(ema(mean)), color=PROTOCOL_COLOR[p],
                        lw=1.6 if s == "TU" else 1.2, ls=SIGNAL_LINESTYLE[s], alpha=fade)
        ax.set_xlim(-5, 300)
        ax.set_xlabel("CCSDT-stage epoch")
        ax.grid(axis="x", visible=False)
        if log:
            log_axis(ax)
    ax_hf_ause.set_title("CCSDT test set, CCSDT stage: ranking", loc="left", pad=3)
    ax_hf_ence.set_title("CCSDT test set, CCSDT stage: calibration", loc="left", pad=3)
    ax_hf_ause.set_ylabel(AUSE_LABEL)
    ax_hf_ence.set_ylabel(ENCE_LABEL)

    # Rows 2-3: LF->HF over the whole trajectory
    color = PROTOCOL_COLOR["LF→HF"]
    for s in SIGNALS:
        line = dict(color=color, lw=1.6 if s == "TU" else 1.2, ls=SIGNAL_LINESTYLE[s])
        for ax_a, ax_e, data in ((ax_tr_ause, ax_tr_ence, "train"), (ax_te_ause, ax_te_ence, "test")):
            fetch = lambda _kind, split, data=data: lf_to_hf(split, data)  # noqa: E731
            curve(ax_a, None, lambda e, s=s: e[("AUSE", s)], False, fetch=fetch, **line)
            curve(ax_e, None, lambda e, s=s: e[("ENCE", s)], True, fetch=fetch,
                  **dict(line, alpha=CAL_ALPHA[s]))
    for ax, title, ylabel, log in ((ax_tr_ause, "LF→HF, train set: ranking", AUSE_LABEL, False),
                                   (ax_tr_ence, "LF→HF, train set: calibration", ENCE_LABEL, True),
                                   (ax_te_ause, "LF→HF, test set: ranking", AUSE_LABEL, False),
                                   (ax_te_ence, "LF→HF, test set: calibration", ENCE_LABEL, True)):
        trajectory_axes(ax, title, ylabel, log)

    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter, x=-0.01, y=1.0)
    handles = [Line2D([], [], color=PROTOCOL_COLOR[p], lw=1.6, label=p) for p in PROTOCOLS]
    handles += [Line2D([], [], color=MUTED, lw=1.2, ls=SIGNAL_LINESTYLE[s], label=s)
                for s in SIGNALS]
    handles.append(Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI, 5 splits"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.955),
               handlelength=1.8, columnspacing=1.2)
    assert PRETRAIN == 300
    save(fig, "figW_dynamics")


if __name__ == "__main__":
    main()
