#!/usr/bin/env python3
"""Poster versions of the paper's result figures (A1 portrait poster), CPU only.

Same data, runs, aggregation (five splits, 95% Student-t intervals, geometric means
on log axes, EMA smoothing) and colour conventions as the paper figures: every value
comes from the functions the paper scripts use (common.final_metrics / interval,
epoch_data.across_splits / band / ema, fig6_acquisition.gain_rows / selected_quantiles,
fig4_rank_vs_calibration.point). Only size, layout and typography differ.

Paper figure -> paper script -> poster file (repository root, poster_figures/):
  Fig. 2  fig2_final_summary.py        poster_fig2   (sizes: SIZE below)
  Fig. 3  fig4_rank_vs_calibration.py  poster_fig3
  Fig. 4  fig5_decomposition.py        poster_fig4   (all six panels, 2 rows x 3 columns)
  Fig. 5  fig3_hf_dynamics.py          poster_fig5
  Fig. 6  fig6_acquisition.py          poster_fig6
Fig. 1 (Workflow.pdf) is a Figma export with no source in the repository: not produced.

Each figure is drawn at its final printed size (figsize), laid out by matplotlib's
constrained layout with 0.05 in padding, and saved without bbox cropping, so the
file is exactly figsize: PDF (TrueType fonts embedded) and PNG at 600 dpi.

Usage: python poster_figures.py [fig2 fig3 fig4 fig5 fig6]   (default: the figures in POSTER)
"""

import sys

import numpy as np

from common import (
    CAL_ALPHA, EXPERIMENT, INK, MUTED, PROTOCOL_COLOR, PROTOCOLS, ROOT, SHIFT_COLOR, SIGNAL_COLOR,
    SIGNAL_LINESTYLE, SIGNAL_MARKER, SIGNALS, TESTS, final_metrics, interval,
)
from epoch_data import across_splits, band, ema, hf_stage, lf_to_hf, lf_to_hf_id, lf_to_hf_train

OUT = ROOT / "poster_figures"
# Final printed sizes (in), one per poster card.
SIZE = {"fig2": (10.3, 4.1), "fig3": (9.3, 4.1), "fig5": (9.3, 4.1), "fig6": (10.3, 4.1),
        "fig4": (10.75, 4.4)}  # fig4: round 3, no longer on the poster (not rebuilt by default)
PAD = 0.05            # in, around the content
FONT = {"tick": 12.5, "label": 14, "title": 15, "legend": 13, "letter": 16, "note": 12}
MS = 1.5              # marker scale relative to the paper
BAND = 0.6            # CI band alpha relative to the paper
LW, LW_TU = 2.0, 2.6  # data line widths (pt); TU drawn slightly heavier, as in the paper
PRETRAIN = 300
LEGEND_PAD = PAD * 72 / FONT["legend"]  # legend-to-edge gap of PAD inches, in font units


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT["label"], "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"], "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"], "axes.edgecolor": "#444444", "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 4,
        "ytick.major.size": 4, "xtick.major.pad": 3, "ytick.major.pad": 3,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
        "grid.linewidth": 0.9, "axes.axisbelow": True, "legend.frameon": False,
        "axes.titlepad": 6, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100,
        "lines.linewidth": LW, "axes.titlelocation": "left", "legend.borderpad": 0.0,
    })
    return plt


def figure(plt, size, **layout):
    fig = plt.figure(figsize=size, facecolor="white", layout="constrained")
    fig.get_layout_engine().set(w_pad=PAD, h_pad=PAD, **layout)
    return fig


def letter(ax, text, dx=-5):
    """Bold panel letter on the title baseline, just left of the left-aligned title."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(dx, plt_rc("axes.titlepad")),
                textcoords="offset points", ha="right", va="baseline",
                fontsize=FONT["letter"], fontweight="bold")


def plt_rc(key):
    import matplotlib

    return matplotlib.rcParams[key]


def fixed_ticks(axis, ticks, fmt="{:g}"):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))


def log_ticks(lo, hi):
    """Coarsest 1-10 / 1-3-10 / 1-2-5 sequence giving 3-5 ticks inside [lo, hi]."""
    for mantissas in ((1,), (1, 3), (1, 2, 5)):
        ticks = [m * 10.0 ** e for e in range(-4, 5) for m in mantissas if lo <= m * 10.0 ** e <= hi]
        if len(ticks) >= 3:
            return ticks[:5] if len(ticks) <= 5 else ticks[::2][:5]
    return ticks


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.get_tightbbox(renderer)  # inches
    w, h = fig.get_size_inches()
    margins = dict(left=box.x0, bottom=box.y0, right=w - box.x1, top=h - box.y1)
    fig.get_layout_engine().set(w_pad=PAD, h_pad=PAD)
    fig.set_layout_engine("none")  # freeze the layout so pdf and png are identical
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=600 if suffix == "png" else None, facecolor="white")
    print(f"wrote {OUT / name}.{{pdf,png}}  {w:g} x {h:g} in; content margins (in): "
          + ", ".join(f"{k} {v:.3f}" for k, v in margins.items()))


def line_handle(color, label, ls="-", lw=LW, alpha=1.0):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def marker_handle(color, marker, label, ms, **kw):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, marker=marker, ls="", ms=ms, label=label, **kw)


def band_handle(label):
    from matplotlib.patches import Patch

    return Patch(color=MUTED, alpha=0.3, lw=0, label=label)


# ------------------------------------------------------------------ Fig. 2 (paper)

def fig2(plt):
    """Paper Fig. 2 (fig2_final_summary.py): RMSE / AUSE / ENCE dot plot."""
    columns = (("RMSE", "RMSE (meV/atom)  ←", True), ("AUSE", "AUSE  ← (ranking)", False),
               ("ENCE", "ENCE  ← (calibration)", True))
    fig = figure(plt, SIZE["fig2"], wspace=0.06)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.15])
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[0, i], sharey=axes[0]) for i in (1, 2)]

    slot, bands = {}, []
    y = 0.0
    for _, _, label in TESTS:
        ys = []
        for p in PROTOCOLS:
            for s in SIGNALS:
                slot[(label, p, s)] = y
                ys.append(y)
                y -= 0.42
            y -= 0.25
        bands.append((label, float(np.mean(ys)), ys[0] + 0.3, ys[-1] - 0.3))
        y -= 0.45

    for ax, (metric, title, log) in zip(axes, columns):
        for i, (_, _, top, bottom) in enumerate(bands):
            if i % 2 == 0:
                ax.axhspan(bottom, top, color="#f4f4f2", zorder=0, lw=0)
        for kind, test, label in TESTS:
            for p in PROTOCOLS:
                per_split = final_metrics(EXPERIMENT[(kind, p)], test)
                if metric == "RMSE":
                    y0 = float(np.mean([slot[(label, p, s)] for s in SIGNALS]))
                    c, lo, hi = interval([m["RMSE"] for m in per_split], log)
                    ax.errorbar(c, y0, xerr=[[c - lo], [hi - c]], fmt="o", ms=4.2 * MS,
                                color=PROTOCOL_COLOR[p], mec="white", mew=0.8,
                                elinewidth=LW, capsize=0, zorder=3)
                    continue
                for s in SIGNALS:
                    c, lo, hi = interval([m[(metric, s)] for m in per_split], log)
                    big = metric == "ENCE" and s == "TU"
                    ax.errorbar(c, slot[(label, p, s)], xerr=[[c - lo], [hi - c]], fmt=SIGNAL_MARKER[s],
                                alpha=CAL_ALPHA[s] if metric == "ENCE" else 1.0,
                                ms=(4.3 if big else 3.6) * MS, color=PROTOCOL_COLOR[p],
                                mec="white", mew=0.8, elinewidth=LW, capsize=0, zorder=3)
        ax.set_title(title, color=INK)
        if log:
            ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks([c for _, c, _, _ in bands], [l for l, _, _, _ in bands], fontsize=FONT["label"])
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    axes[0].set_ylim(bands[-1][3] - 0.1, bands[0][2] + 0.1)
    fixed_ticks(axes[0].xaxis, (5, 10, 20, 50, 100))
    axes[0].set_xlim(4.5, 110)
    from matplotlib.ticker import MaxNLocator
    fixed_ticks(axes[1].xaxis, (0.2, 0.3, 0.4), "{:.1f}")
    fixed_ticks(axes[2].xaxis, (0.2, 1, 5))
    for ax, l in zip(axes, "abc"):
        letter(ax, l)

    handles = [marker_handle(PROTOCOL_COLOR[p], "o", p, 4.5 * MS) for p in PROTOCOLS]
    handles += [marker_handle(MUTED, SIGNAL_MARKER[s], s, 4 * MS) for s in SIGNALS]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, handletextpad=0.3,
               columnspacing=1.6, borderaxespad=LEGEND_PAD)
    save(fig, "poster_fig2")


# ------------------------------------------------------------------ Fig. 3 (paper)

def fig3(plt):
    """Paper Fig. 3 (fig4_rank_vs_calibration.py): ENCE vs AUSE, ID (hollow) -> OOD (filled)."""
    from fig4_rank_vs_calibration import point

    fig = figure(plt, SIZE["fig3"], wspace=0.02)
    grid = fig.add_gridspec(1, 2)
    plane = [fig.add_subplot(grid[0, 0])]
    plane.append(fig.add_subplot(grid[0, 1], sharex=plane[0], sharey=plane[0]))
    for ax, (kind, title) in zip(plane, (("system", "System shift"), ("energy", "Energy shift"))):
        for p in PROTOCOLS:
            color = PROTOCOL_COLOR[p]
            for s in SIGNALS:
                (x0, y0), (x1, y1) = point(kind, p, "id", s), point(kind, p, "ood", s)
                fade, ms = CAL_ALPHA[s], (4.8 if s == "TU" else 3.8) * MS
                z = 4 if s == "TU" else 3
                ax.plot([x0, x1], [y0, y1], color=color, lw=LW_TU if s == "TU" else LW,
                        alpha=0.7 * fade, zorder=z - 1, solid_capstyle="round")
                ax.plot([x0], [y0], SIGNAL_MARKER[s], ms=ms, mfc="white", mec=color, mew=1.8,
                        alpha=fade, zorder=z)
                ax.plot([x1], [y1], SIGNAL_MARKER[s], ms=ms, color=color, mec=color, mew=1.8,
                        alpha=fade, zorder=z)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xlabel("ENCE  ← (calibration)")
    plane[0].set_ylabel("AUSE  ← (ranking)")  # rotated label: "←" renders pointing down
    plt.setp(plane[1].get_yticklabels(), visible=False)
    plane[0].set_xlim(0.15, 25)
    plane[0].set_ylim(0.1, 0.42)
    fixed_ticks(plane[0].xaxis, (0.2, 1, 5, 20))
    fixed_ticks(plane[0].yaxis, (0.1, 0.2, 0.3, 0.4), "{:.1f}")
    for ax, l in zip(plane, "ab"):
        ax.text(0.03, 0.03, "better", transform=ax.transAxes, fontsize=FONT["note"], color=MUTED,
                ha="left", va="bottom", style="italic")
        letter(ax, l)

    handles = [line_handle(PROTOCOL_COLOR[p], p) for p in PROTOCOLS]
    handles += [marker_handle(MUTED, SIGNAL_MARKER[s], s, 4 * MS) for s in SIGNALS]
    handles += [marker_handle(MUTED, "o", "ID", 4 * MS, mfc="white", mec=MUTED, mew=1.8),
                marker_handle(MUTED, "o", "OOD", 4 * MS, mew=1.8)]
    fig.legend(handles=handles, loc="outside upper center", ncol=7, handletextpad=0.35,
               columnspacing=1.4, handlelength=1.8, borderaxespad=LEGEND_PAD)
    save(fig, "poster_fig3")


# ------------------------------------------------------------- epoch-curve helpers

def curve(ax, epochs, values, log, color, ls, lw, alpha, phased):
    """Five-split mean +/- 95% band, EMA-smoothed (restarted at epoch 300 when phased),
    as fig5_decomposition.curve / fig3_hf_dynamics; band alpha scaled by BAND."""
    mean, half = band(values)
    back = (lambda v: 10 ** v) if log else (lambda v: v)
    masks = (epochs < PRETRAIN, epochs >= PRETRAIN) if phased else (np.ones_like(epochs, bool),)
    for mask in masks:
        if not mask.any():
            continue
        lo, mid, hi = (ema(v[mask]) for v in (mean - half, mean, mean + half))
        ax.fill_between(epochs[mask], back(lo), back(hi), color=color, alpha=band_alpha * alpha,
                        lw=0, zorder=1)
        ax.plot(epochs[mask], back(mid), color=color, ls=ls, lw=lw, alpha=alpha, zorder=3)
    return back(mean - half).min(), back(mean + half).max()


band_alpha = 0.13 * BAND  # paper: 0.13 (Fig. 5 decomposition), 0.10 (Fig. 3 HF dynamics)


def log_axis(ax, lo, hi):
    ax.set_yscale("log")
    fixed_ticks(ax.yaxis, log_ticks(*ax.get_ylim()))


# ------------------------------------------------------------------ Fig. 4 (paper)

def fig4(plt):
    """Paper Fig. 4 (fig5_decomposition.py), all six panels: columns train / ID test / OOD test,
    rows AUSE (a, c, e) and ENCE (b, d, f, log y), letters as in the paper."""
    global band_alpha
    band_alpha = 0.13 * BAND
    fig = figure(plt, SIZE["fig4"], wspace=0.02, hspace=0.02)
    axes = fig.subplots(2, 3, sharex=True)
    columns = (("Train", lf_to_hf_train, "ab"), ("ID test", lf_to_hf_id, "cd"), ("OOD test", lf_to_hf, "ef"))
    rows = (("AUSE", "AUSE  ←", False), ("ENCE", "ENCE  ←", True))  # rotated: "←" renders as ↓
    for c, (title, fetch, letters) in enumerate(columns):
        for r, (metric, ylabel, log) in enumerate(rows):
            ax = axes[r, c]
            ax.axvspan(PRETRAIN - 2.5, 400, color="#ececea", zorder=0, lw=0)
            ax.axvline(PRETRAIN - 2.5, color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=1)
            ax.grid(axis="x", visible=False)
            for kind in ("system", "energy"):
                for s in SIGNALS:
                    epochs, values = across_splits(lambda x: fetch(kind, x), lambda e, s=s: e[(metric, s)],
                                                   log=log)
                    if epochs is None:
                        raise RuntimeError(f"missing per-epoch data: {title} {kind} {metric}")
                    cal = metric == "ENCE"
                    curve(ax, epochs, values, log, SHIFT_COLOR[kind], SIGNAL_LINESTYLE[s],
                          LW_TU if (cal and s == "TU") else LW, CAL_ALPHA[s] if cal else 1.0, phased=True)
            ax.set_xlim(0, 400)
            if log:
                log_axis(ax, *ax.get_ylim())
            else:
                from matplotlib.ticker import MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10], min_n_ticks=3))
            if r == 0:  # headroom above the curves for the region labels (axis limit only)
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.3 * (hi - lo))
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(ylabel)
            letter(ax, letters[r])
    fixed_ticks(axes[1, 0].xaxis, (0, 100, 200, 300))
    fig.supxlabel("epoch", fontsize=FONT["label"])
    region_labels(axes[0])

    handles = [line_handle(SHIFT_COLOR["system"], "System split"), line_handle(SHIFT_COLOR["energy"], "Energy split")]
    handles += [line_handle(MUTED, s, SIGNAL_LINESTYLE[s]) for s in SIGNALS] + [band_handle("95% CI")]
    fig.legend(handles=handles, loc="outside upper center", ncol=6, handlelength=1.8,
               handletextpad=0.4, columnspacing=1.4, borderaxespad=LEGEND_PAD)
    save(fig, "poster_fig4")


def region_labels(axes):
    """'DFT pretrain' / 'CCSD(T) fine-tune' inside the top of each given axes: the pretraining
    label centred over epochs 0-300, the fine-tuning label on two lines in the shaded band."""
    for ax in axes:
        trans = ax.get_xaxis_transform()
        ax.text(150, 0.97, "DFT pretrain", transform=trans, ha="center", va="top",
                fontsize=11, color=MUTED, zorder=5)
        ax.text(350, 0.97, "CCSD(T)\nfine-tune", transform=trans, ha="center", va="top",
                fontsize=11, color=MUTED, linespacing=1.0, zorder=5)


# ------------------------------------------------------------------ Fig. 5 (paper)

def fig5(plt):
    """Paper Fig. 5 (fig3_hf_dynamics.py): AUSE / ENCE vs HF-stage epoch."""
    global band_alpha
    band_alpha = 0.10 * BAND
    rows = (("AUSE", "AUSE  ←", False), ("ENCE", "ENCE  ←", True))
    columns = (("system", "System-OOD"), ("energy", "Energy-OOD"))
    fig = figure(plt, SIZE["fig5"], wspace=0.02, hspace=0.02)
    axes = fig.subplots(2, 2, sharex=True, sharey="row")
    for c, (kind, title) in enumerate(columns):
        for r, (metric, ylabel, log) in enumerate(rows):
            ax = axes[r, c]
            for p in PROTOCOLS:
                for s in SIGNALS:
                    epochs, values = across_splits(lambda x: hf_stage(kind, p, x),
                                                   lambda e, s=s: e[(metric, s)], log=log)
                    if epochs is None:
                        raise RuntimeError(f"missing per-epoch data: {title} {p}")
                    curve(ax, epochs, values, log, PROTOCOL_COLOR[p], SIGNAL_LINESTYLE[s],
                          LW_TU if s == "TU" else LW, CAL_ALPHA[s] if metric == "ENCE" else 1.0,
                          phased=False)
            if log:
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(ylabel)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.grid(axis="x", visible=False)
    axes[0, 0].set_xlim(-5, 300)
    fixed_ticks(axes[1, 0].xaxis, (0, 100, 200, 300))
    from matplotlib.ticker import MaxNLocator
    fixed_ticks(axes[0, 0].yaxis, (0.2, 0.4, 0.6), "{:.1f}")
    fixed_ticks(axes[1, 0].yaxis, log_ticks(*axes[1, 0].get_ylim()))
    fig.supxlabel("HF-stage epoch", fontsize=FONT["label"])
    for ax, l in zip(axes.flat, "abcd"):
        letter(ax, l)

    handles = [line_handle(PROTOCOL_COLOR[p], p) for p in PROTOCOLS]
    handles += [line_handle(MUTED, s, SIGNAL_LINESTYLE[s]) for s in SIGNALS] + [band_handle("95% CI")]
    fig.legend(handles=handles, loc="outside upper center", ncol=6, handlelength=1.8,
               handletextpad=0.4, columnspacing=1.4, borderaxespad=LEGEND_PAD)
    save(fig, "poster_fig5")


# ------------------------------------------------------------------ Fig. 6 (paper)

def fig6(plt):
    """Paper Fig. 6 (fig6_acquisition.py): (a) RMSE gain over random, (b) selected quantiles."""
    from fig6_acquisition import RANDOM_COLOR, REGIME, gain_rows, selected_quantiles

    rows = gain_rows()
    fig = figure(plt, SIZE["fig6"], wspace=0.06, hspace=0.03)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3])
    ax_a = fig.add_subplot(grid[:, 0])
    ax_q = [fig.add_subplot(grid[0, 1])]
    ax_q.append(fig.add_subplot(grid[1, 1], sharex=ax_q[0], sharey=ax_q[0]))

    x, ticks = 0.0, []
    for p in PROTOCOLS:
        for s in SIGNALS:
            row = rows[(REGIME[p], "energy_ood", f"{s.lower()}_gain_over_random_meV_per_atom")]
            ax_a.scatter(x + (np.arange(5) - 2) * 0.07, row["split_values"], s=7 * MS ** 2, color="#9a9a9a",
                         edgecolors="white", linewidths=0.5, zorder=2)
            ax_a.errorbar(x, row["mean"], yerr=[[row["mean"] - row["ci95_lower"]],
                                                [row["ci95_upper"] - row["mean"]]],
                          fmt=SIGNAL_MARKER[s], ms=4.4 * MS, color=PROTOCOL_COLOR[p], mec="white",
                          mew=0.8, elinewidth=LW, capsize=0, zorder=3)
            ticks.append((x, s))
            x += 1.0
        x += 0.45
    ax_a.axhline(0, color="#555555", lw=1.2, zorder=1)
    ax_a.set_xticks([t for t, _ in ticks], [s for _, s in ticks])
    ax_a.set_xlim(-0.6, x - 0.85)
    ax_a.grid(axis="x", visible=False)
    for i, p in enumerate(PROTOCOLS):
        center = float(np.mean([t for t, _ in ticks[3 * i:3 * i + 3]]))
        ax_a.annotate(p, xy=(center, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                      textcoords="offset points", ha="center", va="top", fontsize=FONT["legend"],
                      color=PROTOCOL_COLOR[p], fontweight="bold")
    ax_a.set_ylabel("RMSE gain over random\n(meV/atom)  →")  # rotated: "→" renders as an upward arrow
    ax_a.set_title("Held-out Energy-OOD")
    ax_a.set_ylim(-0.3, 3.4)
    fixed_ticks(ax_a.yaxis, (0, 1, 2, 3))
    letter(ax_a, "a")

    bins = np.linspace(0.55, 1.0, 10)
    random_q = selected_quantiles("random")
    for ax, p in zip(ax_q, PROTOCOLS):
        weights = np.full(len(random_q), 1 / len(random_q))
        ax.hist(random_q, bins=bins, histtype="stepfilled", color="#e9e9e7", zorder=1, weights=weights,
                label="random")
        ax.hist(random_q, bins=bins, histtype="step", color=RANDOM_COLOR, lw=LW, zorder=2, weights=weights)
        for s in SIGNALS:
            q = selected_quantiles(f"{REGIME[p]}_{s.lower()}")
            ax.hist(q, bins=bins, histtype="step", color=SIGNAL_COLOR[s], lw=LW, zorder=3,
                    weights=np.full(len(q), 1 / len(q)), label=s,
                    linestyle=(0, (3, 1)) if s == "EU" else "-")
        ax.text(0.03, 0.93, p, transform=ax.transAxes, fontsize=FONT["legend"], color=PROTOCOL_COLOR[p],
                fontweight="bold", va="top")
        ax.set_ylabel("fraction")
        ax.grid(axis="x", visible=False)
    plt.setp(ax_q[0].get_xticklabels(), visible=False)
    ax_q[1].set_xlabel("within-system energy quantile $q_s(x)$")
    ax_q[0].set_title("Selected configurations")
    ax_q[0].set_xlim(0.55, 1.0)
    fixed_ticks(ax_q[0].xaxis, (0.6, 0.8, 1.0), "{:.1f}")
    fixed_ticks(ax_q[0].yaxis, (0, 0.1, 0.2, 0.3))
    letter(ax_q[0], "b")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#e9e9e7", edgecolor=RANDOM_COLOR, lw=LW, label="random")]
    handles += [line_handle(SIGNAL_COLOR[s], s, (0, (3, 1)) if s == "EU" else "-") for s in SIGNALS]
    fig.legend(handles=handles, loc="outside upper right", ncol=4, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.4, borderaxespad=LEGEND_PAD)
    save(fig, "poster_fig6")


FIGURES = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6}
POSTER = ("fig2", "fig3", "fig5", "fig6")  # built when no figure is named


def main():
    plt = style()
    for name in sys.argv[1:] or POSTER:
        FIGURES[name](plt)
        plt.close("all")


if __name__ == "__main__":
    main()
