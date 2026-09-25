#!/usr/bin/env python3
"""Appendix figure - training and validation curves (replaces train_curves.svg).

Rows: system split loss, system split RMSE, energy split loss, energy split RMSE.
Columns: LF-only (A/C, DFT), HF-only (E/F, CC), LF->HF fine-tuning stage (B/D, CC,
epoch 0 = first fine-tuning epoch). Solid = training set, dashed = validation set.
Loss is the Gaussian NLL training objective as logged by MACE; RMSE is the
ensemble energy RMSE per atom (meV/atom, log y, fixed range 0.5-300 with bands
clipped to it: a few early-epoch intervals are wider). Epochs every 5.

Data: the five-split aggregates experiment_X/evaluation/train_curves.csv written by
eval/aggregate_replicates.py: loss as arithmetic mean with 95% Student-t interval
(df=4), RMSE as geometric mean with log-space interval. Validation-NLL checkpoint
selection uses the dashed loss curve.
"""

import csv

import numpy as np

from common import INK, PROTOCOL_COLOR, ROOT, panel_label, save, style

COLUMNS = (("LF-only", "LF-only (DFT)"), ("HF-only", "HF-only (CC)"),
           ("LF→HF", "LF→HF, fine-tuning (CC)"))
EXPERIMENTS = {"system": {"LF-only": "A", "HF-only": "E", "LF→HF": "B"},
               "energy": {"LF-only": "C", "HF-only": "F", "LF→HF": "D"}}
ROWS = (("system", "loss"), ("system", "rmse"), ("energy", "loss"), ("energy", "rmse"))


def read(experiment):
    path = ROOT / f"experiment_{experiment}" / "evaluation" / "train_curves.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if any(int(r["train_loss_n"]) != 5 for r in rows):
        raise RuntimeError(f"{path}: not every epoch has five splits")
    return rows


def columns(rows, quantity, subset):
    epochs = np.array([int(r["epoch"]) for r in rows])
    if quantity == "loss":
        key = f"{subset}_loss"
        get = lambda suffix: np.array([float(r[f"{key}_{suffix}"]) for r in rows])  # noqa: E731
        return epochs, get("mean"), get("ci95_low"), get("ci95_high")
    key = f"{subset}_rmse_e_atom_geometric"
    get = lambda suffix: 1000 * np.array([float(r[f"{key}_{suffix}"]) for r in rows])  # noqa: E731
    return epochs, get("mean"), get("ci95_low"), get("ci95_high")


def main():
    plt = style()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(4, 3, figsize=(5.5, 5.6), sharey="row",
                             gridspec_kw={"hspace": 0.3, "wspace": 0.1})
    for c, (protocol, title) in enumerate(COLUMNS):
        color = PROTOCOL_COLOR[protocol]
        for r, (kind, quantity) in enumerate(ROWS):
            ax = axes[r, c]
            rows = read(EXPERIMENTS[kind][protocol])
            for subset, ls in (("train", "-"), ("val", (0, (3, 1.5)))):
                epochs, mean, lo, hi = columns(rows, quantity, subset)
                ax.fill_between(epochs, lo, hi, color=color, alpha=0.13, lw=0)
                ax.plot(epochs, mean, color=color, lw=1.1, ls=ls)
            if quantity == "rmse":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
                ax.set_ylim(0.5, 300)  # bands clipped to the visible range
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(0, 300 if protocol != "LF→HF" else 100)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(title, loc="left", pad=3)
            if c == 0:
                # rotated label: "←" renders as a downward arrow
                label = "loss (GNLL)  ←" if quantity == "loss" else "RMSE (meV/atom)  ←"
                ax.set_ylabel(f"{kind.capitalize()} split\n{label}")
            if r == len(ROWS) - 1:
                ax.set_xlabel("epoch" if protocol != "LF→HF" else "fine-tuning epoch")
    for ax, letter in zip(axes.flat, "abcdefghijkl"):
        panel_label(ax, letter, x=-0.02, y=0.98)
    handles = [Line2D([], [], color=INK, lw=1.1, ls="-", label="training set"),
               Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 1.5)), label="validation set")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
    save(fig, "figA_train_curves")


if __name__ == "__main__":
    main()
