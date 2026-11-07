#!/usr/bin/env python3
"""Appendix figure (rebuttal item B) - sensitivity of both protocols to the learning rate.

Rows: system split, energy split. Columns: RMSE (meV/atom), TU Spearman, TU ENCE of the
final, uncalibrated, support-filtered ensembles. x: learning rate of the trained stage
(HF-only: 1e-2 as in the paper, 3e-3, 1e-3; LF->HF fine-tuning: 1e-4, 1e-3 as in the
paper, 1e-2; LF pretraining unchanged). ID solid, OOD dashed; the paper's setting is
ringed. Five-split values from the aggregate summaries
(``experiments/<id>/evaluation/reliability_<test>_nocal_summary.csv``): RMSE and ENCE as
geometric means with log-space 95% Student-t intervals, Spearman as arithmetic means
(df = 4). An experiment whose aggregate is missing or has fewer than five splits is left
out and reported, never filled in.
"""

import csv

import yaml

from common import MUTED, PROTOCOL_COLOR, panel_label, save, style
from uq import paths

PROTOCOL_EXPERIMENTS = {
    "HF-only": ("hf-only", "hf-only-lr3e-3", "hf-only-lr1e-3"),
    "LF→HF": ("lf-to-hf-lr1e-4", "lf-to-hf", "lf-to-hf-lr1e-2"),
}
PAPER = {"HF-only": "hf-only", "LF→HF": "lf-to-hf"}
SPLITS_ROWS = (("system", "System split"), ("energy", "Energy split"))
# (metric, uncertainty_type, label, scale, geometric)
COLUMNS = (("rmse", "ensemble", "RMSE (meV/atom) ↓", 1000.0, True),
           ("spearman", "total", "TU Spearman ↑", 1.0, False),
           ("ence", "total", "TU ENCE ↓", 1.0, True))
TEST_STYLE = {"id": ("-", "ID"), "ood": ((0, (4, 2)), "OOD")}


def experiment_id(kind, name):
    return f"ani1x-{kind}/mace-mve/{name}"


def learning_rate(exp_id):
    with (paths.experiment_dir(exp_id) / "config.yaml").open() as handle:
        return float(yaml.safe_load(handle)["lr"])


def summary(exp_id, test):
    path = paths.evaluation_dir(exp_id) / f"reliability_{test}_nocal_summary.csv"
    if not path.exists():
        return None
    with path.open() as handle:
        rows = {(r["metric"], r["uncertainty_type"]): r for r in csv.DictReader(handle)}
    if any(int(r["value_n"]) != 5 for r in rows.values()):
        return None
    return rows


def value(row, scale, geometric):
    prefix = "value_geometric" if geometric else "value"
    return tuple(scale * float(row[f"{prefix}_{k}"]) for k in ("mean", "ci95_low", "ci95_high"))


def main():
    plt = style()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(len(SPLITS_ROWS), len(COLUMNS), figsize=(7.0, 3.6),
                             gridspec_kw={"wspace": 0.32, "hspace": 0.42})
    missing = []
    for r, (kind, row_label) in enumerate(SPLITS_ROWS):
        for c, (metric, signal, label, scale, geometric) in enumerate(COLUMNS):
            ax = axes[r, c]
            for protocol, names in PROTOCOL_EXPERIMENTS.items():
                color = PROTOCOL_COLOR[protocol]
                for t, (test, (ls, _)) in enumerate(TEST_STYLE.items()):
                    xs, ys, los, his, paper_point = [], [], [], [], None
                    for name in names:
                        exp_id = experiment_id(kind, name)
                        rows = summary(exp_id, test)
                        if rows is None:
                            missing.append((exp_id, test))
                            continue
                        lr = learning_rate(exp_id) * (1.06 if t else 1 / 1.06)  # dodge ID / OOD
                        m, lo, hi = value(rows[(metric, signal)], scale, geometric)
                        xs.append(lr), ys.append(m), los.append(m - lo), his.append(hi - m)
                        if name == PAPER[protocol]:
                            paper_point = (lr, m)
                    ax.errorbar(xs, ys, yerr=[los, his], color=color, ls=ls, lw=1.2, marker="o", ms=3.5,
                                mfc=color if test == "id" else "white", mec=color, mew=0.9,
                                elinewidth=0.8, capsize=0)
                    if paper_point:
                        ax.plot(*paper_point, ls="", marker="o", ms=7.5, mfc="none", mec=MUTED, mew=0.7)
            ax.set_xscale("log")
            ax.set_xticks([1e-4, 1e-3, 1e-2], ["1e-4", "1e-3", "1e-2"])
            ax.minorticks_off()
            if geometric:
                ax.set_yscale("log")
                from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
                lo, hi = ax.get_ylim()
                subs = (1.0, 2.0, 5.0) if hi / lo > 4 else (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0)
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            if r == 0:
                ax.set_title(label, loc="left", pad=3)
            if r == len(SPLITS_ROWS) - 1:
                ax.set_xlabel("learning rate")
            if c == 0:
                ax.set_ylabel(row_label)
            panel_label(ax, "abcdef"[r * len(COLUMNS) + c], x=-0.08, y=1.0)
    handles = [Line2D([], [], color=PROTOCOL_COLOR[p], lw=1.5, label=p) for p in PROTOCOL_EXPERIMENTS]
    handles += [Line2D([], [], color=MUTED, ls=ls, marker="o", ms=3.5, mfc=MUTED if t == "id" else "white",
                       label=name) for t, (ls, name) in TEST_STYLE.items()]
    handles.append(Line2D([], [], ls="", marker="o", ms=7.5, mfc="none", mec=MUTED, label="paper's setting"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03),
               columnspacing=1.2, handlelength=2.2)
    save(fig, "figA_lr_sensitivity")
    if missing:
        print(f"MISSING five-split aggregates ({len(missing)}):", sorted(set(missing)))


if __name__ == "__main__":
    main()
