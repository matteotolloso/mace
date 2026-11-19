#!/usr/bin/env python3
"""Appendix figure (rebuttal item D) - uncertainty estimators: MSE ensemble, MVE ensemble, one MVE model.

Fig. 2 layout (dotplot.py): rows = four test sets, columns = RMSE, AUSE, ENCE; one figure per
protocol (``figA_estimators_hf`` HF-only, ``figA_estimators_lfhf`` LF->HF, in the protocol's colour,
on common x ranges). Four signals: EU of a ten-member MSE ensemble (hollow square;
its only signal), EU (square) and TU (diamond) of the paper's ten-member MVE ensemble, and AU of
a single MVE member (hollow circle; mean over the ten members). RMSE: MSE ensemble, MVE ensemble
and single member, in the slots of their signal. Uncalibrated, per atom; five-split means with 95%
Student-t intervals (geometric, log-space for RMSE and ENCE), read from
``results/tables/mse_vs_mve.csv`` (``python -m uq.evaluation.compare_uq``; ledger 2026-11-03/04
HF-only, 2026-11-08 LF->HF). Tests whose MSE ensemble is not evaluated on all five splits are drawn
as "pending". In the ENCE column only the MVE-ensemble EU is faded: the other three signals are the
predictive variance of their estimator. The paper's estimator (the ten-member MVE ensemble, EU and
TU slots) is marked "(ours)" in the legend.
"""

import csv

from matplotlib.lines import Line2D

from common import MUTED, PROTOCOL_COLOR, ROOT, TESTS, save, style
from dotplot import draw

TABLE = ROOT / "results" / "tables" / "mse_vs_mve.csv"
PROTOCOLS = (("hf-only", "HF-only", "hf"), ("lf-to-hf", "LF→HF", "lfhf"))
SIGNALS = ("mse_eu", "mve_eu", "mve_tu", "mve_single_au")
MARKER = {"mse_eu": "s", "mve_eu": "s", "mve_tu": "D", "mve_single_au": "o"}
HOLLOW = {"mse_eu", "mve_single_au"}
RMSE_OF = {"mse_eu": "mse_ens", "mve_tu": "mve_ens", "mve_single_au": "mve_single"}
LABEL = {"mse_eu": "MSE ens. EU", "mve_eu": "MVE ens. EU", "mve_tu": "MVE ens. TU", "mve_single_au": "one MVE AU"}
OURS = {"mve_eu", "mve_tu"}  # the paper's estimator: ten-member mean-variance ensemble
TEST_LABEL = {(kind, test): label for kind, test, label in TESTS}


def load():
    out, missing = {}, set()
    with TABLE.open() as handle:
        for row in csv.DictReader(handle):
            label = TEST_LABEL[(row["dataset"].removeprefix("ani1x-"), row["test"])]
            if row["missing"]:
                missing.add((label, row["protocol"]))
                continue
            if int(float(row["n_splits"])) != 5:
                raise RuntimeError(f"{TABLE.name}: {row['n_splits']} splits for {row}")
            scale = 1000.0 if row["metric"] == "rmse" else 1.0
            out[(label, row["protocol"], row["metric"], row["signal"])] = tuple(
                scale * float(row[k]) for k in ("value_mean", "value_ci95_low", "value_ci95_high"))
    return out, missing


def xlims(data):
    """Common x range per column over both protocols, so the two figures can be compared."""
    out = {}
    for metric, log in (("rmse", True), ("ause", False), ("ence", True)):
        values = [v for (_, _, m, _), triple in data.items() if m == metric for v in triple[1:]]
        lo, hi = min(values), max(values)
        out[metric] = (lo / 1.15, hi * 1.15) if log else (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    return out


def main():
    plt = style()
    data, missing = load()
    limits = xlims(data)

    def value(label, protocol, signal, metric):
        if metric == "RMSE":
            if signal not in RMSE_OF:
                return False
            metric, signal = "rmse", RMSE_OF[signal]
        if (label, protocol) in missing:
            return None
        return data[(label, protocol, metric.lower(), signal)]

    for protocol, name, suffix in PROTOCOLS:
        colour = PROTOCOL_COLOR[name]
        fig, axes = draw(plt, [(protocol, colour)], lambda key: SIGNALS, value,
                         hollow=lambda key, s: s in HOLLOW, marker=lambda s: MARKER[s],
                         fade=lambda s: 0.35 if s == "mve_eu" else 1.0, rmse_per_signal=True,
                         figsize=(7.0, 2.9), rmse_ticks=(5, 10, 20, 50, 100),
                         ence_ticks=(0.2, 0.5, 1, 2, 5, 10, 20))
        for ax, metric in zip(axes, ("rmse", "ause", "ence")):
            ax.set_xlim(*limits[metric])
        handles = [Line2D([], [], color=colour, marker=MARKER[s], ls="", ms=4,
                          label=LABEL[s] + (" (ours)" if s in OURS else ""),
                          mfc="white" if s in HOLLOW else colour, mec=colour) for s in SIGNALS]
        fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.52, 1.075),
                   handletextpad=0.3, columnspacing=1.6, handlelength=1.2)
        fig.text(0.52, 1.095, name, ha="center", va="bottom", fontsize=8, fontweight="bold", color=colour)
        save(fig, f"figA_estimators_{suffix}")

if __name__ == "__main__":
    main()
