#!/usr/bin/env python3
"""Appendix figure (rebuttal item D) - uncertainty estimators: MSE ensemble, MVE ensemble, one MVE model.

Fig. 2 layout (dotplot.py): rows = four test sets, columns = RMSE, AUSE, ENCE; colour = protocol
(HF-only, LF->HF). Per protocol, four signals: EU of a ten-member MSE ensemble (hollow square;
its only signal), EU (square) and TU (diamond) of the paper's ten-member MVE ensemble, and AU of
a single MVE member (hollow circle; mean over the ten members). RMSE: MSE ensemble, MVE ensemble
and single member, in the slots of their signal. Uncalibrated, per atom; five-split means with 95%
Student-t intervals (geometric, log-space for RMSE and ENCE), read from
``results/tables/mse_vs_mve.csv`` (``python -m uq.evaluation.compare_uq``; ledger 2026-11-03/04
HF-only, 2026-11-08 LF->HF). Tests whose MSE ensemble is not evaluated on all five splits are drawn
as "pending". In the ENCE column only the MVE-ensemble EU is faded: the other three signals are the
predictive variance of their estimator.
"""

import csv

from matplotlib.lines import Line2D

from common import MUTED, PROTOCOL_COLOR, ROOT, TESTS, save, style
from dotplot import draw

TABLE = ROOT / "results" / "tables" / "mse_vs_mve.csv"
PROTOCOLS = (("hf-only", "HF-only"), ("lf-to-hf", "LF→HF"))
SIGNALS = ("mse_eu", "mve_eu", "mve_tu", "mve_single_au")
MARKER = {"mse_eu": "s", "mve_eu": "s", "mve_tu": "D", "mve_single_au": "o"}
HOLLOW = {"mse_eu", "mve_single_au"}
RMSE_OF = {"mse_eu": "mse_ens", "mve_tu": "mve_ens", "mve_single_au": "mve_single"}
LABEL = {"mse_eu": "MSE ens. EU", "mve_eu": "MVE ens. EU", "mve_tu": "MVE ens. TU", "mve_single_au": "one MVE AU"}
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


def main():
    plt = style()
    data, missing = load()

    def value(label, protocol, signal, metric):
        if metric == "RMSE":
            if signal not in RMSE_OF:
                return False
            metric, signal = "rmse", RMSE_OF[signal]
        if (label, protocol) in missing:
            return None
        return data[(label, protocol, metric.lower(), signal)]

    fig, _ = draw(plt, [(p, PROTOCOL_COLOR[name]) for p, name in PROTOCOLS], lambda key: SIGNALS, value,
                  hollow=lambda key, s: s in HOLLOW, marker=lambda s: MARKER[s],
                  fade=lambda s: 0.35 if s == "mve_eu" else 1.0, rmse_per_signal=True,
                  figsize=(7.0, 4.0), rmse_ticks=(10, 20, 50, 100), ence_ticks=(0.2, 0.5, 1, 2, 5, 10, 20))
    handles = [Line2D([], [], color=PROTOCOL_COLOR[name], marker="o", ls="", ms=4.5, label=name)
               for _, name in PROTOCOLS]
    handles += [Line2D([], [], color=MUTED, marker=MARKER[s], ls="", ms=4, label=LABEL[s],
                       mfc="white" if s in HOLLOW else MUTED, mec=MUTED) for s in SIGNALS]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.52, 1.05),
               handletextpad=0.3, columnspacing=1.1)
    save(fig, "figA_estimators")


if __name__ == "__main__":
    main()
