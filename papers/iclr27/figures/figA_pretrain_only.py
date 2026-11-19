#!/usr/bin/env python3
"""Appendix figure (rebuttal item A) - pretrain-only (LF-only) baseline on the CC test sets.

Fig. 2 layout (dotplot.py): rows = four test sets, columns = RMSE, AUSE, ENCE of the final
ensembles (members at their best validation-NLL checkpoint), uncalibrated, per atom,
support-filtered; colour = protocol, marker = signal. Protocols: LF-only with per-element
reference energies refit to the CC training residuals (step ``lfonly_baseline``:
``lfonly_reliability_<test>_nocal_raw.csv`` in the B / D split folders), HF-only (E / F) and
LF->HF (B / D). Five-split means with 95% Student-t intervals (geometric, log-space for RMSE
and ENCE), recomputed from the per-configuration rows as in Fig. 2 (ledger 2026-11-07
"LF-only (pretrain-only) ensembles on the CC labels").
"""

import csv

from common import (
    EXPERIMENT, PROTOCOL_COLOR, SIGNALS, SPLITS, TESTS, experiment_kind, interval, legend_handles,
    final_metrics, metrics, save, split_dir, style, supported, test_file,
)
from dotplot import draw

SERIES = (("LF-only", "LF-only (E$_0$ refit)"), ("HF-only", "HF-only"), ("LF→HF", "LF→HF"))


def lf_only_metrics(kind, test):
    """Per-split metrics of the LF-only members on the CC test set (E0 refit), support-filtered."""
    host = EXPERIMENT[(kind, "LF→HF")]  # the step writes its outputs into the LF->HF folders
    out = []
    for s in SPLITS:
        path = split_dir(host, s) / f"lfonly_reliability_{test}_nocal_raw.csv"
        with path.open() as handle:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
        keep = supported(experiment_kind(host), s, test_file(host, test))
        if len(keep) != len(rows):
            raise RuntimeError(f"{path}: {len(rows)} rows vs {len(keep)} geometries")
        out.append(metrics([r for r, k in zip(rows, keep) if k]))
    return out


def main():
    plt = style()
    per_split = {}
    for kind, test, label in TESTS:
        per_split[(label, "LF-only")] = lf_only_metrics(kind, test)
        for p in ("HF-only", "LF→HF"):
            per_split[(label, p)] = final_metrics(EXPERIMENT[(kind, p)], test)

    def value(label, key, signal, metric):
        values = per_split[(label, key)]
        log = metric in ("RMSE", "ENCE")
        return interval([m["RMSE"] if signal is None else m[(metric, signal)] for m in values], log)

    fig, _ = draw(plt, [(k, PROTOCOL_COLOR[k]) for k, _ in SERIES], lambda key: SIGNALS, value,
                  figsize=(7.0, 3.6), ence_ticks=(0.2, 0.5, 1, 2, 5, 10, 20))
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=PROTOCOL_COLOR[k], marker="o", ls="", ms=4.5, label=label)
               for k, label in SERIES] + legend_handles()[2:]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.52, 1.06),
               handletextpad=0.3, columnspacing=1.2)
    save(fig, "figA_pretrain_only")


if __name__ == "__main__":
    main()
