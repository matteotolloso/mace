#!/usr/bin/env python3
"""Effect of LF pretraining on each uncertainty signal: LF->HF minus HF-only, paired per split (CPU only).

Reads ``results/tables/mse_vs_mve.csv`` (``uq.evaluation.compare_uq``), whose ``per_split``
column holds the five per-split values of every (dataset, protocol, test, metric, signal), and
pairs, on the same split, the ``lf-to-hf`` row with the ``hf-only`` row. The question (rebuttal
WP3.6): does fine-tuning from LF change the EU of a plain MSE ensemble the way it changes the
EU / TU of the MVE ensemble?

Output (``--output``, default ``results/tables/mse_vs_mve_pretraining.csv``), one row per dataset,
test, metric and signal: five-split mean of each protocol and the paired contrast
``lf-to-hf - hf-only`` (Spearman, AUSE: difference) or ``lf-to-hf / hf-only`` (RMSE, ENCE:
geometric mean ratio, log-space interval), 95% Student-t, df = 4. A dataset whose LF->HF or
HF-only rows are missing a split is reported as missing, never aggregated over fewer splits.

    python -m uq.evaluation.compare_uq_protocols
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from uq import paths
from uq.evaluation.compare_uq import GEOMETRIC
from uq.evaluation.replicate_statistics import summarize, summarize_log, write_csv

PROTOCOLS = ("lf-to-hf", "hf-only")


def compare(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    table = {(r["dataset"], r["protocol"], r["test"], r["metric"], r["signal"]): r for r in rows if r["metric"]}
    missing = {(r["dataset"], r["protocol"], r["test"]) for r in rows if r["missing"]}
    out = []
    for (dataset, protocol, test, metric, signal), hf in sorted(table.items()):
        if protocol != "hf-only":
            continue
        base = {"dataset": dataset, "test": test, "metric": metric, "signal": signal}
        lfhf = table.get((dataset, "lf-to-hf", test, metric, signal))
        gaps = [p for p in PROTOCOLS if (dataset, p, test) in missing]
        if gaps or lfhf is None:
            out.append({**base, "missing": " ".join(gaps) or "lf-to-hf"})
            continue
        a = np.array([float(v) for v in lfhf["per_split"].split()])
        b = np.array([float(v) for v in hf["per_split"].split()])
        if a.size != 5 or b.size != 5:
            out.append({**base, "missing": f"splits lf-to-hf {a.size}, hf-only {b.size}"})
            continue
        geometric = metric in GEOMETRIC
        stats = summarize_log(a / b) if geometric else summarize(a - b)
        out.append({**base, "missing": "", "lf_to_hf_mean": lfhf["value_mean"], "hf_only_mean": hf["value_mean"],
                    "contrast": "ratio" if geometric else "difference", "contrast_mean": stats["mean"],
                    "contrast_ci95_low": stats["ci95_low"], "contrast_ci95_high": stats["ci95_high"],
                    "n_splits": stats["n"]})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=paths.RESULTS / "tables" / "mse_vs_mve.csv")
    parser.add_argument("--output", type=Path, default=paths.RESULTS / "tables" / "mse_vs_mve_pretraining.csv")
    args = parser.parse_args(argv)
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = compare(rows)
    write_csv(args.output, out)
    print(f"wrote {args.output} ({len(out)} rows)")
    for row in out:
        if row["missing"]:
            print(f"  missing: {row['dataset']} {row['test']} {row['metric']} {row['signal']}: {row['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
