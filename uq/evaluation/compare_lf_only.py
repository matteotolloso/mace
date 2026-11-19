#!/usr/bin/env python3
"""LF-only ensemble on HF labels against LF->HF and HF-only, paired per split (CPU only; rebuttal WP2).

For every LF->HF experiment with ``lf_only_baseline`` (B, D), reads per split the support-filtered
per-configuration rows of

    lf_only   its parent's ensemble after the E0 refit   (``lfonly_reliability_<test>_nocal_raw.csv``)
    lf_to_hf  the fine-tuned ensemble                    (``reliability_<test>_nocal_raw.csv``)
    hf_only   the HF-only twin (same dataset, CC files)  (``<dataset>/<model>-<uq>/hf-only``)

computes the ensemble RMSE and, for AU, EU and TU, Spearman, AUSE and ENCE exactly as the five-split
aggregate does (same functions, same support filter, 15 bins), and writes the five-split mean with 95%
Student-t interval (df = 4) of each value and of the paired contrasts ``lf_to_hf - lf_only`` and
``hf_only - lf_only`` (``rmse``, ``ence``: geometric mean and ratio, log-space intervals). A contrast
interval that excludes 0 (a ratio interval that excludes 1) means the two differ on that test.

    python -m uq.evaluation.compare_lf_only        # -> results/tables/lf_only_vs_hf.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from uq import paths, registry
from uq.evaluation.aggregate import UNCERTAINTIES, supported_replicates
from uq.evaluation.reliability import (build_binned_rows, compute_ause_summary, compute_ence_summary,
                                       compute_energy_rmse, compute_spearman_summary)
from uq.evaluation.replicate_statistics import summarize, summarize_log, write_csv

GEOMETRIC = {"rmse", "ence"}


def split_metrics(exp_id, split, test, filename):
    path = paths.split_evaluation_dir(exp_id, split) / filename
    if not path.is_file():
        return None
    with path.open() as handle:
        raw = list(csv.DictReader(handle))
    rows = supported_replicates([raw], exp_id, test, [split])[0]
    bins = build_binned_rows(rows, num_bins=15)
    out = {("rmse", "ensemble"): compute_energy_rmse(rows)}
    for metric, values in (("spearman", compute_spearman_summary(rows)), ("ause", compute_ause_summary(rows)),
                           ("ence", compute_ence_summary(bins))):
        out.update({(metric, unc): float(values[unc]) for unc in UNCERTAINTIES})
    return out


def interval(values, geometric):
    stats = summarize_log(values) if geometric else summarize(values)
    return stats["mean"], stats["ci95_low"], stats["ci95_high"]


def compare(child, test):
    hf_only = registry.experiment(child.id.rsplit("/", 1)[0] + "/hf-only")
    sources = {"lf_only": (child.id, f"lfonly_reliability_{test}_nocal_raw.csv"),
               "lf_to_hf": (child.id, f"reliability_{test}_nocal_raw.csv"),
               "hf_only": (hf_only.id, f"reliability_{test}_nocal_raw.csv")}
    tables = {name: [split_metrics(exp_id, s, test, f) for s in child.splits] for name, (exp_id, f) in sources.items()}
    base = {"dataset": child.dataset_id, "test": test, "lf_to_hf": child.id, "lf_only": child.parent.id,
            "hf_only": hf_only.id}
    missing = [f"{name} split {s}" for name, ts in tables.items() for s, t in zip(child.splits, ts) if t is None]
    if missing:
        return [{**base, "missing": "; ".join(missing)}]
    rows = []
    for key in tables["lf_only"][0]:
        metric, unc = key
        geometric = metric in GEOMETRIC
        ref = [t[key] for t in tables["lf_only"]]
        for name in ("lf_only", "lf_to_hf", "hf_only"):
            values = [t[key] for t in tables[name]]
            if not all(np.isfinite(values)):
                continue
            mean, low, high = interval(values, geometric)
            row = {**base, "metric": metric, "uncertainty_type": unc, "model": name, "value_mean": mean,
                   "value_ci95_low": low, "value_ci95_high": high, "n_splits": len(values),
                   "per_split": " ".join(f"{v:.6g}" for v in values), "missing": ""}
            if name != "lf_only":
                contrast = [v / r if r > 0 else math.nan for v, r in zip(values, ref)] if geometric \
                    else [v - r for v, r in zip(values, ref)]
                c_mean, c_low, c_high = interval(contrast, geometric)
                row.update({"contrast": ("ratio" if geometric else "difference") + " vs lf_only",
                            "contrast_mean": c_mean, "contrast_ci95_low": c_low, "contrast_ci95_high": c_high})
            rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=paths.RESULTS / "tables" / "lf_only_vs_hf.csv")
    args = parser.parse_args(argv)
    rows = []
    for exp in registry.experiments().values():
        if exp.parent and exp.evaluation.get("lf_only_baseline"):
            for test in exp.evaluation.get("reliability", exp.dataset.tests):
                rows.extend(compare(exp, test))
    fields = ["dataset", "test", "metric", "uncertainty_type", "model", "value_mean", "value_ci95_low",
              "value_ci95_high", "contrast", "contrast_mean", "contrast_ci95_low", "contrast_ci95_high",
              "n_splits", "per_split", "lf_only", "lf_to_hf", "hf_only", "missing"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, [{f: row.get(f, "") for f in fields} for row in rows])
    print(f"wrote {paths.rel(args.output)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
