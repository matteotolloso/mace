#!/usr/bin/env python3
"""Rebuttal ablations against their reference experiment, paired per split (CPU only).

Each ablation differs from its reference in one setting and shares its data, splits and member seeds
(LF->HF ablations also share the parent members). For every pair and test role, reads the
support-filtered per-configuration rows of both final ensembles (``reliability_<test>_nocal_raw.csv``),
computes RMSE and, for AU, EU and TU, Spearman, AUSE and ENCE as the five-split aggregate does
(``compare_lf_only.split_metrics``), and writes the five-split mean [95% Student-t interval, df = 4] of
the ablation's value and of the paired contrast ``ablation - reference`` (RMSE, ENCE: geometric mean and
ratio). Pairs whose outputs are missing for any split are reported as missing, never aggregated.

    python -m uq.evaluation.compare_ablations        # -> results/tables/ablations_vs_reference.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from uq import paths, registry
from uq.evaluation.compare_lf_only import GEOMETRIC, interval, split_metrics
from uq.evaluation.replicate_statistics import write_csv

# ablation protocol suffix -> (reference protocol, what changes)
ABLATIONS = {
    "lf-to-hf-reset-var": ("lf-to-hf", "variance head re-initialized at the start of fine-tuning (WP3.2)"),
    "lf-to-hf-beta-nll": ("lf-to-hf", "beta-NLL fine-tuning, beta = 0.5 (WP3.3)"),
    "lf-to-hf-lr1e-4": ("lf-to-hf", "fine-tuning lr 1e-4 instead of 1e-3 (WP3.5/4.3)"),
    "lf-to-hf-lr1e-2": ("lf-to-hf", "fine-tuning lr 1e-2 instead of 1e-3 (WP4.3)"),
    "hf-only-lr1e-3": ("hf-only", "HF-only lr 1e-3 instead of 1e-2 (WP4.2)"),
    "hf-only-lr3e-3": ("hf-only", "HF-only lr 3e-3 instead of 1e-2 (WP4.2)"),
}


def pairs():
    out = []
    for exp in registry.experiments().values():
        group, protocol = exp.id.rsplit("/", 1)
        if protocol in ABLATIONS:
            reference, change = ABLATIONS[protocol]
            out.append((exp, registry.experiment(f"{group}/{reference}"), change))
    return out


def compare(ablation, reference, change, test):
    filename = f"reliability_{test}_nocal_raw.csv"
    tables = {side: [split_metrics(exp.id, s, test, filename) for s in reference.splits]
              for side, exp in (("ablation", ablation), ("reference", reference))}
    base = {"dataset": ablation.dataset_id, "test": test, "ablation": ablation.id, "reference": reference.id,
            "change": change}
    missing = [f"{side} split {s}" for side, ts in tables.items() for s, t in zip(reference.splits, ts) if t is None]
    if missing:
        return [{**base, "missing": "; ".join(missing)}]
    rows = []
    for key in tables["reference"][0]:
        metric, unc = key
        geometric = metric in GEOMETRIC
        values = [t[key] for t in tables["ablation"]]
        ref = [t[key] for t in tables["reference"]]
        if not (all(np.isfinite(values)) and all(np.isfinite(ref))):
            continue
        contrast = [v / r if r > 0 else math.nan for v, r in zip(values, ref)] if geometric \
            else [v - r for v, r in zip(values, ref)]
        mean, low, high = interval(values, geometric)
        r_mean, r_low, r_high = interval(ref, geometric)
        c_mean, c_low, c_high = interval(contrast, geometric)
        rows.append({**base, "metric": metric, "uncertainty_type": unc,
                     "ablation_mean": mean, "ablation_ci95_low": low, "ablation_ci95_high": high,
                     "reference_mean": r_mean, "reference_ci95_low": r_low, "reference_ci95_high": r_high,
                     "contrast": "ratio" if geometric else "difference", "contrast_mean": c_mean,
                     "contrast_ci95_low": c_low, "contrast_ci95_high": c_high, "n_splits": len(values),
                     "per_split_ablation": " ".join(f"{v:.6g}" for v in values),
                     "per_split_reference": " ".join(f"{v:.6g}" for v in ref), "missing": ""})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=paths.RESULTS / "tables" / "ablations_vs_reference.csv")
    args = parser.parse_args(argv)
    rows = []
    for ablation, reference, change in pairs():
        for test in reference.evaluation.get("reliability", reference.dataset.tests):
            rows.extend(compare(ablation, reference, change, test))
    fields = ["dataset", "test", "ablation", "reference", "change", "metric", "uncertainty_type", "ablation_mean",
              "ablation_ci95_low", "ablation_ci95_high", "reference_mean", "reference_ci95_low",
              "reference_ci95_high", "contrast", "contrast_mean", "contrast_ci95_low", "contrast_ci95_high",
              "n_splits", "per_split_ablation", "per_split_reference", "missing"]
    write_csv(args.output, [{f: row.get(f, "") for f in fields} for row in rows])
    complete = len({(r["ablation"], r["test"]) for r in rows if not r.get("missing")})
    print(f"wrote {paths.rel(args.output)}: {complete} complete (ablation, test) pairs, "
          f"{sum(1 for r in rows if r.get('missing'))} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
