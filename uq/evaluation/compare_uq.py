#!/usr/bin/env python3
"""MSE ensemble (EU only) against its MVE twin, paired per split (CPU only).

For every ``<dataset>/mace-mse/<protocol>`` experiment with ``member_predictions`` and
its twin ``<dataset>/mace-mve/<protocol>``, reads the per-split
``ensemble_size_<role>.csv`` of both (``uq.evaluation.ensemble_size``) and compares,
on the same split (same data, same member seeds), the uncertainty signals

    mse_eu          MSE ensemble, EU (its only signal), M = 10
    mve_eu          MVE ensemble, EU, M = 10        (effect of the loss on EU)
    mve_tu          MVE ensemble, TU = AU + EU, M = 10   (the paper's predictive variance)
    mve_au          MVE ensemble, AU, M = 10
    mve_single_au   one MVE member, AU (mean over the ten members), M = 1

and the RMSE of the predictions they come with (``mse_ens``, ``mve_ens``, ``mve_single``).

Output (``--output``, default ``results/tables/mse_vs_mve.csv``), one row per
dataset, protocol, test role, metric and signal: the five-split mean and 95% Student-t
interval of the value and of the paired difference ``signal - mse_eu`` (``rmse``
and ``ence``: geometric mean of the value and of the ratio ``signal / mse_eu`` with
log-space intervals). A difference interval that excludes 0 (a ratio interval that
excludes 1) means the signal differs from the MSE-ensemble EU on that test. Pairs with
a missing split are reported as missing, never aggregated over fewer splits.

    python -m uq.evaluation.compare_uq
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from uq import paths, registry
from uq.evaluation.replicate_statistics import summarize, summarize_log, write_csv

SIGNALS = {  # name -> (members, uncertainty_type) inside ensemble_size_<role>.csv, and the experiment side
    "mse_eu": ("mse", 10, "epistemic"),
    "mve_eu": ("mve", 10, "epistemic"),
    "mve_tu": ("mve", 10, "total"),
    "mve_au": ("mve", 10, "aleatoric"),
    "mve_single_au": ("mve", 1, "aleatoric"),
}
PREDICTIONS = {"mse_ens": ("mse", 10), "mve_ens": ("mve", 10), "mve_single": ("mve", 1)}
METRICS = ("spearman", "ause", "ence")
GEOMETRIC = {"rmse", "ence"}
REFERENCE = {"signal": "mse_eu", "prediction": "mse_ens"}


def pairs() -> List[tuple]:
    """(mse experiment, mve twin) for every MSE experiment that saves member predictions."""
    out = []
    for exp in registry.experiments().values():
        if exp.uq != "mse-ensemble" or not exp.evaluation.get("member_predictions"):
            continue
        twin_id = exp.id.replace(f"/{exp.model}-mse/", f"/{exp.model}-mve/")
        out.append((exp, registry.experiment(twin_id)))
    return out


def read_table(exp, split: int, role: str) -> Optional[Dict[tuple, float]]:
    path = paths.split_evaluation_dir(exp.id, split) / f"ensemble_size_{role}.csv"
    if not path.is_file():
        return None
    with path.open() as handle:
        return {(int(row["members"]), row["metric"], row["uncertainty_type"]): float(row["value"])
                for row in csv.DictReader(handle)}


def interval(values: List[float], geometric: bool) -> Dict[str, float]:
    stats = summarize_log(values) if geometric else summarize(values)
    return {"mean": stats["mean"], "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"]}


def compare(mse, mve, role: str) -> List[Dict[str, object]]:
    tables = {"mse": [read_table(mse, s, role) for s in mse.splits],
              "mve": [read_table(mve, s, role) for s in mve.splits]}
    missing = [f"{side} split {s}" for side, exp in (("mse", mse), ("mve", mve))
               for s, table in zip(exp.splits, tables[side]) if table is None]
    base = {"dataset": mse.dataset_id, "protocol": mse.protocol, "test": role,
            "mse_experiment": mse.id, "mve_experiment": mve.id}
    if missing:
        return [{**base, "metric": "", "signal": "", "missing": "; ".join(missing)}]
    rows = []
    items = [("rmse", name, side, members, "ensemble") for name, (side, members) in PREDICTIONS.items()]
    items += [(metric, name, side, members, unc) for metric in METRICS
              for name, (side, members, unc) in SIGNALS.items()]
    for metric, name, side, members, unc in items:
        reference = REFERENCE["prediction" if metric == "rmse" else "signal"]
        ref_side, ref_members, ref_unc = (PREDICTIONS[reference] + ("ensemble",)) if metric == "rmse" \
            else SIGNALS[reference]
        values = [table[(members, metric, unc)] for table in tables[side]]
        ref_values = [table[(ref_members, metric, ref_unc)] for table in tables[ref_side]]
        geometric = metric in GEOMETRIC
        if geometric:
            contrast = [v / r if r > 0 else math.nan for v, r in zip(values, ref_values)]
        else:
            contrast = [v - r for v, r in zip(values, ref_values)]
        if not all(np.isfinite(values)):
            continue  # undefined signal (e.g. constant AU); never summarized over fewer splits
        value = interval(values, geometric)
        diff = interval(contrast, geometric)
        rows.append({**base, "metric": metric, "signal": name, "reference": reference,
                     "value_mean": value["mean"], "value_ci95_low": value["ci95_low"],
                     "value_ci95_high": value["ci95_high"],
                     "contrast": "ratio" if geometric else "difference",
                     "contrast_mean": diff["mean"], "contrast_ci95_low": diff["ci95_low"],
                     "contrast_ci95_high": diff["ci95_high"], "n_splits": len(values),
                     "per_split": " ".join(f"{v:.6g}" for v in values), "missing": ""})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=paths.RESULTS / "tables" / "mse_vs_mve.csv")
    args = parser.parse_args(argv)
    rows = []
    for mse, mve in pairs():
        for role in mse.evaluation["member_predictions"]:
            rows.extend(compare(mse, mve, role))
    write_csv(args.output, rows)
    print(f"wrote {paths.rel(args.output)} ({len(rows)} rows)")
    for row in rows:
        if row.get("missing"):
            print(f"  missing: {row['mse_experiment']} vs {row['mve_experiment']} ({row['test']}): {row['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
