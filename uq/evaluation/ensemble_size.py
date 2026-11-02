#!/usr/bin/env python3
"""Reliability metrics as a function of ensemble size, from saved member predictions (CPU).

Reads ``members_<role>.npz`` (``uq.evaluation.member_predictions``) of one split and,
for every ensemble size M = 1..10, evaluates the ensembles formed by the ten cyclic
windows of M consecutive member seeds ({m, m+1, ..., m+M-1} mod 10) and averages each
metric over the windows (all ten windows are the full ensemble when M = 10). M = 1 is
the single-model arm: each member alone, AU = its own predicted variance, EU = 0.

Per window and per configuration (per atom, as in ``reliability``): prediction = mean
member energy, AU = mean member variance, EU = population variance of member energies,
TU = AU + EU. Metrics are those of ``reliability`` (RMSE of the window's mean
prediction, Spearman, AUSE, ENCE with 15 equal-count bins), computed after the
training-support filter (datasets with ``support_filter: true``), like the
five-split aggregates. A constant signal (EU at M = 1, AU of an MSE ensemble) has
undefined metrics, written as NaN.

    python -m uq.evaluation.ensemble_size \\
        --members-npz experiments/<id>/evaluation/split_<s>/members_<role>.npz \\
        --output-csv experiments/<id>/evaluation/split_<s>/ensemble_size_<role>.csv

The experiment, split and test role are read from the ``--members-npz`` path.

``uq aggregate`` summarizes ``ensemble_size_<role>.csv`` over splits (mean and 95%
Student-t interval; geometric for RMSE and ENCE) into ``evaluation/ensemble_size_<role>.csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from uq import registry
from uq.evaluation.reliability import (
    build_binned_rows,
    compute_ause_summary,
    compute_ence_summary,
    compute_energy_rmse,
    compute_spearman_summary,
)
from uq.evaluation.replicate_statistics import write_csv
from uq.evaluation.support_filter import experiment_from_path, experiment_kind, supported, test_file

LOGGER = logging.getLogger(__name__)
UNCERTAINTIES = ("aleatoric", "epistemic", "total")
NUM_BINS = 15


def load_members(path: Path) -> Dict[str, np.ndarray]:
    """Per-atom member energies/variances and reference, members ordered by seed."""
    data = np.load(path)
    order = np.argsort(data["member_seeds"])
    atoms = data["num_atoms"].astype(float)
    return {
        "seeds": data["member_seeds"][order],
        "energy": data["pred_energy"][order].astype(float) / atoms,
        "var": data["pred_var"][order].astype(float) / atoms**2,
        "ref": data["ref_energy"].astype(float) / atoms,
    }


def support_mask(experiment: str, role: str, split: int, count: int) -> np.ndarray:
    kind = experiment_kind(experiment)
    if kind is None:
        return np.ones(count, dtype=bool)
    keep = supported(kind, split, test_file(experiment, role))
    if len(keep) != count:
        raise RuntimeError(f"{experiment} split {split} {role}: {count} predictions but {len(keep)} geometries")
    return keep


def window_metrics(energy: np.ndarray, var: np.ndarray, ref: np.ndarray) -> Dict[tuple, float]:
    """Metrics of one ensemble (rows of ``energy``/``var`` are its members)."""
    pred = energy.mean(axis=0)
    au = var.mean(axis=0)
    eu = energy.var(axis=0)          # population variance (ddof=0), as reliability
    error = pred - ref
    rows = [{"config_index": i, "error": float(e), "sq_error": float(e * e), "aleatoric_var": float(a),
             "epistemic_var": float(u), "total_var": float(a + u)}
            for i, (e, a, u) in enumerate(zip(error, au, eu))]
    out = {("rmse", "ensemble"): compute_energy_rmse(rows)}
    constant = {name for name, values in (("aleatoric", au), ("epistemic", eu), ("total", au + eu))
                if np.ptp(values) == 0.0}
    summaries = {"spearman": compute_spearman_summary(rows), "ause": compute_ause_summary(rows),
                 "ence": compute_ence_summary(build_binned_rows(rows, num_bins=NUM_BINS))}
    for metric, values in summaries.items():
        for name in UNCERTAINTIES:
            out[(metric, name)] = float("nan") if name in constant else float(values[name])
    return out


def ensemble_size_rows(members: Dict[str, np.ndarray], keep: np.ndarray) -> List[Dict[str, object]]:
    energy, var, ref = members["energy"][:, keep], members["var"][:, keep], members["ref"][keep]
    total = energy.shape[0]
    rows = []
    for size in range(1, total + 1):
        windows = [[(start + j) % total for j in range(size)] for start in range(total if size < total else 1)]
        per_window = [window_metrics(energy[w], var[w], ref) for w in windows]
        for key in per_window[0]:
            values = np.asarray([metrics[key] for metrics in per_window], dtype=float)
            rows.append({"members": size, "metric": key[0], "uncertainty_type": key[1],
                         "value": float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan"),
                         "windows": len(windows)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members-npz", type=Path, required=True,
                        help="experiments/<id>/evaluation/split_<s>/members_<role>.npz")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    exp_id = experiment_from_path(args.members_npz)
    if exp_id is None or not args.members_npz.parent.name.startswith("split_") \
            or not args.members_npz.name.startswith("members_"):
        raise SystemExit(f"{args.members_npz}: expected experiments/<id>/evaluation/split_<s>/members_<role>.npz")
    exp = registry.experiment(exp_id)
    split = int(args.members_npz.parent.name.removeprefix("split_"))
    role = args.members_npz.stem.removeprefix("members_")
    members = load_members(args.members_npz)
    if list(members["seeds"]) != list(exp.members):
        raise RuntimeError(f"{args.members_npz}: members {list(members['seeds'])}, spec has {list(exp.members)}")
    keep = support_mask(exp.id, role, split, members["ref"].size)
    LOGGER.info("%s split %d %s: %d of %d configurations inside training support",
                exp.id, split, role, int(keep.sum()), keep.size)
    write_csv(args.output_csv, ensemble_size_rows(members, keep))
    LOGGER.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
