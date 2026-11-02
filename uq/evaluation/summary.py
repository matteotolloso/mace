"""Headline metrics of one experiment from its tracked five-split aggregates.

    uq summary B                 # every test role, uncalibrated
    uq summary B --test ood --cal
    uq summary B --test ood --members 1   # ensemble of M members (1 = single model), if aggregated

Reads ``experiments/<id>/evaluation/reliability_<test>_<nocal|cal>_summary.csv``
(written by ``uq aggregate``; support-filtered rows) and prints, per metric and
signal, the five-split mean with its 95% Student-t interval (df = n - 1).
RMSE and ENCE are also shown as geometric means with log-space intervals, the
convention of the paper figures. RMSE is in meV/atom. With ``--members M`` it reads
``ensemble_size_<test>.csv`` instead (``uq.evaluation.ensemble_size``: uncalibrated,
mean over the ten cyclic windows of M members; M = 10 is the final ensemble).
"""

from __future__ import annotations

import csv

from uq import paths, registry

SIGNALS = (("aleatoric", "AU"), ("epistemic", "EU"), ("total", "TU"))
METRICS = ("rmse", "spearman", "ause", "ence")


def _cell(row, prefix="value", scale=1.0):
    try:
        mean, low, high = (float(row[f"{prefix}_{k}"]) * scale for k in ("mean", "ci95_low", "ci95_high"))
    except (KeyError, ValueError):
        return "-"
    if mean != mean:  # NaN
        return "-"
    digits = 1 if scale != 1.0 else 3
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _geo(row, scale=1.0):
    return _cell(row, "value_geometric", scale) if row.get("value_geometric_n", "0") not in ("", "0") else "-"


def main(experiment, test=None, calibrated=False, members=None) -> int:
    exp = registry.experiment(experiment)
    tag = "cal" if calibrated else "nocal"
    if members is not None and calibrated:
        raise SystemExit("--members reads the uncalibrated ensemble-size aggregate; drop --cal")
    default_roles = exp.evaluation.get("member_predictions" if members is not None else "reliability",
                                       exp.dataset.tests)
    roles = [test] if test else list(default_roles)
    status = 0
    for role in roles:
        name = f"ensemble_size_{role}.csv" if members is not None else f"reliability_{role}_{tag}_summary.csv"
        path = paths.evaluation_dir(exp.id) / name
        if not path.is_file():
            print(f"{exp.label}: no {paths.rel(path)} (run `uq aggregate {exp.legacy_id or exp.id}`)")
            status = 1
            continue
        with path.open() as handle:
            rows = {(r["metric"], r["uncertainty_type"]): r for r in csv.DictReader(handle)
                    if members is None or int(r["members"]) == members}
        if not rows:
            print(f"{exp.label}: no rows for {members} members in {paths.rel(path)}")
            status = 1
            continue
        n = next(iter(rows.values())).get("value_n", "?")
        size = f" | M={members} members" if members is not None else ""
        print(f"{exp.label} | {role} | {'isotonic-recalibrated' if calibrated else 'uncalibrated'}{size} | "
              f"n={n} splits | {paths.rel(path)}")
        rmse = rows.get(("rmse", "ensemble"), {})
        print(f"  RMSE (meV/atom)  mean {_cell(rmse, scale=1000)}   geometric {_geo(rmse, scale=1000)}")
        width = 26
        print(f"  {'metric':<9}" + "".join(f"{label:<{width}}" for _, label in SIGNALS))
        for metric in METRICS[1:]:
            cells = [_cell(rows.get((metric, name), {})) for name, _ in SIGNALS]
            print(f"  {metric:<9}" + "".join(f"{c:<{width}}" for c in cells))
        ence = [_geo(rows.get(("ence", name), {})) for name, _ in SIGNALS]
        print(f"  {'ence geo':<9}" + "".join(f"{c:<{width}}" for c in ence))
    return status
