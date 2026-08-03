"""Shared statistics for independent dataset-split replicates."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# Two-sided 95% Student-t critical values. Five splits use df=4 -> 2.776.
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty aggregate CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: Iterable[float]) -> Dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    n = int(finite.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n": 0,
        }

    mean = float(np.mean(finite))
    if n == 1:
        return {
            "mean": mean,
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n": 1,
        }

    std = float(np.std(finite, ddof=1))
    critical = T_CRITICAL_95.get(n - 1, 1.96)
    half_width = critical * std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "n": n,
    }


def aggregate_rows(
    replicate_rows: Sequence[Sequence[Dict[str, str | float | int]]],
    *,
    key_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(
        lambda: {metric: [] for metric in metric_fields}
    )
    key_values: Dict[Tuple[str, ...], Dict[str, object]] = {}

    for rows in replicate_rows:
        seen = set()
        for row in rows:
            key = tuple(str(row[field]) for field in key_fields)
            if key in seen:
                raise RuntimeError(f"Duplicate key {key} within one replicate CSV.")
            seen.add(key)
            key_values.setdefault(key, {field: row[field] for field in key_fields})
            for metric in metric_fields:
                try:
                    grouped[key][metric].append(float(row[metric]))
                except (KeyError, TypeError, ValueError):
                    grouped[key][metric].append(float("nan"))

    output: List[Dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: tuple(_sort_value(value) for value in item)):
        row_out = dict(key_values[key])
        for metric in metric_fields:
            stats = summarize(grouped[key][metric])
            row_out[metric] = stats["mean"]
            for suffix in ("mean", "std", "ci95_low", "ci95_high", "n"):
                row_out[f"{metric}_{suffix}"] = stats[suffix]
        output.append(row_out)
    return output


def confidence_arrays(
    rows: Sequence[Dict[str, object]], metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray([float(row.get(f"{metric}_mean", row[metric])) for row in rows])
    low = np.asarray([float(row.get(f"{metric}_ci95_low", np.nan)) for row in rows])
    high = np.asarray([float(row.get(f"{metric}_ci95_high", np.nan)) for row in rows])
    return mean, low, high


def _sort_value(value: str) -> Tuple[int, object]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)
