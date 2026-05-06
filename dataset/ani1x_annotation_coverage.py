#!/usr/bin/env python3
"""Utilities to summarize label coverage for ANI-1x theory keys."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


DEFAULT_THEORY_KEYS: Tuple[str, ...] = (
    "ccsd(t)_cbs.energy",
    "hf_dz.energy",
    "hf_tz.energy",
    "hf_qz.energy",
    "mp2_dz.corr_energy",
    "mp2_tz.corr_energy",
    "mp2_qz.corr_energy",
    "npno_ccsd(t)_dz.corr_energy",
    "npno_ccsd(t)_tz.corr_energy",
    "tpno_ccsd(t)_dz.corr_energy",
    "wb97x_dz.energy",
    "wb97x_tz.energy",
)


@dataclass
class CoverageResult:
    summary: pd.DataFrame
    system_details: pd.DataFrame
    total_systems: int
    total_configs: int


def _finite_config_mask(values: np.ndarray, n_configs: int) -> np.ndarray:
    """Return one boolean per configuration indicating a valid annotation."""
    arr = np.asarray(values)

    if arr.ndim == 0:
        return np.full(n_configs, bool(np.isfinite(arr).all()), dtype=bool)

    if arr.shape[0] != n_configs:
        return np.zeros(n_configs, dtype=bool)

    if arr.ndim == 1:
        return np.isfinite(arr)

    flattened = arr.reshape(n_configs, -1)
    return np.all(np.isfinite(flattened), axis=1)


def summarize_annotation_coverage(
    h5_path: str | Path,
    theory_keys: Sequence[str] = DEFAULT_THEORY_KEYS,
) -> CoverageResult:
    """
    Summarize how many ANI-1x systems/configurations are annotated for each key.

    A configuration counts as annotated when the value for that configuration is finite.
    """
    summary_stats: Dict[str, Dict[str, float]] = {
        key: {
            "systems_with_dataset": 0,
            "systems_with_any_valid_configs": 0,
            "systems_with_all_configs_valid": 0,
            "systems_with_partial_configs_valid": 0,
            "systems_with_zero_valid_configs": 0,
            "configs_in_present_systems": 0,
            "valid_configs": 0,
        }
        for key in theory_keys
    }
    system_rows: List[Dict[str, object]] = []

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as handle:
        total_systems = len(handle.keys())
        total_configs = 0

        for system_name in handle.keys():
            group = handle[system_name]
            n_configs = int(group["coordinates"].shape[0])
            total_configs += n_configs

            for key in theory_keys:
                if key not in group:
                    continue

                mask = _finite_config_mask(group[key][()], n_configs)
                valid_configs = int(mask.sum())
                coverage_fraction = valid_configs / n_configs if n_configs else 0.0

                if valid_configs == 0:
                    coverage_type = "none"
                elif valid_configs == n_configs:
                    coverage_type = "all"
                else:
                    coverage_type = "partial"

                system_rows.append(
                    {
                        "theory_key": key,
                        "system": system_name,
                        "n_configs": n_configs,
                        "valid_configs": valid_configs,
                        "missing_configs": n_configs - valid_configs,
                        "coverage_fraction": coverage_fraction,
                        "coverage_type": coverage_type,
                    }
                )

                stats = summary_stats[key]
                stats["systems_with_dataset"] += 1
                stats["configs_in_present_systems"] += n_configs
                stats["valid_configs"] += valid_configs

                if coverage_type == "all":
                    stats["systems_with_any_valid_configs"] += 1
                    stats["systems_with_all_configs_valid"] += 1
                elif coverage_type == "partial":
                    stats["systems_with_any_valid_configs"] += 1
                    stats["systems_with_partial_configs_valid"] += 1
                else:
                    stats["systems_with_zero_valid_configs"] += 1

    summary_rows: List[Dict[str, object]] = []
    for key in theory_keys:
        stats = summary_stats[key]
        present_systems = int(stats["systems_with_dataset"])
        present_configs = int(stats["configs_in_present_systems"])
        valid_configs = int(stats["valid_configs"])
        summary_rows.append(
            {
                "theory_key": key,
                **{name: int(value) for name, value in stats.items()},
                "system_fraction": present_systems / total_systems if total_systems else 0.0,
                "valid_config_fraction": (
                    valid_configs / present_configs if present_configs else 0.0
                ),
                "dataset_fraction_overall": (
                    valid_configs / total_configs if total_configs else 0.0
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        by=["systems_with_any_valid_configs", "valid_configs", "theory_key"],
        ascending=[False, False, True],
    )
    system_details = pd.DataFrame(system_rows).sort_values(
        by=["theory_key", "coverage_type", "coverage_fraction", "system"],
        ascending=[True, True, True, True],
    )
    return CoverageResult(
        summary=summary.reset_index(drop=True),
        system_details=system_details.reset_index(drop=True),
        total_systems=total_systems,
        total_configs=total_configs,
    )


def partial_coverage_details(
    system_details: pd.DataFrame, theory_key: str, limit: int | None = None
) -> pd.DataFrame:
    """Return only partially annotated systems for a specific key."""
    partial = system_details[
        (system_details["theory_key"] == theory_key)
        & (system_details["coverage_type"] == "partial")
    ].copy()
    if limit is not None:
        partial = partial.head(limit)
    return partial.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ANI-1x annotation coverage for selected theory keys."
    )
    parser.add_argument("--h5", required=True, help="Path to the ANI-1x HDF5 file.")
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="Theory key to analyze. Repeat to pass multiple keys.",
    )
    parser.add_argument(
        "--partial-key",
        default=None,
        help="If set, also print systems with partial coverage for this key.",
    )
    parser.add_argument(
        "--partial-limit",
        type=int,
        default=20,
        help="Maximum number of partially covered systems to print.",
    )
    return parser.parse_args()


def _format_fraction_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: f"{value:.3%}")
    return formatted


def main() -> None:
    args = _parse_args()
    theory_keys = tuple(args.keys) if args.keys else DEFAULT_THEORY_KEYS
    result = summarize_annotation_coverage(args.h5, theory_keys=theory_keys)

    summary = _format_fraction_columns(
        result.summary,
        columns=("system_fraction", "valid_config_fraction", "dataset_fraction_overall"),
    )
    print(
        f"Total systems: {result.total_systems} | Total configurations: {result.total_configs}"
    )
    print(summary.to_string(index=False))

    if args.partial_key:
        partial = partial_coverage_details(
            result.system_details, args.partial_key, limit=args.partial_limit
        )
        print()
        if partial.empty:
            print(f"No partially covered systems found for {args.partial_key}.")
        else:
            print(f"Partial coverage for {args.partial_key}:")
            print(partial.to_string(index=False))


if __name__ == "__main__":
    main()
