#!/usr/bin/env python3
"""Recompute AL metrics restricted to the training data's geometric support region.

Held-out ANI Energy-OOD contains configurations compressed far below anything in
training (down to 0.646 A between two atoms, against a training minimum of about
0.83 A). Models extrapolate catastrophically there, and because RMSE averages
squares a single such configuration can supply over 99% of a reported RMSE. The
split-to-split scatter this produces is an extrapolation artifact, not
experimental variance between repetitions.

This program restricts every test to the region where the models have training
support. For each dataset split the threshold is the smallest interatomic
distance occurring anywhere in that split's own training data (cc_train, cc_val,
dft_train, dft_val); held-out configurations below it are excluded. The rule is
derived from the training data alone, never from predictions or errors, and the
same configurations are removed for every condition, regime and test, so the
paired comparisons stay valid.

Nothing existing is modified. Cached predictions are read and verified against
the hashes recorded in each run's metrics; all output goes to a new aggregate
directory. No model inference, training or GPU is required.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import (
    ACQUISITION_METRICS, HERE, ROOT, comparison, inventory, load_json, save_json, verify_inventory,
)
from five_splits import REGIMES, TESTS, aggregate, write_report

TRAINING_FILES = ("cc_train", "cc_val", "dft_train", "dft_val")
CONDITIONS = tuple(name for regime in REGIMES for name in
                   (f"before_{regime}", f"{regime}_random",
                    *(f"{regime}_{m}" for m in ACQUISITION_METRICS)))


def minimum_distances(path):
    """Smallest interatomic distance in each frame, in Angstrom."""
    from ase.io import read

    out = []
    for atoms in read(str(path), index=":"):
        if len(atoms) < 2:
            out.append(math.inf)
            continue
        distances = atoms.get_all_distances()
        np.fill_diagonal(distances, np.inf)
        out.append(float(distances.min()))
    return out


def frame_ids(path):
    from ase.io import read

    ids = []
    for atoms in read(str(path), index=":"):
        info = atoms.info
        if "al_id" in info:
            ids.append(str(info["al_id"]))
        else:
            ids.append(f"{info['system']}:{info['conf_idx']}")
    return ids


def support_bound(split_seed):
    """Smallest interatomic distance present anywhere in this split's training data."""
    dataset = ROOT / "dataset" / f"ani1x_energy_split_{split_seed}"
    return min(min(minimum_distances(dataset / f"{name}.xyz")) for name in TRAINING_FILES)


def split_report(run, split_seed, override=None):
    manifest = load_json(run / "manifest.json")
    bound = support_bound(split_seed) if override is None else override
    dataset = Path(manifest["source_dataset"])

    distances = {
        "energy_ood": dict(zip(frame_ids(run / "data" / "heldout_ood.xyz"),
                               minimum_distances(run / "data" / "heldout_ood.xyz"))),
        "energy_id": dict(zip(frame_ids(dataset / "cc_test_id.xyz"),
                              minimum_distances(dataset / "cc_test_id.xyz"))),
    }
    dropped = {test: sorted(key for key, value in mapping.items() if value < bound)
               for test, mapping in distances.items()}

    rmse = {}
    counts = {}
    read_files = []
    for test in TESTS:
        for name in CONDITIONS:
            record = load_json(run / "metrics" / f"{name}_{test}.json")
            # Prove these are the same predictions that produced the published numbers.
            verify_inventory({record["prediction_file"]: record["prediction_sha256"]})
            rows = load_json(record["prediction_file"])["data"]
            read_files.append(Path(record["prediction_file"]))
            missing = [row["al_id"] for row in rows if row["al_id"] not in distances[test]]
            if missing:
                raise RuntimeError(f"{name}/{test}: {len(missing)} predictions have no matching geometry.")
            kept = [row for row in rows if distances[test][row["al_id"]] >= bound]
            if not kept:
                raise RuntimeError(f"{name}/{test}: the support filter removed every configuration.")
            value = math.sqrt(sum(row["sq_error"] for row in kept) / len(kept))
            if not math.isfinite(value):
                raise RuntimeError(f"{name}/{test}: non-finite filtered RMSE.")
            rmse[(name, test)] = 1000 * value
            counts[(name, test)] = len(kept)

    rows = []
    for test in TESTS:
        if len({counts[(name, test)] for name in CONDITIONS}) != 1:
            raise RuntimeError(f"{test}: conditions no longer share one test set.")
        for regime in REGIMES:
            stats = {}
            for method in ACQUISITION_METRICS:
                stats.update(comparison(rmse[(f"before_{regime}", test)],
                                        rmse[(f"{regime}_random", test)],
                                        rmse[(f"{regime}_{method}", test)], method))
            rows.append({"regime": regime, "test": test,
                         "count": counts[(f"before_{regime}", test)], **stats})

    return {
        "settings": manifest["settings"], "rows": rows, "common_evaluator": [],
        "support_filter": {
            "split_seed": split_seed,
            "min_interatomic_distance_A": bound,
            "threshold_source": "explicit --min-distance override" if override is not None else
                                "minimum interatomic distance in this split's cc_train/cc_val/dft_train/dft_val",
            "excluded": {test: {"count": len(dropped[test]), "al_ids": dropped[test]} for test in TESTS},
            "retained": {test: counts[(f"before_{REGIMES[0]}", test)] for test in TESTS},
        },
        "inputs": inventory(read_files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="split_{split}_epochs_50",
                        help="Run directory name pattern containing {split}")
    parser.add_argument("--output", default="aggregate_epochs_50_supported",
                        help="New aggregate directory under runs/ (never an existing one)")
    parser.add_argument("--min-distance", type=float,
                        help="Override the per-split threshold with one fixed value in Angstrom")
    args = parser.parse_args()

    if args.min_distance is not None and args.min_distance <= 0:
        parser.error("--min-distance must be positive.")
    destination = (HERE / "runs" / args.output).resolve()
    if not destination.is_relative_to(HERE / "runs"):
        parser.error("Output must stay inside the isolated AL runs directory.")

    reports = []
    for split in range(5):
        run = (HERE / "runs" / args.runs.format(split=split)).resolve()
        if not (run / "manifest.json").exists():
            raise SystemExit(f"Missing run: {run}")
        report = split_report(run, split, args.min_distance)
        info = report["support_filter"]
        print(f"split {split}: support bound {info['min_interatomic_distance_A']:.4f} A; "
              f"excluded {info['excluded']['energy_ood']['count']} OOD and "
              f"{info['excluded']['energy_id']['count']} ID configurations; "
              f"retained {info['retained']['energy_ood']}/{info['retained']['energy_id']}", flush=True)
        reports.append(report)

    settings = {key: value for key, value in reports[0]["settings"].items() if key != "split_seed"}
    rows = aggregate(reports, settings)
    inputs = {"support_filter": [report["support_filter"] for report in reports],
              "files": {path: digest for report in reports
                        for path, digest in report["inputs"].items()}}
    write_report(destination, rows, inputs, settings)
    save_json(destination / "support_filter.json",
              {"rule": "Exclude test configurations whose minimum interatomic distance is below "
                       "the minimum present in that split's own training data. Derived from the "
                       "training geometries only; independent of predictions and errors; applied "
                       "identically to every condition, regime and test.",
               "splits": [report["support_filter"] for report in reports]})
    print(f"Support-filter provenance: {destination}/support_filter.json", flush=True)


if __name__ == "__main__":
    main()
