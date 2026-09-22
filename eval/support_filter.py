#!/usr/bin/env python3
"""Training-support filter for the ANI test sets, shared by every analysis.

Held-out test sets contain a handful of configurations compressed below the
smallest interatomic distance present anywhere in that split's own training data
(``cc_train``, ``cc_val``, ``dft_train``, ``dft_val``). The models extrapolate
catastrophically there, and because RMSE and ENCE average squares, one such
configuration can dominate a whole metric. Every test configuration below its
split's bound is excluded, for every model and every uncertainty signal alike.

The rule reads geometry only. It never inspects predictions, errors or
uncertainties, so it cannot select on the quantity being evaluated - unlike the
``reliability.py --trim`` option, which drops the most and least *uncertain*
configurations and ranks them by total uncertainty even for the AU and EU rows.

It removes 13-16 of 5000 Energy-OOD and 0-2 Energy-ID configurations per split,
and essentially nothing on the system split, whose DFT training data already
covers compressed geometries.

The per-file geometry facts are cached, in file order, under
``eval/cache/geometry/<kind>_<split>_<file>.npz``:

  dmin      smallest interatomic distance (Angstrom)
  quantile  within-system energy quantile q_s(x) (energy split only, else NaN)
  system    molecular system label
  conf_idx  configuration index inside the system

Reading the 50k-frame XYZ files is the slow part, so the cache is built once, in
parallel, and reused by `new_figures/`, by the aggregation step and by anything
else that needs it:

    python -B eval/support_filter.py --build     # once, CPU, ~5 min
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEOMETRY_CACHE = HERE / "cache" / "geometry"

KINDS = ("energy", "system")
FILES = ("cc_train", "cc_val", "dft_train", "dft_val",
         "cc_test_id", "cc_test_ood", "dft_test_id", "dft_test_ood")
TRAIN_FILES = ("cc_train", "cc_val", "dft_train", "dft_val")

# ANI experiment families. Water (wA/wB/wC) has no ANI geometry cache and is
# never filtered; experiment_kind() returns None for it.
SYSTEM_EXPERIMENTS = ("A", "B", "E")
ENERGY_EXPERIMENTS = ("C", "D", "F")
#: LF-only models are scored against DFT labels, every other family against CC.
DFT_EXPERIMENTS = ("A", "C")


def cache_path(kind, split, name):
    return GEOMETRY_CACHE / f"{kind}_{split}_{name}.npz"


def split_file(kind, split, name):
    return ROOT / "dataset" / f"ani1x_{kind}_split_{split}" / f"{name}.xyz"


def _build_one(job):
    kind, split, name = job
    out = cache_path(kind, split, name)
    if out.exists():
        return f"cached {out.name}"
    from ase.io import read

    dmin, quantile, system, conf = [], [], [], []
    for atoms in read(str(split_file(kind, split, name)), index=":"):
        pos = atoms.get_positions()
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        dmin.append(dist.min() if len(atoms) > 1 else np.inf)
        info = atoms.info
        quantile.append(float(info.get("energy_quantile", np.nan)))
        system.append(str(info["system"]))
        conf.append(int(info["conf_idx"]))
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, dmin=np.asarray(dmin), quantile=np.asarray(quantile),
             system=np.asarray(system), conf_idx=np.asarray(conf))
    tmp.replace(out)
    return f"built  {out.name} ({len(dmin)} frames)"


def build_cache(processes=48):
    """Build every missing geometry cache file. Existing files are kept."""
    GEOMETRY_CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s, f) for k in KINDS for s in range(5) for f in FILES]
    with Pool(min(len(jobs), processes)) as pool:
        for message in pool.imap_unordered(_build_one, jobs):
            print(message, flush=True)


@lru_cache(None)
def geometry(kind, split, name):
    path = cache_path(kind, split, name)
    if not path.exists():
        raise RuntimeError(
            f"Missing geometry cache {path}. Build it once with "
            f"'python -B eval/support_filter.py --build'."
        )
    return dict(np.load(path))


@lru_cache(None)
def support_bound(kind, split):
    """Smallest interatomic distance anywhere in that split's training data."""
    return min(float(geometry(kind, split, name)["dmin"].min()) for name in TRAIN_FILES)


def supported(kind, split, name):
    """Boolean mask over the file's configurations: inside the training support."""
    return geometry(kind, split, name)["dmin"] >= support_bound(kind, split)


def experiment_kind(experiment):
    """'system', 'energy', or None when the family has no ANI geometry cache."""
    if experiment in SYSTEM_EXPERIMENTS:
        return "system"
    if experiment in ENERGY_EXPERIMENTS:
        return "energy"
    return None


def test_file(experiment, test):
    """Name of the split file a family is evaluated on ('id' or 'ood')."""
    return f"{'dft' if experiment in DFT_EXPERIMENTS else 'cc'}_test_{test}"


def experiment_from_path(path):
    """Experiment letter from any path inside experiment_X/, else None."""
    for part in Path(path).resolve().parts:
        if part.startswith("experiment_"):
            return part[len("experiment_"):]
    return None


def filter_raw_rows(rows, experiment, test, split):
    """Drop out-of-support rows from one split's per-configuration reliability CSV.

    ``rows`` must be in file order, as written by reliability.py. Water and any
    unknown family are returned unchanged.
    """
    kind = experiment_kind(experiment)
    if kind is None:
        return list(rows), 0
    keep = supported(kind, split, test_file(experiment, test))
    if len(keep) != len(rows):
        raise RuntimeError(
            f"experiment_{experiment} split {split} {test}: {len(rows)} prediction rows "
            f"but {len(keep)} cached geometries. Rebuild the geometry cache or the "
            f"reliability cache; they must describe the same file."
        )
    indices = [int(row["config_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(
            f"experiment_{experiment} split {split} {test}: reliability rows are not in "
            f"file order, so they cannot be matched to geometries."
        )
    filtered = [row for row, ok in zip(rows, keep) if ok]
    return filtered, len(rows) - len(filtered)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="Build any missing geometry cache files.")
    parser.add_argument("--processes", type=int, default=48)
    parser.add_argument("--report", action="store_true",
                        help="Print each split's support bound and how many test configurations it removes.")
    args = parser.parse_args()
    if args.build:
        build_cache(args.processes)
    if args.report or not args.build:
        for kind in KINDS:
            for split in range(5):
                bound = support_bound(kind, split)
                removed = {
                    name: int((~supported(kind, split, name)).sum())
                    for name in ("dft_test_id", "dft_test_ood", "cc_test_id", "cc_test_ood")
                }
                print(f"{kind} split {split}: bound {bound:.3f} A, removed {removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
