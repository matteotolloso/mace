#!/usr/bin/env python3
"""Training-support filter for held-out test sets, shared by every analysis.

Held-out test sets contain a handful of configurations compressed below the
smallest interatomic distance present anywhere in that split's own training data
(the train and val files of every fidelity, e.g. ``cc_train``, ``cc_val``,
``dft_train``, ``dft_val``). The models extrapolate catastrophically there, and
because RMSE and ENCE average squares, one such configuration can dominate a whole
metric. Every test configuration below its split's bound is excluded, for every
model and every uncertainty signal alike.

The rule reads geometry only. It never inspects predictions, errors or
uncertainties, so it cannot select on the quantity being evaluated - unlike the
``reliability --trim`` option, which drops the most and least *uncertain*
configurations and ranks them by total uncertainty even for the AU and EU rows.

On ANI-1x it removes 13-16 of 5000 Energy-OOD and 0-2 Energy-ID configurations per
split, and essentially nothing on the system split, whose DFT training data already
covers compressed geometries. Datasets opt in with ``support_filter: true`` in their
``dataset.yaml``; water does not.

The per-file geometry facts are cached, in file order, next to the data:
``data/<dataset>/split_<s>/geometry/<file>.npz`` with

  dmin      smallest interatomic distance (Angstrom)
  quantile  within-system energy quantile q_s(x) (energy split only, else NaN)
  system    molecular system label
  conf_idx  configuration index inside the system

Reading the 50k-frame XYZ files is the slow part, so the cache is built once, in
parallel, and reused by the aggregation step, the paper figures and active learning:

    uq data support --build      # once, CPU, ~5 min
    uq data support              # report the bound and removals per split

``kind`` arguments accept a dataset id (``ani1x-energy``) or the short ANI names
``system`` / ``energy``; ``experiment`` arguments accept ids or legacy letters.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from uq import paths, registry

TRAIN_ROLES = ("train", "val")


def dataset_id(kind):
    """``system`` / ``energy`` (legacy ANI names) or a dataset id -> dataset id."""
    return f"ani1x-{kind}" if kind in ("system", "energy") else kind


def filtered_datasets():
    return [data for data in registry.datasets().values() if data.support_filter]


def _stems(kind, roles=None):
    data = registry.dataset(dataset_id(kind))
    return [Path(name).stem for fidelity in data.fidelities.values()
            for role, name in fidelity.files.items() if roles is None or role in roles]


def cache_path(kind, split, name):
    return paths.geometry_cache(dataset_id(kind), split, name)


def split_file(kind, split, name):
    return paths.data_split_dir(dataset_id(kind), split) / f"{name}.xyz"


def _build_one(job):
    kind, split, name = job
    out = cache_path(kind, split, name)
    if out.exists():
        return f"cached {paths.rel(out)}"
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
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, dmin=np.asarray(dmin), quantile=np.asarray(quantile),
             system=np.asarray(system), conf_idx=np.asarray(conf))
    tmp.replace(out)
    return f"built  {paths.rel(out)} ({len(dmin)} frames)"


def build_cache(processes=48):
    """Build every missing geometry cache file. Existing files are kept."""
    jobs = [(data.id, s, stem) for data in filtered_datasets() for s in data.splits for stem in _stems(data.id)]
    with Pool(min(len(jobs), processes)) as pool:
        for message in pool.imap_unordered(_build_one, jobs):
            print(message, flush=True)


@lru_cache(None)
def geometry(kind, split, name):
    path = cache_path(kind, split, name)
    if not path.exists():
        raise RuntimeError(f"Missing geometry cache {path}. Build it once with 'uq data support --build'.")
    return dict(np.load(path))


@lru_cache(None)
def support_bound(kind, split):
    """Smallest interatomic distance anywhere in that split's training data."""
    return min(float(geometry(kind, split, name)["dmin"].min()) for name in _stems(kind, TRAIN_ROLES))


def supported(kind, split, name):
    """Boolean mask over the file's configurations: inside the training support."""
    return geometry(kind, split, name)["dmin"] >= support_bound(kind, split)


def experiment_kind(experiment):
    """Dataset id of an experiment whose dataset uses the filter, else None.

    For ANI experiments this is ``ani1x-system`` / ``ani1x-energy``; the legacy
    short names are accepted wherever a ``kind`` is expected."""
    exp = registry.experiment(experiment)
    return exp.dataset_id if exp.dataset.support_filter else None


def test_file(experiment, test):
    """File stem an experiment is evaluated on for a test role ('id', 'ood', ...):
    the target fidelity's file, e.g. ``dft_test_id`` for LF-only, ``cc_test_ood`` otherwise."""
    exp = registry.experiment(experiment)
    return Path(exp.fidelity.files[test]).stem


def experiment_from_path(path):
    """Experiment id of any path inside ``experiments/<id>/``, else None."""
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent / "experiment.yaml").is_file():
            return parent.relative_to(paths.EXPERIMENTS).as_posix()
    return None


def filter_raw_rows(rows, experiment, test, split):
    """Drop out-of-support rows from one split's per-configuration reliability CSV.

    ``rows`` must be in file order, as written by reliability. Experiments whose
    dataset does not use the filter (water) are returned unchanged.
    """
    kind = experiment_kind(experiment)
    if kind is None:
        return list(rows), 0
    keep = supported(kind, split, test_file(experiment, test))
    if len(keep) != len(rows):
        raise RuntimeError(
            f"{experiment} split {split} {test}: {len(rows)} prediction rows "
            f"but {len(keep)} cached geometries. Rebuild the geometry cache or the "
            f"reliability cache; they must describe the same file."
        )
    indices = [int(row["config_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(
            f"{experiment} split {split} {test}: reliability rows are not in "
            f"file order, so they cannot be matched to geometries."
        )
    filtered = [row for row, ok in zip(rows, keep) if ok]
    return filtered, len(rows) - len(filtered)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="Build any missing geometry cache files.")
    parser.add_argument("--processes", type=int, default=48)
    parser.add_argument("--report", action="store_true",
                        help="Print each split's support bound and how many test configurations it removes.")
    args = parser.parse_args(argv)
    if args.build:
        build_cache(args.processes)
    if args.report or not args.build:
        for data in filtered_datasets():
            tests = [Path(f.files[t]).stem for f in data.fidelities.values() for t in data.tests]
            for split in data.splits:
                bound = support_bound(data.id, split)
                removed = {name: int((~supported(data.id, split, name)).sum()) for name in tests}
                print(f"{data.id} split {split}: bound {bound:.3f} A, removed {removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
