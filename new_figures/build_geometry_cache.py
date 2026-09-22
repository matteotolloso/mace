#!/usr/bin/env python3
"""Cache per-configuration geometry facts needed by the figures.

For every ANI split file this stores, in file order:
  dmin      smallest interatomic distance (Angstrom)
  quantile  within-system energy quantile q_s(x) (energy split only, else NaN)
  system    molecular system label
  conf_idx  configuration index inside the system

Reading 50k-frame XYZ files is the slow part, so files are processed in parallel
and each result is written once to _cache/geometry/<kind>_<split>_<file>.npz.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "_cache" / "geometry"
KINDS = ("energy", "system")
FILES = ("cc_train", "cc_val", "dft_train", "dft_val",
         "cc_test_id", "cc_test_ood", "dft_test_id", "dft_test_ood")


def target(kind, split, name):
    return OUT / f"{kind}_{split}_{name}.npz"


def build(job):
    kind, split, name = job
    out = target(kind, split, name)
    if out.exists():
        return f"cached {out.name}"
    from ase.io import read

    dmin, quantile, system, conf = [], [], [], []
    for atoms in read(str(ROOT / "dataset" / f"ani1x_{kind}_split_{split}" / f"{name}.xyz"), index=":"):
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


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s, f) for k in KINDS for s in range(5) for f in FILES]
    with Pool(min(len(jobs), 48)) as pool:
        for message in pool.imap_unordered(build, jobs):
            print(message, flush=True)
    sys.exit(0)
