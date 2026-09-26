"""Overlap of the unaligned water splits, and the selected epochs of the water ensembles.

The BLYP and CCSD(T) water files are split independently (see docs/known-issues.md),
so a CCSD(T) test geometry may sit in the BLYP training set of the LF->HF parent.
This counts, per split, how many CCSD(T) geometries of each role appear in each
BLYP role (exact match of positions rounded to 1e-4 A), and reports the epoch that
validation-NLL checkpoint selection picks for every water member.

    python -m uq.datasets.water_overlap > results/tables/water_split_overlap.txt

The printed text is kept identical to the 2026-09-24 ledger entry's output.
"""

from __future__ import annotations

import logging

import numpy as np
from ase.io import read

from uq import paths, registry
from uq.evaluation.reliability import CHECKPOINT_PATTERN, select_best_checkpoints

DATASET = "water-random"
WATER = ("wA", "wB", "wC")


def key(atoms):
    return np.round(atoms.get_positions(), 4).tobytes()


def main():
    logging.disable(logging.CRITICAL)
    data = registry.dataset(DATASET)
    print("## Geometry overlap between CCSDT and BLYP roles (exact match of positions rounded to 1e-4 A)")
    print("split | CCSDT test (50) in BLYP train / val / test | CCSDT val in BLYP train | "
          "CCSDT train (130) in BLYP train / val / test")
    for s in data.splits:
        b = {r: {key(a) for a in read(data.file("lf", r, s), ":")} for r in ("train", "val", "test")}
        c = {r: [key(a) for a in read(data.file("hf", r, s), ":")] for r in ("train", "val", "test")}
        allb = b["train"] | b["val"] | b["test"]
        assert all(k in allb for r in c for k in c[r]), "CCSDT geometry not found in BLYP"
        cnt = lambda cr, br: sum(k in b[br] for k in c[cr])  # noqa: E731
        print(f"{s} | n_test={len(c['test'])}: {cnt('test','train')} / {cnt('test','val')} / {cnt('test','test')} | "
              f"n_val={len(c['val'])}: {cnt('val','train')} | n_train={len(c['train'])}: "
              f"{cnt('train','train')} / {cnt('train','val')} / {cnt('train','test')}")

    print("\n## Validation-NLL selected epoch per member (0-based, as eval/reliability.py selects it)")
    for letter in WATER:
        exp = registry.experiment(letter)
        ep = np.array([[int(CHECKPOINT_PATTERN.match(p.name).group("epoch"))
                        for p in select_best_checkpoints(paths.checkpoints_dir(exp.id, s), paths.results_dir(exp.id, s),
                                                         None, "loss", "min", None)] for s in exp.splits])
        print(f"{letter}: members {ep.size}, median {np.median(ep):.1f}, range {ep.min()}-{ep.max()}, "
              f"per-split medians {np.median(ep, axis=1).tolist()}")


if __name__ == "__main__":
    main()
