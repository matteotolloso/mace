#!/usr/bin/env python3
"""
Build MACE-ready extended-XYZ datasets from ANI-1x (HDF5) with a compound-heterogeneous split.

What it does
- Reads an ANI-1x style HDF5: top-level groups = compounds, each with datasets:
  atomic_numbers (Nat,), coordinates (Nconf,Nat,3), plus many per-conf labels.
- Creates train/val/test splits by sampling configurations across *different compounds*
  (i.e., not taking all configs from the same compound).
- Writes extended XYZ files (ASE extxyz) containing:
    - positions (from coordinates)
    - atomic numbers (converted to symbols)
    - per-atom arrays: forces / charges / etc. when shape matches (Nat,) or (Nat,3) or (Nat,k)
    - per-config info: energies / dipoles / etc. when shape matches ( ) / (3,) / (k,)
- Also writes a CSV manifest recording which (compound, conf_idx) went to which split.

Typical MACE expectations
- energy key: "energy"
- forces key: "forces"
This script lets you map any HDF5 key -> those canonical names while still storing *all* keys.

Example
python build_ani1x_extxyz.py \
  --h5 ./ani1x-release.h5 \
  --outdir ./mace_xyz \
  --n-train 50000 --n-val 5000 --n-test 5000 \
  --energy-key "ccsd(t)_cbs.energy" --forces-key "wb97x_dz.forces" \
  --seed 0 --max-per-compound 50

Notes
- If you want strict “different compounds” in the strong sense (no compound appearing in multiple splits),
  set --exclusive-compounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write


# ---------------------------- helpers ----------------------------

def z_to_symbols(z: np.ndarray) -> List[str]:
    # z is (Nat,) int
    syms = []
    for zi in z.tolist():
        if zi < 1 or zi >= len(chemical_symbols) or chemical_symbols[zi] == "":
            raise ValueError(f"Invalid atomic number {zi}")
        syms.append(chemical_symbols[zi])
    return syms


def is_scalar(x: np.ndarray) -> bool:
    return x.shape == () or x.shape == (1,)


def maybe_to_python_scalar(x: np.ndarray) -> Any:
    if x.shape == (1,):
        return x.reshape(()).item()
    if x.shape == ():
        return x.item()
    return x


def sanitize_key(k: str) -> str:
    # extxyz generally tolerates many characters, but dots can confuse some downstream tools.
    # We keep both original and sanitized mapping (if requested).
    return k.replace(".", "_").replace("(", "_").replace(")", "_").replace("/", "_")


@dataclass
class KeyConfig:
    energy_key: Optional[str]
    forces_key: Optional[str]
    canonical_energy: str = "energy"
    canonical_forces: str = "forces"
    sanitize: bool = False


def extract_conf_labels(
    grp: h5py.Group,
    conf_idx: int,
    nat: int,
    keycfg: KeyConfig,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Returns:
      info_dict: per-configuration properties (scalars/vectors/matrices/etc as small arrays)
      arrays_dict: per-atom arrays (Nat,), (Nat,3), (Nat,k), ...
    """
    info: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {}

    # iterate all datasets except atomic_numbers and coordinates
    for k in grp.keys():
        if k in ("atomic_numbers", "coordinates"):
            continue

        ds = grp[k]
        if not isinstance(ds, h5py.Dataset):
            continue

        x = ds[()]  # load whole dataset for this key
        # x is typically (Nconf, ...)

        if x.ndim == 0:
            # weird constant dataset
            val = maybe_to_python_scalar(x)
            outk = sanitize_key(k) if keycfg.sanitize else k
            info[outk] = val
            continue

        if x.shape[0] <= conf_idx:
            continue

        v = x[conf_idx]

        outk = sanitize_key(k) if keycfg.sanitize else k

        # Decide per-atom vs per-config
        # Per-atom patterns: (Nat,), (Nat,3), (Nat,k), (Nat,3,3) etc.
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == nat:
            arrays[outk] = np.array(v)
        else:
            # per-config: scalar or small vector/matrix
            if isinstance(v, np.ndarray) and is_scalar(v):
                info[outk] = maybe_to_python_scalar(v)
            else:
                info[outk] = np.array(v)

    # Add canonical MACE keys if requested
    if keycfg.energy_key:
        src = sanitize_key(keycfg.energy_key) if keycfg.sanitize else keycfg.energy_key
        if src in info:
            info[keycfg.canonical_energy] = info[src]
        else:
            # Some energies might be stored as (Nconf,) dataset; we already placed in info
            # but if not found, do nothing.
            pass

    if keycfg.forces_key:
        src = sanitize_key(keycfg.forces_key) if keycfg.sanitize else keycfg.forces_key
        if src in arrays:
            arrays[keycfg.canonical_forces] = arrays[src]

    return info, arrays


# ---------------------------- splitting ----------------------------

def sample_split_indices(
    rng: np.random.Generator,
    compounds: Sequence[str],
    nconf_per_compound: Dict[str, int],
    n_target: int,
    max_per_compound: Optional[int],
) -> List[Tuple[str, int]]:
    """
    Sample (compound, conf_idx) pairs up to n_target, spreading across compounds.
    """
    comp_list = list(compounds)
    rng.shuffle(comp_list)

    chosen: List[Tuple[str, int]] = []
    remaining = n_target

    # pre-shuffle indices per compound to avoid repeated random calls
    per_comp_indices: Dict[str, np.ndarray] = {}
    for c in comp_list:
        nconf = nconf_per_compound[c]
        idxs = np.arange(nconf, dtype=int)
        rng.shuffle(idxs)
        per_comp_indices[c] = idxs

    # Round-robin take from each compound (1..k), respecting max_per_compound
    taken_per_comp: Dict[str, int] = {c: 0 for c in comp_list}
    ptr_per_comp: Dict[str, int] = {c: 0 for c in comp_list}

    while remaining > 0:
        progressed = False
        for c in comp_list:
            if remaining <= 0:
                break
            if max_per_compound is not None and taken_per_comp[c] >= max_per_compound:
                continue
            ptr = ptr_per_comp[c]
            if ptr >= nconf_per_compound[c]:
                continue
            conf_idx = int(per_comp_indices[c][ptr])
            ptr_per_comp[c] += 1
            taken_per_comp[c] += 1
            chosen.append((c, conf_idx))
            remaining -= 1
            progressed = True
        if not progressed:
            # No more available samples under constraints
            break

    if len(chosen) < n_target:
        raise RuntimeError(
            f"Could only sample {len(chosen)} configs (target {n_target}). "
            f"Try increasing --max-per-compound, or reduce requested sizes."
        )

    return chosen


def split_compounds_exclusive(
    rng: np.random.Generator,
    compounds: Sequence[str],
    nconf_per_compound: Dict[str, int],
    n_train: int,
    n_val: int,
    n_test: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split compounds so no compound appears in multiple splits.
    Greedy by number of conformations (largest first) to hit target counts approximately.
    """
    comps = list(compounds)
    rng.shuffle(comps)
    comps.sort(key=lambda c: nconf_per_compound[c], reverse=True)

    targets = {"train": n_train, "val": n_val, "test": n_test}
    buckets = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for c in comps:
        # assign to bucket with largest remaining need (normalized)
        best = None
        best_score = -1e18
        for b in ("train", "val", "test"):
            need = targets[b] - counts[b]
            score = need
            if score > best_score:
                best_score = score
                best = b
        buckets[best].append(c)
        counts[best] += nconf_per_compound[c]

    # We will sample within each bucket up to exact target counts later.
    return buckets["train"], buckets["val"], buckets["test"]


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to ani1x-release.h5")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--n-val", type=int, required=True)
    ap.add_argument("--n-test", type=int, required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-compound", type=int, default=None,
                    help="Maximum configs sampled from any single compound (per split).")
    ap.add_argument("--exclusive-compounds", action="store_true",
                    help="Ensure no compound appears in multiple splits (strong heterogeneity).")

    ap.add_argument("--energy-key", type=str, default=None,
                    help='HDF5 key to map to canonical "energy" (e.g. ccsd(t)_cbs.energy)')
    ap.add_argument("--forces-key", type=str, default=None,
                    help='HDF5 key to map to canonical "forces" (e.g. wb97x_dz.forces)')
    ap.add_argument("--sanitize-keys", action="store_true",
                    help="Replace '.', '()' etc. in keys to improve downstream compatibility.")

    ap.add_argument("--train-file", default="train.xyz")
    ap.add_argument("--val-file", default="val.xyz")
    ap.add_argument("--test-file", default="test.xyz")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--keymap", default="keymap.json",
                    help="If --sanitize-keys is used, write original->sanitized mapping here.")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    keycfg = KeyConfig(
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        sanitize=args.sanitize_keys,
    )

    # Read metadata about compounds / conformations
    with h5py.File(args.h5, "r") as f:
        compounds = list(f.keys())
        nconf_per_compound: Dict[str, int] = {}
        nat_per_compound: Dict[str, int] = {}

        for c in compounds:
            grp = f[c]
            z = grp["atomic_numbers"][()]
            coords = grp["coordinates"]
            nat_per_compound[c] = int(len(z))
            nconf_per_compound[c] = int(coords.shape[0])

        # Decide compound pools for each split
        if args.exclusive_compounds:
            train_comps, val_comps, test_comps = split_compounds_exclusive(
                rng, compounds, nconf_per_compound, args.n_train, args.n_val, args.n_test
            )
        else:
            train_comps = compounds
            val_comps = compounds
            test_comps = compounds

        # Sample (compound, conf) pairs
        train_pairs = sample_split_indices(
            rng, train_comps, nconf_per_compound, args.n_train, args.max_per_compound
        )
        val_pairs = sample_split_indices(
            rng, val_comps, nconf_per_compound, args.n_val, args.max_per_compound
        )
        test_pairs = sample_split_indices(
            rng, test_comps, nconf_per_compound, args.n_test, args.max_per_compound
        )

        # If not exclusive, avoid accidental duplicates across splits (optional but nice)
        if not args.exclusive_compounds:
            used = set(train_pairs)
            def resample_avoiding(pairs, pool_comps, n_target):
                out = []
                attempts = 0
                while len(out) < n_target:
                    attempts += 1
                    if attempts > 50:
                        raise RuntimeError("Too many attempts to avoid duplicates; consider --exclusive-compounds.")
                    cand = sample_split_indices(rng, pool_comps, nconf_per_compound, n_target, args.max_per_compound)
                    out = [p for p in cand if p not in used]
                return out[:n_target]
            val_pairs = resample_avoiding(val_pairs, val_comps, args.n_val)
            used |= set(val_pairs)
            test_pairs = resample_avoiding(test_pairs, test_comps, args.n_test)

        # Build a key mapping (original -> sanitized) if requested
        keymap: Dict[str, str] = {}
        if args.sanitize_keys:
            # Discover keys from first compound only (good enough for map)
            first = compounds[0]
            for k in f[first].keys():
                keymap[k] = sanitize_key(k)

        def build_atoms(c: str, conf_idx: int) -> Atoms:
            grp = f[c]
            z = grp["atomic_numbers"][()]
            coords = grp["coordinates"][conf_idx]  # (Nat,3)
            nat = int(len(z))

            atoms = Atoms(symbols=z_to_symbols(z), positions=np.array(coords))

            info, arrays = extract_conf_labels(grp, conf_idx, nat, keycfg)

            # Always include provenance
            info["compound"] = c
            info["conf_idx"] = int(conf_idx)

            atoms.info.update(info)
            for ak, av in arrays.items():
                atoms.arrays[ak] = np.array(av)

            return atoms

        def write_split(pairs: List[Tuple[str, int]], filename: str):
            path = os.path.join(args.outdir, filename)
            atoms_list = [build_atoms(c, i) for (c, i) in pairs]
            # extxyz supports arrays + info
            write(path, atoms_list, format="extxyz")
            return path

        train_path = write_split(train_pairs, args.train_file)
        val_path = write_split(val_pairs, args.val_file)
        test_path = write_split(test_pairs, args.test_file)

    # Write manifest
    manifest_path = os.path.join(args.outdir, args.manifest)
    with open(manifest_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["split", "compound", "conf_idx"])
        for c, i in train_pairs:
            w.writerow(["train", c, i])
        for c, i in val_pairs:
            w.writerow(["val", c, i])
        for c, i in test_pairs:
            w.writerow(["test", c, i])

    # Write keymap if needed
    if args.sanitize_keys:
        keymap_path = os.path.join(args.outdir, args.keymap)
        with open(keymap_path, "w") as fp:
            json.dump(keymap, fp, indent=2, sort_keys=True)

    print("Wrote:")
    print(" ", os.path.join(args.outdir, args.train_file))
    print(" ", os.path.join(args.outdir, args.val_file))
    print(" ", os.path.join(args.outdir, args.test_file))
    print(" ", manifest_path)
    if args.sanitize_keys:
        print(" ", os.path.join(args.outdir, args.keymap))
    print()
    print("Canonical keys:")
    print(f'  energy: {args.energy_key!r} -> "energy" (if provided and present)')
    print(f'  forces: {args.forces_key!r} -> "forces" (if provided and present)')


if __name__ == "__main__":
    main()
