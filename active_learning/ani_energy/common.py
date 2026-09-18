"""Local data/provenance helpers for the isolated ANI active-learning POC."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT, ROOT / "eval", ROOT / "dataset"):
    sys.path.insert(0, str(directory))

import numpy as np
from ase import Atoms
from ase.io import read, write

ENERGY_KEY = "ccsd(t)_cbs.energy"
BUDGET = 500
AL_LEARNING_RATE = 0.001
AL_EPOCHS = 100
RUN_TAG = ""
ACQUISITION_METRICS = {"au": "aleatoric_var", "eu": "epistemic_var", "tu": "total_var"}
MEMBERS = tuple(range(10))
REGIMES = {"hf_only": "F", "lf_hf": "D"}
CASES = {
    f"{regime}_{method}": (regime, "random" if method == "random" else f"{regime}_{method}")
    for regime in REGIMES for method in ("random", *ACQUISITION_METRICS)
}
CONTROL = "lf_hf_from_hf_only_tu"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(paths):
    return {str(Path(path).resolve()): sha256(path) for path in paths}


def verify_inventory(records):
    for path, expected in records.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing/changed artifact: {path}. Use a new --name; do not mix runs.")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def load_cached(path, inputs):
    if not path.exists():
        return None
    cached = load_json(path)
    if cached["inputs"] != inputs:
        raise RuntimeError(f"Stale cache: {path}. Use a new --name to change an experiment.")
    verify_inventory(cached.get("artifacts", {}))
    return cached["data"]


def save_cached(path, inputs, data, artifacts=()):
    save_json(path, {"inputs": inputs, "data": data, "artifacts": inventory(artifacts)})


def config_id(atoms):
    return f"{atoms.info['system']}:{int(atoms.info['conf_idx'])}"


def geometry_only(atoms):
    """Do not propagate calculators, arrays, energies, quantiles or label-derived metadata."""
    clean = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    clean.info = {"al_id": config_id(atoms), "system": str(atoms.info["system"]),
                  "conf_idx": int(atoms.info["conf_idx"])}
    return clean


def partition_ood(atoms_list, seed):
    from ani1x_energy_splitter import _allocate_stratified_quotas, _sample_indices_by_quota

    ids = [config_id(atoms) for atoms in atoms_list]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate (system, conf_idx) IDs in the original CC OOD data.")
    if len(ids) < 2 * BUDGET:
        raise ValueError("Need at least 1000 OOD configurations for a half-pool and B=500.")
    grouped = {}
    for index, atoms in enumerate(atoms_list):
        grouped.setdefault(str(atoms.info["system"]), []).append(index)
    grouped = {key: np.asarray(grouped[key], dtype=int) for key in sorted(grouped)}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    # The input is already system-balanced. Preserve its composition in both halves.
    quotas = _allocate_stratified_quotas(
        {key: len(value) for key, value in grouped.items()}, len(ids) // 2, None, rng
    )
    sampled = _sample_indices_by_quota(grouped, quotas, rng)
    pool = sorted(int(index) for indices in sampled.values() for index in indices)
    heldout = sorted(set(range(len(ids))) - set(pool))
    return pool, heldout, {
        key: {"pool": quotas[key], "heldout": len(grouped[key]) - quotas[key]}
        for key in grouped
    }


def prepare_data(source, output, seed):
    output.mkdir(parents=True, exist_ok=True)
    original = read(source / "cc_test_ood.xyz", index=":")
    pool, heldout, systems = partition_ood(original, seed)
    ood_ids = {config_id(atoms) for atoms in original}
    counts = {}
    for theory in ("cc", "dft"):
        for split in ("train", "val", "test_id"):
            name = f"{theory}_{split}.xyz"
            configurations = read(source / name, index=":")
            ids = [config_id(atoms) for atoms in configurations]
            if len(ids) != len(set(ids)) or ood_ids.intersection(ids):
                raise ValueError(f"Duplicate IDs or OOD leakage in {name}.")
            counts[name] = len(ids)
    public_pool = [geometry_only(original[index]) for index in pool]
    oracle = {}
    for index in pool:
        energy = float(original[index].info[ENERGY_KEY])
        if not math.isfinite(energy):
            raise ValueError("Non-finite CC oracle energy.")
        oracle[config_id(original[index])] = energy
    test_atoms = []
    for index in heldout:
        atoms = geometry_only(original[index])
        atoms.info[ENERGY_KEY] = float(original[index].info[ENERGY_KEY])
        if not math.isfinite(atoms.info[ENERGY_KEY]):
            raise ValueError("Non-finite held-out CC energy.")
        test_atoms.append(atoms)
    write(output / "pool.xyz", public_pool, format="extxyz")
    write(output / "heldout_ood.xyz", test_atoms, format="extxyz")
    save_json(output / "oracle.json", oracle)
    split = {
        "seed": seed, "budget": BUDGET, "pool_fraction": 0.5,
        "pool_ids": [config_id(original[index]) for index in pool],
        "heldout_ids": [config_id(original[index]) for index in heldout],
        "pool_source_indices": pool, "heldout_source_indices": heldout,
        "systems": systems, "original_low_energy_counts": counts,
    }
    save_json(output / "split.json", split)
    return split


def select_ids(pool_ids, method, seed, scores=None):
    if len(pool_ids) < BUDGET or len(set(pool_ids)) != len(pool_ids):
        raise ValueError("The pool must have at least B unique IDs.")
    if method == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        chosen = rng.choice(len(pool_ids), BUDGET, replace=False)
        return [pool_ids[int(index)] for index in chosen]
    if method not in ACQUISITION_METRICS or scores is None or set(scores) != set(pool_ids):
        raise ValueError("Uncertainty selection requires exactly one score for every pool ID.")
    if any(not math.isfinite(value) or value < 0 for value in scores.values()):
        raise ValueError("Uncertainty contains negative/non-finite values; no samples may be silently dropped.")
    # A stable ID tie-break avoids any dependence on file/dictionary iteration order.
    return sorted(pool_ids, key=lambda key: (-scores[key], key))[:BUDGET]


def augment_training(base, pool_path, oracle_path, selected_ids, heldout_ids, destination):
    if len(selected_ids) != BUDGET or len(set(selected_ids)) != BUDGET:
        raise ValueError("Acquisition must contain exactly 500 unique IDs.")
    if set(selected_ids).intersection(heldout_ids):
        raise ValueError("Held-out OOD IDs cannot be acquired.")
    pool = {config_id(atoms): atoms for atoms in read(pool_path, index=":")}
    if not set(selected_ids).issubset(pool):
        raise ValueError("Selection includes an ID outside the acquisition pool.")
    base_atoms = read(base, index=":")
    if {config_id(atoms) for atoms in base_atoms}.intersection(selected_ids):
        raise ValueError("Acquired IDs already occur in the original HF training data.")
    config_types = {atoms.info.get("config_type", "Default") for atoms in base_atoms}
    if len(config_types) != 1:
        raise ValueError("Expected one training config_type in the current ANI setup.")
    # This is the first acquisition-stage access to CC labels, after IDs are persisted.
    oracle = load_json(oracle_path)
    additions = []
    for key in sorted(selected_ids):
        atoms = geometry_only(pool[key])
        atoms.info[ENERGY_KEY] = float(oracle[key])
        atoms.info["config_type"] = next(iter(config_types))
        if not math.isfinite(atoms.info[ENERGY_KEY]):
            raise ValueError(f"Non-finite revealed CC energy for {key}.")
        additions.append(atoms)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base, destination)
    with destination.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.write(b"\n")
    write(destination, additions, format="extxyz", append=True)
    return {"original_count": len(base_atoms), "acquired_count": BUDGET,
            "total_count": len(base_atoms) + BUDGET,
            "selected_system_counts": dict(Counter(pool[key].info["system"] for key in selected_ids))}


def comparison(before, random, tu, method="tu"):
    if method not in ACQUISITION_METRICS:
        raise ValueError(f"Unknown acquisition metric: {method}")
    if not all(math.isfinite(value) and value >= 0 for value in (before, random, tu)):
        raise ValueError("RMSEs must be finite and non-negative.")
    random_percent = 100 * (before - random) / before if before else None
    tu_percent = 100 * (before - tu) / before if before else None
    return {
        "rmse_before_meV_per_atom": before,
        "rmse_random500_meV_per_atom": random,
        f"rmse_{method}500_meV_per_atom": tu,
        "relative_improvement_random_percent": random_percent,
        f"relative_improvement_{method}_percent": tu_percent,
        f"{method}_gain_over_random_meV_per_atom": random - tu,
        f"{method}_gain_over_random_percentage_points": tu_percent - random_percent if before else None,
    }
