#!/usr/bin/env python3
"""Build ANI-1x energy-based train/val/test splits and save them as MACE-ready extxyz files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import h5py
import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz


SPLIT_NAMES = ("train", "val", "test_id", "test_ood")
ENERGY_BANDS = ("low", "middle", "high")
DEFAULT_DFT_KEY = "wb97x_tz.energy"
DEFAULT_CC_KEY = "ccsd(t)_cbs.energy"
AUTO_FORCES_KEY = "__AUTO__"
HARTREE_TO_EV = 27.211386245988
OUTPUT_LENGTH_UNITS = "Angstrom"
OUTPUT_ENERGY_UNITS = "eV"
OUTPUT_FORCE_UNITS = "eV/Angstrom"


@dataclass
class SystemEnergyPartition:
    """Store energy-band pools, split assignments and availability for one molecular system."""

    system: str
    n_configs: int
    n_atoms: int
    valid_energy_configs: int
    is_energy_split_excluded: bool
    energy_band_pools: Dict[str, np.ndarray]
    config_pools: Dict[str, np.ndarray]
    dft_pools: Dict[str, np.ndarray]
    cc_pools: Dict[str, np.ndarray]


@dataclass
class SplitSelection:
    """Store the final DFT/CC selections for one split."""

    split_name: str
    requested_dft: int | None
    requested_cc: int | None
    resolved_dft_target: int
    resolved_cc_target: int
    dft_available: int
    cc_available: int
    dft_selected: Dict[str, np.ndarray]
    cc_selected: Dict[str, np.ndarray]


@dataclass(frozen=True)
class TheoryOutputSpec:
    """Describe how one theory-specific split file should be written."""

    name: str
    energy_key: str
    forces_key: str | None



def _finite_config_mask(values: np.ndarray, n_configs: int) -> np.ndarray:
    """Return one boolean per configuration indicating whether the annotation is finite."""
    array = np.asarray(values)

    if array.ndim == 0:
        return np.full(n_configs, bool(np.isfinite(array).all()), dtype=bool)

    if array.shape[0] != n_configs:
        return np.zeros(n_configs, dtype=bool)

    if array.ndim == 1:
        return np.isfinite(array)

    flattened = array.reshape(n_configs, -1)
    return np.all(np.isfinite(flattened), axis=1)



def _read_energy_array(
    group: h5py.Group, theory_key: str, n_configs: int, system_name: str
) -> np.ndarray:
    """Load one scalar energy per configuration, returning NaN where the label is invalid."""
    if theory_key not in group:
        return np.full(n_configs, np.nan, dtype=float)

    raw_values = np.asarray(group[theory_key][()])
    valid_mask = _finite_config_mask(raw_values, n_configs)
    energies = np.full(n_configs, np.nan, dtype=float)

    if raw_values.ndim == 0:
        energies[valid_mask] = float(raw_values)
        return energies

    if raw_values.shape[0] != n_configs:
        return energies

    if raw_values.ndim == 1:
        flattened = raw_values.astype(float, copy=False)
    else:
        reshaped = raw_values.reshape(n_configs, -1)
        if reshaped.shape[1] != 1:
            raise ValueError(
                f"{system_name}:{theory_key} is not a scalar-per-configuration energy dataset."
            )
        flattened = reshaped[:, 0].astype(float, copy=False)

    energies[valid_mask] = flattened[valid_mask]
    return energies



def _read_optional_array(
    group: h5py.Group, array_key: str | None, n_configs: int
) -> np.ndarray | None:
    """Load an optional per-configuration array dataset when present and shape-compatible."""
    if not array_key or array_key not in group:
        return None

    values = np.asarray(group[array_key][()])
    if values.ndim == 0 or values.shape[0] != n_configs:
        return None
    return values



def _empty_index_array() -> np.ndarray:
    """Create an empty integer index array with a consistent dtype."""
    return np.array([], dtype=np.int64)



def _counts_from_fractions(total: int, fractions: Iterable[float]) -> list[int]:
    """Convert fractional split weights into integer counts that sum exactly to total."""
    fractions_array = np.asarray(tuple(fractions), dtype=float)
    raw_counts = total * fractions_array
    counts = np.floor(raw_counts).astype(int)
    remainder = int(total - counts.sum())

    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for index in order[:remainder]:
            counts[index] += 1

    return counts.tolist()



def _split_low_energy_configurations(
    low_indices: np.ndarray,
    p_train: float,
    p_val: float,
    p_test_id: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Shuffle the low-energy pool of one system and split it into train/val/test-ID."""
    shuffled_indices = rng.permutation(low_indices)
    counts = _counts_from_fractions(low_indices.size, (p_train, p_val, p_test_id))

    train_end = counts[0]
    val_end = counts[0] + counts[1]

    return {
        "train": shuffled_indices[:train_end],
        "val": shuffled_indices[train_end:val_end],
        "test_id": shuffled_indices[val_end:],
    }


def _split_exactly_four_configurations(
    valid_indices: np.ndarray,
    relative_energies: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Assign systems with exactly four valid energies to four disjoint final splits."""
    order = np.argsort(relative_energies[valid_indices], kind="mergesort")
    ordered_indices = valid_indices[order]

    if ordered_indices.size != 4:
        raise ValueError("The exact-four split helper requires exactly four valid configurations.")

    lowest_three = ordered_indices[:3].copy()
    rng.shuffle(lowest_three)

    energy_band_pools = {
        "low": ordered_indices[:3].copy(),
        "middle": _empty_index_array(),
        "high": ordered_indices[3:].copy(),
    }
    config_pools = {
        "train": np.array([lowest_three[0]], dtype=np.int64),
        "val": np.array([lowest_three[1]], dtype=np.int64),
        "test_id": np.array([lowest_three[2]], dtype=np.int64),
        "test_ood": ordered_indices[3:].copy(),
    }
    return energy_band_pools, config_pools



def _filter_valid_indices(pool_indices: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Keep only configuration indices that have a valid annotation."""
    if pool_indices.size == 0:
        return _empty_index_array()
    return pool_indices[valid_mask[pool_indices]]



def _compute_relative_energies_and_quantiles(energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-system relative energies and empirical quantile ranks on valid entries only."""
    relative_energies = np.full_like(energies, np.nan, dtype=float)
    quantile_ranks = np.full_like(energies, np.nan, dtype=float)

    valid_indices = np.flatnonzero(np.isfinite(energies))
    if valid_indices.size == 0:
        return relative_energies, quantile_ranks

    valid_energies = energies[valid_indices]
    valid_relative = valid_energies - valid_energies.min()
    relative_energies[valid_indices] = valid_relative

    if valid_indices.size == 1:
        quantile_ranks[valid_indices] = 0.0
        return relative_energies, quantile_ranks

    order = np.argsort(valid_relative, kind="mergesort")
    ranks = np.empty(valid_indices.size, dtype=float)
    ranks[order] = np.arange(valid_indices.size, dtype=float) / (valid_indices.size - 1)
    quantile_ranks[valid_indices] = ranks
    return relative_energies, quantile_ranks



def _infer_forces_key(energy_key: str) -> str | None:
    """Infer the corresponding forces key when the energy key follows the '*.energy' convention."""
    if energy_key.endswith(".energy"):
        return energy_key[:-len(".energy")] + ".forces"
    return None



def _normalize_optional_key(value: str | None) -> str | None:
    """Normalize optional CLI keys so empty strings disable the corresponding dataset."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None



def _validate_probability(name: str, value: float) -> None:
    """Validate that one split probability is inside the closed unit interval."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")



def _validate_requested_size(name: str, value: int | None) -> None:
    """Validate that one requested dataset size is non-negative when provided."""
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")



def _validate_args(args: argparse.Namespace) -> None:
    """Check quantiles, split probabilities and requested target sizes."""
    _validate_probability("q_low", args.q_low)
    _validate_probability("q_high", args.q_high)
    _validate_probability("p_train", args.p_train)
    _validate_probability("p_val", args.p_val)
    _validate_probability("p_test_id", args.p_test_id)

    if not args.q_low < args.q_high:
        raise ValueError("q_low must be strictly smaller than q_high.")

    if not math.isclose(args.p_train + args.p_val + args.p_test_id, 1.0, abs_tol=1e-9):
        raise ValueError("p_train + p_val + p_test_id must sum to 1.")

    if args.max_per_system is not None and args.max_per_system <= 0:
        raise ValueError("--max-per-system must be strictly positive when provided.")

    for name in (
        "n_train_dft",
        "n_val_dft",
        "n_test_id_dft",
        "n_test_ood_dft",
        "n_train_cc",
        "n_val_cc",
        "n_test_id_cc",
        "n_test_ood_cc",
    ):
        _validate_requested_size(name, getattr(args, name))

    requested_pairs = (
        ("train", args.n_train_dft, args.n_train_cc),
        ("val", args.n_val_dft, args.n_val_cc),
        ("test_id", args.n_test_id_dft, args.n_test_id_cc),
        ("test_ood", args.n_test_ood_dft, args.n_test_ood_cc),
    )
    for split_name, requested_dft, requested_cc in requested_pairs:
        if requested_dft is not None and requested_cc is not None and requested_cc > requested_dft:
            raise ValueError(
                f"{split_name}: requested CC size ({requested_cc}) cannot exceed DFT size ({requested_dft})."
            )



def build_energy_partitions(
    h5_path: str | Path,
    energy_key: str,
    dft_key: str,
    cc_key: str,
    q_low: float,
    q_high: float,
    p_train: float,
    p_val: float,
    p_test_id: float,
    rng: np.random.Generator,
) -> Dict[str, SystemEnergyPartition]:
    """Create low-/high-energy pools per system and derive the final train/val/test-ID/test-OOD pools."""
    partitions: Dict[str, SystemEnergyPartition] = {}
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as handle:
        for system_name in sorted(handle.keys()):
            group = handle[system_name]
            coordinates = group["coordinates"]
            n_configs = int(coordinates.shape[0])
            n_atoms = int(coordinates.shape[1])

            split_energies = _read_energy_array(group, energy_key, n_configs, system_name)
            dft_energies = _read_energy_array(group, dft_key, n_configs, system_name)
            cc_energies = _read_energy_array(group, cc_key, n_configs, system_name)

            relative_energies, quantile_ranks = _compute_relative_energies_and_quantiles(split_energies)
            valid_split_energy = np.isfinite(split_energies)
            valid_indices = np.flatnonzero(valid_split_energy)
            valid_energy_configs = int(valid_indices.size)

            if valid_energy_configs < 4:
                energy_band_pools = {band_name: _empty_index_array() for band_name in ENERGY_BANDS}
                config_pools = {split_name: _empty_index_array() for split_name in SPLIT_NAMES}
                is_energy_split_excluded = True
            elif valid_energy_configs == 4:
                energy_band_pools, config_pools = _split_exactly_four_configurations(
                    valid_indices=valid_indices,
                    relative_energies=relative_energies,
                    rng=rng,
                )
                is_energy_split_excluded = False
            else:
                low_indices = np.flatnonzero(valid_split_energy & (quantile_ranks <= q_low))
                high_indices = np.flatnonzero(valid_split_energy & (quantile_ranks >= q_high))
                middle_indices = np.flatnonzero(
                    valid_split_energy & (quantile_ranks > q_low) & (quantile_ranks < q_high)
                )

                low_split_pools = _split_low_energy_configurations(
                    low_indices=low_indices,
                    p_train=p_train,
                    p_val=p_val,
                    p_test_id=p_test_id,
                    rng=rng,
                )
                energy_band_pools = {
                    "low": low_indices,
                    "middle": middle_indices,
                    "high": high_indices,
                }
                config_pools = {
                    "train": low_split_pools["train"],
                    "val": low_split_pools["val"],
                    "test_id": low_split_pools["test_id"],
                    "test_ood": high_indices,
                }
                is_energy_split_excluded = False

            dft_valid = np.isfinite(dft_energies)
            cc_valid = np.isfinite(cc_energies)
            dft_pools = {
                split_name: _filter_valid_indices(pool_indices, dft_valid)
                for split_name, pool_indices in config_pools.items()
            }
            cc_pools = {
                split_name: _filter_valid_indices(pool_indices, cc_valid)
                for split_name, pool_indices in config_pools.items()
            }

            partitions[system_name] = SystemEnergyPartition(
                system=system_name,
                n_configs=n_configs,
                n_atoms=n_atoms,
                valid_energy_configs=valid_energy_configs,
                is_energy_split_excluded=is_energy_split_excluded,
                energy_band_pools=energy_band_pools,
                config_pools=config_pools,
                dft_pools=dft_pools,
                cc_pools=cc_pools,
            )

    return partitions



def _resolved_target_count(requested: int | None, available: int, label: str) -> int:
    """Resolve an optional requested size, or fail loudly when the target is infeasible."""
    if requested is None:
        return available
    if requested > available:
        raise ValueError(f"{label}: requested {requested} samples but only {available} are available.")
    return requested



def _available_counts_by_system(system_to_indices: Mapping[str, np.ndarray]) -> Dict[str, int]:
    """Summarize how many candidate configurations each system contributes."""
    return {
        system_name: int(indices.size)
        for system_name, indices in system_to_indices.items()
        if indices.size > 0
    }



def _allocate_stratified_quotas(
    available_by_system: Mapping[str, int],
    target: int,
    max_per_system: int | None,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """Allocate an exact target count across systems with optional soft capping."""
    if target < 0:
        raise ValueError(f"Target must be non-negative, got {target}.")

    positive_items = [(system, count) for system, count in available_by_system.items() if count > 0]
    if target == 0 or not positive_items:
        return {system: 0 for system in available_by_system}

    total_available = sum(count for _, count in positive_items)
    if target > total_available:
        raise ValueError(f"Requested target {target} exceeds the available pool {total_available}.")

    systems = [system for system, _ in positive_items]
    counts = np.asarray([count for _, count in positive_items], dtype=int)

    if max_per_system is None:
        weights = counts.astype(float)
    else:
        weights = np.minimum(counts, max_per_system).astype(float)

    if np.all(weights == 0.0):
        weights = counts.astype(float)

    scaled = target * weights / weights.sum()
    quotas = np.floor(scaled).astype(int)
    quotas = np.minimum(quotas, counts)

    fractional_parts = scaled - quotas
    tie_breakers = rng.random(len(systems))
    order = np.lexsort((tie_breakers, -fractional_parts))

    remaining = int(target - quotas.sum())
    while remaining > 0:
        progress = False
        for index in order:
            if quotas[index] >= counts[index]:
                continue
            quotas[index] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            raise RuntimeError("Unable to allocate the requested quotas without exceeding availability.")

    all_quotas = {system: 0 for system in available_by_system}
    for system, quota in zip(systems, quotas):
        all_quotas[system] = int(quota)
    return all_quotas



def _sample_indices_by_quota(
    candidates_by_system: Mapping[str, np.ndarray],
    quotas_by_system: Mapping[str, int],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Sample the requested number of indices for each system without replacement."""
    sampled: Dict[str, np.ndarray] = {}
    for system_name, candidates in candidates_by_system.items():
        quota = int(quotas_by_system.get(system_name, 0))
        if quota == 0:
            sampled[system_name] = _empty_index_array()
            continue
        if quota > candidates.size:
            raise ValueError(
                f"{system_name}: requested {quota} samples but only {candidates.size} candidates exist."
            )
        choice = rng.choice(candidates, size=quota, replace=False)
        sampled[system_name] = np.sort(choice.astype(np.int64, copy=False))
    return sampled



def _subtract_selected_indices(candidates: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Remove already reserved indices from a candidate pool."""
    if selected.size == 0:
        return candidates.copy()
    keep_mask = np.isin(candidates, selected, assume_unique=False, invert=True)
    return candidates[keep_mask]



def _combine_disjoint_index_sets(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Merge two disjoint sampled index sets into one sorted array."""
    if first.size == 0:
        return second.copy()
    if second.size == 0:
        return first.copy()
    return np.sort(np.concatenate((first, second)).astype(np.int64, copy=False))



def build_split_selection(
    partitions: Mapping[str, SystemEnergyPartition],
    split_name: str,
    requested_dft: int | None,
    requested_cc: int | None,
    max_per_system: int | None,
    rng: np.random.Generator,
) -> SplitSelection:
    """Build one split with stratified DFT sampling and a CC reservation inside DFT."""
    dft_candidates = {
        system_name: partition.dft_pools[split_name]
        for system_name, partition in partitions.items()
    }
    cc_candidates = {
        system_name: np.intersect1d(
            partition.dft_pools[split_name],
            partition.cc_pools[split_name],
            assume_unique=True,
        )
        for system_name, partition in partitions.items()
    }

    dft_available = sum(indices.size for indices in dft_candidates.values())
    cc_available = sum(indices.size for indices in cc_candidates.values())

    resolved_dft_target = _resolved_target_count(
        requested=requested_dft,
        available=dft_available,
        label=f"{split_name} DFT",
    )
    resolved_cc_target = _resolved_target_count(
        requested=requested_cc,
        available=cc_available,
        label=f"{split_name} CC",
    )

    if resolved_cc_target > resolved_dft_target:
        raise ValueError(
            f"{split_name}: resolved CC target ({resolved_cc_target}) exceeds DFT target ({resolved_dft_target})."
        )

    if resolved_cc_target == 0:
        reserved_cc_inside_dft = {system_name: _empty_index_array() for system_name in partitions}
    else:
        cc_reserve_quotas = _allocate_stratified_quotas(
            available_by_system=_available_counts_by_system(cc_candidates),
            target=resolved_cc_target,
            max_per_system=max_per_system,
            rng=rng,
        )
        reserved_cc_inside_dft = _sample_indices_by_quota(
            candidates_by_system=cc_candidates,
            quotas_by_system=cc_reserve_quotas,
            rng=rng,
        )

    remaining_dft_target = resolved_dft_target - resolved_cc_target
    remaining_dft_candidates = {
        system_name: _subtract_selected_indices(
            candidates=dft_candidates[system_name],
            selected=reserved_cc_inside_dft[system_name],
        )
        for system_name in partitions
    }

    if remaining_dft_target == 0:
        extra_dft = {system_name: _empty_index_array() for system_name in partitions}
    else:
        extra_dft_quotas = _allocate_stratified_quotas(
            available_by_system=_available_counts_by_system(remaining_dft_candidates),
            target=remaining_dft_target,
            max_per_system=max_per_system,
            rng=rng,
        )
        extra_dft = _sample_indices_by_quota(
            candidates_by_system=remaining_dft_candidates,
            quotas_by_system=extra_dft_quotas,
            rng=rng,
        )

    dft_selected = {
        system_name: _combine_disjoint_index_sets(
            reserved_cc_inside_dft[system_name],
            extra_dft[system_name],
        )
        for system_name in partitions
    }

    eligible_cc_from_dft = {
        system_name: np.intersect1d(
            dft_selected[system_name],
            cc_candidates[system_name],
            assume_unique=True,
        )
        for system_name in partitions
    }

    if resolved_cc_target == 0:
        cc_selected = {system_name: _empty_index_array() for system_name in partitions}
    else:
        cc_selection_quotas = _allocate_stratified_quotas(
            available_by_system=_available_counts_by_system(eligible_cc_from_dft),
            target=resolved_cc_target,
            max_per_system=max_per_system,
            rng=rng,
        )
        cc_selected = _sample_indices_by_quota(
            candidates_by_system=eligible_cc_from_dft,
            quotas_by_system=cc_selection_quotas,
            rng=rng,
        )

    return SplitSelection(
        split_name=split_name,
        requested_dft=requested_dft,
        requested_cc=requested_cc,
        resolved_dft_target=resolved_dft_target,
        resolved_cc_target=resolved_cc_target,
        dft_available=dft_available,
        cc_available=cc_available,
        dft_selected=dft_selected,
        cc_selected=cc_selected,
    )



def _build_atoms(
    partition: SystemEnergyPartition,
    split_name: str,
    output_spec: TheoryOutputSpec,
    config_index: int,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    energy_basis_key: str,
    energy_basis_value: float,
    relative_energy: float,
    quantile_rank: float,
    dft_key: str,
    cc_key: str,
    dft_energy: float,
    cc_energy: float,
    dft_forces_key: str | None,
    dft_forces: np.ndarray | None,
    cc_forces_key: str | None,
    cc_forces: np.ndarray | None,
) -> Atoms:
    """Create one ASE Atoms object with geometry, energies and energy-band metadata."""
    atoms = Atoms(numbers=atomic_numbers, positions=coordinates[config_index], pbc=False)
    atoms.info["compound"] = partition.system
    atoms.info["system"] = partition.system
    atoms.info["conf_idx"] = int(config_index)
    atoms.info["split"] = split_name
    atoms.info["energy_domain"] = "high" if split_name == "test_ood" else "low"
    atoms.info["config_type"] = f"{output_spec.name}_{split_name}"

    if np.isfinite(energy_basis_value) and energy_basis_key not in {dft_key, cc_key}:
        atoms.info[energy_basis_key] = float(energy_basis_value * HARTREE_TO_EV)
    if np.isfinite(relative_energy):
        atoms.info["relative_energy_eV"] = float(relative_energy * HARTREE_TO_EV)
    if np.isfinite(quantile_rank):
        atoms.info["energy_quantile"] = float(quantile_rank)

    if np.isfinite(dft_energy):
        atoms.info[dft_key] = float(dft_energy * HARTREE_TO_EV)
    if np.isfinite(cc_energy):
        atoms.info[cc_key] = float(cc_energy * HARTREE_TO_EV)

    if output_spec.name == "dft" and dft_forces_key and dft_forces is not None:
        force_row = np.asarray(dft_forces[config_index], dtype=float)
        if np.all(np.isfinite(force_row)):
            atoms.arrays[dft_forces_key] = force_row * HARTREE_TO_EV

    if output_spec.name == "cc" and cc_forces_key and cc_forces is not None:
        force_row = np.asarray(cc_forces[config_index], dtype=float)
        if np.all(np.isfinite(force_row)):
            atoms.arrays[cc_forces_key] = force_row * HARTREE_TO_EV

    return atoms



def write_xyz_files(
    h5_path: str | Path,
    outdir: str | Path,
    partitions: Mapping[str, SystemEnergyPartition],
    selections: Mapping[str, SplitSelection],
    energy_key: str,
    dft_key: str,
    cc_key: str,
    dft_forces_key: str | None,
    cc_forces_key: str | None,
) -> Dict[str, Dict[str, int]]:
    """Write the sampled DFT and CC splits as MACE-compatible extxyz files."""
    outdir = Path(outdir)
    output_specs = {
        "dft": TheoryOutputSpec(name="dft", energy_key=dft_key, forces_key=dft_forces_key),
        "cc": TheoryOutputSpec(name="cc", energy_key=cc_key, forces_key=cc_forces_key),
    }

    file_handles: Dict[tuple[str, str], object] = {}
    counts = {
        "dft": {split_name: 0 for split_name in SPLIT_NAMES},
        "cc": {split_name: 0 for split_name in SPLIT_NAMES},
    }

    try:
        for theory_name in ("dft", "cc"):
            for split_name in SPLIT_NAMES:
                path = outdir / f"{theory_name}_{split_name}.xyz"
                file_handles[(theory_name, split_name)] = path.open("w", encoding="utf-8")

        with h5py.File(Path(h5_path), "r") as dataset_handle:
            for system_name in sorted(partitions):
                partition = partitions[system_name]
                group = dataset_handle[system_name]

                atomic_numbers = np.asarray(group["atomic_numbers"][()], dtype=int)
                coordinates = np.asarray(group["coordinates"][()], dtype=float)
                split_energies = _read_energy_array(group, energy_key, partition.n_configs, system_name)
                relative_energies, quantile_ranks = _compute_relative_energies_and_quantiles(split_energies)
                dft_energies = _read_energy_array(group, dft_key, partition.n_configs, system_name)
                cc_energies = _read_energy_array(group, cc_key, partition.n_configs, system_name)
                dft_forces = _read_optional_array(group, dft_forces_key, partition.n_configs)
                cc_forces = _read_optional_array(group, cc_forces_key, partition.n_configs)

                for split_name in SPLIT_NAMES:
                    split_selection = selections[split_name]

                    for config_index in split_selection.dft_selected[system_name]:
                        atoms = _build_atoms(
                            partition=partition,
                            split_name=split_name,
                            output_spec=output_specs["dft"],
                            config_index=int(config_index),
                            atomic_numbers=atomic_numbers,
                            coordinates=coordinates,
                            energy_basis_key=energy_key,
                            energy_basis_value=split_energies[config_index],
                            relative_energy=relative_energies[config_index],
                            quantile_rank=quantile_ranks[config_index],
                            dft_key=dft_key,
                            cc_key=cc_key,
                            dft_energy=dft_energies[config_index],
                            cc_energy=cc_energies[config_index],
                            dft_forces_key=dft_forces_key,
                            dft_forces=dft_forces,
                            cc_forces_key=cc_forces_key,
                            cc_forces=cc_forces,
                        )
                        write_extxyz(
                            file_handles[("dft", split_name)],
                            [atoms],
                            write_info=True,
                            write_results=False,
                        )
                        counts["dft"][split_name] += 1

                    for config_index in split_selection.cc_selected[system_name]:
                        atoms = _build_atoms(
                            partition=partition,
                            split_name=split_name,
                            output_spec=output_specs["cc"],
                            config_index=int(config_index),
                            atomic_numbers=atomic_numbers,
                            coordinates=coordinates,
                            energy_basis_key=energy_key,
                            energy_basis_value=split_energies[config_index],
                            relative_energy=relative_energies[config_index],
                            quantile_rank=quantile_ranks[config_index],
                            dft_key=dft_key,
                            cc_key=cc_key,
                            dft_energy=dft_energies[config_index],
                            cc_energy=cc_energies[config_index],
                            dft_forces_key=dft_forces_key,
                            dft_forces=dft_forces,
                            cc_forces_key=cc_forces_key,
                            cc_forces=cc_forces,
                        )
                        write_extxyz(
                            file_handles[("cc", split_name)],
                            [atoms],
                            write_info=True,
                            write_results=False,
                        )
                        counts["cc"][split_name] += 1
    finally:
        for handle in file_handles.values():
            handle.close()

    return counts



def write_system_assignment_file(
    outdir: str | Path,
    partitions: Mapping[str, SystemEnergyPartition],
) -> None:
    """Write one CSV summarizing low/middle/high bands and split sizes per system."""
    outdir = Path(outdir)
    path = outdir / "system_assignments.csv"
    fieldnames = [
        "system",
        "n_atoms",
        "n_configs",
        "valid_energy_configs",
        "energy_split_excluded",
        "low_energy_configs",
        "middle_energy_configs",
        "high_energy_configs",
        "train_configs",
        "val_configs",
        "test_id_configs",
        "test_ood_configs",
        "train_dft_available",
        "val_dft_available",
        "test_id_dft_available",
        "test_ood_dft_available",
        "train_cc_available",
        "val_cc_available",
        "test_id_cc_available",
        "test_ood_cc_available",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system_name in sorted(partitions):
            partition = partitions[system_name]
            writer.writerow(
                {
                    "system": partition.system,
                    "n_atoms": partition.n_atoms,
                    "n_configs": partition.n_configs,
                    "valid_energy_configs": partition.valid_energy_configs,
                    "energy_split_excluded": partition.is_energy_split_excluded,
                    "low_energy_configs": int(partition.energy_band_pools["low"].size),
                    "middle_energy_configs": int(partition.energy_band_pools["middle"].size),
                    "high_energy_configs": int(partition.energy_band_pools["high"].size),
                    "train_configs": int(partition.config_pools["train"].size),
                    "val_configs": int(partition.config_pools["val"].size),
                    "test_id_configs": int(partition.config_pools["test_id"].size),
                    "test_ood_configs": int(partition.config_pools["test_ood"].size),
                    "train_dft_available": int(partition.dft_pools["train"].size),
                    "val_dft_available": int(partition.dft_pools["val"].size),
                    "test_id_dft_available": int(partition.dft_pools["test_id"].size),
                    "test_ood_dft_available": int(partition.dft_pools["test_ood"].size),
                    "train_cc_available": int(partition.cc_pools["train"].size),
                    "val_cc_available": int(partition.cc_pools["val"].size),
                    "test_id_cc_available": int(partition.cc_pools["test_id"].size),
                    "test_ood_cc_available": int(partition.cc_pools["test_ood"].size),
                }
            )



def _pool_counts(
    partitions: Mapping[str, SystemEnergyPartition], attribute_name: str
) -> Dict[str, int]:
    """Aggregate how many configurations belong to each split for one pool type."""
    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    for partition in partitions.values():
        pools = getattr(partition, attribute_name)
        for split_name in SPLIT_NAMES:
            counts[split_name] += int(pools[split_name].size)
    return counts



def _energy_band_counts(partitions: Mapping[str, SystemEnergyPartition]) -> Dict[str, int]:
    """Aggregate the low/middle/high energy counts across all systems."""
    counts = {band_name: 0 for band_name in ENERGY_BANDS}
    for partition in partitions.values():
        for band_name in ENERGY_BANDS:
            counts[band_name] += int(partition.energy_band_pools[band_name].size)
    return counts



def _small_system_counts(partitions: Mapping[str, SystemEnergyPartition]) -> Dict[str, int]:
    """Count systems excluded or handled specially because they have very few valid energies."""
    excluded = sum(1 for partition in partitions.values() if partition.is_energy_split_excluded)
    exact_four = sum(1 for partition in partitions.values() if partition.valid_energy_configs == 4)
    return {
        "excluded_lt4": excluded,
        "exactly4_special_case": exact_four,
    }



def build_summary(
    args: argparse.Namespace,
    partitions: Mapping[str, SystemEnergyPartition],
    selections: Mapping[str, SplitSelection],
    xyz_counts: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    """Collect a compact JSON summary of the energy split and final sampled datasets."""
    return {
        "input_h5": str(Path(args.h5).resolve()),
        "output_dir": str(Path(args.outdir).resolve()),
        "seed": args.seed,
        "energy_key": args.energy_key,
        "dft_key": args.dft_key,
        "cc_key": args.cc_key,
        "dft_forces_key": args.dft_forces_key,
        "cc_forces_key": args.cc_forces_key,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "discard_middle_band": True,
        "output_units": {
            "positions": OUTPUT_LENGTH_UNITS,
            "energies": OUTPUT_ENERGY_UNITS,
            "forces": OUTPUT_FORCE_UNITS if (args.dft_forces_key or args.cc_forces_key) else None,
        },
        "p_train": args.p_train,
        "p_val": args.p_val,
        "p_test_id": args.p_test_id,
        "max_per_system": args.max_per_system,
        "num_systems": len(partitions),
        "small_system_policy": {
            "exclude_if_valid_energy_configs_lt": 4,
            "exactly4_assignment": "three lowest energies go to train/val/test_id, highest goes to test_ood",
        },
        "small_system_counts": _small_system_counts(partitions),
        "energy_band_counts": _energy_band_counts(partitions),
        "config_pool_counts": _pool_counts(partitions, "config_pools"),
        "dft_available_pool_counts": _pool_counts(partitions, "dft_pools"),
        "cc_available_pool_counts": _pool_counts(partitions, "cc_pools"),
        "requested_targets": {
            "dft": {
                "train": args.n_train_dft,
                "val": args.n_val_dft,
                "test_id": args.n_test_id_dft,
                "test_ood": args.n_test_ood_dft,
            },
            "cc": {
                "train": args.n_train_cc,
                "val": args.n_val_cc,
                "test_id": args.n_test_id_cc,
                "test_ood": args.n_test_ood_cc,
            },
        },
        "resolved_targets": {
            split_name: {
                "dft_available": selections[split_name].dft_available,
                "cc_available": selections[split_name].cc_available,
                "dft_selected": selections[split_name].resolved_dft_target,
                "cc_selected": selections[split_name].resolved_cc_target,
            }
            for split_name in SPLIT_NAMES
        },
        "written_xyz_counts": {
            "dft": dict(xyz_counts["dft"]),
            "cc": dict(xyz_counts["cc"]),
        },
    }



def write_summary_file(outdir: str | Path, summary: Mapping[str, object]) -> None:
    """Persist the JSON summary alongside the split files."""
    outdir = Path(outdir)
    path = outdir / "split_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)



def _print_summary(summary: Mapping[str, object]) -> None:
    """Print a short terminal summary after the split files have been created."""
    band_counts = summary["energy_band_counts"]
    print(
        "Energy bands:",
        f"low={band_counts['low']}",
        f"middle={band_counts['middle']}",
        f"high={band_counts['high']}",
    )

    resolved_targets = summary["resolved_targets"]
    for split_name in SPLIT_NAMES:
        split_summary = resolved_targets[split_name]
        print(
            f"{split_name}:",
            f"DFT {split_summary['dft_selected']}/{split_summary['dft_available']}",
            f"CC {split_summary['cc_selected']}/{split_summary['cc_available']}",
        )



def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ANI-1x energy splitter."""
    parser = argparse.ArgumentParser(
        description=(
            "Build ANI-1x energy-based splits with low-energy ID pools, high-energy OOD pools, "
            "stratified DFT sampling, CC subsets constrained to lie inside DFT, "
            "and MACE-ready extxyz outputs."
        )
    )
    parser.add_argument("--h5", required=True, help="Path to the ANI-1x HDF5 file.")
    parser.add_argument("--outdir", required=True, help="Directory where split files are written.")
    parser.add_argument(
        "--energy-key",
        default=None,
        help=(
            "Energy key used to define the low/high energy domains. "
            "If omitted, the value of --dft-key is used."
        ),
    )
    parser.add_argument(
        "--dft-key",
        default=DEFAULT_DFT_KEY,
        help=f"Energy key used for the DFT dataset. Default: {DEFAULT_DFT_KEY}",
    )
    parser.add_argument(
        "--cc-key",
        default=DEFAULT_CC_KEY,
        help=f"Energy key used for the CC dataset. Default: {DEFAULT_CC_KEY}",
    )
    parser.add_argument(
        "--dft-forces-key",
        default=AUTO_FORCES_KEY,
        help=(
            "Optional forces key written to the DFT xyz files. "
            "If omitted, it is inferred from --dft-key when possible. "
            "Pass an empty string to disable forces export."
        ),
    )
    parser.add_argument(
        "--cc-forces-key",
        default=AUTO_FORCES_KEY,
        help=(
            "Optional forces key written to the CC xyz files. "
            "If omitted, it is inferred from --cc-key when possible. "
            "Pass an empty string to disable forces export."
        ),
    )
    parser.add_argument("--q-low", type=float, default=0.8, help="Upper quantile threshold for the low-energy ID pool.")
    parser.add_argument("--q-high", type=float, default=0.9, help="Lower quantile threshold for the high-energy OOD pool.")
    parser.add_argument(
        "--p-train",
        type=float,
        default=0.8,
        help="Fraction of low-energy configurations assigned to the train ID pool.",
    )
    parser.add_argument(
        "--p-val",
        type=float,
        default=0.1,
        help="Fraction of low-energy configurations assigned to the validation ID pool.",
    )
    parser.add_argument(
        "--p-test-id",
        type=float,
        default=0.1,
        help="Fraction of low-energy configurations assigned to the test-ID pool.",
    )
    parser.add_argument(
        "--max-per-system",
        type=int,
        default=None,
        help=(
            "Optional soft cap used in the stratified quota weights. "
            "Large systems can still contribute more if needed to reach the target."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for all shuffles and samples.")

    parser.add_argument("--n-train-dft", type=int, default=None, help="Target size for DFT_train.")
    parser.add_argument("--n-val-dft", type=int, default=None, help="Target size for DFT_val.")
    parser.add_argument("--n-test-id-dft", type=int, default=None, help="Target size for DFT_test_ID.")
    parser.add_argument("--n-test-ood-dft", type=int, default=None, help="Target size for DFT_test_OOD.")
    parser.add_argument("--n-train-cc", type=int, default=None, help="Target size for CC_train.")
    parser.add_argument("--n-val-cc", type=int, default=None, help="Target size for CC_val.")
    parser.add_argument("--n-test-id-cc", type=int, default=None, help="Target size for CC_test_ID.")
    parser.add_argument("--n-test-ood-cc", type=int, default=None, help="Target size for CC_test_OOD.")
    return parser.parse_args()



def main() -> None:
    """Run the full ANI-1x energy split protocol and save the resulting extxyz files."""
    args = _parse_args()
    args.energy_key = args.dft_key if args.energy_key is None else args.energy_key

    if args.dft_forces_key == AUTO_FORCES_KEY:
        args.dft_forces_key = _infer_forces_key(args.dft_key)
    else:
        args.dft_forces_key = _normalize_optional_key(args.dft_forces_key)

    if args.cc_forces_key == AUTO_FORCES_KEY:
        args.cc_forces_key = _infer_forces_key(args.cc_key)
    else:
        args.cc_forces_key = _normalize_optional_key(args.cc_forces_key)

    _validate_args(args)

    rng = np.random.default_rng(args.seed)
    partitions = build_energy_partitions(
        h5_path=args.h5,
        energy_key=args.energy_key,
        dft_key=args.dft_key,
        cc_key=args.cc_key,
        q_low=args.q_low,
        q_high=args.q_high,
        p_train=args.p_train,
        p_val=args.p_val,
        p_test_id=args.p_test_id,
        rng=rng,
    )

    selections = {
        "train": build_split_selection(
            partitions=partitions,
            split_name="train",
            requested_dft=args.n_train_dft,
            requested_cc=args.n_train_cc,
            max_per_system=args.max_per_system,
            rng=rng,
        ),
        "val": build_split_selection(
            partitions=partitions,
            split_name="val",
            requested_dft=args.n_val_dft,
            requested_cc=args.n_val_cc,
            max_per_system=args.max_per_system,
            rng=rng,
        ),
        "test_id": build_split_selection(
            partitions=partitions,
            split_name="test_id",
            requested_dft=args.n_test_id_dft,
            requested_cc=args.n_test_id_cc,
            max_per_system=args.max_per_system,
            rng=rng,
        ),
        "test_ood": build_split_selection(
            partitions=partitions,
            split_name="test_ood",
            requested_dft=args.n_test_ood_dft,
            requested_cc=args.n_test_ood_cc,
            max_per_system=args.max_per_system,
            rng=rng,
        ),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_system_assignment_file(outdir=outdir, partitions=partitions)
    xyz_counts = write_xyz_files(
        h5_path=args.h5,
        outdir=outdir,
        partitions=partitions,
        selections=selections,
        energy_key=args.energy_key,
        dft_key=args.dft_key,
        cc_key=args.cc_key,
        dft_forces_key=args.dft_forces_key,
        cc_forces_key=args.cc_forces_key,
    )
    summary = build_summary(
        args=args,
        partitions=partitions,
        selections=selections,
        xyz_counts=xyz_counts,
    )
    write_summary_file(outdir=outdir, summary=summary)
    _print_summary(summary)


if __name__ == "__main__":
    main()
