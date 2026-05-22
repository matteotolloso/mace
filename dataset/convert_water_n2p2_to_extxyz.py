#!/usr/bin/env python3
"""Convert water n2p2 input.data files into MACE-ready extxyz splits."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz
from ase.units import Bohr, Hartree


FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANG = Hartree / Bohr


@dataclass
class N2P2Config:
    atoms: Atoms
    has_nonzero_forces: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert each dataset/water/<theory>/input.data from n2p2 format to "
            "train.xyz, val.xyz, and test.xyz extended XYZ files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/water"),
        help="Root folder containing theory subfolders with input.data files.",
    )
    parser.add_argument("--train", type=float, default=80.0, help="Train percentage.")
    parser.add_argument("--val", type=float, default=10.0, help="Validation percentage.")
    parser.add_argument("--test", type=float, default=10.0, help="Test percentage.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic shuffled splits.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep the input order instead of shuffling before splitting.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap atomic positions back into the periodic cell before writing.",
    )
    parser.add_argument(
        "--energy-key",
        default="REF_energy",
        help="Energy key written to Atoms.info.",
    )
    parser.add_argument(
        "--forces-key",
        default="REF_forces",
        help="Forces key written to Atoms.arrays when forces are available.",
    )
    parser.add_argument(
        "--all-zero-forces",
        choices=("keep", "drop"),
        default="drop",
        help=(
            "Whether to keep or drop force arrays for configurations whose parsed "
            "forces are all zero."
        ),
    )
    parser.add_argument(
        "--position-unit",
        choices=("bohr", "angstrom"),
        default="bohr",
        help="Unit used by n2p2 positions and lattice vectors.",
    )
    parser.add_argument(
        "--energy-unit",
        choices=("hartree", "ev"),
        default="hartree",
        help="Unit used by n2p2 total energies.",
    )
    parser.add_argument(
        "--force-unit",
        choices=("hartree/bohr", "ev/angstrom"),
        default="hartree/bohr",
        help="Unit used by n2p2 force columns.",
    )
    return parser.parse_args()


def split_counts(n_items: int, percentages: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_items < 0:
        raise ValueError("Number of items cannot be negative.")

    total = sum(percentages)
    if not math.isclose(total, 100.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(f"Split percentages must sum to 100, got {total}.")
    if any(value < 0.0 for value in percentages):
        raise ValueError("Split percentages cannot be negative.")

    raw_counts = [n_items * value / 100.0 for value in percentages]
    counts = [math.floor(value) for value in raw_counts]
    missing = n_items - sum(counts)

    remainders = sorted(
        enumerate(value - math.floor(value) for value in raw_counts),
        key=lambda item: (-item[1], item[0]),
    )
    for index, _ in remainders[:missing]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def parse_n2p2_input(
    path: Path,
    *,
    energy_key: str,
    forces_key: str,
    position_scale: float,
    energy_scale: float,
    force_scale: float,
    keep_all_zero_forces: bool,
    wrap: bool,
    config_type_prefix: str,
) -> list[N2P2Config]:
    configs: list[N2P2Config] = []
    in_block = False
    comment = ""
    lattice: list[list[float]] = []
    symbols: list[str] = []
    positions: list[list[float]] = []
    forces: list[list[float]] = []
    energy: float | None = None
    charge: float | None = None

    def reset_block() -> None:
        nonlocal comment, lattice, symbols, positions, forces, energy, charge
        comment = ""
        lattice = []
        symbols = []
        positions = []
        forces = []
        energy = None
        charge = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            keyword = parts[0].lower()

            if keyword == "begin":
                if in_block:
                    raise ValueError(f"Nested begin at {path}:{line_number}")
                in_block = True
                reset_block()
                continue

            if keyword == "end":
                if not in_block:
                    raise ValueError(f"Unexpected end at {path}:{line_number}")
                if len(lattice) != 3:
                    raise ValueError(f"Expected 3 lattice lines before {path}:{line_number}")
                if energy is None:
                    raise ValueError(f"Missing energy before {path}:{line_number}")
                if not symbols:
                    raise ValueError(f"Missing atoms before {path}:{line_number}")

                atoms = Atoms(
                    symbols=symbols,
                    positions=np.asarray(positions, dtype=float) * position_scale,
                    cell=np.asarray(lattice, dtype=float) * position_scale,
                    pbc=True,
                )
                atoms.info[energy_key] = energy * energy_scale
                atoms.info["config_type"] = config_type_prefix
                if comment:
                    atoms.info["n2p2_comment"] = comment
                if charge is not None:
                    atoms.info["charge"] = charge

                force_array = np.asarray(forces, dtype=float) * force_scale
                has_nonzero_forces = bool(np.any(np.abs(force_array) > 0.0))
                if keep_all_zero_forces or has_nonzero_forces:
                    atoms.arrays[forces_key] = force_array
                if wrap:
                    atoms.wrap()

                configs.append(N2P2Config(atoms=atoms, has_nonzero_forces=has_nonzero_forces))
                in_block = False
                continue

            if not in_block:
                raise ValueError(f"Found data outside begin/end block at {path}:{line_number}")

            if keyword == "comment":
                comment = raw_line.strip()[len("comment") :].strip()
            elif keyword == "lattice":
                if len(parts) != 4:
                    raise ValueError(f"Malformed lattice line at {path}:{line_number}")
                lattice.append([float(value) for value in parts[1:4]])
            elif keyword == "atom":
                if len(parts) < 10:
                    raise ValueError(f"Malformed atom line at {path}:{line_number}")
                positions.append([float(value) for value in parts[1:4]])
                symbols.append(parts[4])
                forces.append([float(value) for value in parts[7:10]])
            elif keyword == "energy":
                if len(parts) != 2:
                    raise ValueError(f"Malformed energy line at {path}:{line_number}")
                energy = float(parts[1])
            elif keyword == "charge":
                if len(parts) != 2:
                    raise ValueError(f"Malformed charge line at {path}:{line_number}")
                charge = float(parts[1])

    if in_block:
        raise ValueError(f"Unclosed begin/end block in {path}")
    return configs


def write_split(path: Path, configs: Iterable[Atoms]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        write_extxyz(handle, list(configs))


def convert_theory_folder(
    theory_dir: Path,
    *,
    percentages: tuple[float, float, float],
    seed: int,
    shuffle: bool,
    args: argparse.Namespace,
) -> tuple[int, int, int, int, int]:
    input_path = theory_dir / "input.data"
    position_scale = Bohr if args.position_unit == "bohr" else 1.0
    energy_scale = Hartree if args.energy_unit == "hartree" else 1.0
    force_scale = (
        FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANG
        if args.force_unit == "hartree/bohr"
        else 1.0
    )

    parsed = parse_n2p2_input(
        input_path,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        position_scale=position_scale,
        energy_scale=energy_scale,
        force_scale=force_scale,
        keep_all_zero_forces=args.all_zero_forces == "keep",
        wrap=args.wrap,
        config_type_prefix=theory_dir.name,
    )
    configs = [item.atoms for item in parsed]
    nonzero_force_configs = sum(item.has_nonzero_forces for item in parsed)

    indices = list(range(len(configs)))
    if shuffle:
        rng = random.Random(seed + sum(ord(char) for char in theory_dir.name))
        rng.shuffle(indices)

    n_train, n_val, n_test = split_counts(len(indices), percentages)
    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    for split_name, selected_indices in split_indices.items():
        for index in selected_indices:
            configs[index].info["config_type"] = f"{theory_dir.name}_{split_name}"
        write_split(theory_dir / f"{split_name}.xyz", (configs[index] for index in selected_indices))

    return len(configs), n_train, n_val, n_test, nonzero_force_configs


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")

    theory_dirs = sorted(path for path in root.iterdir() if (path / "input.data").is_file())
    if not theory_dirs:
        raise SystemExit(f"No theory folders with input.data found under {root}")

    percentages = (args.train, args.val, args.test)
    split_counts(1, percentages)

    for theory_dir in theory_dirs:
        total, n_train, n_val, n_test, nonzero_forces = convert_theory_folder(
            theory_dir,
            percentages=percentages,
            seed=args.seed,
            shuffle=not args.no_shuffle,
            args=args,
        )
        force_note = (
            f"{nonzero_forces} configs with nonzero forces"
            if nonzero_forces
            else "no nonzero forces"
        )
        print(
            f"{theory_dir}: {total} configs -> "
            f"train={n_train}, val={n_val}, test={n_test}; {force_note}"
        )


if __name__ == "__main__":
    main()
