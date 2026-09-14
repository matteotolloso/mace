"""Reuse the project's MVE evaluation and graph construction without pool labels."""

from __future__ import annotations

from pathlib import Path

from common import ENERGY_KEY, config_id, geometry_only

import torch
from ase.io import read
from mace import data, tools
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools import torch_geometric

import reliability


def load_ensemble(paths, device):
    models = reliability.load_models([Path(path) for path in paths], torch.device(device))
    for model in models:
        if not getattr(model, "predict_mve", False):
            raise ValueError("Every ensemble member must predict MVE mean and variance.")
        if len(getattr(model, "heads", ["Default"])) != 1:
            raise ValueError("This POC supports the existing single-head ANI ensembles only.")
    return models


def make_loader(atoms_list, model, batch_size, *, labeled, training=False, seed=0,
                num_workers=0, pin_memory=False):
    head = getattr(model, "heads", ["Default"])[0]
    keys = KeySpecification(info_keys={"energy": ENERGY_KEY}, arrays_keys={})
    configs = []
    for atoms in atoms_list:
        if labeled:
            if ENERGY_KEY not in atoms.info:
                raise ValueError("Missing CC label in labeled training/evaluation data.")
        else:
            atoms = geometry_only(atoms)
        # Missing energy is represented by a zero-weight placeholder in AtomicData.
        # Shared evaluate_split computes TU independently of that placeholder.
        configs.append(config_from_atoms(atoms, key_specification=keys, head_name=head))
    table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    dataset = [data.AtomicData.from_config(
        config, z_table=table, cutoff=float(model.r_max), heads=getattr(model, "heads", None)
    ) for config in configs]
    if not dataset or (training and len(dataset) < batch_size):
        raise ValueError("Empty loader (or training set smaller than its batch size).")
    return torch_geometric.dataloader.DataLoader(
        dataset, batch_size=batch_size, shuffle=training, drop_last=training,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )


def predict(paths, xyz_path, device, batch_size, *, labeled):
    tools.set_default_dtype("float32")
    models = load_ensemble(paths, device)
    atoms_list = read(xyz_path, index=":")
    loader = make_loader(atoms_list, models[0], batch_size, labeled=labeled)
    rows = reliability.evaluate_split(
        models, loader, torch.device(device), per_atom=True, head_name=None,
        split_name=Path(xyz_path).stem, log_every_batches=10,
    )
    if len(rows) != len(atoms_list):
        raise RuntimeError("Inference did not cover every configuration.")
    if not labeled:
        allowed = ("config_index", "num_atoms", "pred_energy", "aleatoric_var",
                   "epistemic_var", "total_var")
        rows = [{key: row[key] for key in allowed} for row in rows]
    for atoms, row in zip(atoms_list, rows):
        row["al_id"] = config_id(atoms)
    del models, loader
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows
