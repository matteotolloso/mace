"""Experiment and dataset specs.

Datasets are declared in ``data/<dataset>/dataset.yaml`` and experiments in
``experiments/<dataset>/<model>-<uq>/<protocol>/experiment.yaml``; the experiment
id is that directory path relative to ``experiments/``. Field meanings are
documented in ``docs/experiments.md`` and ``docs/datasets.md``.

    >>> from uq import registry
    >>> exp = registry.experiment("B")            # legacy letter or full id
    >>> exp.id, exp.parent.id, exp.data_file("id", split=0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Tuple

import yaml

from uq import paths

PROTOCOLS = ("lf-only", "lf-to-hf", "hf-only")   # also the listing order
# (model, uq) pairs with a training launcher and an evaluation backend. Add a pair only
# together with its code (docs/extending.md); specs naming anything else fail validation.
BACKENDS = {
    ("mace", "mve-ensemble"),   # MVE members (Gaussian NLL): AU = mean member variance, EU = member spread
    ("mace", "mse-ensemble"),   # MSE members, no variance head: AU = 0, EU = member spread (concepts.md)
}


class SpecError(ValueError):
    """An experiment or dataset spec is missing, ambiguous or inconsistent."""


@dataclass(frozen=True)
class Fidelity:
    key: str                      # "lf" or "hf"
    name: str                     # short name used in titles, e.g. "DFT", "CC"
    method: str                   # level of theory
    energy_key: str               # extxyz info key of the energy label
    files: Mapping[str, str]      # role (train, val, id, ood, test) -> path inside split_<s>/


@dataclass(frozen=True)
class Dataset:
    id: str
    description: str
    splits: Tuple[int, ...]
    tests: Tuple[str, ...]
    support_filter: bool
    fidelities: Mapping[str, Fidelity]
    build: Mapping = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return paths.dataset_dir(self.id)

    def split_dir(self, split: int) -> Path:
        return paths.data_split_dir(self.id, split)

    def file(self, fidelity: str, role: str, split: int) -> Path:
        try:
            name = self.fidelities[fidelity].files[role]
        except KeyError as exc:
            raise SpecError(f"{self.id}: no {fidelity!r} file for role {role!r}") from exc
        return self.split_dir(split) / name


@dataclass(frozen=True)
class Experiment:
    id: str
    legacy_id: Optional[str]
    dataset_id: str
    model: str
    uq: str
    protocol: str
    target: str
    parent_id: Optional[str]
    splits: Tuple[int, ...]
    members: Tuple[int, ...]
    training: Mapping
    evaluation: Mapping

    # -- related specs ---------------------------------------------------------------
    @property
    def dataset(self) -> Dataset:
        return dataset(self.dataset_id)

    @property
    def parent(self) -> Optional["Experiment"]:
        return experiment(self.parent_id) if self.parent_id else None

    @property
    def fidelity(self) -> Fidelity:
        """Fidelity of the training labels."""
        return self.dataset.fidelities[self.target]

    # -- files -------------------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return paths.experiment_dir(self.id)

    @property
    def config_path(self) -> Path:
        return self.dir / self.training.get("config", "config.yaml")

    @property
    def run_name(self) -> str:
        """MACE ``name`` from the config: checkpoints are ``<name>_run-<member>...``."""
        return _config(self.config_path).get("name", "MACE_model")

    def data_file(self, role: str, split: int, fidelity: Optional[str] = None) -> Path:
        return self.dataset.file(fidelity or self.target, role, split)

    def final_model(self, split: int, member: int) -> Path:
        return paths.checkpoints_dir(self.id, split) / f"{self.run_name}_run-{member}.model"

    def train_log(self, split: int, member: int) -> Path:
        return paths.results_dir(self.id, split) / f"{self.run_name}_run-{member}_train.txt"

    def member_complete(self, split: int, member: int) -> bool:
        """Complete = final model and MACE result log both exist (as check_experiments.sh did)."""
        return self.final_model(split, member).is_file() and self.train_log(split, member).is_file()

    @property
    def label(self) -> str:
        legacy = f" ({self.legacy_id})" if self.legacy_id else ""
        return f"{self.id}{legacy}"


# -- loading --------------------------------------------------------------------------------

@lru_cache(None)
def _config(path: Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle) or {}


def _load_yaml(path: Path) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SpecError(f"{path}: expected a mapping")
    return data


@lru_cache(None)
def datasets() -> Mapping[str, Dataset]:
    out = {}
    for path in sorted(paths.DATA.glob("*/dataset.yaml")):
        raw = _load_yaml(path)
        dataset_id = path.parent.name
        fidelities = {
            key: Fidelity(key=key, name=spec["name"], method=spec.get("method", spec["name"]),
                          energy_key=spec["energy_key"], files=dict(spec["files"]))
            for key, spec in raw["fidelities"].items()
        }
        out[dataset_id] = Dataset(
            id=dataset_id, description=raw.get("description", ""), splits=tuple(raw["splits"]),
            tests=tuple(raw["tests"]), support_filter=bool(raw.get("support_filter", False)),
            fidelities=fidelities, build=raw.get("build", {}),
        )
    return out


@lru_cache(None)
def experiments() -> Mapping[str, Experiment]:
    out = {}
    specs = sorted(paths.EXPERIMENTS.glob("**/experiment.yaml"))
    order = {p: i for i, p in enumerate(PROTOCOLS)}
    specs.sort(key=lambda p: (p.parent.parent.as_posix(), order.get(p.parent.name, len(order)), p.as_posix()))
    for path in specs:
        raw = _load_yaml(path)
        experiment_id = path.parent.relative_to(paths.EXPERIMENTS).as_posix()
        out[experiment_id] = Experiment(
            id=experiment_id, legacy_id=raw.get("legacy_id"), dataset_id=raw["dataset"],
            model=raw["model"], uq=raw["uq"], protocol=raw["protocol"], target=raw["target"],
            parent_id=raw.get("parent"), splits=tuple(raw["splits"]), members=tuple(raw["members"]),
            training=raw.get("training", {}), evaluation=raw.get("evaluation", {}),
        )
    return out


def dataset(name: str) -> Dataset:
    try:
        return datasets()[name]
    except KeyError as exc:
        raise SpecError(f"unknown dataset {name!r}; known: {', '.join(datasets())}") from exc


def experiment(name: str) -> Experiment:
    """Look up an experiment by id (``ani1x-system/mace-mve/lf-to-hf``), by its path
    (``experiments/.../``) or by its legacy letter (``B``)."""
    table = experiments()
    key = str(name).strip().rstrip("/")
    key = key.removeprefix("experiments/")
    if key in table:
        return table[key]
    legacy = [exp for exp in table.values() if exp.legacy_id == key]
    if len(legacy) == 1:
        return legacy[0]
    hint = [exp_id for exp_id in table if key and key in exp_id]
    raise SpecError(f"unknown experiment {name!r}" + (f"; did you mean {', '.join(hint)}?" if hint else
                                                      f"; known: {', '.join(table)}"))


def require_backend(exp: "Experiment") -> None:
    """Refuse to build commands for a model/UQ method that has no implementation yet."""
    if (exp.model, exp.uq) not in BACKENDS:
        raise SpecError(f"{exp.label}: no training/evaluation backend for model {exp.model!r} with uq {exp.uq!r}; "
                        f"implemented: {sorted(BACKENDS)} (docs/extending.md)")


def resolve(names) -> list:
    """Experiments for CLI arguments; no argument means all of them."""
    return [experiment(name) for name in names] if names else list(experiments().values())


def validate() -> list:
    """Return a list of human-readable problems with the specs (empty if all good)."""
    problems = []
    for exp in experiments().values():
        where = f"experiments/{exp.id}/experiment.yaml"
        if exp.protocol not in PROTOCOLS:
            problems.append(f"{where}: protocol {exp.protocol!r} not in {PROTOCOLS}")
        if (exp.model, exp.uq) not in BACKENDS:
            problems.append(f"{where}: no backend for model {exp.model!r} with uq {exp.uq!r} "
                            f"(implemented: {sorted(BACKENDS)}; see docs/extending.md)")
        try:
            data = exp.dataset
        except SpecError as error:
            problems.append(f"{where}: {error}")
            continue
        if exp.target not in data.fidelities:
            problems.append(f"{where}: target {exp.target!r} not a fidelity of {data.id}")
        if (exp.protocol == "lf-to-hf") != bool(exp.parent_id):
            problems.append(f"{where}: lf-to-hf experiments need a parent, others must not have one")
        if exp.parent_id:
            try:
                parent = exp.parent
            except SpecError as error:
                problems.append(f"{where}: {error}")
            else:
                if (parent.dataset_id, parent.splits, parent.members) != (exp.dataset_id, exp.splits, exp.members):
                    problems.append(f"{where}: parent must share dataset, splits and members")
        if not exp.config_path.is_file():
            problems.append(f"{where}: missing {exp.config_path.name}")
        for role in {*exp.evaluation.get("reliability", []), *exp.evaluation.get("sets", []),
                     *exp.evaluation.get("member_predictions", []),
                     exp.training.get("test_role", "id")}:
            if role not in exp.fidelity.files:
                problems.append(f"{where}: role {role!r} has no {exp.target} file in {data.id}")
    legacy = [exp.legacy_id for exp in experiments().values() if exp.legacy_id]
    if len(legacy) != len(set(legacy)):
        problems.append("duplicate legacy_id values")
    return problems
