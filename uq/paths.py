"""Directory layout of the repository.

This is the only module that knows where things live. Every other module, the
CLI and the paper figure scripts build paths through these helpers, so changing
the layout means editing this file (and ``docs/layout.md``) only.

Heavy run artifacts default to ``<repo>/runs``; set ``UQ_RUNS`` to put them
elsewhere (for example on another disk).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"
RUNS = Path(os.environ.get("UQ_RUNS", ROOT / "runs"))
RESULTS = ROOT / "results"
PAPERS = ROOT / "papers"
DOCS = ROOT / "docs"

MACE_RUN_TRAIN = ROOT / "mace" / "cli" / "run_train.py"
WANDB = RUNS / "wandb"
SCHEDULER_LOGS = RUNS / "_logs"
ACTIVE_LEARNING_RUNS = RUNS / "active-learning"


def rel(path: Path) -> Path:
    """Path relative to the repository root when possible (commands use these)."""
    path = Path(path)
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


# -- datasets ---------------------------------------------------------------------------

def dataset_dir(dataset: str) -> Path:
    return DATA / dataset


def data_split_dir(dataset: str, split: int) -> Path:
    return dataset_dir(dataset) / f"split_{split}"


def geometry_cache(dataset: str, split: int, stem: str) -> Path:
    """Minimum-interatomic-distance cache of one split file (support filter)."""
    return data_split_dir(dataset, split) / "geometry" / f"{stem}.npz"


# -- experiments ------------------------------------------------------------------------

def experiment_dir(experiment_id: str) -> Path:
    return EXPERIMENTS / experiment_id


def evaluation_dir(experiment_id: str) -> Path:
    """Five-split aggregates (tracked)."""
    return experiment_dir(experiment_id) / "evaluation"


def split_evaluation_dir(experiment_id: str, split: int) -> Path:
    """Per-split evaluation outputs (tracked; GPU-costly to regenerate)."""
    return evaluation_dir(experiment_id) / f"split_{split}"


# -- runs (git-ignored) -----------------------------------------------------------------

def run_dir(experiment_id: str, split: int) -> Path:
    return RUNS / experiment_id / f"split_{split}"


def checkpoints_dir(experiment_id: str, split: int) -> Path:
    """Per-epoch checkpoints ``<name>_run-<member>_epoch-<e>.pt`` and final ``.model``."""
    return run_dir(experiment_id, split) / "checkpoints"


def models_dir(experiment_id: str, split: int) -> Path:
    return run_dir(experiment_id, split) / "models"


def logs_dir(experiment_id: str, split: int) -> Path:
    return run_dir(experiment_id, split) / "logs"


def results_dir(experiment_id: str, split: int) -> Path:
    """MACE per-epoch result logs, used for checkpoint selection (validation NLL)."""
    return run_dir(experiment_id, split) / "results"
