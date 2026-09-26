"""Bridge to the pre-2026-09 layout (tag ``iclr27-submission``).

Old ledger entries, paper provenance comments and the active-learning run records
use the old paths (``experiment_D/checkpoints_3/...``, ``dataset/water_0/...``).
``translate`` maps any such path, relative or absolute, to its current location;
``LETTERS`` maps the old experiment letters to experiment ids. The full table is in
``docs/migration-2026-09.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Tuple

from uq import paths

LETTERS = {
    "A": "ani1x-system/mace-mve/lf-only",
    "B": "ani1x-system/mace-mve/lf-to-hf",
    "E": "ani1x-system/mace-mve/hf-only",
    "C": "ani1x-energy/mace-mve/lf-only",
    "D": "ani1x-energy/mace-mve/lf-to-hf",
    "F": "ani1x-energy/mace-mve/hf-only",
    "wA": "water-random/mace-mve/lf-only",
    "wB": "water-random/mace-mve/lf-to-hf",
    "wC": "water-random/mace-mve/hf-only",
}
DATASETS = {"ani1x_system_split": "ani1x-system", "ani1x_energy_split": "ani1x-energy", "water": "water-random"}

# Renamed code modules (old repo-relative path -> new repo-relative path).
CODE = {
    "eval/reliability.py": "uq/evaluation/reliability.py",
    "eval/epoch_quality.py": "uq/evaluation/epoch_quality.py",
    "eval/epoch_quality_finetune_mixed_dataset.py": "uq/evaluation/epoch_quality_mixed.py",
    "eval/epoch_quality_finetune_same_dataset.py": "uq/evaluation/epoch_quality_same.py",
    "eval/epoch_raw.py": "uq/evaluation/epoch_raw.py",
    "eval/train_curves.py": "uq/evaluation/train_curves.py",
    "eval/distribution.py": "uq/evaluation/distribution.py",
    "eval/energy_ood.py": "uq/evaluation/energy_ood.py",
    "eval/finetune.py": "uq/evaluation/finetune_comparison.py",
    "eval/aggregate_replicates.py": "uq/evaluation/aggregate.py",
    "eval/replicate_statistics.py": "uq/evaluation/replicate_statistics.py",
    "eval/support_filter.py": "uq/evaluation/support_filter.py",
    "eval/csv_cache_utils.py": "uq/evaluation/csv_cache.py",
    "eval/plot_style.py": "uq/evaluation/plot_style.py",
    "eval/plt_member_pred_var_grid.py": "uq/evaluation/extra/member_pred_var_grid.py",
    "eval/plt_epoch_au_members_configs.py": "uq/evaluation/extra/epoch_au_members_configs.py",
    "eval/plot_idealized_au_eu.py": "uq/evaluation/extra/idealized_au_eu.py",
    "eval/test_plotting.py": "uq/tests/test_evaluation.py",
    "new_figures/epoch_predictions.py": "uq/evaluation/epoch_predictions.py",
    "dataset/ani1x_system_splitter.py": "uq/datasets/ani1x_system.py",
    "dataset/ani1x_energy_splitter.py": "uq/datasets/ani1x_energy.py",
    "dataset/convert_water_n2p2_to_extxyz.py": "uq/datasets/water_n2p2.py",
    "experiments/scripts/water_split_overlap.py": "uq/datasets/water_overlap.py",
    "experiments/scripts/train_water_splits.py": "uq/training/scheduler.py",
}
for _name in ("common", "five_splits", "migrate_run_names", "plot_results", "runtime", "support_filter",
              "train_member", "workflow"):
    CODE[f"active_learning/ani_energy/{_name}.py"] = f"uq/active_learning/{_name}.py"
for _name in ("test_five_splits", "test_reporting", "test_workflow"):
    CODE[f"active_learning/ani_energy/{_name}.py"] = f"uq/tests/active_learning/{_name}.py"


def _experiment(letter: str) -> str:
    return LETTERS[letter]


_EXP = r"experiment_(?P<x>[A-F]|w[ABC])"
# (pattern on a repo-relative POSIX path, replacement builder) — first match wins.
RULES: List[Tuple[re.Pattern, Callable[[re.Match], str]]] = [
    (re.compile(rf"^{_EXP}/(?P<kind>checkpoints|models|logs|results)_(?P<s>\d+)(?P<rest>/.*)?$"),
     lambda m: f"runs/{_experiment(m['x'])}/split_{m['s']}/{m['kind']}{m['rest'] or ''}"),
    (re.compile(rf"^{_EXP}/evaluation/cache/split_(?P<s>\d+)(?P<rest>/.*)?$"),
     lambda m: f"experiments/{_experiment(m['x'])}/evaluation/split_{m['s']}{m['rest'] or ''}"),
    (re.compile(rf"^{_EXP}/evaluation/cache/?$"),   # the old per-split cache root
     lambda m: f"experiments/{_experiment(m['x'])}/evaluation"),
    (re.compile(rf"^{_EXP}/evaluation(?P<rest>/.*)?$"),
     lambda m: f"experiments/{_experiment(m['x'])}/evaluation{m['rest'] or ''}"),
    (re.compile(rf"^{_EXP}/config_(?:[A-F]|w[ABC])\.yml$"),
     lambda m: f"experiments/{_experiment(m['x'])}/config.yaml"),
    (re.compile(r"^dataset/ani1x-release\.h5$"), lambda m: "data/raw/ani1x-release.h5"),
    (re.compile(r"^dataset/ani1x-processed(?P<rest>/.*)?$"),
     lambda m: f"data/legacy/ani1x-processed{m['rest'] or ''}"),
    (re.compile(r"^dataset/(?P<d>ani1x_system_split|ani1x_energy_split|water)_(?P<s>\d+)(?P<rest>/.*)?$"),
     lambda m: f"data/{DATASETS[m['d']]}/split_{m['s']}{m['rest'] or ''}"),
    (re.compile(r"^eval/cache/geometry/(?P<k>system|energy)_(?P<s>\d+)_(?P<f>[a-z_]+)\.npz$"),
     lambda m: f"data/ani1x-{m['k']}/split_{m['s']}/geometry/{m['f']}.npz"),
    (re.compile(r"^new_figures/_cache/epochs/(?P<x>[A-F])_(?P<s>\d+)\.(?P<ext>npz|log)$"),
     lambda m: f"experiments/{_experiment(m['x'])}/evaluation/split_{m['s']}/epoch_predictions_cc_test_ood.{m['ext']}"),
    (re.compile(r"^active_learning/ani_energy/runs(?P<rest>/.*)?$"),
     lambda m: f"runs/active-learning/ani1x-energy{m['rest'] or ''}"),
    (re.compile(r"^wandb(?P<rest>/.*)?$"), lambda m: f"runs/wandb{m['rest'] or ''}"),
    (re.compile(r"^experiments/runs(?P<rest>/.*)?$"), lambda m: f"runs/_logs{m['rest'] or ''}"),
    (re.compile(r"^new_figures/out(?P<rest>/.*)?$"), lambda m: f"papers/iclr27/figures/out{m['rest'] or ''}"),
    (re.compile(r"^new_figures/results\.tex$"), lambda m: "papers/iclr27/drafts/results.tex"),
    (re.compile(r"^new_figures/README\.md$"), lambda m: "papers/iclr27/README.md"),
    (re.compile(r"^new_figures/(?P<rest>[^/]+\.(?:py|sh|md))$"),
     lambda m: f"papers/iclr27/figures/{m['rest']}"),
]


def translate_relative(old: str) -> str:
    """New repo-relative path for an old repo-relative path (unchanged if no rule applies)."""
    while old.startswith("./"):
        old = old[2:]
    if old in CODE:
        return CODE[old]
    for pattern, build in RULES:
        match = pattern.match(old)
        if match:
            return build(match)
    return old


def _repo_relative(text: str):
    """Repo-relative part of an absolute path written by any checkout, or None.

    Old records hold absolute paths of the machine that wrote them. The checkout root is
    recognised by what follows it (the leftmost remainder that a rule translates or that
    exists in this checkout), so no machine-specific root is stored here.
    """
    root = str(paths.ROOT)
    if text == root or text.startswith(root + "/"):
        return text[len(root) + 1:]
    for index in (i for i, char in enumerate(text) if char == "/"):
        rest = text[index + 1:]
        if rest and (translate_relative(rest) != rest or (paths.ROOT / rest).exists()):
            return rest
    return None


def translate(old) -> Path:
    """Current location of an old path; absolute inputs give absolute outputs."""
    text = str(old)
    if not text.startswith("/"):
        return Path(translate_relative(text))
    relative = _repo_relative(text)
    if relative is None:
        return Path(text)
    new = translate_relative(relative)
    if new.startswith("runs/") and paths.RUNS != paths.ROOT / "runs":
        return paths.RUNS / new[len("runs/"):]
    return paths.ROOT / new
