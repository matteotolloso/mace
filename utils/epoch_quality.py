#!/usr/bin/env python3
"""Plot epoch-wise uncertainty quality metrics from ensemble checkpoints.

This script evaluates an ensemble directly from checkpoint files and measures how
well uncertainty predicts the true error over training.

For each evaluated epoch it computes, separately for aleatoric, epistemic, and
total uncertainty:
- Pearson correlation between per-system uncertainty and per-system squared error
- Spearman correlation between per-system uncertainty and per-system squared error
- AUSE (Area Under the Sparsification Error), computed from the ordering induced
  by per-system uncertainty variance (i.e. predicted sigma^2) compared with the
  oracle ordering induced by the true per-system absolute error. The
  sparsification curves themselves are MAE-based, following the formal
  definition of AUSE.
- ENCE (Expected Normalized Calibration Error), computed from RMSE-vs-RMV
  reliability bins on the same per-system rows.

The split is provided through ``--split-path`` / ``--energy-key`` and epochs are
sampled using ``--every-n-epochs``.

Per-atom mode is the default:
- energies are normalized as ``E / N``
- variances are normalized as ``var / N^2``

Use ``--total`` to switch to total-system quantities.

Unlike a reliability diagram, this script does not use RMSE-vs-RMV binning for
all metrics. Pearson, Spearman, and AUSE are computed directly from per-system
uncertainty variance (i.e. predicted sigma^2). Pearson and Spearman use
per-system squared error, AUSE uses per-system absolute error, and ENCE is
computed from reliability bins with configurable ``--num-bins``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn

from csv_cache_utils import load_cached_csv_rows, parse_float, parse_int
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
    module=r"e3nn\.o3\._wigner",
)

from mace import data
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools import torch_geometric, torch_tools, utils as mace_utils
from reliability import build_binned_rows, compute_ence_summary, trim_rows_by_key


CHECKPOINT_PATTERN = re.compile(
    r"^(?P<experiment>.+)_run-(?P<seed>\d+)_epoch-(?P<epoch>\d+)\.pt$"
)
LOGGER = logging.getLogger(__name__)
UNCERTAINTY_SPECS = [
    ("aleatoric", "aleatoric_var"),
    ("epistemic", "epistemic_var"),
    ("total", "total_var"),
]


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=False,
        help="Folder containing checkpoints named <experiment>_run-<seed>_epoch-<epoch>.pt and companion <experiment>_run-<seed>.model files.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        help="Experiment name prefix used to filter checkpoints inside --checkpoints-dir.",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        required=False,
        help="Path to the .xyz split used for epoch-wise evaluation.",
    )
    parser.add_argument(
        "--energy-key",
        type=str,
        required=False,
        help="Energy key to read from --split-path.",
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Optional model head to use for multi-head models.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu"],
        default="cpu",
        help="Evaluation device.",
    )
    parser.add_argument(
        "--default-dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Torch default dtype. If it does not match the checkpoint dtype, the checkpoint dtype is used for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--every-n-epochs",
        type=int,
        default=1,
        help="Evaluate only epochs divisible by this value. Example: 5 -> evaluate epochs 0, 5, 10, ...",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.0,
        help="Symmetric fraction to trim from the lowest and highest total uncertainty values before summarizing each epoch. Example: 0.005 trims 0.5%% on each side.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of uncertainty bins used for ENCE.",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system energies and variances instead of per-atom values.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="epoch_unc_quality.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="epoch_unc_quality.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--drop-first-k-epochs",
        type=int,
        default=0,
        help="Exclude the first k evaluated epochs from the plot only.",
    )
    parser.add_argument(
        "--free-scale",
        action="store_true",
        help="Also save a second autoscaled plot with '-free-scale' appended to the output filename.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Verbosity for progress logging.",
    )
    parser.set_defaults(per_atom=True)
    return parser.parse_args()


def discover_checkpoint_map(
    checkpoints_dir: Path,
    experiment_name: Optional[str],
) -> Dict[int, Dict[int, Path]]:
    start_time = time.perf_counter()
    by_seed: Dict[int, Dict[int, Path]] = {}
    experiments = set()

    for path in checkpoints_dir.glob("*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        experiment = match.group("experiment")
        if experiment_name is not None and experiment != experiment_name:
            continue
        seed = int(match.group("seed"))
        epoch = int(match.group("epoch"))
        by_seed.setdefault(seed, {})[epoch] = path
        experiments.add(experiment)

    if not by_seed:
        experiment_msg = (
            f" for experiment '{experiment_name}'" if experiment_name is not None else ""
        )
        raise RuntimeError(
            f"No checkpoints matching <experiment>_run-<seed>_epoch-<epoch>.pt were found in {checkpoints_dir}{experiment_msg}."
        )

    if experiment_name is None and len(experiments) > 1:
        raise RuntimeError(
            f"Multiple experiments were found in {checkpoints_dir}: {sorted(experiments)}. Pass --experiment-name to disambiguate."
        )

    if len(by_seed) < 2:
        raise RuntimeError(
            f"At least 2 ensemble members are required, but only seeds {sorted(by_seed)} were found."
        )

    LOGGER.info(
        "Discovered %d ensemble members in %.2fs from %s%s",
        len(by_seed),
        time.perf_counter() - start_time,
        checkpoints_dir,
        f" (experiment={experiment_name})" if experiment_name is not None else "",
    )
    return dict(sorted(by_seed.items()))


def select_common_epochs(
    checkpoint_map: Dict[int, Dict[int, Path]],
    every_n_epochs: int,
) -> List[int]:
    if every_n_epochs <= 0:
        raise ValueError("--every-n-epochs must be >= 1")

    common_epochs = sorted(
        set.intersection(*(set(epoch_map.keys()) for epoch_map in checkpoint_map.values()))
    )
    if not common_epochs:
        raise RuntimeError("No common epochs found across ensemble members.")

    filtered_epochs = [epoch for epoch in common_epochs if epoch % every_n_epochs == 0]
    if not filtered_epochs:
        raise RuntimeError(
            f"No common epochs survived the --every-n-epochs={every_n_epochs} filter. Common epochs were: {common_epochs}"
        )

    LOGGER.info(
        "Using %d common epochs after every-%d filtering: %s%s",
        len(filtered_epochs),
        every_n_epochs,
        filtered_epochs[:10],
        " ..." if len(filtered_epochs) > 10 else "",
    )
    return filtered_epochs


def _get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.dtype
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.dtype
    return torch.get_default_dtype()


def _cast_batch_dict_dtype(
    batch_dict: Dict[str, torch.Tensor], dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    casted: Dict[str, torch.Tensor] = {}
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            casted[key] = value.to(dtype=dtype)
        else:
            casted[key] = value
    return casted


def _select_head_tensor(
    x: Optional[torch.Tensor],
    batch_heads: Optional[torch.Tensor],
    head_name: Optional[str],
    model_heads: Optional[List[str]],
) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.ndim >= 2 and model_heads is not None and x.shape[1] == len(model_heads):
        if head_name is not None:
            if head_name not in model_heads:
                raise ValueError(
                    f"Requested head '{head_name}' not in model heads {model_heads}."
                )
            head_idx = model_heads.index(head_name)
            return x[:, head_idx, ...]
        if batch_heads is not None:
            graph_idx = torch.arange(x.shape[0], device=x.device)
            return x[graph_idx, batch_heads.long(), ...]
        return x[:, 0, ...]
    return x


def load_models(checkpoint_paths: List[Path], device: torch.device) -> List[torch.nn.Module]:
    total_start = time.perf_counter()
    models: List[torch.nn.Module] = []
    for model_idx, path in enumerate(checkpoint_paths, start=1):
        checkpoint_obj = torch.load(
            f=str(path), map_location=str(device), weights_only=False
        )
        if isinstance(checkpoint_obj, torch.nn.Module):
            model = checkpoint_obj
        elif isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
            tag = path.name.rsplit("_epoch-", 1)[0]
            companion_model_path = path.with_name(f"{tag}.model")
            if not companion_model_path.exists():
                raise RuntimeError(
                    "Checkpoint file contains only a state_dict and no companion "
                    f"serialized model was found at {companion_model_path}."
                )
            model = torch.load(
                f=str(companion_model_path),
                map_location=str(device),
                weights_only=False,
            )
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError(
                    f"Companion model file {companion_model_path} did not contain a torch.nn.Module."
                )
            model.load_state_dict(checkpoint_obj["model"], strict=True)
        else:
            raise RuntimeError(
                f"Unsupported checkpoint contents in {path}: expected torch.nn.Module or dict with key 'model'."
            )

        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
        LOGGER.debug("Loaded model %d/%d from %s", model_idx, len(checkpoint_paths), path.name)

    model0 = models[0]
    for model in models[1:]:
        if hasattr(model, "atomic_numbers") and hasattr(model0, "atomic_numbers"):
            if not torch.equal(model.atomic_numbers.cpu(), model0.atomic_numbers.cpu()):
                raise RuntimeError("Ensemble checkpoints have different atomic_numbers.")
        if float(model.r_max) != float(model0.r_max):
            raise RuntimeError("Ensemble checkpoints have different r_max values.")
        if getattr(model, "heads", None) != getattr(model0, "heads", None):
            raise RuntimeError("Ensemble checkpoints have different heads.")

    LOGGER.info(
        "Loaded %d ensemble members in %.2fs on %s",
        len(models),
        time.perf_counter() - total_start,
        device,
    )
    return models


def build_dataloader(
    xyz_path: Path,
    energy_key: str,
    model: torch.nn.Module,
    batch_size: int,
    head_name: Optional[str],
):
    total_start = time.perf_counter()
    LOGGER.info("Reading split from %s with energy key '%s'", xyz_path, energy_key)
    atoms_list = ase.io.read(str(xyz_path), index=":")
    if len(atoms_list) == 0:
        raise RuntimeError(f"No configurations found in {xyz_path}.")

    keyspec = KeySpecification(info_keys={"energy": energy_key}, arrays_keys={})
    effective_head = head_name or (
        getattr(model, "heads", ["Default"])[0]
        if getattr(model, "heads", None)
        else "Default"
    )

    configs = []
    for idx, atoms in enumerate(atoms_list):
        if energy_key not in atoms.info:
            raise KeyError(
                f"Energy key '{energy_key}' not found in configuration index {idx} of {xyz_path}."
            )
        configs.append(
            config_from_atoms(atoms, key_specification=keyspec, head_name=effective_head)
        )

    z_table = mace_utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", None)
    dataset = [
        data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(model.r_max),
            heads=heads,
        )
        for config in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    LOGGER.info(
        "Prepared dataloader for %s: %d configs, batch_size=%d, total prep %.2fs",
        xyz_path,
        len(dataset),
        batch_size,
        time.perf_counter() - total_start,
    )
    return loader


def evaluate_split(
    models: List[torch.nn.Module],
    data_loader,
    device: torch.device,
    per_atom: bool,
    head_name: Optional[str],
    split_name: str,
) -> List[Dict[str, float]]:
    raw_rows: List[Dict[str, float]] = []
    model_heads = getattr(models[0], "heads", None)
    model_dtype = _get_model_dtype(models[0])
    running_index = 0
    num_batches = len(data_loader)
    LOGGER.info(
        "Starting %s evaluation: %d configs across %d batches with %d ensemble members on %s",
        split_name,
        len(data_loader.dataset),
        num_batches,
        len(models),
        device,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, start=1):
            batch = batch.to(device)
            batch_dict = _cast_batch_dict_dtype(batch.to_dict(), model_dtype)
            batch_heads = batch["head"] if "head" in batch.keys else None
            num_graphs = int(batch.num_graphs)
            num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(device=device, dtype=model_dtype)

            member_preds = []
            member_vars = []
            for model in models:
                output = model(
                    batch_dict,
                    training=False,
                    compute_force=False,
                    compute_virials=False,
                    compute_stress=False,
                    compute_hessian=False,
                    compute_edge_forces=False,
                )
                pred_energy = _select_head_tensor(
                    output.get("energy_mean", output.get("energy")),
                    batch_heads,
                    head_name,
                    model_heads,
                )
                pred_var = output.get("energy_var", None)
                if pred_var is None and output.get("energy_logvar", None) is not None:
                    pred_var = torch.exp(output["energy_logvar"])
                pred_var = _select_head_tensor(
                    pred_var,
                    batch_heads,
                    head_name,
                    model_heads,
                )
                if pred_var is None:
                    pred_var = torch.zeros_like(pred_energy)
                member_preds.append(pred_energy)
                member_vars.append(pred_var)

            pred_stack_total = torch.stack(member_preds, dim=0)
            var_stack_total = torch.stack(member_vars, dim=0)
            ref_energy_total = batch.energy.to(device=device, dtype=model_dtype)

            ensemble_pred_total = torch.mean(pred_stack_total, dim=0)
            aleatoric_var_total = torch.mean(var_stack_total, dim=0)
            epistemic_var_total = torch.var(pred_stack_total, dim=0, unbiased=False)
            total_var_total = aleatoric_var_total + epistemic_var_total

            config_weight = (
                batch.weight.to(device=device, dtype=model_dtype)
                if "weight" in batch.keys
                else torch.ones(num_graphs, device=device, dtype=model_dtype)
            )
            config_energy_weight = (
                batch.energy_weight.to(device=device, dtype=model_dtype)
                if "energy_weight" in batch.keys
                else torch.ones(num_graphs, device=device, dtype=model_dtype)
            )
            eps = torch.tensor(1e-16, device=device, dtype=model_dtype)
            total_var_safe = total_var_total + eps
            log_2pi = math.log(2.0 * math.pi)
            nll_energy = (
                config_weight
                * config_energy_weight
                * 0.5
                * (((ref_energy_total - ensemble_pred_total) ** 2) / total_var_safe + torch.log(total_var_safe) + log_2pi)
                / num_atoms
            )

            if per_atom:
                pred_stack = pred_stack_total / num_atoms.unsqueeze(0)
                var_stack = var_stack_total / (num_atoms.unsqueeze(0) ** 2)
                ref_energy = ref_energy_total / num_atoms
            else:
                pred_stack = pred_stack_total
                var_stack = var_stack_total
                ref_energy = ref_energy_total

            ensemble_pred = torch.mean(pred_stack, dim=0)
            aleatoric_var = torch.mean(var_stack, dim=0)
            epistemic_var = torch.var(pred_stack, dim=0, unbiased=False)
            total_var = aleatoric_var + epistemic_var
            sq_error = (ensemble_pred - ref_energy) ** 2

            for local_idx in range(num_graphs):
                raw_rows.append(
                    {
                        "config_index": running_index + local_idx,
                        "sq_error": float(sq_error[local_idx].detach().cpu().item()),
                        "aleatoric_var": float(aleatoric_var[local_idx].detach().cpu().item()),
                        "epistemic_var": float(epistemic_var[local_idx].detach().cpu().item()),
                        "total_var": float(total_var[local_idx].detach().cpu().item()),
                        "nll_energy": float(nll_energy[local_idx].detach().cpu().item()),
                    }
                )
            running_index += num_graphs

            if batch_idx == 1 or batch_idx == num_batches:
                LOGGER.info(
                    "%s progress: batch %d/%d, rows=%d",
                    split_name,
                    batch_idx,
                    num_batches,
                    len(raw_rows),
                )

    return raw_rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    xv = x[valid]
    yv = y[valid]
    if np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    xv = x[valid]
    yv = y[valid]
    x_rank = _rankdata(xv)
    y_rank = _rankdata(yv)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def compute_ause(scores: np.ndarray, abs_errors: np.ndarray) -> float:
    valid = np.isfinite(scores) & np.isfinite(abs_errors)
    if np.count_nonzero(valid) < 2:
        return float("nan")

    scores = scores[valid]
    abs_errors = abs_errors[valid]
    if np.any(abs_errors < 0.0):
        return float("nan")

    n = len(scores)
    unc_order = np.argsort(scores)[::-1]
    oracle_order = np.argsort(abs_errors)[::-1]

    unc_errors = abs_errors[unc_order]
    oracle_errors = abs_errors[oracle_order]

    unc_tail_mae = np.cumsum(unc_errors[::-1])[::-1] / np.arange(n, 0, -1)
    oracle_tail_mae = np.cumsum(oracle_errors[::-1])[::-1] / np.arange(n, 0, -1)

    baseline = unc_tail_mae[0]
    if baseline <= 0.0 or not np.isfinite(baseline):
        return float("nan")

    unc_curve = unc_tail_mae / baseline
    oracle_curve = oracle_tail_mae / baseline
    sparsification_error = unc_curve - oracle_curve
    fractions = np.arange(n, dtype=float) / float(n)
    return _trapezoid_integral(sparsification_error, fractions)


def summarize_epoch(raw_rows: List[Dict[str, float]], num_bins: int) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    sq_errors = np.array([row["sq_error"] for row in raw_rows], dtype=float)
    abs_errors = np.sqrt(np.clip(sq_errors, a_min=0.0, a_max=None))
    nll_values = np.array([row["nll_energy"] for row in raw_rows], dtype=float)
    binned_rows = build_binned_rows(raw_rows, num_bins=num_bins)
    ence_summary = compute_ence_summary(binned_rows)
    for unc_name, unc_key in UNCERTAINTY_SPECS:
        scores = np.array([row[unc_key] for row in raw_rows], dtype=float)
        summary[f"pearson_{unc_name}"] = compute_pearson(scores, sq_errors)
        summary[f"spearman_{unc_name}"] = compute_spearman(scores, sq_errors)
        summary[f"ause_{unc_name}"] = compute_ause(scores, abs_errors)
        summary[f"ence_{unc_name}"] = ence_summary[unc_name]
        finite_scores = scores[np.isfinite(scores)]
        summary[f"magnitude_{unc_name}"] = (
            float(np.mean(np.abs(finite_scores))) if finite_scores.size > 0 else float("nan")
        )
    finite_sq_errors = sq_errors[np.isfinite(sq_errors)]
    summary["rmse_e_atom"] = (
        float(np.sqrt(np.mean(finite_sq_errors))) if finite_sq_errors.size > 0 else float("nan")
    )
    finite_nll = nll_values[np.isfinite(nll_values)]
    summary["nll_energy"] = float(np.mean(finite_nll)) if finite_nll.size > 0 else float("nan")
    return summary


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "pearson_aleatoric",
        "pearson_epistemic",
        "pearson_total",
        "spearman_aleatoric",
        "spearman_epistemic",
        "spearman_total",
        "ause_aleatoric",
        "ause_epistemic",
        "ause_total",
        "ence_aleatoric",
        "ence_epistemic",
        "ence_total",
        "magnitude_aleatoric",
        "magnitude_epistemic",
        "magnitude_total",
        "rmse_e_atom",
        "nll_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> Optional[List[Dict[str, float]]]:
    metric_fields = [
        "pearson_aleatoric",
        "pearson_epistemic",
        "pearson_total",
        "spearman_aleatoric",
        "spearman_epistemic",
        "spearman_total",
        "ause_aleatoric",
        "ause_epistemic",
        "ause_total",
        "ence_aleatoric",
        "ence_epistemic",
        "ence_total",
        "magnitude_aleatoric",
        "magnitude_epistemic",
        "magnitude_total",
        "rmse_e_atom",
        "nll_energy",
    ]
    field_parsers = {"epoch": parse_int}
    for field in metric_fields:
        field_parsers[field] = parse_float
    cached_rows = load_cached_csv_rows(
        path,
        required_fields=["epoch", *metric_fields],
        field_parsers=field_parsers,
        logger=LOGGER,
        label="epoch_quality",
        key_fields=["epoch"],
    )
    if cached_rows is None:
        return None
    return cached_rows


def with_free_scale_suffix(path: Path) -> Path:
    return path.with_name(f"{path.stem}-free-scale{path.suffix}")


def write_plot(
    path: Path,
    rows: List[Dict[str, float]],
    drop_first_k_epochs: int,
    *,
    fixed_scales: bool = True,
    title_suffix: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_plot = rows[drop_first_k_epochs:] if drop_first_k_epochs > 0 else rows
    if not rows_plot:
        raise RuntimeError("No epochs left to plot after applying --drop-first-k-epochs.")

    epochs = np.array([row["epoch"] for row in rows_plot], dtype=int)
    style_map = {
        "aleatoric": {"color": "tab:blue", "label": "AU"},
        "epistemic": {"color": "tab:orange", "label": "EU"},
        "total": {"color": "tab:green", "label": "Total"},
    }
    metric_specs = [
        ("spearman", "Spearman ↑"),
        ("ause", "AUSE ↓"),
        ("ence", "ENCE ↓"),
        ("magnitude", "|Uncertainty|"),
    ]

    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14
    suptitle_fontsize = 20

    def plot_series_with_clipped_markers(
        ax,
        x_values: np.ndarray,
        y_values: np.ndarray,
        *,
        color: str,
        label: str,
        y_limits: tuple[float, float],
        log_scale: bool = False,
    ) -> None:
        values = np.array(y_values, dtype=float)
        finite_mask = np.isfinite(values)
        low, high = y_limits
        low_mask = finite_mask & (values < low)
        high_mask = finite_mask & (values > high)
        in_range_mask = finite_mask & ~(low_mask | high_mask)

        clipped = values.copy()
        clipped[low_mask] = low
        clipped[high_mask] = high

        if np.any(finite_mask):
            ax.plot(
                x_values[finite_mask],
                clipped[finite_mask],
                color=color,
                linewidth=2.3,
                label=label,
            )
        if np.any(in_range_mask):
            ax.plot(
                x_values[in_range_mask],
                clipped[in_range_mask],
                linestyle="None",
                marker="o",
                color=color,
                markersize=6.0,
            )
        out_of_range_mask = low_mask | high_mask
        if np.any(out_of_range_mask):
            ax.plot(
                x_values[out_of_range_mask],
                clipped[out_of_range_mask],
                linestyle="None",
                marker="*",
                color=color,
                markersize=10.0,
            )

    fig, axes = plt.subplots(6, 1, figsize=(11, 20), sharex=True)
    for ax, (metric_prefix, title) in zip(axes[:4], metric_specs):
        plotted_uncertainties = (
            [("aleatoric", "aleatoric_var"), ("epistemic", "epistemic_var")]
            if metric_prefix == "magnitude"
            else UNCERTAINTY_SPECS
        )
        y_limits = {
            "spearman": (0.0, 1.0),
            "ause": (0.0, 0.5),
            "ence": (0.0, 2.0),
            "magnitude": (1e-6, 1e2),
        }[metric_prefix]
        for unc_name, _ in plotted_uncertainties:
            values = np.array([row[f"{metric_prefix}_{unc_name}"] for row in rows_plot], dtype=float)
            if metric_prefix == "magnitude":
                values = np.where(values > 0.0, values, np.nan)
            if fixed_scales:
                plot_series_with_clipped_markers(
                    ax,
                    epochs,
                    values,
                    color=style_map[unc_name]["color"],
                    label=style_map[unc_name]["label"],
                    y_limits=y_limits,
                    log_scale=(metric_prefix == "magnitude"),
                )
            else:
                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    color=style_map[unc_name]["color"],
                    label=style_map[unc_name]["label"],
                    linewidth=2.3,
                    markersize=6.0,
                )
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_ylabel(title, fontsize=label_fontsize)
        if fixed_scales:
            if metric_prefix == "spearman":
                ax.set_ylim(0.0, 1.0)
            elif metric_prefix == "ause":
                ax.set_ylim(0.0, 0.5)
            elif metric_prefix == "ence":
                ax.set_ylim(0.0, 2.0)
        if metric_prefix == "magnitude":
            ax.set_yscale("log", base=10)
            if fixed_scales:
                ax.set_ylim(1e-6, 1e2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=legend_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    rmse_values = np.array([row["rmse_e_atom"] for row in rows_plot], dtype=float)
    rmse_values = np.where(rmse_values > 0.0, rmse_values, np.nan)
    if fixed_scales:
        plot_series_with_clipped_markers(
            axes[4],
            epochs,
            rmse_values,
            color="black",
            label="RMSE",
            y_limits=(1e-6, 1e2),
            log_scale=True,
        )
    else:
        axes[4].plot(epochs, rmse_values, marker="o", color="black", label="RMSE", linewidth=2.3, markersize=6.0)
    axes[4].set_title("RMSE_E_per_atom ↓", fontsize=title_fontsize)
    axes[4].set_ylabel("RMSE", fontsize=label_fontsize)
    axes[4].set_yscale("log", base=10)
    if fixed_scales:
        axes[4].set_ylim(1e-6, 1e2)
    axes[4].grid(alpha=0.3)
    axes[4].legend(fontsize=legend_fontsize)
    axes[4].tick_params(axis="both", labelsize=tick_fontsize)

    nll_values = np.array([row["nll_energy"] for row in rows_plot], dtype=float)
    if fixed_scales:
        plot_series_with_clipped_markers(
            axes[5],
            epochs,
            nll_values,
            color="black",
            label="NLL",
            y_limits=(-0.5, 0.5),
        )
    else:
        axes[5].plot(epochs, nll_values, marker="o", color="black", label="NLL", linewidth=2.3, markersize=6.0)
    axes[5].set_title("Weighted Gaussian NLL Energy ↓", fontsize=title_fontsize)
    axes[5].set_ylabel("NLL", fontsize=label_fontsize)
    if fixed_scales:
        axes[5].set_ylim(-0.5, 0.5)
    axes[5].grid(alpha=0.3)
    axes[5].legend(fontsize=legend_fontsize)
    axes[5].tick_params(axis="both", labelsize=tick_fontsize)

    axes[-1].set_xlabel("Epoch", fontsize=label_fontsize)
    fig.suptitle(f"Uncertainty quality vs epoch{title_suffix}", fontsize=suptitle_fontsize)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def release_models(models: List[torch.nn.Module], device: torch.device) -> None:
    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    main_start = time.perf_counter()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)
    LOGGER.info("Using device=%s default_dtype=%s", device, args.default_dtype)
    def compute_rows() -> List[Dict[str, float]]:
        if not all([args.checkpoints_dir, args.split_path, args.energy_key]):
            raise RuntimeError("You must pass --checkpoints-dir, --split-path, and --energy-key.")

        checkpoint_map = discover_checkpoint_map(Path(args.checkpoints_dir), args.experiment_name)
        epochs = select_common_epochs(checkpoint_map, args.every_n_epochs)
        seeds = sorted(checkpoint_map)

        sample_epoch = epochs[0]
        sample_model = load_models([checkpoint_map[seeds[0]][sample_epoch]], device=device)[0]
        model_dtype = _get_model_dtype(sample_model)
        requested_dtype = torch.get_default_dtype()
        if model_dtype != requested_dtype:
            LOGGER.warning(
                "Requested default dtype %s does not match checkpoint dtype %s; using checkpoint dtype for dataset/inference.",
                requested_dtype,
                model_dtype,
            )
            torch.set_default_dtype(model_dtype)
        data_loader = build_dataloader(
            Path(args.split_path),
            args.energy_key,
            sample_model,
            batch_size=args.batch_size,
            head_name=args.head,
        )
        release_models([sample_model], device)

        rows: List[Dict[str, float]] = []
        for epoch in epochs:
            epoch_start = time.perf_counter()
            checkpoint_paths = [checkpoint_map[seed][epoch] for seed in seeds]
            LOGGER.info("Evaluating epoch %d with %d ensemble members", epoch, len(checkpoint_paths))
            models = load_models(checkpoint_paths, device=device)
            raw_rows = evaluate_split(
                models=models,
                data_loader=data_loader,
                device=device,
                per_atom=args.per_atom,
                head_name=args.head,
                split_name=f"epoch {epoch}",
            )
            raw_rows = trim_rows_by_key(raw_rows, args.trim, key="total_var")
            summary = summarize_epoch(raw_rows, num_bins=args.num_bins)
            summary["epoch"] = epoch
            rows.append(summary)
            LOGGER.info(
                "Epoch %d summary: pearson_total=%.4f spearman_total=%.4f ause_total=%.4f ence_total=%.4f rmse=%.6f nll=%.6f (computed in %.2fs)",
                epoch,
                summary["pearson_total"],
                summary["spearman_total"],
                summary["ause_total"],
                summary["ence_total"],
                summary["rmse_e_atom"],
                summary["nll_energy"],
                time.perf_counter() - epoch_start,
            )
            release_models(models, device)
        return rows

    metric_fields = [
        "pearson_aleatoric",
        "pearson_epistemic",
        "pearson_total",
        "spearman_aleatoric",
        "spearman_epistemic",
        "spearman_total",
        "ause_aleatoric",
        "ause_epistemic",
        "ause_total",
        "ence_aleatoric",
        "ence_epistemic",
        "ence_total",
        "magnitude_aleatoric",
        "magnitude_epistemic",
        "magnitude_total",
        "rmse_e_atom",
        "nll_energy",
    ]
    field_parsers = {"epoch": parse_int}
    for field in metric_fields:
        field_parsers[field] = parse_float
    cached_rows = load_cached_csv_rows(
        Path(args.output_csv),
        required_fields=["epoch", *metric_fields],
        field_parsers=field_parsers,
        logger=LOGGER,
        label="epoch_quality",
        key_fields=["epoch"],
        compute_missing_rows=compute_rows,
    )
    if cached_rows is not None:
        output_plot = Path(args.output_plot)
        write_plot(output_plot, cached_rows, drop_first_k_epochs=args.drop_first_k_epochs)
        if args.free_scale:
            write_plot(
                with_free_scale_suffix(output_plot),
                cached_rows,
                drop_first_k_epochs=args.drop_first_k_epochs,
                fixed_scales=False,
                title_suffix=" - free-scale",
            )
        LOGGER.info("Saved plot: %s", args.output_plot)
        LOGGER.info("Total runtime: %.2fs", time.perf_counter() - main_start)
        print(f"Saved CSV: {args.output_csv}")
        print(f"Saved plot: {args.output_plot}")
        return

    rows = compute_rows()

    write_csv(Path(args.output_csv), rows)
    output_plot = Path(args.output_plot)
    write_plot(output_plot, rows, drop_first_k_epochs=args.drop_first_k_epochs)
    if args.free_scale:
        write_plot(
            with_free_scale_suffix(output_plot),
            rows,
            drop_first_k_epochs=args.drop_first_k_epochs,
            fixed_scales=False,
            title_suffix=" - free-scale",
        )

    LOGGER.info("Saved CSV: %s", args.output_csv)
    LOGGER.info("Saved plot: %s", args.output_plot)
    LOGGER.info("Total runtime: %.2fs", time.perf_counter() - main_start)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
