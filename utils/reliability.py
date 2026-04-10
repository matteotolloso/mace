#!/usr/bin/env python3
"""Plot uncertainty-vs-error reliability diagrams from an ensemble.

Current workflow:
1. Read one ``<experiment>_run-<seed>_train.txt`` file per ensemble member from
   ``--results-dir``. These files contain batch logs (``mode == "opt"``) and
   validation epoch logs (``mode == "eval"``).
2. For each seed, select the best validation epoch independently using
   ``--selection-key`` and ``--selection-mode``. If ``--head`` is provided, only
   validation rows for that head are considered.
3. Enforce checkpoint completeness: every logged validation epoch for the chosen
   seed/head must have a matching checkpoint file
   ``<experiment>_run-<seed>_epoch-<epoch>.pt`` in ``--checkpoints-dir``. If
   any validation epoch is missing its checkpoint, the script raises an error
   and stops.
4. Load the selected checkpoint for each seed. Checkpoints can be saved either
   as full serialized modules or as state dictionaries paired with
   ``<experiment>_run-<seed>.model`` companion files.
5. Evaluate the resulting ensemble on the selection split and on the test split.
   The selection split is used for optional isotonic calibration; the test split
   is used for the final reliability diagram.
6. For each configuration, compute per-member energies and predicted variances.
   In the default per-atom mode, each member energy is divided by ``N`` and each
   member variance by ``N^2`` before ensemble aggregation. Then compute:
   - aleatoric variance = mean predicted variance across members
   - epistemic variance = variance of member predictions
   - total variance = aleatoric + epistemic
7. Optionally fit isotonic regressors on the selection split and apply them to
   the test split. Calibration is learned independently for aleatoric,
   epistemic, and total variance against squared error.
8. Sort test configurations independently by aleatoric, epistemic, and total
   uncertainty, bin them into equal-count bins, and plot RMSE vs RMV.

Metrics reported by the script:
- Ensemble ``rmse_e_per_atom`` on selection and test splits in default mode
  (or ``rmse_e`` in ``--total`` mode).
- Reliability diagram quantities are computed after sorting systems by
  uncertainty and binning them. For each bin, the script computes
  ``RMSE = sqrt(mean(squared error))`` and
  ``RMV = sqrt(mean(predicted variance))``, then plots ``RMSE`` vs ``RMV``.
- ENCE is computed from those binned reliability quantities only.
- Pearson correlation is computed on all systems, without binning, between
  per-system uncertainty variance (i.e. predicted sigma^2) and per-system squared error.
- Spearman correlation is computed on all systems, without binning, between
  per-system uncertainty variance (i.e. predicted sigma^2) and per-system squared error.
- AUSE (Area Under the Sparsification Error) is computed on all systems,
  without binning, from the ordering induced by each uncertainty variance
  against the oracle ordering induced by absolute error. The sparsification
  curves themselves are MAE-based, following the formal AUSE definition.

Outputs:
- ``--output-csv-raw`` stores per-configuration predictions and uncertainties for
  the plotted split.
- ``--output-csv-bins`` stores the binned reliability data, including ENCE
  terms.
- ``--output-plot`` stores the parity-style reliability diagram.

Notes:
- ``--input-csv`` bypasses checkpoint loading and rebuilds only the bins/plot
  from a previously saved raw CSV.
- Per-atom mode is the default. Use ``--total`` to switch to total-system
  energies and variances.
"""

import argparse
import csv
import json
import logging
import warnings
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
    module=r"e3nn\.o3\._wigner",
)

from mace import data
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools import torch_geometric, torch_tools, utils as mace_utils


CHECKPOINT_PATTERN = re.compile(
    r"^(?P<experiment>.+)_run-(?P<seed>\d+)_epoch-(?P<epoch>\d+)\.pt$"
)
RESULTS_PATTERN = re.compile(
    r"^(?P<experiment>.+)_run-(?P<seed>\d+)_train\.txt$"
)
ConfigKey = int
LOGGER = logging.getLogger(__name__)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in {"true", "1", "yes", "y"}:
        return True
    if value_lower in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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
        "--results-dir",
        type=str,
        required=False,
        help="Folder containing result logs named <experiment>_run-<seed>_train.txt used to select the best epoch independently for each seed.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        help="Experiment name prefix used to filter both result files in --results-dir and checkpoints in --checkpoints-dir.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=(
            "Optional raw per-config CSV produced by this script. If set, skip "
            "result parsing, checkpoint loading, and split evaluation, and rebuild "
            "only the bins/plot from the saved data."
        ),
    )
    parser.add_argument(
        "--validation-split",
        type=str,
        required=False,
        help="Selection/validation .xyz file used both for best-epoch selection diagnostics and optional isotonic calibration fitting.",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        required=False,
        help="Test .xyz file used for the final reliability diagram and reported ensemble test RMSE.",
    )
    parser.add_argument(
        "--energy-key-val",
        type=str,
        required=False,
        help="Energy key to read from the selection/validation .xyz file.",
    )
    parser.add_argument(
        "--energy-key-test",
        type=str,
        required=False,
        help="Energy key to read from the test .xyz file.",
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Optional model head to use for multi-head models. When multiple validation heads are present in the results file, this is also used to select the metric rows.",
    )
    parser.add_argument(
        "--selection-key",
        type=str,
        default="rmse_e_per_atom",
        help="Validation metric key used to select the best epoch independently for each seed from the *_train.txt files.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["auto", "min", "max"],
        default="auto",
        help="Whether to minimize or maximize --selection-key when selecting the best checkpoint per seed.",
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
        help="Torch default dtype.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of equal-count uncertainty bins used in the reliability diagram.",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system energy errors and variances instead of per-atom values.",
    )
    parser.add_argument(
        "--isotonic-calibration",
        type=str2bool,
        default=False,
        help=(
            "Whether to fit isotonic regressors on the validation split and apply them to "
            "the test split before binning. Calibration is fit against squared error. "
            "Use explicit values such as True or False."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Verbosity for timing and progress logging.",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=10,
        help="Emit an inference progress log every N batches for validation/test evaluation.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="unc_vs_error.png",
        help="Output reliability plot path.",
    )
    parser.add_argument(
        "--output-csv-raw",
        type=str,
        default="unc_vs_error_raw.csv",
        help="Output raw per-configuration CSV path for the plotted split.",
    )
    parser.add_argument(
        "--output-csv-bins",
        type=str,
        default="unc_vs_error_bins.csv",
        help="Output binned reliability CSV path.",
    )
    parser.set_defaults(per_atom=True)
    return parser.parse_args()


def discover_checkpoints(checkpoints_dir: Path, experiment_name: Optional[str]) -> List[Path]:
    start_time = time.perf_counter()
    model_paths: List[Tuple[int, Path]] = []
    for path in checkpoints_dir.glob("*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        if experiment_name is not None and match.group("experiment") != experiment_name:
            continue
        model_paths.append((int(match.group("seed")), path))

    if not model_paths:
        experiment_msg = (
            f" for experiment '{experiment_name}'" if experiment_name is not None else ""
        )
        raise RuntimeError(
            "No checkpoints matching '*_run-<seed>_epoch-<best_epoch>.pt' "
            f"were found in {checkpoints_dir}{experiment_msg}."
        )

    model_paths.sort(key=lambda item: item[0])
    checkpoint_list = [path for _, path in model_paths]
    LOGGER.info(
        "Discovered %d checkpoints in %.2fs from %s%s",
        len(checkpoint_list),
        time.perf_counter() - start_time,
        checkpoints_dir,
        f" (experiment={experiment_name})" if experiment_name is not None else "",
    )
    return checkpoint_list


def _resolve_selection_mode(selection_key: str, selection_mode: str) -> str:
    if selection_mode != "auto":
        return selection_mode
    key = selection_key.lower()
    minimize_tokens = ("loss", "rmse", "mae", "q95", "error", "var")
    maximize_tokens = ("acc", "accuracy", "auc", "pearson", "spearman", "r2")
    if any(token in key for token in maximize_tokens):
        return "max"
    if any(token in key for token in minimize_tokens):
        return "min"
    return "min"


def _read_jsonl_dicts(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse JSON on line {line_number} of {path}: {exc}") from exc
            if isinstance(record, dict):
                rows.append(record)
    return rows


def discover_result_files(results_dir: Path, experiment_name: Optional[str]) -> List[Tuple[int, Path, str]]:
    start_time = time.perf_counter()
    result_files: List[Tuple[int, Path, str]] = []
    experiments = set()
    for path in results_dir.glob("*_train.txt"):
        match = RESULTS_PATTERN.match(path.name)
        if match is None:
            continue
        experiment = match.group("experiment")
        if experiment_name is not None and experiment != experiment_name:
            continue
        seed = int(match.group("seed"))
        result_files.append((seed, path, experiment))
        experiments.add(experiment)

    if not result_files:
        experiment_msg = f" for experiment '{experiment_name}'" if experiment_name is not None else ""
        raise RuntimeError(
            f"No result files matching <experiment>_run-<seed>_train.txt were found in {results_dir}{experiment_msg}."
        )

    if experiment_name is None and len(experiments) > 1:
        raise RuntimeError(
            f"Multiple experiments were found in {results_dir}: {sorted(experiments)}. Pass --experiment-name to disambiguate."
        )

    result_files.sort(key=lambda item: item[0])
    LOGGER.info(
        "Discovered %d result files in %.2fs from %s%s",
        len(result_files),
        time.perf_counter() - start_time,
        results_dir,
        f" (experiment={experiment_name})" if experiment_name is not None else "",
    )
    return result_files


def select_best_checkpoints(
    checkpoints_dir: Path,
    results_dir: Path,
    experiment_name: Optional[str],
    selection_key: str,
    selection_mode: str,
    head_name: Optional[str],
) -> List[Path]:
    resolved_mode = _resolve_selection_mode(selection_key, selection_mode)
    result_files = discover_result_files(results_dir, experiment_name)
    selected_paths: List[Tuple[int, Path]] = []

    for seed, result_path, experiment in result_files:
        rows = _read_jsonl_dicts(result_path)
        eval_rows = [
            row
            for row in rows
            if row.get("mode") == "eval" and row.get("epoch") is not None
        ]
        if head_name is not None:
            eval_rows = [row for row in eval_rows if row.get("head") == head_name]
        else:
            heads = sorted({str(row.get("head")) for row in eval_rows if row.get("head") is not None})
            if len(heads) > 1:
                raise RuntimeError(
                    f"Result file {result_path} contains multiple validation heads {heads}. Pass --head to select one."
                )

        available_checkpoints = []
        for checkpoint_path in checkpoints_dir.glob(f"{experiment}_run-{seed}_epoch-*.pt"):
            match = CHECKPOINT_PATTERN.match(checkpoint_path.name)
            if match is None:
                continue
            available_checkpoints.append((int(match.group("epoch")), checkpoint_path))
        available_checkpoints.sort(key=lambda item: item[0])
        available_epochs = {epoch for epoch, _ in available_checkpoints}
        if not available_checkpoints:
            raise RuntimeError(
                f"No checkpoints were found for seed {seed} in {checkpoints_dir} with experiment prefix '{experiment}'."
            )

        logged_eval_epochs = sorted({int(row["epoch"]) for row in eval_rows})
        missing_epochs = [epoch for epoch in logged_eval_epochs if epoch not in available_epochs]
        if missing_epochs:
            raise RuntimeError(
                f"Result file {result_path} contains validation epochs without matching checkpoints for seed {seed}: {missing_epochs}. All logged eval epochs must have checkpoint files in {checkpoints_dir}."
            )

        metric_rows = []
        for row in eval_rows:
            value = row.get(selection_key)
            if value is None:
                continue
            try:
                metric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(metric_value):
                continue
            metric_rows.append((row, metric_value))

        if not metric_rows:
            available_keys = sorted(
                {key for row in eval_rows for key, value in row.items() if isinstance(value, (int, float))}
            )
            raise RuntimeError(
                f"No finite '{selection_key}' values were found in {result_path}. Available numeric keys: {available_keys}"
            )

        best_row, best_metric = (
            min(metric_rows, key=lambda item: item[1])
            if resolved_mode == "min"
            else max(metric_rows, key=lambda item: item[1])
        )
        best_epoch = int(best_row["epoch"])
        checkpoint_path = checkpoints_dir / f"{experiment}_run-{seed}_epoch-{best_epoch}.pt"
        selected_paths.append((seed, checkpoint_path))
        LOGGER.info(
            "Selected seed %d from %s: epoch=%d using %s=%s (%s)",
            seed,
            result_path.name,
            best_epoch,
            selection_key,
            best_metric,
            resolved_mode,
        )

    return [path for _, path in selected_paths]


def _get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.dtype
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.dtype
    return torch.get_default_dtype()


def _cast_batch_dict_dtype(batch_dict: Dict[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
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
        model_start = time.perf_counter()
        checkpoint_obj = torch.load(f=str(path), map_location=str(device), weights_only=False)
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
            model = torch.load(f=str(companion_model_path), map_location=str(device), weights_only=False)
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
        LOGGER.info(
            "Loaded model %d/%d from %s in %.2fs",
            model_idx,
            len(checkpoint_paths),
            path.name,
            time.perf_counter() - model_start,
        )

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
    read_start = time.perf_counter()
    atoms_list = ase.io.read(str(xyz_path), index=":")
    if len(atoms_list) == 0:
        raise RuntimeError(f"No configurations found in {xyz_path}.")

    LOGGER.info("Read %d configurations from %s in %.2fs", len(atoms_list), xyz_path, time.perf_counter() - read_start)

    keyspec = KeySpecification(info_keys={"energy": energy_key}, arrays_keys={})
    effective_head = head_name or (
        getattr(model, "heads", ["Default"])[0] if getattr(model, "heads", None) else "Default"
    )

    configs_start = time.perf_counter()
    configs = []
    for idx, atoms in enumerate(atoms_list):
        if energy_key not in atoms.info:
            raise KeyError(
                f"Energy key '{energy_key}' not found in configuration index {idx} of {xyz_path}."
            )
        configs.append(config_from_atoms(atoms, key_specification=keyspec, head_name=effective_head))

    LOGGER.info("Converted ASE atoms to configs in %.2fs", time.perf_counter() - configs_start)

    z_table = mace_utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", None)
    dataset_start = time.perf_counter()
    dataset = [
        data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(model.r_max),
            heads=heads,
        )
        for config in configs
    ]
    LOGGER.info("Built AtomicData dataset in %.2fs", time.perf_counter() - dataset_start)

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
    return loader, len(dataset)


def evaluate_split(
    models: List[torch.nn.Module],
    data_loader,
    device: torch.device,
    per_atom: bool,
    head_name: Optional[str],
    split_name: str,
    log_every_batches: int,
) -> List[Dict[str, float]]:
    raw_rows: List[Dict[str, float]] = []
    model_heads = getattr(models[0], "heads", None)
    model_dtype = _get_model_dtype(models[0])
    running_index = 0
    num_batches = len(data_loader)
    split_start = time.perf_counter()
    cumulative_model_time = 0.0
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
            batch_start = time.perf_counter()
            batch = batch.to(device)
            batch_dict = _cast_batch_dict_dtype(batch.to_dict(), model_dtype)
            batch_heads = batch["head"] if "head" in batch.keys else None
            num_graphs = int(batch.num_graphs)
            num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(device=device, dtype=model_dtype)

            member_preds = []
            member_vars = []
            for model in models:
                model_forward_start = time.perf_counter()
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
                cumulative_model_time += time.perf_counter() - model_forward_start
                member_preds.append(pred_energy)
                member_vars.append(pred_var)

            pred_stack = torch.stack(member_preds, dim=0)
            var_stack = torch.stack(member_vars, dim=0)
            ref_energy = batch.energy.to(device=device, dtype=model_dtype)

            if per_atom:
                pred_stack = pred_stack / num_atoms.unsqueeze(0)
                var_stack = var_stack / (num_atoms.unsqueeze(0) ** 2)
                ref_energy = ref_energy / num_atoms

            ensemble_pred = torch.mean(pred_stack, dim=0)
            aleatoric_var = torch.mean(var_stack, dim=0)
            epistemic_var = torch.var(pred_stack, dim=0, unbiased=False)
            total_var = aleatoric_var + epistemic_var
            sq_error = (ensemble_pred - ref_energy) ** 2

            for local_idx in range(num_graphs):
                raw_rows.append(
                    {
                        "config_index": running_index + local_idx,
                        "num_atoms": float(num_atoms[local_idx].detach().cpu().item()),
                        "ref_energy": float(ref_energy[local_idx].detach().cpu().item()),
                        "pred_energy": float(ensemble_pred[local_idx].detach().cpu().item()),
                        "error": float((ensemble_pred[local_idx] - ref_energy[local_idx]).detach().cpu().item()),
                        "sq_error": float(sq_error[local_idx].detach().cpu().item()),
                        "aleatoric_var": float(aleatoric_var[local_idx].detach().cpu().item()),
                        "epistemic_var": float(epistemic_var[local_idx].detach().cpu().item()),
                        "total_var": float(total_var[local_idx].detach().cpu().item()),
                        "aleatoric_var_raw": float(aleatoric_var[local_idx].detach().cpu().item()),
                        "epistemic_var_raw": float(epistemic_var[local_idx].detach().cpu().item()),
                        "total_var_raw": float(total_var[local_idx].detach().cpu().item()),
                    }
                )
            running_index += num_graphs

            batch_elapsed = time.perf_counter() - batch_start
            if batch_idx == 1 or batch_idx == num_batches or (log_every_batches > 0 and batch_idx % log_every_batches == 0):
                elapsed = time.perf_counter() - split_start
                LOGGER.info(
                    "%s progress: batch %d/%d, rows=%d, batch %.2fs, elapsed %.2fs, avg/batch %.2fs, avg model forward %.2fs",
                    split_name,
                    batch_idx,
                    num_batches,
                    len(raw_rows),
                    batch_elapsed,
                    elapsed,
                    elapsed / batch_idx,
                    cumulative_model_time / max(batch_idx * len(models), 1),
                )

    total_elapsed = time.perf_counter() - split_start
    LOGGER.info(
        "Finished %s evaluation in %.2fs for %d configs (%d batches); cumulative model forward time %.2fs",
        split_name,
        total_elapsed,
        len(raw_rows),
        num_batches,
        cumulative_model_time,
    )
    return raw_rows


def compute_energy_rmse(raw_rows: List[Dict[str, float]]) -> float:
    if not raw_rows:
        return float("nan")
    sq_errors = np.array([row["sq_error"] for row in raw_rows if np.isfinite(row["sq_error"])], dtype=float)
    if sq_errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(sq_errors)))


def fit_isotonic_calibrators(selection_rows: List[Dict[str, float]]) -> Dict[str, object]:
    start_time = time.perf_counter()
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:
        raise ImportError(
            "Isotonic calibration requires scikit-learn to be installed."
        ) from exc

    calibrators: Dict[str, object] = {}
    for unc_name, unc_key in (
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ):
        valid_rows = [
            row
            for row in selection_rows
            if np.isfinite(row[unc_key]) and np.isfinite(row["sq_error"])
        ]
        if len(valid_rows) < 2:
            continue

        x = np.array([row[unc_key] for row in valid_rows], dtype=float)
        y = np.array([row["sq_error"] for row in valid_rows], dtype=float)
        if np.unique(x).size < 2:
            continue

        regressor = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds="clip")
        regressor.fit(x, y)
        calibrators[unc_name] = regressor

    LOGGER.info(
        "Fit %d isotonic calibrators on %d validation rows in %.2fs",
        len(calibrators),
        len(selection_rows),
        time.perf_counter() - start_time,
    )
    return calibrators


def apply_isotonic_calibration(
    rows: List[Dict[str, float]],
    calibrators: Dict[str, object],
) -> List[Dict[str, float]]:
    start_time = time.perf_counter()
    calibrated_rows: List[Dict[str, float]] = []
    for row in rows:
        updated_row = dict(row)
        for unc_name, unc_key in (
            ("aleatoric", "aleatoric_var"),
            ("epistemic", "epistemic_var"),
            ("total", "total_var"),
        ):
            regressor = calibrators.get(unc_name)
            if regressor is not None and np.isfinite(row[unc_key]):
                calibrated_value = float(regressor.predict([row[unc_key]])[0])
                updated_row[unc_key] = max(calibrated_value, 0.0)
        calibrated_rows.append(updated_row)
    LOGGER.info(
        "Applied isotonic calibration to %d rows in %.2fs",
        len(rows),
        time.perf_counter() - start_time,
    )
    return calibrated_rows


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_raw_csv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed: Dict[str, float] = {
                "config_index": int(row["config_index"]),
                "num_atoms": float(row["num_atoms"]) if row["num_atoms"] else np.nan,
                "ref_energy": float(row["ref_energy"]),
                "pred_energy": float(row["pred_energy"]),
                "error": float(row["error"]),
                "sq_error": float(row["sq_error"]),
                "aleatoric_var": float(row["aleatoric_var"]) if row.get("aleatoric_var") else np.nan,
                "epistemic_var": float(row["epistemic_var"]) if row.get("epistemic_var") else np.nan,
                "total_var": float(row["total_var"]) if row.get("total_var") else np.nan,
                "aleatoric_var_raw": float(row["aleatoric_var_raw"]) if row.get("aleatoric_var_raw") else np.nan,
                "epistemic_var_raw": float(row["epistemic_var_raw"]) if row.get("epistemic_var_raw") else np.nan,
                "total_var_raw": float(row["total_var_raw"]) if row.get("total_var_raw") else np.nan,
            }
            rows.append(parsed)
    return rows


def build_binned_rows(raw_rows: List[Dict[str, float]], num_bins: int) -> List[Dict[str, float]]:
    start_time = time.perf_counter()
    if not raw_rows:
        raise RuntimeError("Cannot build reliability bins from empty raw rows.")

    binned_rows: List[Dict[str, float]] = []
    uncertainty_specs = [
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ]

    for unc_name, unc_key in uncertainty_specs:
        valid_rows = [row for row in raw_rows if np.isfinite(row[unc_key])]
        if not valid_rows:
            continue
        valid_rows = sorted(valid_rows, key=lambda row: row[unc_key])
        effective_bins = min(num_bins, len(valid_rows))
        for bin_idx, bin_rows in enumerate(
            np.array_split(np.array(valid_rows, dtype=object), effective_bins)
        ):
            bin_list = list(bin_rows)
            if not bin_list:
                continue
            sq_errors = np.array([row["sq_error"] for row in bin_list], dtype=float)
            variances = np.array([row[unc_key] for row in bin_list], dtype=float)
            mean_sq_error = float(np.mean(sq_errors))
            mean_variance = float(np.mean(variances))
            ence_term = (
                float(abs(mean_variance - mean_sq_error) / mean_variance)
                if mean_variance > 0.0
                else float("nan")
            )
            binned_rows.append(
                {
                    "uncertainty_type": unc_name,
                    "bin_index": bin_idx,
                    "count": len(bin_list),
                    "rmse": float(np.sqrt(mean_sq_error)),
                    "rmv": float(np.sqrt(mean_variance)),
                    "mse": mean_sq_error,
                    "mean_variance": mean_variance,
                    "ence_term": ence_term,
                    "uncertainty_min": float(np.min(variances)),
                    "uncertainty_max": float(np.max(variances)),
                }
            )
    LOGGER.info(
        "Built %d binned reliability rows from %d raw rows in %.2fs",
        len(binned_rows),
        len(raw_rows),
        time.perf_counter() - start_time,
    )
    return binned_rows


def compute_ence_summary(binned_rows: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for unc_name in ["aleatoric", "epistemic", "total"]:
        rows = [row for row in binned_rows if row["uncertainty_type"] == unc_name]
        finite_terms = np.array(
            [row["ence_term"] for row in rows if np.isfinite(row["ence_term"])],
            dtype=float,
        )
        summary[unc_name] = (
            float(np.mean(finite_terms)) if finite_terms.size > 0 else float("nan")
        )
    return summary


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


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


def compute_pearson_summary(raw_rows: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    sq_errors = np.array([row["sq_error"] for row in raw_rows], dtype=float)

    for unc_name, unc_key in (
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ):
        variances = np.array([row[unc_key] for row in raw_rows], dtype=float)
        valid = np.isfinite(variances) & np.isfinite(sq_errors)
        if np.count_nonzero(valid) < 2:
            summary[unc_name] = float("nan")
            continue
        x = variances[valid]
        y = sq_errors[valid]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            summary[unc_name] = float("nan")
            continue
        summary[unc_name] = float(np.corrcoef(x, y)[0, 1])
    return summary


def compute_spearman_summary(raw_rows: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    sq_errors = np.array([row["sq_error"] for row in raw_rows], dtype=float)

    for unc_name, unc_key in (
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ):
        variances = np.array([row[unc_key] for row in raw_rows], dtype=float)
        valid = np.isfinite(variances) & np.isfinite(sq_errors)
        if np.count_nonzero(valid) < 2:
            summary[unc_name] = float("nan")
            continue
        x = variances[valid]
        y = sq_errors[valid]
        x_rank = _rankdata(x)
        y_rank = _rankdata(y)
        if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
            summary[unc_name] = float("nan")
            continue
        summary[unc_name] = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return summary


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


def compute_ause_summary(raw_rows: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    abs_errors = np.array(
        [
            abs(row["error"]) if np.isfinite(row.get("error", np.nan)) else np.sqrt(max(row["sq_error"], 0.0))
            for row in raw_rows
        ],
        dtype=float,
    )

    for unc_name, unc_key in (
        ("aleatoric", "aleatoric_var"),
        ("epistemic", "epistemic_var"),
        ("total", "total_var"),
    ):
        variances = np.array([row[unc_key] for row in raw_rows], dtype=float)
        summary[unc_name] = compute_ause(variances, abs_errors)
    return summary


def write_plot(
    path: Path,
    binned_rows: List[Dict[str, float]],
    per_atom: bool,
    isotonic_calibration: bool,
    raw_rows: List[Dict[str, float]],
) -> None:
    start_time = time.perf_counter()
    if not binned_rows:
        raise RuntimeError("Cannot create reliability plot from empty binned rows.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 7))
    fig.subplots_adjust(right=0.66)

    style_map = {
        "aleatoric": {"marker": "o", "label": "Aleatoric"},
        "epistemic": {"marker": "s", "label": "Epistemic"},
        "total": {"marker": "^", "label": "Total"},
    }
    ence_summary = compute_ence_summary(binned_rows)
    pearson_summary = compute_pearson_summary(raw_rows)
    spearman_summary = compute_spearman_summary(raw_rows)
    ause_summary = compute_ause_summary(raw_rows)
    max_xy = 0.0
    for unc_name in ["aleatoric", "epistemic", "total"]:
        rows = [row for row in binned_rows if row["uncertainty_type"] == unc_name]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["bin_index"])
        rmv = np.array([row["rmv"] for row in rows], dtype=float)
        rmse = np.array([row["rmse"] for row in rows], dtype=float)
        max_xy = max(max_xy, float(np.max(rmv)), float(np.max(rmse)))
        ax.plot(
            rmv,
            rmse,
            marker=style_map[unc_name]["marker"],
            label=style_map[unc_name]["label"],
        )

    parity_max = max_xy * 1.05 if max_xy > 0 else 1.0
    ax.plot([0.0, parity_max], [0.0, parity_max], linestyle="--", color="black", label="Ideal")
    ax.set_xlim(0.0, parity_max)
    ax.set_ylim(0.0, parity_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    ax.set_title("Reliability Diagram (per-atom energies)" if per_atom else "Reliability Diagram")
    if isotonic_calibration:
        ax.set_title(ax.get_title() + " with Isotonic Calibration")
    ax.grid(alpha=0.3)
    ax.legend()

    display_names = {
        "aleatoric": "Aleatoric",
        "epistemic": "Epistemic",
        "total": "Total",
    }
    summary_blocks = []
    for unc_name in ["aleatoric", "epistemic", "total"]:
        block_lines = [display_names[unc_name]]
        if np.isfinite(ence_summary[unc_name]):
            block_lines.append(f"ENCE     {ence_summary[unc_name]:.4f}")
        if np.isfinite(pearson_summary[unc_name]):
            block_lines.append(f"Pearson  {pearson_summary[unc_name]:.4f}")
        if np.isfinite(spearman_summary[unc_name]):
            block_lines.append(f"Spearman {spearman_summary[unc_name]:.4f}")
        if np.isfinite(ause_summary[unc_name]):
            block_lines.append(f"AUSE     {ause_summary[unc_name]:.4f}")
        if len(block_lines) > 1:
            summary_blocks.append("\n".join(block_lines))
    if summary_blocks:
        fig.text(
            0.70,
            0.93,
            "\n\n".join(summary_blocks),
            va="top",
            ha="left",
            family="monospace",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "alpha": 0.9},
        )

    plt.savefig(path, dpi=200)
    plt.close()
    LOGGER.info("Saved reliability plot to %s in %.2fs", path, time.perf_counter() - start_time)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    main_start = time.perf_counter()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)
    LOGGER.info("Using device=%s default_dtype=%s", device, args.default_dtype)

    if args.input_csv is not None:
        raw_rows = read_raw_csv(Path(args.input_csv))
    else:
        if not all([
            args.checkpoints_dir,
            args.results_dir,
            args.validation_split,
            args.test_split,
            args.energy_key_val,
            args.energy_key_test,
        ]):
            raise RuntimeError(
                "When --input-csv is not provided, you must pass --checkpoints-dir, "
                "--results-dir, --validation-split, --test-split, --energy-key-val, and --energy-key-test."
            )

        checkpoint_paths = select_best_checkpoints(
            checkpoints_dir=Path(args.checkpoints_dir),
            results_dir=Path(args.results_dir),
            experiment_name=args.experiment_name,
            selection_key=args.selection_key,
            selection_mode=args.selection_mode,
            head_name=args.head,
        )
        models = load_models(checkpoint_paths, device=device)
        model_dtype = _get_model_dtype(models[0])
        requested_dtype = torch.get_default_dtype()
        if model_dtype != requested_dtype:
            LOGGER.warning(
                "Requested default dtype %s does not match checkpoint dtype %s; using checkpoint dtype for dataset/inference.",
                requested_dtype,
                model_dtype,
            )
            torch.set_default_dtype(model_dtype)
        LOGGER.info("Ensemble parameter dtype: %s", model_dtype)
        val_loader, _ = build_dataloader(
            Path(args.validation_split),
            args.energy_key_val,
            models[0],
            batch_size=args.batch_size,
            head_name=args.head,
        )
        test_loader, _ = build_dataloader(
            Path(args.test_split),
            args.energy_key_test,
            models[0],
            batch_size=args.batch_size,
            head_name=args.head,
        )

        val_rows = evaluate_split(
            models=models,
            data_loader=val_loader,
            device=device,
            per_atom=args.per_atom,
            head_name=args.head,
            split_name="validation",
            log_every_batches=args.log_every_batches,
        )
        raw_rows = evaluate_split(
            models=models,
            data_loader=test_loader,
            device=device,
            per_atom=args.per_atom,
            head_name=args.head,
            split_name="test",
            log_every_batches=args.log_every_batches,
        )

        selection_rmse = compute_energy_rmse(val_rows)
        test_rmse = compute_energy_rmse(raw_rows)
        rmse_label = "rmse_e_per_atom" if args.per_atom else "rmse_e"
        LOGGER.info("Ensemble %s on selection split: %.6f", rmse_label, selection_rmse)
        LOGGER.info("Ensemble %s on test split: %.6f", rmse_label, test_rmse)
        print(f"Selection {rmse_label}: {selection_rmse:.6f}")
        print(f"Test {rmse_label}: {test_rmse:.6f}")

        if args.isotonic_calibration:
            calibrators = fit_isotonic_calibrators(val_rows)
            raw_rows = apply_isotonic_calibration(raw_rows, calibrators)

        raw_fieldnames = [
            "config_index",
            "num_atoms",
            "ref_energy",
            "pred_energy",
            "error",
            "sq_error",
            "aleatoric_var",
            "epistemic_var",
            "total_var",
            "aleatoric_var_raw",
            "epistemic_var_raw",
            "total_var_raw",
        ]
        write_csv(Path(args.output_csv_raw), raw_rows, raw_fieldnames)

    binned_rows = build_binned_rows(raw_rows, num_bins=args.num_bins)
    bins_fieldnames = [
        "uncertainty_type",
        "bin_index",
        "count",
        "rmse",
        "rmv",
        "mse",
        "mean_variance",
        "ence_term",
        "uncertainty_min",
        "uncertainty_max",
    ]
    write_csv(Path(args.output_csv_bins), binned_rows, bins_fieldnames)
    write_plot(
        Path(args.output_plot),
        binned_rows,
        per_atom=args.per_atom,
        isotonic_calibration=args.isotonic_calibration,
        raw_rows=raw_rows,
    )

    LOGGER.info("Total runtime: %.2fs", time.perf_counter() - main_start)
    print(f"Saved raw CSV: {args.output_csv_raw}")
    print(f"Saved binned CSV: {args.output_csv_bins}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
