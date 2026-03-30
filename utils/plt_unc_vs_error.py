#!/usr/bin/env python3
"""Plot uncertainty-vs-error reliability diagrams from ensemble checkpoints.

Workflow:
1. Discover ensemble checkpoints in ``--checkpoints-dir`` matching the pattern
   ``<experiment_name>_run-<seed>_epoch-<best_epoch>.pt``.
2. Load the ensemble directly from those checkpoints.
3. Evaluate the validation and test ``.xyz`` splits using the requested energy
   keys.
4. On each configuration, compute:
   - aleatoric variance = mean predicted variance across ensemble members
   - epistemic variance = variance of ensemble member predictions
   - total variance = aleatoric + epistemic
5. Optionally fit isotonic regressors on the validation split and apply them to
   the test split.
6. Sort test systems independently by aleatoric, epistemic, and total
   uncertainty, bin them into equal-count bins, and plot RMSE vs RMV.

Reported summary metrics:
- ENCE is computed from the bins only.
- Pearson correlation is computed on all systems, without binning.

Outputs:
- ``--output-csv-raw`` stores per-configuration data for the plotted split.
- ``--output-csv-bins`` stores the binned reliability data, including ENCE
  terms.
"""

import argparse
import csv
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn

from mace import data
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools import torch_geometric, torch_tools, utils as mace_utils


CHECKPOINT_PATTERN = re.compile(
    r"^(?P<experiment>.+)_run-(?P<seed>\d+)_epoch-(?P<epoch>\d+)\.pt$"
)
ConfigKey = int
LOGGER = logging.getLogger(__name__)


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
        help="Folder containing ensemble checkpoints named *_run-<seed>_epoch-<best_epoch>.pt.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        help="Experiment name prefix used to filter checkpoints inside --checkpoints-dir.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=(
            "Optional raw per-config CSV produced by this script. If set, skip "
            "checkpoint inference and rebuild bins/plot from the saved data."
        ),
    )
    parser.add_argument(
        "--validation-split",
        type=str,
        required=False,
        help="Validation .xyz file used for isotonic calibration fitting.",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        required=False,
        help="Test .xyz file used for the final reliability diagram.",
    )
    parser.add_argument(
        "--energy-key-val",
        type=str,
        required=False,
        help="Energy key to read from the validation .xyz file.",
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
        "--per-atom",
        dest="per_atom",
        action="store_true",
        help="Use per-atom energy errors and variances: E/N and var/N^2 (default).",
    )
    parser.add_argument(
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system energy errors and variances instead of per-atom values.",
    )
    parser.add_argument(
        "--isotonic-calibration",
        action="store_true",
        help=(
            "Fit isotonic regressors on the validation split and apply them to "
            "the test split before binning. Calibration is fit against squared error."
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
        checkpoint_obj = torch.load(f=str(path), map_location=str(device))
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
            model = torch.load(f=str(companion_model_path), map_location=str(device))
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
    fig, ax = plt.subplots(figsize=(7, 7))

    style_map = {
        "aleatoric": {"marker": "o", "label": "Aleatoric"},
        "epistemic": {"marker": "s", "label": "Epistemic"},
        "total": {"marker": "^", "label": "Total"},
    }
    ence_summary = compute_ence_summary(binned_rows)
    pearson_summary = compute_pearson_summary(raw_rows)
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

    summary_lines = []
    for unc_name in ["aleatoric", "epistemic", "total"]:
        if np.isfinite(ence_summary[unc_name]):
            summary_lines.append(
                f"{unc_name.capitalize()} ENCE (binned) = {ence_summary[unc_name]:.4f}"
            )
        if np.isfinite(pearson_summary[unc_name]):
            summary_lines.append(
                f"{unc_name.capitalize()} Pearson (all systems) = {pearson_summary[unc_name]:.4f}"
            )
    if summary_lines:
        ax.text(
            0.04,
            0.96,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    plt.tight_layout()
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
            args.validation_split,
            args.test_split,
            args.energy_key_val,
            args.energy_key_test,
        ]):
            raise RuntimeError(
                "When --input-csv is not provided, you must pass --checkpoints-dir, "
                "--validation-split, --test-split, --energy-key-val, and --energy-key-test."
            )

        checkpoint_paths = discover_checkpoints(Path(args.checkpoints_dir), args.experiment_name)
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
