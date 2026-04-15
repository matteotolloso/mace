#!/usr/bin/env python3
"""Plot epoch-wise AU/EU/TU and energy RMSE from ensemble checkpoints.

This script no longer depends on ``*_epoch_outputs.txt`` files.
Instead, it evaluates ensemble checkpoints directly on a chosen split.

Workflow:
1. Discover checkpoints in ``--checkpoints-dir`` matching
   ``<experiment>_run-<seed>_epoch-<epoch>.pt``.
2. Group checkpoints by seed and epoch.
3. Keep only epochs that are present for every ensemble member.
4. Evaluate the ensemble every ``--every-n-epochs`` epochs on the requested
   ``.xyz`` split.
5. At each evaluated epoch compute:
   - aleatoric variance = mean predicted variance across members
   - epistemic variance = variance of member predictions
   - total variance = aleatoric + epistemic
   - energy RMSE = ensemble RMSE on the same split

Per-atom mode is the default:
- energies are normalized as ``E / N``
- variances are normalized as ``var / N^2``

Use ``--total`` to switch to total-system quantities.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from reliability import trim_rows_by_key


CHECKPOINT_PATTERN = re.compile(
    r"^(?P<experiment>.+)_run-(?P<seed>\d+)_epoch-(?P<epoch>\d+)\.pt$"
)
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
        "--total",
        dest="per_atom",
        action="store_false",
        help="Use total-system energies and variances instead of per-atom values.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="epoch_au_eu.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="epoch_au_eu.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--drop-first-k-epochs",
        type=int,
        default=0,
        help="Exclude the first k evaluated epochs from the plot only.",
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
        model_start = time.perf_counter()
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
        LOGGER.debug(
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
    split_start = time.perf_counter()
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
                        "sq_error": float(sq_error[local_idx].detach().cpu().item()),
                        "aleatoric_var": float(aleatoric_var[local_idx].detach().cpu().item()),
                        "epistemic_var": float(epistemic_var[local_idx].detach().cpu().item()),
                        "total_var": float(total_var[local_idx].detach().cpu().item()),
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

    LOGGER.info(
        "Finished %s evaluation in %.2fs for %d configs",
        split_name,
        time.perf_counter() - split_start,
        len(raw_rows),
    )
    return raw_rows


def summarize_epoch(raw_rows: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    if not raw_rows:
        return float("nan"), float("nan"), float("nan"), float("nan")
    au = np.array([row["aleatoric_var"] for row in raw_rows], dtype=float)
    eu = np.array([row["epistemic_var"] for row in raw_rows], dtype=float)
    tu = np.array([row["total_var"] for row in raw_rows], dtype=float)
    sq_err = np.array([row["sq_error"] for row in raw_rows], dtype=float)
    return (
        float(np.nanmean(au)),
        float(np.nanmean(eu)),
        float(np.nanmean(tu)),
        float(np.sqrt(np.nanmean(sq_err))),
    )


def write_csv(path: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "au_var", "eu_var", "tu_var", "rmse_energy"])
        writer.writerows(rows)


def read_csv(path: Path) -> Optional[List[Tuple[int, float, float, float, float]]]:
    cached_rows = load_cached_csv_rows(
        path,
        required_fields=["epoch", "au_var", "eu_var", "tu_var", "rmse_energy"],
        field_parsers={
            "epoch": parse_int,
            "au_var": parse_float,
            "eu_var": parse_float,
            "tu_var": parse_float,
            "rmse_energy": parse_float,
        },
        logger=LOGGER,
        label="epoch_raw",
        key_fields=["epoch"],
    )
    if cached_rows is None:
        return None
    return [
        (
            int(row["epoch"]),
            float(row["au_var"]),
            float(row["eu_var"]),
            float(row["tu_var"]),
            float(row["rmse_energy"]),
        )
        for row in cached_rows
    ]


def write_plot(
    path: Path,
    rows: List[Tuple[int, float, float, float, float]],
    drop_first_k_epochs: int,
    per_atom: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_plot = rows[drop_first_k_epochs:] if drop_first_k_epochs > 0 else rows
    if not rows_plot:
        raise RuntimeError("No epochs left to plot after applying --drop-first-k-epochs.")

    epochs = [row[0] for row in rows_plot]
    ale = np.array([row[1] for row in rows_plot], dtype=float)
    epi = np.array([row[2] for row in rows_plot], dtype=float)
    total = np.array([row[3] for row in rows_plot], dtype=float)
    rmse_energy = np.array([row[4] for row in rows_plot], dtype=float)

    if not (np.any(np.isfinite(ale)) and np.any(np.isfinite(epi)) and np.any(np.isfinite(total))):
        raise RuntimeError("Cannot plot: uncertainties contain no finite values.")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, ale, marker="o", label="Aleatoric")
    ax1.plot(epochs, epi, marker="o", label="Epistemic")
    ax1.plot(epochs, total, marker="o", label="Total")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Uncertainty per atom^2" if per_atom else "Uncertainty")
    ax1.set_title(
        "Aleatoric/Epistemic/Total Uncertainty per Atom vs Epoch"
        if per_atom
        else "Aleatoric/Epistemic/Total Uncertainty vs Epoch"
    )
    ax1.grid(alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    rmse_display = rmse_energy * 1e3 if per_atom else rmse_energy
    finite_rmse = np.isfinite(rmse_display)
    if np.any(finite_rmse):
        ax2 = ax1.twinx()
        (rmse_line,) = ax2.plot(
            np.array(epochs)[finite_rmse],
            rmse_display[finite_rmse],
            color="black",
            linestyle="--",
            marker="s",
            label="RMSE_E_per_atom (meV)" if per_atom else "RMSE_E (eV)",
        )
        ax2.set_ylabel("RMSE_E_per_atom (meV)" if per_atom else "RMSE_E (eV)")
        handles.append(rmse_line)
        labels.append("RMSE_E_per_atom (meV)" if per_atom else "RMSE_E (eV)")

    ax1.legend(handles, labels)
    fig.tight_layout()
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

    def compute_rows() -> List[Tuple[int, float, float, float, float]]:
        if not all([args.checkpoints_dir, args.split_path, args.energy_key]):
            raise RuntimeError(
                "You must pass --checkpoints-dir, --split-path, and --energy-key."
            )

        checkpoint_map = discover_checkpoint_map(
            Path(args.checkpoints_dir), args.experiment_name
        )
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
        LOGGER.info("Model parameter dtype: %s", model_dtype)
        data_loader = build_dataloader(
            Path(args.split_path),
            args.energy_key,
            sample_model,
            batch_size=args.batch_size,
            head_name=args.head,
        )
        release_models([sample_model], device)

        rows: List[Tuple[int, float, float, float, float]] = []
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
            au, eu, tu, rmse_energy = summarize_epoch(raw_rows)
            rows.append((epoch, au, eu, tu, rmse_energy))
            LOGGER.info(
                "Epoch %d summary: AU=%.6e EU=%.6e TU=%.6e %s=%.6e (computed in %.2fs)",
                epoch,
                au,
                eu,
                tu,
                "rmse_e_per_atom" if args.per_atom else "rmse_e",
                rmse_energy,
                time.perf_counter() - epoch_start,
            )
            release_models(models, device)
        return rows

    cached_rows = load_cached_csv_rows(
        Path(args.output_csv),
        required_fields=["epoch", "au_var", "eu_var", "tu_var", "rmse_energy"],
        field_parsers={
            "epoch": parse_int,
            "au_var": parse_float,
            "eu_var": parse_float,
            "tu_var": parse_float,
            "rmse_energy": parse_float,
        },
        logger=LOGGER,
        label="epoch_raw",
        key_fields=["epoch"],
        compute_missing_rows=lambda: [
            {
                "epoch": epoch,
                "au_var": au,
                "eu_var": eu,
                "tu_var": tu,
                "rmse_energy": rmse_energy,
            }
            for epoch, au, eu, tu, rmse_energy in compute_rows()
        ],
    )
    if cached_rows is not None:
        rows = [
            (
                int(row["epoch"]),
                float(row["au_var"]),
                float(row["eu_var"]),
                float(row["tu_var"]),
                float(row["rmse_energy"]),
            )
            for row in cached_rows
        ]
        write_plot(
            Path(args.output_plot),
            rows,
            drop_first_k_epochs=args.drop_first_k_epochs,
            per_atom=args.per_atom,
        )
        LOGGER.info("Saved plot: %s", args.output_plot)
        LOGGER.info("Total runtime: %.2fs", time.perf_counter() - main_start)
        print(f"Saved CSV: {args.output_csv}")
        print(f"Saved plot: {args.output_plot}")
        return

    rows = compute_rows()

    write_csv(Path(args.output_csv), rows)
    write_plot(
        Path(args.output_plot),
        rows,
        drop_first_k_epochs=args.drop_first_k_epochs,
        per_atom=args.per_atom,
    )

    LOGGER.info("Saved CSV: %s", args.output_csv)
    LOGGER.info("Saved plot: %s", args.output_plot)
    LOGGER.info("Total runtime: %.2fs", time.perf_counter() - main_start)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
