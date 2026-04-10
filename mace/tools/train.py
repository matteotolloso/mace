###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
    sanitize_for_json,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )


def _wandb_metric_key(split: str, metric: str, head_name: str) -> str:
    if head_name == "Default":
        return f"{split}_{metric}"
    return f"{split}_{head_name}_{metric}"


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
    epoch_train_loaders: Optional[Dict[str, DataLoader]] = None,
    epoch_test_loaders: Optional[Dict[str, DataLoader]] = None,
    log_epoch_outputs: bool = False,
    epoch_logger: Optional[MetricsLogger] = None,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    if log_epoch_outputs and distributed and rank == 0:
        logging.warning(
            "Per-configuration epoch outputs are disabled in distributed mode; only split summaries will be logged."
        )
    epoch = start_epoch

    # log validation loss before _any_ training
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None
    # this is a bugfix of the original code. This allows to use the patient stopping mechanism even when not using distributed training. Before, if distributed was False, exit_now would be None and the check `if exit_now is not None` would fail, meaning that the loop would never break based on patience. 
    exit_now_single = False if not distributed else None
    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        train_epoch_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_dict = {"epoch": epoch}
                if log_wandb and rank == 0:
                    train_head_name = (
                        next(iter(valid_loaders.keys()))
                        if len(valid_loaders) == 1
                        else "Default"
                    )
                    if train_epoch_metrics.get("loss") is not None:
                        wandb_log_dict[
                            _wandb_metric_key("train", "loss", train_head_name)
                        ] = train_epoch_metrics["loss"]
                    if train_epoch_metrics.get("rmse_e_per_atom") is not None:
                        wandb_log_dict[
                            _wandb_metric_key(
                                "train", "rmse_e_per_atom", train_head_name
                            )
                        ] = train_epoch_metrics["rmse_e_per_atom"]
                    if train_epoch_metrics.get("rmse_f") is not None:
                        wandb_log_dict[
                            _wandb_metric_key("train", "rmse_f", train_head_name)
                        ] = train_epoch_metrics["rmse_f"]
                    if train_epoch_metrics.get("var_e_per_atom_2") is not None:
                        wandb_log_dict[
                            _wandb_metric_key(
                                "train",
                                "var_e_per_atom_2",
                                train_head_name,
                            )
                        ] = train_epoch_metrics["var_e_per_atom_2"]
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            wandb_log_dict[
                                _wandb_metric_key("valid", "loss", valid_loader_name)
                            ] = valid_loss_head
                            if eval_metrics.get("rmse_e_per_atom") is not None:
                                wandb_log_dict[
                                    _wandb_metric_key(
                                        "valid", "rmse_e_per_atom", valid_loader_name
                                    )
                                ] = eval_metrics["rmse_e_per_atom"]
                            if eval_metrics.get("rmse_f") is not None:
                                wandb_log_dict[
                                    _wandb_metric_key("valid", "rmse_f", valid_loader_name)
                                ] = eval_metrics["rmse_f"]
                            if eval_metrics.get("var_e_per_atom_2") is not None:
                                wandb_log_dict[
                                    _wandb_metric_key(
                                        "valid",
                                        "var_e_per_atom_2",
                                        valid_loader_name,
                                    )
                                ] = eval_metrics["var_e_per_atom_2"]
                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")
                valid_loss = (
                    valid_loss_head  # consider only the last head for the checkpoint
                )
            if log_wandb:
                wandb.log(sanitize_for_json(wandb_log_dict))
            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement"
                            )
                            if exit_now is not None:
                                exit_now.fill_(1)
                            else: # not distributed case
                                exit_now_single = True
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints

        if log_epoch_outputs:
            model_to_evaluate = model if distributed_model is None else distributed_model
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                if epoch_train_loaders:
                    for train_loader_name, train_loader_eval in epoch_train_loaders.items():
                        train_loss_head, train_metrics = evaluate(
                            model=model_to_evaluate,
                            loss_fn=loss_fn,
                            data_loader=train_loader_eval,
                            output_args=output_args,
                            device=device,
                        )
                        train_metrics["mode"] = "epoch_outputs"
                        train_metrics["split"] = "train"
                        train_metrics["loader"] = train_loader_name
                        train_metrics["epoch"] = epoch
                        train_metrics["head"] = train_loader_name
                        train_metrics["split_loss"] = train_loss_head
                        if rank == 0 and epoch_logger is not None:
                            epoch_logger.log(train_metrics)
                        if not distributed and rank == 0 and epoch_logger is not None:
                            for row in collect_config_predictions(
                                model=model_to_evaluate,
                                data_loader=train_loader_eval,
                                output_args=output_args,
                                device=device,
                                split="train",
                                loader_name=train_loader_name,
                                epoch=epoch,
                                rank=rank,
                            ):
                                epoch_logger.log(row)

                for valid_loader_name, valid_loader_eval in valid_loaders.items():
                    valid_loss_head_ep, valid_metrics_ep = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader_eval,
                        output_args=output_args,
                        device=device,
                    )
                    valid_metrics_ep["mode"] = "epoch_outputs"
                    valid_metrics_ep["split"] = "valid"
                    valid_metrics_ep["loader"] = valid_loader_name
                    valid_metrics_ep["epoch"] = epoch
                    valid_metrics_ep["head"] = valid_loader_name
                    valid_metrics_ep["split_loss"] = valid_loss_head_ep
                    if rank == 0 and epoch_logger is not None:
                        epoch_logger.log(valid_metrics_ep)
                    if not distributed and rank == 0 and epoch_logger is not None:
                        for row in collect_config_predictions(
                            model=model_to_evaluate,
                            data_loader=valid_loader_eval,
                            output_args=output_args,
                            device=device,
                            split="valid",
                            loader_name=valid_loader_name,
                            epoch=epoch,
                            rank=rank,
                        ):
                            epoch_logger.log(row)

                if epoch_test_loaders:
                    for test_loader_name, test_loader in epoch_test_loaders.items():
                        test_loss_head, test_metrics = evaluate(
                            model=model_to_evaluate,
                            loss_fn=loss_fn,
                            data_loader=test_loader,
                            output_args=output_args,
                            device=device,
                        )
                        test_metrics["mode"] = "epoch_outputs"
                        test_metrics["split"] = "test"
                        test_metrics["loader"] = test_loader_name
                        test_metrics["epoch"] = epoch
                        test_metrics["head"] = test_loader_name
                        test_metrics["split_loss"] = test_loss_head
                        if rank == 0 and epoch_logger is not None:
                            epoch_logger.log(test_metrics)
                        if not distributed and rank == 0 and epoch_logger is not None:
                            for row in collect_config_predictions(
                                model=model_to_evaluate,
                                data_loader=test_loader,
                                output_args=output_args,
                                device=device,
                                split="test",
                                loader_name=test_loader_name,
                                epoch=epoch,
                                rank=rank,
                            ):
                                epoch_logger.log(row)
        if distributed:
            torch.distributed.barrier()
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            if exit_now == 1:
                break
        else: # not distributed case
            if exit_now_single:
                break

        epoch += 1

    logging.info("Training complete")


def collect_config_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
    split: str,
    loader_name: str,
    epoch: int,
    rank: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    running_index = 0

    def _batch_field(b, key: str):
        return getattr(b, key) if key in b.keys else None

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )

            ptr = batch.ptr
            num_graphs = int(batch.num_graphs)
            heads_batch = _batch_field(batch, "head")

            pred_energy = output.get("energy")
            if pred_energy is not None and pred_energy.ndim >= 2 and heads_batch is not None:
                graph_idx = torch.arange(num_graphs, device=pred_energy.device)
                pred_energy = pred_energy[graph_idx, heads_batch.long()]

            pred_energy_var = output.get("energy_var", None)
            if (
                pred_energy_var is not None
                and pred_energy_var.ndim >= 2
                and heads_batch is not None
            ):
                graph_idx = torch.arange(num_graphs, device=pred_energy_var.device)
                pred_energy_var = pred_energy_var[graph_idx, heads_batch.long()]

            for i in range(num_graphs):
                start = int(ptr[i].item())
                end = int(ptr[i + 1].item())
                n_atoms = end - start

                row: Dict[str, Any] = {
                    "mode": "epoch_outputs_config",
                    "split": split,
                    "loader": loader_name,
                    "epoch": epoch,
                    "rank": rank,
                    "config_index": running_index + i,
                    "num_atoms": n_atoms,
                }

                if heads_batch is not None:
                    row["head_index"] = int(heads_batch[i].item())

                if pred_energy is not None:
                    row["pred_energy"] = float(pred_energy[i].detach().cpu().item())
                if pred_energy_var is not None:
                    pred_energy_var_i = float(pred_energy_var[i].detach().cpu().item())
                    row["var_e_per_atom_2"] = pred_energy_var_i / (max(float(n_atoms), 1.0) ** 2)
                if _batch_field(batch, "energy") is not None:
                    ref_e = float(batch.energy[i].detach().cpu().item())
                    row["ref_energy"] = ref_e
                    if pred_energy is not None:
                        row["delta_energy"] = float(
                            pred_energy[i].detach().cpu().item() - ref_e
                        )
                        if n_atoms > 0:
                            row["delta_energy_per_atom"] = row["delta_energy"] / n_atoms
                            # for a single configuration, RMSE is the absolute error
                            row["rmse_e_per_atom"] = abs(row["delta_energy_per_atom"])
                            row["rmse_e_per_atom_meV"] = 1e3 * row["rmse_e_per_atom"]

                if output.get("forces", None) is not None and _batch_field(batch, "forces") is not None:
                    f_ref = batch.forces[start:end]
                    f_pred = output["forces"][start:end]
                    row["mae_f"] = float(torch.mean(torch.abs(f_ref - f_pred)).detach().cpu().item())
                    row["rmse_f"] = float(
                        torch.sqrt(torch.mean((f_ref - f_pred) ** 2)).detach().cpu().item()
                    )

                rows.append(row)

            running_index += num_graphs

    return rows


def _batch_field(batch, key: str):
    return getattr(batch, key) if key in batch.keys else None


def _num_atoms_per_graph(
    batch: torch_geometric.batch.Batch, like: torch.Tensor
) -> torch.Tensor:
    return (batch.ptr[1:] - batch.ptr[:-1]).to(device=like.device, dtype=like.dtype)


def _select_active_head_values(
    values: Optional[torch.Tensor], batch: torch_geometric.batch.Batch
) -> Optional[torch.Tensor]:
    if values is None:
        return None
    heads_batch = _batch_field(batch, "head")
    if values.ndim >= 2 and heads_batch is not None:
        graph_idx = torch.arange(values.shape[0], device=values.device)
        values = values[graph_idx, heads_batch.long()]
    return values


def _variance_e_per_atom_2(
    values: Optional[torch.Tensor], batch: torch_geometric.batch.Batch
) -> Optional[torch.Tensor]:
    values = _select_active_head_values(values, batch)
    if values is None:
        return None
    values = torch.clamp(values.detach(), min=0.0)
    num_atoms = torch.clamp(_num_atoms_per_graph(batch, values), min=1.0)
    return values / (num_atoms**2)


def _summarize_pred_energy_uncertainty(
    values: Optional[torch.Tensor],
    batch: torch_geometric.batch.Batch,
) -> Dict[str, float]:
    values = _variance_e_per_atom_2(values, batch)
    if values is None or values.numel() == 0:
        return {}
    return {
        "var_e_per_atom_2": float(values.mean().cpu().item()),
    }


def _init_train_metric_totals(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "loss_sum": torch.tensor(0.0, device=device),
        "loss_count": torch.tensor(0.0, device=device),
        "energy_sq_sum": torch.tensor(0.0, device=device),
        "energy_count": torch.tensor(0.0, device=device),
        "force_sq_sum": torch.tensor(0.0, device=device),
        "force_count": torch.tensor(0.0, device=device),
        "var_sum": torch.tensor(0.0, device=device),
        "var_count": torch.tensor(0.0, device=device),
    }


def _accumulate_train_metric_totals(
    totals: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    output: Dict[str, torch.Tensor],
    batch: torch_geometric.batch.Batch,
) -> None:
    num_graphs = float(batch.num_graphs)
    totals["loss_sum"] += loss.detach() * num_graphs
    totals["loss_count"] += num_graphs

    pred_energy = _select_active_head_values(output.get("energy"), batch)
    if pred_energy is not None and _batch_field(batch, "energy") is not None:
        num_atoms = torch.clamp(_num_atoms_per_graph(batch, pred_energy), min=1.0)
        delta_e_per_atom = (batch.energy - pred_energy.detach()) / num_atoms
        totals["energy_sq_sum"] += torch.sum(delta_e_per_atom**2)
        totals["energy_count"] += float(delta_e_per_atom.numel())

    pred_forces = output.get("forces")
    if pred_forces is not None and _batch_field(batch, "forces") is not None:
        delta_f = batch.forces - pred_forces.detach()
        totals["force_sq_sum"] += torch.sum(delta_f**2)
        totals["force_count"] += float(delta_f.numel())

    pred_var = _variance_e_per_atom_2(output.get("energy_var"), batch)
    if pred_var is not None and pred_var.numel() > 0:
        totals["var_sum"] += pred_var.sum()
        totals["var_count"] += float(pred_var.numel())


def _finalize_train_metric_totals(
    totals: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if totals["loss_count"].item() > 0:
        metrics["loss"] = float((totals["loss_sum"] / totals["loss_count"]).cpu().item())
    if totals["energy_count"].item() > 0:
        metrics["rmse_e_per_atom"] = float(
            torch.sqrt(totals["energy_sq_sum"] / totals["energy_count"]).cpu().item()
        )
    if totals["force_count"].item() > 0:
        metrics["rmse_f"] = float(
            torch.sqrt(totals["force_sq_sum"] / totals["force_count"]).cpu().item()
        )
    if totals["var_count"].item() > 0:
        metrics["var_e_per_atom_2"] = float(
            (totals["var_sum"] / totals["var_count"]).cpu().item()
        )
    return metrics


def _merge_train_metric_totals(
    totals: Dict[str, torch.Tensor],
    batch_totals: Dict[str, torch.Tensor],
) -> None:
    for key, value in batch_totals.items():
        totals[key] += value


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
) -> Dict[str, float]:
    model_to_train = model if distributed_model is None else distributed_model
    epoch_metric_totals = _init_train_metric_totals(device)

    if isinstance(optimizer, LBFGS):
        _, opt_metrics, step_metric_totals = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
        )
        _merge_train_metric_totals(epoch_metric_totals, step_metric_totals)
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        for batch in data_loader:
            _, opt_metrics, step_metric_totals = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            _merge_train_metric_totals(epoch_metric_totals, step_metric_totals)
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)

    if distributed:
        for key in epoch_metric_totals:
            torch.distributed.all_reduce(
                epoch_metric_totals[key], op=torch.distributed.ReduceOp.SUM
            )

    return _finalize_train_metric_totals(epoch_metric_totals)


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any], Dict[str, torch.Tensor]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()
    uncertainty_metrics: Dict[str, float] = {}
    batch_metric_totals = _init_train_metric_totals(device)

    def closure():
        nonlocal uncertainty_metrics
        nonlocal batch_metric_totals
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        uncertainty_metrics = _summarize_pred_energy_uncertainty(
            output.get("energy_var"), batch
        )
        batch_metric_totals = _init_train_metric_totals(device)
        _accumulate_train_metric_totals(batch_metric_totals, loss, output, batch)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }
    loss_dict.update(uncertainty_metrics)

    return loss, loss_dict, batch_metric_totals


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tuple[float, Dict[str, Any], Dict[str, torch.Tensor]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None
    uncertainty_metrics: Dict[str, float] = {}
    epoch_metric_totals = _init_train_metric_totals(device)

    def closure():
        nonlocal uncertainty_metrics
        nonlocal epoch_metric_totals
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        total_pred_var_per_atom_2 = torch.tensor(0.0, device=device)
        total_pred_count = torch.tensor(0.0, device=device)
        epoch_metric_totals = _init_train_metric_totals(device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss_raw = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss_raw * (batch.num_graphs / total_sample_count)
            _accumulate_train_metric_totals(
                epoch_metric_totals, batch_loss_raw, output, batch
            )
            pred_energy_var = _variance_e_per_atom_2(
                output.get("energy_var"), batch
            )
            if pred_energy_var is not None and pred_energy_var.numel() > 0:
                total_pred_var_per_atom_2 += pred_energy_var.sum()
                total_pred_count += float(pred_energy_var.numel())

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                total_pred_var_per_atom_2, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(total_pred_count, op=torch.distributed.ReduceOp.SUM)
            for key in epoch_metric_totals:
                torch.distributed.all_reduce(
                    epoch_metric_totals[key], op=torch.distributed.ReduceOp.SUM
                )
        uncertainty_metrics = {}
        if total_pred_count.item() > 0:
            uncertainty_metrics = {
                "var_e_per_atom_2": float(
                    (total_pred_var_per_atom_2 / total_pred_count)
                    .detach()
                    .cpu()
                    .item()
                ),
            }
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }
    loss_dict.update(uncertainty_metrics)

    return loss, loss_dict, epoch_metric_totals


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    for param in model.parameters():
        param.requires_grad = False

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        avg_loss, aux = metrics(batch, output)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        ### MVE ###
        self.add_state("var_e_per_atom_2", default=[], dist_reduce_fx="cat")
        ### /MVE ###
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        ### MVE ###
        pred_energy = output.get("energy")

        if pred_energy is not None and batch.energy is not None:
            # graph-level delta and per-atom delta
            self.delta_es.append(batch.energy - pred_energy)
            self.delta_es_per_atom.append(
                (batch.energy - pred_energy) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
        # collect predicted variance (if model provides it) for logging/calibration
        pred_var = _variance_e_per_atom_2(output.get("energy_var"), batch)

        if pred_var is not None:
            self.var_e_per_atom_2.append(pred_var)
        ### /MVE ###

        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        
        ### MVE ###
        # compute a simple summary stat for predicted variance
        if self.var_e_per_atom_2:
            pred_vars = self.convert(self.var_e_per_atom_2)
            try:
                mean_pred_var = float(np.mean(pred_vars))
            except Exception:
                mean_pred_var = None
            aux["var_e_per_atom_2"] = mean_pred_var
        ### /MVE ###

        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        return aux["loss"], aux
