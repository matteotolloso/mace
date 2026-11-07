"""LF-only ("pretrain-only") ensemble scored on the HF labels of its LF->HF child (rebuttal WP2, G7/G4).

    python -m uq.evaluation.lf_on_hf --experiment ani1x-energy/mace-mve/lf-to-hf --split 0
    uq evaluate D --gpu 0 --splits 0 --steps lfonly_baseline      # the same, as a pipeline step

The members of the *parent* LF-only experiment, each at its own best LF validation checkpoint (the
ensemble that its evaluation reports), predict the child's HF files (``train``, ``val``, ``id``,
``ood`` roles, e.g. ``cc_*.xyz`` with the CC energy key). DFT and CC total energies differ by large
per-element offsets, so each member gets its atomic reference energies refit, as LF->HF fine-tuning
does before its first step: the per-element shift ``d_m`` minimising
``sum_i (E_HF,i - E_m,i - n_i . d_m)^2`` over the HF *training* set (``n_i`` = element counts).
Nothing else changes: the backbone, the mean readout and the variance head are the LF model's, and
the variance does not depend on the reference energies. This is the most favourable zero-shot use
of the LF ensemble (the shift is fitted, per member, to exactly the residuals it is scored on).

Outputs, in the child's ``evaluation/split_<s>/`` (new names; nothing existing is touched), in the
schema of ``reliability`` so that ``uq aggregate`` and the support filter treat them alike:

  lfonly_reliability_<id|ood>_<nocal|cal>_{raw,bins}.csv, .svg   per-atom rows, 15 equal-count bins
                                                                 (``cal``: isotonic on HF ``val``)
  lfonly_e0_refit.json                                           checkpoints, shifts, HF-train RMSE
                                                                 before/after the refit per member
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from uq import paths, registry
from uq.evaluation.reliability import (
    _cast_batch_dict_dtype,
    _get_model_dtype,
    _select_head_tensor,
    apply_isotonic_calibration,
    build_binned_rows,
    build_dataloader,
    fit_isotonic_calibrators,
    load_models,
    select_best_checkpoints,
    trim_rows_by_key,
    write_csv,
    write_plot,
)

LOGGER = logging.getLogger("lf_on_hf")
PREFIX = "lfonly_reliability"
RAW_FIELDS = ["config_index", "num_atoms", "ref_energy", "pred_energy", "error", "sq_error",
              "aleatoric_var", "epistemic_var", "total_var",
              "aleatoric_var_raw", "epistemic_var_raw", "total_var_raw"]
BIN_FIELDS = ["uncertainty_type", "bin_index", "count", "rmse", "rmv", "mse", "mean_variance", "ence_term",
              "uncertainty_min", "uncertainty_max"]


def predict(models, loader, device):
    """[M, N] total energies and variances, [N] reference energies and atom counts, [N, Z] element counts."""
    dtype = _get_model_dtype(models[0])
    model_heads = getattr(models[0], "heads", None)
    energies, variances, refs, atoms, counts = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_dict = _cast_batch_dict_dtype(batch.to_dict(), dtype)
            heads = batch["head"] if "head" in batch.keys else None
            preds, vars_ = [], []
            for model in models:
                out = model(batch_dict, training=False, compute_force=False, compute_virials=False,
                            compute_stress=False, compute_hessian=False, compute_edge_forces=False)
                energy = _select_head_tensor(out.get("energy_mean", out.get("energy")), heads, None, model_heads)
                var = out.get("energy_var")
                var = torch.zeros_like(energy) if var is None else _select_head_tensor(var, heads, None, model_heads)
                preds.append(energy)
                vars_.append(var)
            energies.append(torch.stack(preds).double().cpu().numpy())
            variances.append(torch.stack(vars_).double().cpu().numpy())
            refs.append(batch.energy.double().cpu().numpy())
            atoms.append((batch.ptr[1:] - batch.ptr[:-1]).cpu().numpy())
            onehot = batch.node_attrs.double().cpu().numpy()
            graph_counts = np.zeros((int(batch.num_graphs), onehot.shape[1]))
            np.add.at(graph_counts, batch.batch.cpu().numpy(), onehot)
            counts.append(graph_counts)
    return (np.concatenate(energies, axis=1), np.concatenate(variances, axis=1), np.concatenate(refs),
            np.concatenate(atoms).astype(float), np.concatenate(counts))


def fit_shifts(pred_energy, ref_energy, counts):
    """Per-member per-element shifts [M, Z] by least squares on the residuals ref - pred."""
    return np.stack([np.linalg.lstsq(counts, ref_energy - member, rcond=None)[0] for member in pred_energy])


def rows_from(pred_energy, pred_var, ref_energy, num_atoms):
    """Per-atom reliability rows of an ensemble ([M, N] energies and variances), as reliability writes them."""
    per_atom = pred_energy / num_atoms
    var = pred_var / num_atoms ** 2
    ref = ref_energy / num_atoms
    mean = per_atom.mean(axis=0)
    au, eu = var.mean(axis=0), per_atom.var(axis=0)   # EU: population variance (unbiased=False)
    rows = []
    for i in range(per_atom.shape[1]):
        error = float(mean[i] - ref[i])
        rows.append({"config_index": i, "num_atoms": float(num_atoms[i]), "ref_energy": float(ref[i]),
                     "pred_energy": float(mean[i]), "error": error, "sq_error": error ** 2,
                     "aleatoric_var": float(au[i]), "epistemic_var": float(eu[i]), "total_var": float(au[i] + eu[i]),
                     "aleatoric_var_raw": float(au[i]), "epistemic_var_raw": float(eu[i]),
                     "total_var_raw": float(au[i] + eu[i])})
    return rows


def run(experiment, split, device="cuda", batch_size=256, num_bins=15, tests=("id", "ood")) -> Path:
    child = registry.experiment(experiment)
    parent = child.parent
    if parent is None:
        raise SystemExit(f"{child.label} has no LF parent")
    start = time.perf_counter()
    checkpoints = select_best_checkpoints(paths.checkpoints_dir(parent.id, split), paths.results_dir(parent.id, split),
                                          parent.run_name, "loss", "min", None)
    LOGGER.info("%s split %d: parent checkpoints %s", child.label, split, [p.name for p in checkpoints])
    models = load_models(checkpoints, torch.device(device))
    torch.set_default_dtype(_get_model_dtype(models[0]))
    key = child.fidelity.energy_key
    predictions = {}
    for role in ("train", "val", *tests):
        loader, count = build_dataloader(child.data_file(role, split), key, models[0], batch_size, None)
        predictions[role] = predict(models, loader, torch.device(device))
        if predictions[role][0].shape != (len(models), count):
            raise RuntimeError(f"{role}: unexpected prediction shape {predictions[role][0].shape}")
    energy, _, ref, _, counts = predictions["train"]
    shifts = fit_shifts(energy, ref, counts)
    before = np.sqrt(((energy - ref) ** 2).mean(axis=1))
    after = np.sqrt(((energy + shifts @ counts.T - ref) ** 2).mean(axis=1))

    def corrected(role):
        energy, var, ref, atoms, counts = predictions[role]
        return rows_from(energy + shifts @ counts.T, var, ref, atoms)

    out = paths.split_evaluation_dir(child.id, split)
    out.mkdir(parents=True, exist_ok=True)
    calibrators = fit_isotonic_calibrators(corrected("val"))
    for role in tests:
        nocal = corrected(role)
        for tag, rows in (("nocal", nocal), ("cal", apply_isotonic_calibration(nocal, calibrators))):
            stem = f"{PREFIX}_{role}_{tag}"
            write_csv(out / f"{stem}_raw.csv", rows, RAW_FIELDS)
            bins = build_binned_rows(trim_rows_by_key(rows, 0.0, key="total_var"), num_bins=num_bins)
            write_csv(out / f"{stem}_bins.csv", bins, BIN_FIELDS)
            write_plot(out / f"{stem}.svg", bins, per_atom=True, isotonic_calibration=tag == "cal", raw_rows=rows)
    record = {
        "experiment": child.id, "parent": parent.id, "split": split, "energy_key": key,
        "selection": "each parent member's lowest LF validation loss (as the parent's evaluation)",
        "checkpoints": [p.name for p in checkpoints],
        "atomic_numbers": [int(z) for z in models[0].atomic_numbers],
        "shift_ev_per_element": shifts.tolist(),
        "hf_train_rmse_ev_before_refit": before.tolist(), "hf_train_rmse_ev_after_refit": after.tolist(),
        "hf_train_configs": int(counts.shape[0]),
    }
    (out / "lfonly_e0_refit.json").write_text(json.dumps(record, indent=1) + "\n")
    LOGGER.info("wrote %s/%s_* in %.1fs", paths.rel(out), PREFIX, time.perf_counter() - start)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, help="an LF->HF experiment (its parent is scored)")
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args.experiment, args.split, device=args.device, batch_size=args.batch_size, num_bins=args.num_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
