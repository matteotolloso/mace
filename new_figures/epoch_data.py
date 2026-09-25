"""Per-epoch uncertainty metrics on the OOD test sets, for Figs. 3 and 5.

System split: the training-support filter removes no configuration there, so the
existing per-epoch caches are exact and are read directly:
  HF-only             experiment_E/.../epoch_quality_ood.csv
  LF->HF, HF stage    experiment_B/.../epoch_quality_ood.csv
  LF->HF, all epochs  experiment_B/.../epoch_quality_finetune_mixed_ood.csv
Energy split: the caches are dominated by ~13 extrapolation configurations per
split, so metrics are recomputed from per-configuration predictions
(_cache/epochs/{C,D,F}_<split>.npz, from run_epoch_predictions.py) restricted to
the training-support region.

Every series is keyed by epoch and holds, per signal, Spearman, AUSE and ENCE,
the mean AU/EU/TU magnitudes, the ensemble RMSE (eV/atom) and the mean Gaussian
NLL of the ensemble (TU variance), as in eval/epoch_quality.py. Epochs are
sampled every 5.
"""

from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np

from common import (
    EPOCHS, ROOT, SIGNALS, SIGNAL_NAME, SPLITS, T95_DF4, supported,
    build_binned_rows, compute_ause, compute_ence_summary, compute_spearman_summary,
)

CACHE = lambda e, s, name: ROOT / f"experiment_{e}" / "evaluation" / "cache" / f"split_{s}" / name  # noqa: E731


def _from_csv(path, phase=None, offset=0):
    series = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if phase is not None and row["phase"] != phase:
                continue
            epoch = int(row["source_epoch"] if phase else row["epoch"]) + offset
            entry = {"mag_AU": float(row["magnitude_aleatoric"]), "mag_EU": float(row["magnitude_epistemic"]),
                     "mag_TU": float(row["magnitude_total"]), "RMSE": float(row["rmse_e_atom"]),
                     "NLL": float(row["nll_energy"])}
            for s in SIGNALS:
                entry[("Spearman", s)] = float(row[f"spearman_{SIGNAL_NAME[s]}"])
                entry[("AUSE", s)] = float(row[f"ause_{SIGNAL_NAME[s]}"])
                entry[("ENCE", s)] = float(row[f"ence_{SIGNAL_NAME[s]}"])
            series[epoch] = entry
    return series


@lru_cache(None)
def _from_npz(experiment, split, offset=0):
    path = EPOCHS / f"{experiment}_{split}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    keep = supported("energy", split, "cc_test_ood")
    series = {}
    for i, epoch in enumerate(data["epochs"]):
        sq = data["sq_error"][i][keep]
        au = data["aleatoric_var"][i][keep]
        eu = data["epistemic_var"][i][keep]
        tu = au + eu
        rows = [{"sq_error": a, "aleatoric_var": b, "epistemic_var": c, "total_var": d}
                for a, b, c, d in zip(sq, au, eu, tu)]
        ence = compute_ence_summary(build_binned_rows(rows, num_bins=15))
        spearman = compute_spearman_summary(rows)
        safe = tu + 1e-16  # as eval/epoch_quality.py
        entry = {"mag_AU": float(au.mean()), "mag_EU": float(eu.mean()), "mag_TU": float(tu.mean()),
                 "RMSE": float(np.sqrt(sq.mean())),
                 "NLL": float(np.mean(0.5 * (sq / safe + np.log(safe) + np.log(2 * np.pi))))}
        for s, scores in (("AU", au), ("EU", eu), ("TU", tu)):
            entry[("Spearman", s)] = spearman[SIGNAL_NAME[s]]
            entry[("AUSE", s)] = compute_ause(scores, np.sqrt(sq))
            entry[("ENCE", s)] = ence[SIGNAL_NAME[s]]
        series[int(epoch) + offset] = entry
    return series


def hf_stage(kind, protocol, split):
    """HF-training epochs from 0: HF-only from scratch, or LF->HF fine-tuning."""
    if kind == "system":
        experiment = "E" if protocol == "HF-only" else "B"
        return _from_csv(CACHE(experiment, split, "epoch_quality_ood.csv"))
    return _from_npz("F" if protocol == "HF-only" else "D", split)


def lf_to_hf(kind, split, pretrain_epochs=300):
    """LF->HF over pretraining then fine-tuning; fine-tuning epochs follow pretraining."""
    if kind == "system":
        path = CACHE("B", split, "epoch_quality_finetune_mixed_ood.csv")
        return {**_from_csv(path, "pretrain"), **_from_csv(path, "finetune", pretrain_epochs)}
    pre, fine = _from_npz("C", split), _from_npz("D", split, pretrain_epochs)
    if pre is None or fine is None:
        return None
    return {**pre, **fine}


def _mixed_cached(kind, split, test, pretrain_epochs=300):
    path = CACHE("B" if kind == "system" else "D", split, f"epoch_quality_finetune_mixed_{test}.csv")
    return {**_from_csv(path, "pretrain"), **_from_csv(path, "finetune", pretrain_epochs)}


def lf_to_hf_train(kind, split):
    """LF->HF on its own training data: DFT train set while pretraining, CC train set
    while fine-tuning. Training configurations are inside the support by definition,
    so the cached curves need no filtering on either split."""
    return _mixed_cached(kind, split, "train")


def lf_to_hf_id(kind, split):
    """LF->HF on the ID test set: dft_test_id while pretraining, cc_test_id while
    fine-tuning, read from the cached (unfiltered) curves. The support filter would
    remove at most 8 of 50000 DFT and 2 of 5000 CC ID configurations per split; on
    the final models that changes AUSE by <= 0.0005 and ENCE by <= 15% (EU, system)."""
    return _mixed_cached(kind, split, "id")


def hf_only(kind, split, test):
    """HF-only (E/F) on train, ID or OOD. Energy-OOD is support-filtered (npz)."""
    experiment = "E" if kind == "system" else "F"
    if kind == "energy" and test == "ood":
        return _from_npz(experiment, split)
    return _from_csv(CACHE(experiment, split, f"epoch_quality_{test}.csv"))


def lf_to_hf_any(kind, split, test):
    """LF->HF over pretraining + fine-tuning on train, ID or OOD (Energy-OOD filtered)."""
    if test == "ood":
        return lf_to_hf(kind, split)
    return _mixed_cached(kind, split, test)


def across_splits(fetch, key, log=False):
    """Epochs present in all five splits, and the per-split values (5 x E)."""
    series = [fetch(s) for s in SPLITS]
    if any(s is None for s in series):
        return None, None
    epochs = sorted(set.intersection(*(set(s) for s in series)))
    values = np.array([[key(s[e]) for e in epochs] for s in series], dtype=float)
    return np.array(epochs), (np.log10(values) if log else values)


def band(values):
    """Per-epoch five-split mean and 95% Student-t half-width (values: 5 x E)."""
    if values.shape[0] != len(SPLITS):
        raise ValueError(f"expected {len(SPLITS)} splits, got {values.shape[0]}")
    return values.mean(axis=0), T95_DF4 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def ema(values, alpha=0.35):
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out
