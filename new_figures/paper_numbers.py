#!/usr/bin/env python3
"""Print every number quoted in the ICLR27 results (Figs. 2-6) and the appendix checks.

Reads the same caches and helpers as fig2..fig6 and writes nothing: redirect stdout
to keep a copy (results/tables/iclr27_main_text_numbers.txt). CPU only.

Statistics follow the figures: five-split arithmetic means for Spearman and AUSE,
geometric means with log-space intervals for ENCE, RMSE and EU/AU ratios; all
intervals are two-sided 95% Student-t over the five dataset splits (df=4).
Per-epoch values are the raw five-split means at the named epoch (not the EMA
drawn as the bold line).
"""

import json
import math

import numpy as np

import fig5_decomposition as f5
from common import EXPERIMENT, PROTOCOLS, ROOT, SIGNALS, SPLITS, TESTS, ci, filtered_rows, final_metrics
from epoch_data import CACHE, _from_csv, across_splits, band, hf_stage, lf_to_hf, lf_to_hf_id, lf_to_hf_train
from fig6_acquisition import selected_quantiles


def geo(values):
    m, h = ci([math.log(v) for v in values])
    return f"{math.exp(m):.3f} [{math.exp(m - h):.3f}, {math.exp(m + h):.3f}]"


def lin(values):
    m, h = ci(values)
    return f"{m:.3f} +/- {h:.3f}"


def final_models():
    for calibrated in (False, True):
        print(f"\n## Final models, {'isotonic-calibrated' if calibrated else 'uncalibrated'} (Figs. 2, 4)")
        for kind, test, label in TESTS:
            for p in PROTOCOLS:
                ps = final_metrics(EXPERIMENT[(kind, p)], test, calibrated)
                print(f"{label:10s} {p:7s} RMSE meV/atom geo {geo([m['RMSE'] for m in ps])}")
                for s in SIGNALS:
                    print(f"{'':19s}{s} Spearman {lin([m[('Spearman', s)] for m in ps])}"
                          f" | AUSE {lin([m[('AUSE', s)] for m in ps])}"
                          f" | ENCE geo {geo([m[('ENCE', s)] for m in ps])}")


def outlier():
    print("\n## Largest HF-only System-ID error per split (experiment E)")
    for s in SPLITS:
        rows = filtered_rows("E", s, "id")
        sq = np.array([r["sq_error"] for r in rows])
        tu = np.array([r["total_var"] for r in rows])
        i = int(sq.argmax())
        print(f"split {s}: n={len(rows)} max |error| {math.sqrt(sq[i]):.3f} eV/atom, "
              f"TU rank {int((tu > tu[i]).sum()) + 1}, share of split MSE {sq[i] / sq.sum():.3f}")


def epoch_line(tag, fetch, key, log, picks):
    epochs, values = across_splits(fetch, key, log=log)
    mean, half = band(values)
    back = (lambda v: 10 ** v) if log else (lambda v: v)
    cells = []
    for pick in picks:
        if pick == "first":
            i = 0
        elif pick == "last":
            i = len(epochs) - 1
        elif pick == "min":
            i = int(np.argmin(mean))
        elif pick == "max":
            i = int(np.argmax(mean))
        else:
            i = int(np.where(epochs == pick)[0][0])
        cells.append(f"e{epochs[i]}: {back(mean[i]):.3f} [{back(mean[i] - half[i]):.3f}, "
                     f"{back(mean[i] + half[i]):.3f}]")
    print(f"{tag:34s} " + " | ".join(cells))


def dynamics():
    print("\n## HF-stage dynamics, AU/EU/TU on the OOD test sets (Fig. 3)")
    for kind in ("system", "energy"):
        for p in PROTOCOLS:
            for sig in SIGNALS:
                for metric, log in (("AUSE", False), ("ENCE", True)):
                    epoch_line(f"{kind} {p} {sig} {metric}", lambda s: hf_stage(kind, p, s),
                               lambda e: e[(metric, sig)], log, ["first", 5, "min", "max", "last"])
    print("\n## HF-only on its own CC training set (cached, unfiltered; train is in support)")
    for kind, experiment in (("system", "E"), ("energy", "F")):
        for sig in SIGNALS:
            for metric, log in (("AUSE", False), ("ENCE", True)):
                epoch_line(f"train {kind} HF-only {sig} {metric}",
                           lambda s: _from_csv(CACHE(experiment, s, "epoch_quality_train.csv")),
                           lambda e: e[(metric, sig)], log, ["first", "min", "last"])
    print("\n## LF->HF over pretraining (0-295) and fine-tuning (300-395) (Fig. 5a-g)")
    for name, fetch in (("train", lf_to_hf_train), ("id", lf_to_hf_id), ("ood", lf_to_hf)):
        for kind in ("system", "energy"):
            for s in SIGNALS:
                for metric, log in (("AUSE", False), ("ENCE", True)):
                    epoch_line(f"{name} {kind} {s} {metric}", lambda sp: fetch(kind, sp),
                               lambda e: e[(metric, s)], log, ["first", 295, 300, "last"])
    for kind in ("system", "energy"):
        epoch_line(f"ood {kind} mean EU / mean AU", lambda sp: lf_to_hf(kind, sp),
                   lambda e: e["mag_EU"] / e["mag_AU"], True, ["first", "min", "max", 295, 300, "last"])
        epochs, values = across_splits(lambda sp: lf_to_hf(kind, sp),
                                       lambda e: e["mag_EU"] / e["mag_AU"], log=True)
        print(f"{'':34s} epochs with five-split geometric mean >= 1: "
              f"{[int(e) for e, m in zip(epochs, band(values)[0]) if m >= 0]}")


def _scale_csv(path, phase=None, offset=0):
    """Per-epoch mean AU, mean EU and MSE (per atom) from a cached epoch-quality CSV."""
    import csv
    out = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if phase is not None and row["phase"] != phase:
                continue
            epoch = int(row["source_epoch"] if phase else row["epoch"]) + offset
            out[epoch] = {"AU": float(row["magnitude_aleatoric"]), "EU": float(row["magnitude_epistemic"]),
                          "MSE": float(row["rmse_e_atom"]) ** 2}
    return out


def _scale_npz(experiment, split, offset=0):
    """As _scale_csv, from per-configuration Energy-OOD predictions with the support filter."""
    from common import supported
    from epoch_data import EPOCHS
    data = np.load(EPOCHS / f"{experiment}_{split}.npz")
    keep = supported("energy", split, "cc_test_ood")
    return {int(e) + offset: {"AU": float(data["aleatoric_var"][i][keep].mean()),
                              "EU": float(data["epistemic_var"][i][keep].mean()),
                              "MSE": float(data["sq_error"][i][keep].mean())}
            for i, e in enumerate(data["epochs"])}


def error_scale():
    """How far the observed squared error exceeds the mean predicted variance, per test set.

    MSE / mean AU and MSE / mean TU (geometric five-split means). A value near one means
    the variance has the size of the error on average; the ratio is not ENCE (no binning).
    """
    print("\n## Squared error over mean predicted variance (MSE/AU, MSE/TU), per-atom")
    mixed = lambda e, s, t: CACHE(e, s, f"epoch_quality_finetune_mixed_{t}.csv")  # noqa: E731
    sources = {
        ("system", "LF->HF"): {t: (lambda s, t=t: {**_scale_csv(mixed("B", s, t), "pretrain"),
                                                  **_scale_csv(mixed("B", s, t), "finetune", 300)})
                               for t in ("train", "id", "ood")},
        ("energy", "LF->HF"): {**{t: (lambda s, t=t: {**_scale_csv(mixed("D", s, t), "pretrain"),
                                                     **_scale_csv(mixed("D", s, t), "finetune", 300)})
                                  for t in ("train", "id")},
                               "ood": lambda s: {**_scale_npz("C", s), **_scale_npz("D", s, 300)}},
        ("system", "HF-only"): {t: (lambda s, t=t: _scale_csv(CACHE("E", s, f"epoch_quality_{t}.csv")))
                                for t in ("train", "id", "ood")},
        ("energy", "HF-only"): {**{t: (lambda s, t=t: _scale_csv(CACHE("F", s, f"epoch_quality_{t}.csv")))
                                   for t in ("train", "id")},
                                "ood": lambda s: _scale_npz("F", s)},
    }
    for (kind, protocol), tests in sources.items():
        picks = [0, 295] if protocol == "HF-only" else [0, 295, 300, 395]
        for test, fetch in tests.items():
            series = [fetch(s) for s in SPLITS]
            cells = []
            for e in picks:
                au = [x[e]["AU"] for x in series]
                tu = [x[e]["AU"] + x[e]["EU"] for x in series]
                mse = [x[e]["MSE"] for x in series]
                cells.append(f"e{e}: MSE/AU {geo([m / a for m, a in zip(mse, au)])} "
                             f"MSE/TU {geo([m / t for m, t in zip(mse, tu)])} "
                             f"AU {geo(au)} MSE {geo(mse)}")
            print(f"{kind} {protocol} {test}\n    " + "\n    ".join(cells))


def _bin_ratios(sq, var, bins=15):
    """MSE_b / MV_b in equal-count bins of var (ascending), as in reliability.build_binned_rows."""
    order = np.argsort(var, kind="stable")
    return np.array([sq[i].mean() / var[i].mean() for i in np.array_split(order, bins)])


def au_bins():
    """Where the overconfidence sits: per-bin MSE/MV for the lowest, middle and highest
    variance bins (geometric five-split means). Values >1 are overconfident bins."""
    print("\n## Per-bin MSE/MV (bin 1 = lowest predicted variance, 8 = middle, 15 = highest)")
    cases = [("Energy-OOD LF->HF fine-tune e0", lambda s: _npz_epoch("D", s, 0)),
             ("Energy-OOD LF->HF fine-tune e95", lambda s: _npz_epoch("D", s, 95)),
             ("Energy-OOD HF-only e0", lambda s: _npz_epoch("F", s, 0)),
             ("Energy-OOD HF-only e295", lambda s: _npz_epoch("F", s, 295))]
    for experiment, label in (("E", "System-OOD HF-only converged"), ("B", "System-OOD LF->HF converged"),
                              ("F", "Energy-OOD HF-only converged"), ("D", "Energy-OOD LF->HF converged")):
        cases.append((label, lambda s, e=experiment: _rows_arrays(filtered_rows(e, s, "ood"))))
    for label, fetch in cases:
        per_split = {sig: [] for sig in SIGNALS}
        for s in SPLITS:
            sq, au, eu = fetch(s)
            for sig, var in (("AU", au), ("EU", eu), ("TU", au + eu)):
                per_split[sig].append(_bin_ratios(sq, var))
        cells = []
        for sig in SIGNALS:
            r = np.array(per_split[sig])
            cells.append(f"{sig} " + " ".join(f"b{b + 1} {geo(list(r[:, b]))}" for b in (0, 7, 14)))
        print(f"{label}\n    " + "\n    ".join(cells))


def _npz_epoch(experiment, split, epoch):
    from common import supported
    from epoch_data import EPOCHS
    data = np.load(EPOCHS / f"{experiment}_{split}.npz")
    keep = supported("energy", split, "cc_test_ood")
    i = int(np.where(data["epochs"] == epoch)[0][0])
    return data["sq_error"][i][keep], data["aleatoric_var"][i][keep], data["epistemic_var"][i][keep]


def _rows_arrays(rows):
    return (np.array([r["sq_error"] for r in rows]), np.array([r["aleatoric_var"] for r in rows]),
            np.array([r["epistemic_var"] for r in rows]))


def selected_epochs():
    """Epoch of the validation-NLL checkpoint of every member (as eval/reliability.py selects it)."""
    import logging
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from reliability import CHECKPOINT_PATTERN, select_best_checkpoints
    logging.disable(logging.CRITICAL)
    print("\n## Validation-NLL selected epoch per member (HF-stage epoch, 0-based)")
    for experiment in "BDEF":
        root = ROOT / f"experiment_{experiment}"
        epochs = np.array([[int(CHECKPOINT_PATTERN.match(p.name).group("epoch"))
                            for p in select_best_checkpoints(root / f"checkpoints_{s}", root / f"results_{s}",
                                                             None, "loss", "min", None)] for s in SPLITS])
        print(f"{experiment}: members {epochs.size}, median {np.median(epochs):.1f}, range {epochs.min()}-"
              f"{epochs.max()}, per-split medians {np.median(epochs, axis=1).tolist()}")
    logging.disable(logging.NOTSET)


def system_ratio():
    """Median per-configuration EU/AU of the converged system-split models (cf. Fig. A ratio b)."""
    print("\n## Median per-configuration EU/AU, converged system-split models (A on DFT labels)")
    for test in ("id", "ood"):
        for experiment, name in (("A", "LF-only"), ("E", "HF-only"), ("B", "LF->HF")):
            logs = [math.log(np.median([r["epistemic_var"] / r["aleatoric_var"]
                                        for r in filtered_rows(experiment, s, test)])) for s in SPLITS]
            m, h = ci(logs)
            print(f"System-{test.upper():3s} {name:7s} {math.exp(m):.2f} [{math.exp(m - h):.2f}, {math.exp(m + h):.2f}]")


def acquisition_signals():
    """Paired EU-minus-AU and TU-minus-AU acquisition gains on held-out Energy-OOD."""
    path = ROOT / "active_learning" / "ani_energy" / "runs" / "aggregate_epochs_50" / "summary_ci95.json"
    rows = {(r["regime"], r["test"], r["metric"]): r["split_values"]
            for r in json.loads(path.read_text())["rows"] if "split_values" in r}
    print("\n## Paired differences of acquisition gains between signals, Energy-OOD held-out (meV/atom)")
    for regime in ("hf_only", "lf_hf"):
        gain = {s: rows[(regime, "energy_ood", f"{s}_gain_over_random_meV_per_atom")] for s in ("au", "eu", "tu")}
        for a, b in (("eu", "au"), ("eu", "tu"), ("tu", "au")):
            print(f"{regime:8s} {a}-{b}: {lin([x - y for x, y in zip(gain[a], gain[b])])} "
                  f"splits {[round(x - y, 3) for x, y in zip(gain[a], gain[b])]}")


def quantile_bins():
    print("\n## Median per-configuration EU/AU by energy bin, final models (Fig. 5h)")
    for name, experiment, prefix in f5.FINAL:
        per_split = np.array([f5.bin_ratios(experiment, prefix, s) for s in SPLITS])
        cells = []
        for (label, _, _), column in zip(f5.BINS, per_split.T):
            m, h = ci(list(column))
            cells.append(f"{label}: {10 ** m:.2f} [{10 ** (m - h):.2f}, {10 ** (m + h):.2f}]")
        print(f"{name:8s} " + " | ".join(cells))


def acquisition():
    path = ROOT / "active_learning" / "ani_energy" / "runs" / "aggregate_epochs_50" / "summary_ci95.json"
    summary = json.loads(path.read_text())
    print(f"\n## Active learning (Fig. 6), {path.relative_to(ROOT)}; "
          f"support_filter={bool(summary['inputs'].get('support_filter'))}")
    for r in summary["rows"]:
        if r["metric"].endswith(("meV_per_atom",)):
            print(f"{r['regime']:12s} {r['test']:10s} {r['metric']:45s} {r['mean']:.3f} "
                  f"[{r['ci95_lower']:.3f}, {r['ci95_upper']:.3f}]")
    print("\n## Within-system energy quantile of acquired configurations (Fig. 6c)")
    for strategy in ("random", "hf_only_au", "hf_only_eu", "hf_only_tu", "lf_hf_au", "lf_hf_eu", "lf_hf_tu"):
        q = selected_quantiles(strategy)
        print(f"{strategy:11s} n={len(q)} median {np.median(q):.3f} fraction q>=0.9 {(q >= 0.9).mean():.3f}")


if __name__ == "__main__":
    final_models()
    outlier()
    dynamics()
    quantile_bins()
    acquisition()
    error_scale()
    acquisition_signals()
    au_bins()
    selected_epochs()
    system_ratio()
