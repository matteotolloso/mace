#!/usr/bin/env python3
"""Export the values plotted in the paper's Figs. 2-6 to talk/data/ (mace env, CPU, ~1 min).

    python talk/scripts/extract_data.py

Uses the paper's own loaders and statistics, imported read-only from
papers/iclr27/figures/ (common.py, epoch_data.py, fig5_decomposition.py, fig6_acquisition.py):
five dataset splits, training-support filter, no trim, 95% Student-t intervals over splits
(geometric means and log-space intervals for RMSE and ENCE), and the same smoothing of the
epoch curves (EMA; restarted at the fine-tuning boundary in Fig. 4). The slides draw these
numbers as they are; nothing is recomputed in the browser.

Writes data/paper_figures.json (readable) and data/paper_figures.js (the same object as
`window.PAPER_DATA`, loadable from file:// without a server).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TALK = Path(__file__).resolve().parents[1]
ROOT = TALK.parent
sys.path.insert(0, str(ROOT / "papers" / "iclr27" / "figures"))

from common import EXPERIMENT, PROTOCOLS, SIGNALS, TESTS, final_metrics, interval  # noqa: E402
from epoch_data import across_splits, band, ema, hf_stage, lf_to_hf, lf_to_hf_id, lf_to_hf_train  # noqa: E402
from fig5_decomposition import phased_ema  # noqa: E402
from fig6_acquisition import REGIME, gain_rows, selected_quantiles  # noqa: E402

KINDS = ("system", "energy")


def r(values, digits=6):
    return [float(f"{v:.{digits}g}") for v in values]


def final():
    out = {}
    for kind, test, label in TESTS:
        out[label] = {}
        for p in PROTOCOLS:
            per_split = final_metrics(EXPERIMENT[(kind, p)], test)
            entry = {"RMSE": r(interval([m["RMSE"] for m in per_split], log=True))}
            for metric, log in (("AUSE", False), ("ENCE", True)):
                entry[metric] = {s: r(interval([m[(metric, s)] for m in per_split], log=log))
                                 for s in SIGNALS}
            out[label][p] = entry
    return out


def series(epochs, mean, half, log, smooth):
    back = (lambda v: 10 ** v) if log else (lambda v: v)
    m, lo, hi = (back(smooth(v)) for v in (mean, mean - half, mean + half))
    return {"epoch": [int(e) for e in epochs], "mean": r(m), "lo": r(lo), "hi": r(hi)}


def decomposition():
    """Paper Fig. 4 (script fig5_decomposition): LF->HF over pretraining + fine-tuning."""
    fetches = {"train": lf_to_hf_train, "id": lf_to_hf_id, "ood": lf_to_hf}
    out = {}
    for kind in KINDS:
        out[kind] = {}
        for test, fetch in fetches.items():
            out[kind][test] = {}
            for metric, log in (("AUSE", False), ("ENCE", True)):
                out[kind][test][metric] = {}
                for s in SIGNALS:
                    epochs, values = across_splits(lambda x: fetch(kind, x),
                                                   lambda e, s=s: e[(metric, s)], log=log)
                    mean, half = band(values)
                    out[kind][test][metric][s] = series(
                        epochs, mean, half, log, lambda v, e=epochs: phased_ema(e, v))
    return out


def hf_dynamics():
    """Paper Fig. 5 (script fig3_hf_dynamics): HF stage, HF-only vs LF->HF, OOD test sets."""
    out = {}
    for kind in KINDS:
        out[kind] = {}
        for p in PROTOCOLS:
            out[kind][p] = {}
            for metric, log in (("AUSE", False), ("ENCE", True)):
                out[kind][p][metric] = {}
                for s in SIGNALS:
                    epochs, values = across_splits(lambda x: hf_stage(kind, p, x),
                                                   lambda e, s=s: e[(metric, s)], log=log)
                    mean, half = band(values)
                    out[kind][p][metric][s] = series(epochs, mean, half, log, ema)
    return out


def acquisition():
    """Paper Fig. 6: (a) Energy-OOD RMSE gain over random, (b) selected energy quantiles."""
    rows = gain_rows()
    gains = {}
    for p in PROTOCOLS:
        gains[p] = {}
        for s in SIGNALS:
            row = rows[(REGIME[p], "energy_ood", f"{s.lower()}_gain_over_random_meV_per_atom")]
            gains[p][s] = {"mean": row["mean"], "lo": row["ci95_lower"], "hi": row["ci95_upper"],
                           "splits": r(row["split_values"])}
    bins = np.linspace(0.55, 1.0, 10)

    def hist(strategy):
        q = selected_quantiles(strategy)
        return r(np.histogram(q, bins=bins, weights=np.full(len(q), 1 / len(q)))[0])

    quantiles = {"bins": r(bins), "random": hist("random")}
    for p in PROTOCOLS:
        quantiles[p] = {s: hist(f"{REGIME[p]}_{s.lower()}") for s in SIGNALS}
    return {"gains": gains, "quantiles": quantiles}


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    data = {
        "meta": {"source": "papers/iclr27/figures (Figs. 2-6 loaders and statistics)",
                 "commit": commit, "splits": 5, "members": 10, "support_filter": True, "trim": 0.0},
        "final": final(), "decomposition": decomposition(), "hf_dynamics": hf_dynamics(),
        "acquisition": acquisition(),
    }
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (TALK / "data" / "paper_figures.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    (TALK / "data" / "paper_figures.js").write_text(f"window.PAPER_DATA = {text};\n")
    print(f"wrote {TALK / 'data'}/paper_figures.{{json,js}} ({len(text) // 1024} KiB)")


if __name__ == "__main__":
    main()
