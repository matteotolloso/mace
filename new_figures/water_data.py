"""Data access for the periodic-water figures (figW_*.py).

Families: wA = LF-only (BLYP), wC = HF-only (CCSDT), wB = LF->HF (BLYP
pretraining of wA member (s, m), then CCSDT fine-tuning). Five dataset splits,
ten members each; statistics as in common.py.

Differences from the ANI figures:
- one held-out test set per split (`dataset/water_<s>/{blyp,ccsdt}/test.xyz`:
  133 BLYP and 50 CCSDT configurations), no ID/OOD shift;
- no training-support filter: every test configuration is used;
- the BLYP and CCSDT splits are shuffled independently, so a wB test geometry may
  have been in wA's pretraining set (README, audit notes). This favours LF->HF.

Final-model rows come from `reliability_test_nocal_raw.csv`, which holds every
test configuration (the `--trim 0.005` in eval_wX.sh only affects the per-split
diagnostic plots). Per-epoch curves come from `epoch_quality_test.csv` and
`epoch_quality_finetune_mixed_{train,test}.csv`, computed without trim.
"""

from __future__ import annotations

from functools import lru_cache

from common import SPLITS, final_rows, metrics
from epoch_data import CACHE, _from_csv

WATER = {"LF-only": "wA", "HF-only": "wC", "LF→HF": "wB"}
PRETRAIN = 300


@lru_cache(None)
def final_metrics_water(protocol, calibrated=False):
    """Per-split metric dicts (five entries) for the final models on the test set."""
    return tuple(metrics(list(final_rows(WATER[protocol], s, "test", calibrated)))
                 for s in SPLITS)


def test_rows(protocol, split, calibrated=False):
    return list(final_rows(WATER[protocol], split, "test", calibrated))


def hf_stage(protocol, split):
    """CCSDT-training epochs from 0 on the CCSDT test set: wC from scratch, wB fine-tuning."""
    return _from_csv(CACHE(WATER[protocol], split, "epoch_quality_test.csv"))


def lf_to_hf(split, data="test"):
    """wB over the whole trajectory: BLYP pretraining (wA checkpoints, BLYP file and
    labels) at epochs 0-299, then CCSDT fine-tuning at 300-399."""
    path = CACHE("wB", split, f"epoch_quality_finetune_mixed_{data}.csv")
    return {**_from_csv(path, "pretrain"), **_from_csv(path, "finetune", PRETRAIN)}
