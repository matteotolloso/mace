"""Checkpoint selection after `uq prune-checkpoints` (synthetic files, no models)."""

import json
import logging
import tempfile
import unittest
from pathlib import Path

from uq.evaluation.reliability import select_best_checkpoints

logging.disable(logging.CRITICAL)
LOSSES = {0: 5.0, 1: 4.0, 2: 1.0, 3: 3.0, 4: 2.0, 5: 2.5}   # best epoch by loss: 2


class PrunedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ck, self.rs = root / "checkpoints", root / "results"
        self.ck.mkdir()
        self.rs.mkdir()
        rows = [{"mode": "eval", "epoch": e, "loss": v, "head": "Default"} for e, v in LOSSES.items()]
        (self.rs / "mace_run-0_train.txt").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        for epoch in LOSSES:
            (self.ck / f"mace_run-0_epoch-{epoch}.pt").write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def select(self):
        return [p.name for p in select_best_checkpoints(self.ck, self.rs, "mace", "loss", "min", None)]

    def prune(self, epochs, record=True):
        names = [f"mace_run-0_epoch-{e}.pt" for e in epochs]
        for name in names:
            (self.ck / name).unlink()
        if record:
            event = {"deleted": [{"file": n, "bytes": 1} for n in names]}
            (self.ck / "PRUNED.json").write_text(json.dumps({"events": [event]}))

    def test_unpruned_selects_best(self):
        self.assertEqual(self.select(), ["mace_run-0_epoch-2.pt"])

    def test_recorded_pruning_keeps_the_same_best(self):
        self.prune([1, 3, 4])            # keep 0 and 5 (stride 5) and the best, 2
        self.assertEqual(self.select(), ["mace_run-0_epoch-2.pt"])

    def test_unrecorded_gap_is_an_error(self):
        self.prune([1, 3], record=False)
        with self.assertRaises(RuntimeError):
            self.select()

    def test_pruned_best_is_an_error(self):
        self.prune([1, 2, 3, 4])
        with self.assertRaisesRegex(RuntimeError, "was pruned"):
            self.select()


if __name__ == "__main__":
    unittest.main()
