"""`uq sync-back`: remote paths mirror the local layout (no ssh needed)."""

import unittest

from uq import paths, registry
from uq.training import sync


class SyncTests(unittest.TestCase):
    def test_remote_paths_mirror_the_layout(self):
        exp = registry.experiment("ani1x-system/mace-mse/lf-only")
        pairs = sync.transfers(exp, [2], "/r/mace", "/r/runs")
        what = {w: (r, l) for r, l, w in pairs}
        self.assertEqual(set(what), {"results", "logs", "models", "evaluation"})
        self.assertEqual(what["results"][0], "/r/runs/ani1x-system/mace-mse/lf-only/split_2/results")
        self.assertEqual(what["results"][1], paths.run_dir(exp.id, 2) / "results")
        self.assertEqual(what["evaluation"][0], "/r/mace/experiments/ani1x-system/mace-mse/lf-only/evaluation/split_2")

    def test_checkpoints_only_on_request(self):
        exp = registry.experiment("ani1x-system/mace-mse/lf-only")
        self.assertIn("checkpoints", [w for _, _, w in sync.transfers(exp, [0], "/a", "/b", checkpoints=True)])


if __name__ == "__main__":
    unittest.main()
