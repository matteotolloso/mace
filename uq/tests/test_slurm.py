"""`uq train-slurm`: chains (parent then child), 1-GPU packing and the sbatch text (no SLURM needed)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from uq import paths
from uq.training import slurm

PARENT, CHILD = "ani1x-system/mace-mse/lf-only", "ani1x-system/mace-mse/lf-to-hf"


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(paths, "RUNS", Path(self.tmp.name))   # nothing complete
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_children_follow_their_parent(self):
        groups = slurm.chains([PARENT, CHILD], splits=[0], members=[0, 1])
        self.assertEqual(groups, [[(PARENT, 0, 0), (CHILD, 0, 0)], [(PARENT, 0, 1), (CHILD, 0, 1)]])

    def test_child_without_parent_in_invocation_is_skipped(self):
        self.assertEqual(slurm.chains([CHILD], splits=[0], members=[0]), [])

    def test_script_runs_train_member_in_chains_on_one_gpu(self):
        groups = slurm.chains([PARENT, CHILD], splits=[0], members=[0, 1])
        text = slurm.script(groups, account="ACC", partition="part", qos="normal", walltime="24:00:00",
                            pack=2, python="py", resume=True, out_dir=Path("/o"), tag="t")
        for line in ("#SBATCH --account=ACC", "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=8",
                     "#SBATCH --mem=120G", "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"):
            self.assertIn(line, text)
        first = next(line for line in text.splitlines() if line.startswith("( "))
        self.assertLess(first.index(f"train-member {PARENT} 0 0 --resume"), first.index(" && "))
        self.assertIn(f"train-member {CHILD} 0 0 --resume", first)
        self.assertEqual(text.count(") &"), 2)

    def test_children_of_parents_in_flight_form_their_own_chains(self):
        self.assertEqual(slurm.chains([CHILD], splits=[0], members=[0], parents_in_flight=True), [[(CHILD, 0, 0)]])
        text = slurm.script([[(CHILD, 0, 0)]], account="A", partition="p", qos="normal", walltime="1:00:00",
                            pack=1, python="py", resume=False, out_dir=Path("/o"), tag="t", after=["5", "6"])
        self.assertIn("#SBATCH --dependency=afterany:5:6", text)

    def test_dry_run_needs_an_account(self):
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("UQ_SLURM_ACCOUNT", None)
            with self.assertRaises(SystemExit):
                slurm.run([PARENT], dry_run=True)


if __name__ == "__main__":
    unittest.main()
