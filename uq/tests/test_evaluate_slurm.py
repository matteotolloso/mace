"""`uq evaluate-slurm`: one guarded `uq evaluate` job per split, dependencies, optional prune job."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from uq import paths
from uq.evaluation import slurm

PARENT, CHILD = "ani1x-system/mace-mse/lf-only", "ani1x-system/mace-mse/lf-to-hf"


class EvaluateSlurmTests(unittest.TestCase):
    def test_evaluation_script_runs_one_split_without_aggregate(self):
        text = slurm.evaluation_script(CHILD, 3, account="ACC", partition="part", qos="normal",
                                       walltime="24:00:00", python="py", out_dir=Path("/o"),
                                       after=["11", "12"], tag="t")
        for line in ("#SBATCH --account=ACC", "#SBATCH --gres=gpu:1", "#SBATCH --mem=120G",
                     "#SBATCH --dependency=afterany:11:12"):
            self.assertIn(line, text)
        self.assertIn(f'py -m uq evaluate {CHILD} --gpu "$gpu" --splits 3 --no-aggregate', text)
        self.assertLess(text.index("|| exit 1"), text.index("-m uq evaluate"))

    def test_completeness_check_covers_the_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"UQ_RUNS": tmp, "PATH": "/usr/bin:/bin"}
            code = subprocess.run([sys.executable, "-c", slurm.complete_check(CHILD, 0)], env=env,
                                  cwd=paths.ROOT, capture_output=True, text=True)
        self.assertEqual(code.returncode, 1)
        self.assertIn(PARENT, code.stdout)

    def test_prune_job_is_cpu_only_and_waits(self):
        text = slurm.prune_script([PARENT, CHILD], account="ACC", partition="lrd_all_serial", walltime="04:00:00",
                                  python="py", out_dir=Path("/o"), after=["21"], tag="p")
        self.assertNotIn("--gres", text)
        self.assertIn("#SBATCH --dependency=afterok:21", text)
        self.assertIn(f"py -m uq prune-checkpoints {PARENT} {CHILD} --apply", text)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "RUNS", Path(tmp)):
            slurm.run([PARENT], splits=[0], account="ACC", prune=True, dry_run=True)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
