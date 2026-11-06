"""Resume of an interrupted member: `launch.prepare_resume` and the resume command (synthetic files)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from uq import paths, registry
from uq.training import launch

EXP = "ani1x-energy/mace-mse/lf-only"


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(paths, "RUNS", Path(self.tmp.name))
        self.patch.start()
        self.exp = registry.experiment(EXP)
        self.ck, self.rs = paths.checkpoints_dir(EXP, 0), paths.results_dir(EXP, 0)
        self.ck.mkdir(parents=True)
        self.rs.mkdir(parents=True)
        self.log = self.exp.train_log(0, 3)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def write_run(self, last_checkpoint, logged_epochs, cut_line=False):
        for epoch in range(last_checkpoint + 1):
            (self.ck / f"{self.exp.run_name}_run-3_epoch-{epoch}.pt").write_bytes(b"x")
        rows = [{"mode": "eval", "epoch": None, "loss": 9.0}]
        for epoch in logged_epochs:
            rows += [{"mode": "opt", "epoch": epoch, "loss": 1.0}, {"mode": "eval", "epoch": epoch, "loss": 2.0}]
        text = "".join(json.dumps(row) + "\n" for row in rows)
        self.log.write_text(text + ('{"mode": "opt", "ep' if cut_line else ""))
        return text

    def epochs_in_log(self):
        return [json.loads(line).get("epoch") for line in self.log.read_text().splitlines()]

    def test_rows_after_last_checkpoint_are_cut_and_backed_up(self):
        original = self.write_run(last_checkpoint=4, logged_epochs=range(6), cut_line=True)
        self.assertEqual(launch.prepare_resume(EXP, 0, 3), 4)
        self.assertEqual(max(e for e in self.epochs_in_log() if e is not None), 4)
        self.assertIn(None, self.epochs_in_log())   # the pre-training validation row is kept
        backups = list(self.rs.glob(f"{self.log.name}.pre-resume-*"))
        self.assertEqual(len(backups), 1)
        self.assertTrue(backups[0].read_text().startswith(original))

    def test_consistent_log_is_left_alone(self):
        self.write_run(last_checkpoint=4, logged_epochs=range(5))
        before = self.log.read_bytes()
        self.assertEqual(launch.prepare_resume(EXP, 0, 3), 4)
        self.assertEqual(self.log.read_bytes(), before)
        self.assertEqual(list(self.rs.glob("*.pre-resume-*")), [])

    def test_refuses_without_checkpoint_or_when_finished(self):
        self.log.write_text("")
        with self.assertRaises(SystemExit):
            launch.prepare_resume(EXP, 0, 3)            # no epoch checkpoint
        self.write_run(last_checkpoint=2, logged_epochs=range(3))
        self.exp.final_model(0, 3).write_bytes(b"x")
        with self.assertRaises(SystemExit):
            launch.prepare_resume(EXP, 0, 3)            # complete member

    def test_missing_checkpoint_directory(self):
        self.ck.rmdir()
        self.assertIsNone(launch.latest_epoch(EXP, 0, 3))

    def test_resume_only_appends_flags(self):
        plain = launch.command(EXP, 0, 3)
        self.assertEqual(launch.command(EXP, 0, 3, resume=True), plain + ["--restart_latest", "--resume_exact"])


if __name__ == "__main__":
    unittest.main()
