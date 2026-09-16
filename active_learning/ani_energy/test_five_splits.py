"""Independent tests for aggregation; never launch real training."""

import copy
import fcntl
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from common import comparison, inventory, load_json, save_json
import five_splits as five


def fixtures():
    settings = {"seed": 0, "epochs_override": None, "budget": 500,
                "learning_rate": 0.001, "additional_epochs": 100}
    reports = []
    for split in range(5):
        rows = []
        for test in five.TESTS:
            for regime in five.REGIMES:
                before = 100 + 10 * split
                gain = 2 + split + (10 if regime == "lf_hf" else 0)
                rows.append({"regime": regime, "test": test, "count": 100,
                             **comparison(before, before - 10, before - 10 - gain)})
        reports.append({"settings": {**settings, "split_seed": split}, "rows": rows,
                        "inputs": {}, "common_evaluator": []})
    return reports, settings


class AggregationTests(unittest.TestCase):
    def test_ci_and_paired_contrast(self):
        result = five.confidence_interval([1, 2, 3, 4, 5])
        self.assertEqual(result["mean"], 3)
        self.assertAlmostEqual(result["ci95_upper"] - 3, five.T95_DF4 / 2 ** 0.5)
        reports, settings = fixtures()
        rows = five.aggregate(reports, settings)
        paired = next(row for row in rows if row["regime"] == "cross_regime"
                      and row["metric"].endswith("meV_per_atom"))
        self.assertEqual(paired["split_values"], [10] * 5)
        self.assertEqual(paired["ci95_lower"], 10)
        self.assertEqual(paired["ci95_upper"], 10)
        self.assertEqual(len(rows), 32)

    def test_reject_partial_mismatched_and_invalid_results(self):
        reports, settings = fixtures()
        with self.assertRaises(ValueError):
            five.aggregate(reports[:4], settings)
        wrong = copy.deepcopy(reports)
        wrong[2]["settings"]["seed"] = 1
        with self.assertRaises(ValueError):
            five.aggregate(wrong, settings)
        wrong = copy.deepcopy(reports)
        wrong[2]["rows"].pop()
        with self.assertRaises(ValueError):
            five.aggregate(wrong, settings)
        with self.assertRaises(ValueError):
            five.confidence_interval([1, 2, 3, 4, float("nan")])
        self.assertIsNone(five.confidence_interval([1, 2, 3, 4, None])["mean"])
        with self.assertRaises(ValueError):
            five.aggregate(reports, settings, common_evaluator=True)

    def test_locks_provenance_and_cached_outputs(self):
        reports, settings = fixtures()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            run = Path(temp)
            metric = run / "metric.json"
            save_json(metric, {"rmse": 1})
            reports[0]["inputs"] = inventory([metric])
            save_json(run / "manifest.json", {"settings": reports[0]["settings"]})
            save_json(run / "report/summary.json", reports[0])
            with (run / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.assertIsNone(five.read_report(run))
            self.assertEqual(five.read_report(run)["settings"], reports[0]["settings"])
            rows = five.aggregate(reports, settings)
            output = run / "aggregate"
            five.write_report(output, rows, {"version": 1}, settings)
            stamp = (output / "summary_ci95.json").stat().st_mtime_ns
            five.write_report(output, rows, {"version": 1}, settings)
            self.assertEqual(stamp, (output / "summary_ci95.json").stat().st_mtime_ns)
            self.assertEqual(len(load_json(output / "summary_ci95.json")["rows"]), 32)
            save_json(metric, {"rmse": 2})
            with self.assertRaises(RuntimeError):
                five.read_report(run)

    def test_wait_only_never_launches_training(self):
        reports, _ = fixtures()
        for report in reports:
            report["_verified_inputs"] = {}
        with patch("sys.argv", ["five_splits.py", "--aggregate-only", "--wait"]), \
                patch.object(five, "read_report", side_effect=[None] * 5 + reports), \
                patch.object(five.time, "sleep") as sleep, \
                patch.object(five.subprocess, "run") as launch, \
                patch.object(five, "write_report") as write, redirect_stdout(io.StringIO()):
            five.main()
            launch.assert_called_once()
            self.assertTrue(launch.call_args.args[0][2].endswith("plot_results.py"))
            sleep.assert_called_once()
            write.assert_called_once()

    def test_runner_launches_five_splits_then_aggregates(self):
        reports, _ = fixtures()
        for report in reports:
            report["_verified_inputs"] = {}
        with patch("sys.argv", ["five_splits.py", "--gpu", "2"]), \
                patch.object(five, "read_report", side_effect=[None] * 5 + reports), \
                patch.object(five.subprocess, "run") as launch, \
                patch.object(five, "write_report") as write, redirect_stdout(io.StringIO()):
            five.main()
            self.assertEqual(launch.call_count, 6)
            for split, call in enumerate(launch.call_args_list[:5]):
                self.assertEqual(call.args[0][2:], ["2", "--split-seed", str(split), "--seed", "0"])
            self.assertTrue(launch.call_args_list[-1].args[0][2].endswith("plot_results.py"))
            write.assert_called_once()

    def test_completed_runs_only_aggregate_and_plot(self):
        reports, _ = fixtures()
        for report in reports:
            report["_verified_inputs"] = {}
        with patch("sys.argv", ["five_splits.py", "--gpu", "2"]), \
                patch.object(five, "read_report", side_effect=reports + reports), \
                patch.object(five.subprocess, "run") as launch, \
                patch.object(five, "write_report"), redirect_stdout(io.StringIO()):
            five.main()
            launch.assert_called_once()
            self.assertTrue(launch.call_args.args[0][2].endswith("plot_results.py"))
            self.assertTrue(launch.call_args.args[0][-1].endswith("runs/aggregate/summary_ci95.json"))

    def test_legacy_learning_rate_is_not_silently_reused(self):
        reports, _ = fixtures()
        del reports[0]["settings"]["learning_rate"]
        with patch("sys.argv", ["five_splits.py", "--gpu", "2"]), \
                patch.object(five, "read_report", return_value=reports[0]), \
                patch.object(five.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "shared AL learning rate"):
                five.main()
            launch.assert_not_called()

    def test_legacy_epoch_limit_is_not_silently_reused(self):
        reports, _ = fixtures()
        del reports[0]["settings"]["additional_epochs"]
        with patch("sys.argv", ["five_splits.py", "--gpu", "2"]), \
                patch.object(five, "read_report", return_value=reports[0]), \
                patch.object(five.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "shared AL epoch limit"):
                five.main()
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
