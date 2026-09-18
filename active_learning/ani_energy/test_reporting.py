"""Regression tests for path migration and graphical AL reports."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import inventory, load_json, save_json, verify_inventory
from five_splits import aggregate
from migrate_run_names import remap, transform_graph
from plot_results import log_interval, render
from test_five_splits import fixtures


class MigrationTests(unittest.TestCase):
    def test_default_run_names_have_no_extra_seed_layer(self):
        from workflow import parse_args
        with patch("sys.argv", ["workflow.py", "--split-seed", "3"]):
            args = parse_args()
            self.assertEqual(args.name, "split_3")
            self.assertEqual(args.seed, 0)
        with patch("sys.argv", ["workflow.py", "--split-seed", "3", "--epochs", "20"]):
            self.assertEqual(parse_args().name, "split_3_epochs_20")

    def test_recursive_hash_updates_preserve_values(self):
        old, new = "/runs/split_0_seed_0", "/runs/split_0"
        leaf = json.dumps({"model": old + "/best.model", "rmse": 2.3}).encode()
        leaf_hash = hashlib.sha256(leaf).hexdigest()
        parent = json.dumps({"artifacts": {old + "/metrics.json": leaf_hash},
                             "prediction_sha256": leaf_hash, "data": [1.0, 2.0, 3.0]}).encode()
        parent_hash = hashlib.sha256(parent).hexdigest()
        root = json.dumps({"inputs": {old + "/report.json": parent_hash}}).encode()
        contents = {old + "/root.json": root, old + "/report.json": parent,
                    old + "/metrics.json": leaf}
        outputs = transform_graph(contents, {old: new}, {})
        changed_leaf = outputs[old + "/metrics.json"]
        changed_parent = json.loads(outputs[old + "/report.json"])
        digest = hashlib.sha256(changed_leaf).hexdigest()
        self.assertEqual(changed_parent["artifacts"], {new + "/metrics.json": digest})
        self.assertEqual(changed_parent["prediction_sha256"], digest)
        self.assertEqual(changed_parent["data"], [1.0, 2.0, 3.0])
        self.assertEqual(json.loads(changed_leaf)["rmse"], 2.3)
        self.assertEqual(json.loads(outputs[old + "/root.json"])["inputs"],
                         {new + "/report.json": hashlib.sha256(outputs[old + "/report.json"]).hexdigest()})

    def test_yaml_source_hash_and_exact_unchanged_bytes(self):
        old, new = "/runs/split_1_seed_0", "/runs/split_1"
        unchanged = b'{"value": 5}\n'
        yaml = f"train_file: {old}/train.xyz\nseed: 7\n".encode()
        metadata = json.dumps({"sources": {"/workflow.py": "a" * 64}}).encode()
        result = transform_graph({"keep.json": unchanged, "config.yml": yaml, "manifest.json": metadata},
                                 {old: new}, {"a" * 64: "b" * 64})
        self.assertEqual(result["keep.json"], unchanged)
        self.assertIn(f"train_file: {new}/train.xyz".encode(), result["config.yml"])
        self.assertIn(b"seed: 7", result["config.yml"])
        self.assertEqual(json.loads(result["manifest.json"])["sources"]["/workflow.py"], "b" * 64)
        self.assertEqual(remap(old + "0/train.xyz", {old: new}), old + "0/train.xyz")


class PlotTests(unittest.TestCase):
    def test_log_interval_is_symmetric_in_log_coordinates(self):
        original = {"split_values": [0.01, 0.1, 1, 10, 100], "mean": 22.222}
        result = log_interval(original)
        self.assertAlmostEqual(result["mean"], 1)
        self.assertAlmostEqual(math.log(result["mean"] / result["ci95_lower"]),
                               math.log(result["ci95_upper"] / result["mean"]))
        self.assertEqual(original["mean"], 22.222)
        for bad in (0, -1, None, float("nan")):
            with self.assertRaises(ValueError):
                log_interval({"split_values": [bad, 1, 2, 3, 4]})

    def test_plots_are_nonblank_and_cached(self):
        reports, settings = fixtures()
        rows = aggregate(reports, settings)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary_ci95.json"
            save_json(summary, {"rows": rows, "settings": settings})
            source_hash = inventory([summary])
            render(summary)
            self.assertEqual(source_hash, inventory([summary]))
            display = load_json(root / "plots/rmse_display.json")
            self.assertEqual(display["ood_scale"], "log")
            for row in display["rows"]:
                self.assertEqual(row["estimator"], "geometric_mean_log_t" if row["test"] == "energy_ood" else "arithmetic_mean_t")
            import xml.etree.ElementTree as ET
            plots = root / "plots"
            images = list(plots.glob("*.svg"))
            self.assertEqual(len(images), 4)
            self.assertEqual(len(list(plots.glob("*.pdf"))), 0)
            self.assertEqual(len(list(plots.glob("*.png"))), 0)
            for path in images:
                root_svg = ET.parse(path).getroot()
                self.assertGreater(len(root_svg.findall(".//{http://www.w3.org/2000/svg}path")), 20)
                self.assertGreater(len(root_svg.findall(".//{http://www.w3.org/2000/svg}text")), 10)
            metadata = load_json(plots / "plots.json")
            verify_inventory(metadata["artifacts"])
            before = inventory(images)
            stamp = (plots / "plots.json").stat().st_mtime_ns
            render(summary)
            self.assertEqual(stamp, (plots / "plots.json").stat().st_mtime_ns)
            self.assertEqual(before, inventory(images))
            images[0].unlink()
            render(summary)
            self.assertTrue(all(path.is_file() for path in images))
            verify_inventory(load_json(plots / "plots.json")["artifacts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
