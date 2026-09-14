"""Regression tests for path migration and graphical AL reports."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import inventory, load_json, save_json, verify_inventory
from five_splits import aggregate
from migrate_run_names import remap, transform_graph
from plot_results import render
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
    def test_plots_are_nonblank_and_cached(self):
        reports, settings = fixtures()
        rows = aggregate(reports, settings)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary_ci95.json"
            save_json(summary, {"rows": rows})
            render(summary)
            from PIL import Image, ImageStat
            plots = root / "plots"
            images = list(plots.glob("*.png"))
            self.assertEqual(len(images), 4)
            self.assertEqual(len(list(plots.glob("*.pdf"))), 4)
            for path in images:
                with Image.open(path) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertGreater(min(ImageStat.Stat(image.convert("RGB")).stddev), 5)
            metadata = load_json(plots / "plots.json")
            verify_inventory(metadata["artifacts"])
            before = inventory(images)
            stamp = (plots / "plots.json").stat().st_mtime_ns
            render(summary)
            self.assertEqual(stamp, (plots / "plots.json").stat().st_mtime_ns)
            self.assertEqual(before, inventory(images))


if __name__ == "__main__":
    unittest.main(verbosity=2)
