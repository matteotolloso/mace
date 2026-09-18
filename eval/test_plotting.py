"""CPU-only regression tests for cached evaluation plotting."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aggregate_replicates as aggregate
from plot_style import save_svg
from replicate_statistics import aggregate_rows, confidence_arrays, summarize, summarize_log


class PlottingTests(unittest.TestCase):
    def test_log_t_interval_is_multiplicative_and_arithmetic_is_preserved(self):
        values = [0.001, 0.01, 0.1, 1.0, 10.0]
        rows = aggregate_rows([[{"epoch": 0, "rmse": v}] for v in values],
                              key_fields=["epoch"], metric_fields=["rmse"])
        mean, low, high = confidence_arrays(rows, "rmse", geometric=True)
        self.assertAlmostEqual(mean[0], 0.1)
        self.assertAlmostEqual(np.log(mean[0] / low[0]), np.log(high[0] / mean[0]))
        self.assertAlmostEqual(rows[0]["rmse"], np.mean(values))
        self.assertEqual(rows[0]["rmse_ci95_low"], summarize(values)["ci95_low"])

    def test_zero_is_not_dropped_or_replaced_for_log_intervals(self):
        self.assertEqual(summarize_log([0, 1, 2, 3, 4])["n"], 0)
        rows = aggregate_rows([[{"epoch": 0, "rmse": v}] for v in [0, 1, 2, 3, 4]],
                              key_fields=["epoch"], metric_fields=["rmse"])
        fig, ax = plt.subplots()
        self.assertFalse(aggregate.use_log_intervals(ax, rows, ["rmse"]))
        self.assertEqual(ax.get_yscale(), "linear")
        self.assertEqual(rows[0]["rmse_n"], 5)
        plt.close(fig)

    def test_svg_only_even_for_old_filename_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [2, 4, 8])
            save_svg(fig, Path(temporary) / "plot.png")
            self.assertEqual([p.name for p in Path(temporary).iterdir()], ["plot.svg"])
            self.assertIn("<svg", (Path(temporary) / "plot.svg").read_text())
            plt.close(fig)

    def test_legacy_same_dataset_cache_is_skipped_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            (cache / "split_0").mkdir(parents=True)
            old = cache / "split_0/epoch_quality_finetune_same_id.csv"
            old.write_text("preserved legacy cache\n")
            with patch("sys.argv", ["aggregate", "--cache-dir", str(cache),
                                    "--output-dir", str(Path(temporary) / "out")]), \
                    patch.object(aggregate, "load_replicates") as load:
                aggregate.main()
                load.assert_not_called()
            self.assertEqual(old.read_text(), "preserved legacy cache\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
