"""CPU-only regression tests for cached evaluation plotting."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uq.evaluation import aggregate
from uq.evaluation.plot_style import save_svg
from uq.evaluation.replicate_statistics import aggregate_rows, confidence_arrays, summarize, summarize_log


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

    def test_epoch_quality_log_view_uses_log_space_intervals_for_positive_metrics(self):
        metrics = [
            *(f"{prefix}_{uncertainty}" for prefix in ("spearman", "ause", "ence", "magnitude")
              for uncertainty in aggregate.UNCERTAINTIES),
            "rmse_e_atom", "nll_energy",
        ]
        replicates = []
        for scale in (1.0, 2.0, 4.0, 8.0, 16.0):
            row = {"epoch": 0, "rmse_e_atom": 0.01 * scale, "nll_energy": -scale}
            for uncertainty in aggregate.UNCERTAINTIES:
                row[f"spearman_{uncertainty}"] = 0.1 * scale
                row[f"ause_{uncertainty}"] = 0.01 * scale
                row[f"ence_{uncertainty}"] = 0.02 * scale
                row[f"magnitude_{uncertainty}"] = 0.001 * scale
            replicates.append([row])
        rows = aggregate_rows(replicates, key_fields=["epoch"], metric_fields=metrics)
        fig, axes = aggregate.draw_epoch_quality(
            rows, metrics, fixed_scales=False, log_positive_metrics=True
        )
        self.assertEqual([ax.get_yscale() for ax in axes],
                         ["linear", "log", "log", "log", "log", "linear"])
        expected_limits = [
            aggregate.LOG_EPOCH_QUALITY_LIMITS[key]
            for key in ("spearman", "ause", "ence", "magnitude", "rmse_e_atom", "nll_energy")
        ]
        for ax, limits in zip(axes, expected_limits):
            np.testing.assert_allclose(ax.get_ylim(), limits)
        self.assertEqual([ax.get_ylabel() for ax in axes],
                         ["Spearman ↑", "AUSE ↓", "ENCE ↓", "Uncertainty", "RMSE ↓", "GNLL ↓"])
        self.assertNotIn("geometric mean", " ".join(ax.get_ylabel() for ax in axes))
        expected = rows[0]["ause_aleatoric_geometric_mean"]
        self.assertAlmostEqual(axes[1].lines[0].get_ydata()[0], expected)
        plt.close(fig)

    def test_out_of_range_center_is_clipped_and_marked_with_x(self):
        rows = [{"value": 2.0, "value_mean": 2.0,
                 "value_ci95_low": 1.5, "value_ci95_high": 2.5}]
        fig, ax = plt.subplots()
        aggregate.plot_band(
            ax, np.array([3.0]), rows, "value", color="black", label="value",
            y_limits=(0.0, 1.0),
        )
        x_markers = [line for line in ax.lines if line.get_marker() == "x"]
        self.assertEqual(len(x_markers), 1)
        self.assertEqual(x_markers[0].get_ydata()[0], 1.0)
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


class EnsembleSizeTests(unittest.TestCase):
    """uq.evaluation.ensemble_size: windows of M members, single model at M = 1."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.members = {"energy": rng.normal(0.0, 0.05, (10, 300)), "var": rng.uniform(1e-4, 4e-3, (10, 300)),
                        "ref": rng.normal(0.0, 0.05, 300)}
        from uq.evaluation import ensemble_size
        self.module = ensemble_size
        self.rows = ensemble_size.ensemble_size_rows(self.members, np.ones(300, dtype=bool))

    def value(self, members, metric, signal):
        (row,) = [r for r in self.rows if (r["members"], r["metric"], r["uncertainty_type"]) == (members, metric, signal)]
        return row["value"]

    def test_full_ensemble_is_one_window_with_the_reliability_definitions(self):
        full = self.module.window_metrics(self.members["energy"], self.members["var"], self.members["ref"])
        self.assertEqual({r["windows"] for r in self.rows if r["members"] == 10}, {1})
        for (metric, signal), value in full.items():
            self.assertAlmostEqual(self.value(10, metric, signal), value)
        pred = self.members["energy"].mean(0)
        self.assertAlmostEqual(self.value(10, "rmse", "ensemble"), np.sqrt(np.mean((pred - self.members["ref"]) ** 2)))

    def test_single_model_is_the_mean_over_members_and_has_no_epistemic_signal(self):
        singles = [self.module.window_metrics(self.members["energy"][[m]], self.members["var"][[m]],
                                              self.members["ref"]) for m in range(10)]
        self.assertAlmostEqual(self.value(1, "spearman", "aleatoric"),
                               np.mean([s[("spearman", "aleatoric")] for s in singles]))
        self.assertTrue(np.isnan(self.value(1, "spearman", "epistemic")))
        self.assertAlmostEqual(self.value(1, "ause", "total"), self.value(1, "ause", "aleatoric"))

    def test_mse_ensemble_has_no_aleatoric_signal(self):
        members = dict(self.members, var=np.zeros_like(self.members["var"]))
        rows = self.module.ensemble_size_rows(members, np.ones(300, dtype=bool))
        aleatoric = [r["value"] for r in rows if r["uncertainty_type"] == "aleatoric"]
        self.assertTrue(np.all(np.isnan(aleatoric)))
        total = {(r["members"], r["metric"]): r["value"] for r in rows if r["uncertainty_type"] == "total"}
        epistemic = {(r["members"], r["metric"]): r["value"] for r in rows if r["uncertainty_type"] == "epistemic"}
        self.assertEqual(total.keys(), epistemic.keys())
        for key in total:
            np.testing.assert_equal(total[key], epistemic[key])
