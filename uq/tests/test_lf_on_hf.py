"""WP2 LF-only baseline: E0 refit by least squares and per-atom ensemble rows (CPU, synthetic data)."""

import unittest

import numpy as np

from uq.evaluation import lf_on_hf, pipeline


class LfOnHfTests(unittest.TestCase):
    def test_shift_recovers_per_element_offsets(self):
        rng = np.random.default_rng(0)
        counts = rng.integers(0, 6, size=(200, 4)).astype(float)
        offset = np.array([-0.5, -38.0, -54.6, -75.1])
        pred = rng.normal(size=(3, 200))
        ref = pred[1] + counts @ offset + 1e-6 * rng.normal(size=200)
        shifts = lf_on_hf.fit_shifts(pred, ref, counts)
        np.testing.assert_allclose(shifts[1], offset, atol=1e-5)

    def test_rows_follow_the_reliability_definitions(self):
        energy = np.array([[2.0, 6.0], [4.0, 6.0]])   # 2 members x 2 configurations (total energies)
        var = np.array([[4.0, 0.0], [12.0, 0.0]])
        rows = lf_on_hf.rows_from(energy, var, ref_energy=np.array([2.0, 9.0]), num_atoms=np.array([2.0, 3.0]))
        first = rows[0]
        self.assertAlmostEqual(first["pred_energy"], 1.5)            # mean of 1.0 and 2.0 per atom
        self.assertAlmostEqual(first["aleatoric_var"], 2.0)          # mean of 4/4 and 12/4
        self.assertAlmostEqual(first["epistemic_var"], 0.25)         # population variance of 1.0 and 2.0
        self.assertAlmostEqual(first["total_var"], 2.25)
        self.assertAlmostEqual(rows[1]["sq_error"], 1.0)
        self.assertEqual(list(rows[0]), lf_on_hf.RAW_FIELDS)

    def test_pipeline_step_only_where_requested(self):
        names = [step.name for step in pipeline.steps("ani1x-energy/mace-mve/lf-to-hf", 0)]
        self.assertIn("lfonly_baseline", names)
        names = [step.name for step in pipeline.steps("ani1x-energy/mace-mse/lf-to-hf", 0)]
        self.assertNotIn("lfonly_baseline", names)


if __name__ == "__main__":
    unittest.main()
