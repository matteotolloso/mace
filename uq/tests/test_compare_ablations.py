"""compare_ablations: every rebuttal ablation is paired with its one-setting-different reference."""

import unittest

from uq.evaluation import compare_ablations


class CompareAblationsTests(unittest.TestCase):
    def test_pairs_share_dataset_and_parent(self):
        pairs = compare_ablations.pairs()
        self.assertEqual(len(pairs), 12)   # 6 variants x 2 datasets
        for ablation, reference, _ in pairs:
            self.assertEqual(ablation.dataset_id, reference.dataset_id)
            self.assertEqual(ablation.parent_id, reference.parent_id)
            self.assertEqual(list(ablation.splits), list(reference.splits))
            self.assertEqual(list(ablation.members), list(reference.members))


if __name__ == "__main__":
    unittest.main()
