"""Fork option --mve_reset_variance_head (WP3.2): only the variance outputs of the last readout change.

Uses a trained LF-only MVE member on a GPU (saved models hold CUDA TorchScript modules); skipped
without a GPU or where the model file is absent (e.g. a fresh clone).
"""

import copy
import unittest

from uq import registry


class ResetVarianceHeadTests(unittest.TestCase):
    def setUp(self):
        self.parent = registry.experiment("ani1x-energy/mace-mve/lf-only")
        self.path = self.parent.final_model(0, 0)
        if not self.path.is_file():
            self.skipTest(f"{self.path} not available")
        import torch
        if not torch.cuda.is_available():
            self.skipTest("needs a GPU to load the model")

    def _outputs(self, model, batch):
        out = model(batch.to_dict(), training=False, compute_force=False)
        return out["energy"].detach(), out["energy_var"].detach()

    def test_mean_kept_variance_redrawn_deterministically(self):
        import torch
        from mace import data as mace_data
        from mace.tools import torch_geometric, utils
        from mace.tools.finetuning_utils import reset_mve_variance_head
        from ase.io import read

        torch.set_default_dtype(torch.float32)
        model = torch.load(self.path, map_location="cuda", weights_only=False)
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        atoms = read(self.parent.data_file("val", 0), index=":8")
        configs = [mace_data.config_from_atoms(a) for a in atoms]
        data = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=float(model.r_max)) for c in configs]
        batch = next(iter(torch_geometric.dataloader.DataLoader(data, batch_size=8, shuffle=False)))
        batch = batch.to("cuda")
        before = self._outputs(model, batch)

        reset = copy.deepcopy(model)
        weights_before = [p.detach().clone() for p in model.parameters()]
        n = reset_mve_variance_head(reset, seed=3)
        after = self._outputs(reset, batch)
        self.assertEqual(n, model.readouts[-1].linear_2.irreps_in.dim)
        # mean energies unchanged (up to GPU summation order); variances change by far more than that
        self.assertTrue(torch.allclose(before[0], after[0], rtol=1e-6, atol=1e-3))
        self.assertGreater(float((before[1] - after[1]).abs().max() / before[1].abs().max()), 1e-2)
        changed = [i for i, (a, b) in enumerate(zip(weights_before, reset.parameters())) if not torch.equal(a, b)]
        self.assertEqual(len(changed), 1)                             # only linear_2 of the last readout
        self.assertIs(list(reset.parameters())[changed[0]], reset.readouts[-1].linear_2.weight)

        again = copy.deepcopy(model)
        reset_mve_variance_head(again, seed=3)
        self.assertTrue(torch.equal(again.readouts[-1].linear_2.weight, reset.readouts[-1].linear_2.weight))


if __name__ == "__main__":
    unittest.main()
