"""Fork option --mve_beta_nll (WP3.3): the loss value stays the Gaussian NLL, the gradient is beta-NLL's."""

import unittest

import torch

from mace.modules.loss import weighted_gaussian_nll_energy


class _Batch(dict):
    def __init__(self, energy, ptr):
        super().__init__(energy=energy)
        self.ptr, self.weight, self.energy_weight = ptr, torch.ones_like(energy), torch.ones_like(energy)


def _loss(beta):
    mean = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
    var = torch.tensor([0.04, 0.5, 2.0], requires_grad=True)
    ref = _Batch(torch.tensor([1.3, -2.1, 2.0]), torch.tensor([0, 2, 5, 9]))
    loss = weighted_gaussian_nll_energy(ref, {"energy": mean, "energy_var": var}, beta=beta)
    loss.backward()
    return loss.detach(), mean.grad, var.grad, var.detach()


class BetaNllTests(unittest.TestCase):
    def test_value_unchanged_gradient_weighted(self):
        value0, gmean0, gvar0, var = _loss(0.0)
        value, gmean, gvar, _ = _loss(0.5)
        self.assertTrue(torch.allclose(value, value0))
        weight = var ** 0.5
        self.assertTrue(torch.allclose(gmean, weight * gmean0, rtol=1e-5))
        self.assertTrue(torch.allclose(gvar, weight * gvar0, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
