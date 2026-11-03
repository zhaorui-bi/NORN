import copy

import numpy as np
import pytest
import torch

from norn_earth.losses.objective import ReconstructionObjective
from norn_earth.models.factory import build_model
from norn_earth.models.temporal import gp_cholesky, unwhiten_cores, whiten_cores
from norn_earth.training.trainer import anchor_fields, recompute_backward, regularized_loss


def test_two_pass_gradient_includes_core_prior_once(tiny_problem):
    config, data, _ = tiny_problem
    torch.set_num_threads(1)
    torch.manual_seed(7)
    direct = build_model(config)
    recomputed = copy.deepcopy(direct)
    inputs = torch.tensor(data["inputs"])
    ages = torch.tensor(config.anchor_ages)
    objective = ReconstructionObjective(data, config)
    loss, _ = regularized_loss(direct, config, anchor_fields(direct, inputs, ages, 2), objective)
    loss.backward()
    with torch.no_grad():
        cached = anchor_fields(recomputed, inputs, ages, 1)
    leaf = cached.detach().requires_grad_(True)
    other, _ = regularized_loss(recomputed, config, leaf, objective)
    other.backward()
    recompute_backward(recomputed, inputs, ages, leaf.grad, 1)
    torch.testing.assert_close(loss, other)
    for (name, p), (_, q) in zip(direct.named_parameters(), recomputed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name)


def test_dynamic_features_and_age_are_used(tiny_problem):
    config, data, _ = tiny_problem
    model = build_model(config)
    inputs = torch.tensor(data["inputs"][:2])
    ages = torch.tensor([0.0, 20.0])
    with torch.no_grad():
        model.cores[0].fill_(0.0)
        model.cores[1].fill_(5.0)
        fields = model(inputs, ages)
        changed = inputs.clone()
        changed[:, 5] += 0.5
        result = model(changed, ages)
    assert fields.shape == (2, 1, config.grid.nlat, config.grid.nlon)
    assert torch.all(fields > config.model.epsilon_km)
    assert not torch.allclose(fields[0], fields[1])
    assert not torch.allclose(fields[1], result[1])
    with pytest.raises(ValueError, match="Ages"):
        model(inputs[:1], torch.tensor([61.0]))


def test_gp_whitening_roundtrip_and_nonuniform_age_units():
    ages = torch.tensor([0.0, 2.0, 13.0, 60.0], dtype=torch.float64)
    cholesky = gp_cholesky(ages, 10.0, amplitude=2.0)
    x = torch.randn(4, 2, 3, dtype=torch.float64)
    torch.testing.assert_close(whiten_cores(unwhiten_cores(x, cholesky), cholesky), x)
    cov = cholesky @ cholesky.T
    assert np.isclose(float(cov[0, 1] / 4), np.exp(-0.5 * (2 / 10) ** 2))
    assert float(cov[0, 1]) > float(cov[0, -1])
    with pytest.raises(ValueError):
        gp_cholesky(torch.tensor([2.0, 1.0]), 10.0)
