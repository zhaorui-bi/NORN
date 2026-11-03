import math

import numpy as np
import torch

from norn_earth.losses.objective import ReconstructionObjective
from norn_earth.models.posterior import data_potential, pcn_sample
from norn_earth.models.temporal import gp_cholesky, unwhiten_cores


def test_pcn_matches_gaussian_posterior_and_informs_unobserved_time():
    torch.set_num_threads(1)
    # Observe w(0)=1 with variance .25; prior covariance has unit diagonals
    # and off-diagonal rho. w(1) is never directly observed.
    chol = gp_cholesky(torch.tensor([0.0, 1.0]), 2.0)

    def likelihood(z):
        w = unwhiten_cores(z, chol)
        return -0.5 * (w[0, 0] - 1.0).square() / 0.25

    samples, report = pcn_sample(
        likelihood, torch.zeros(4, 2, 1), draws=2500, burnin=600, thin=2, beta=0.65, seed=31
    )
    actual = (chol @ samples.flatten(0, 1)).squeeze(-1)
    rho = math.exp(-0.5 / 4)
    expected_mean = torch.tensor([0.8, rho / 1.25])
    expected_cov = chol @ chol.T
    column = expected_cov[:, :1]
    expected_cov = expected_cov - column @ column.T / (expected_cov[0, 0] + 0.25)
    torch.testing.assert_close(actual.mean(0), expected_mean, rtol=0.05, atol=0.04)
    torch.testing.assert_close(torch.cov(actual.T), expected_cov, rtol=0.15, atol=0.03)
    assert all(0.2 < x < 0.95 for x in report["acceptance_rate_by_chain"])
    assert report["split_rhat_max"] < 1.1


def test_conditioning_ignores_validation_test_and_buffer(tiny_problem):
    config, data, _ = tiny_problem
    data = dict(data)
    # Same train data and fields, arbitrary other labels and age-node values.
    fields = torch.full((config.n_anchors, config.grid.nlat, config.grid.nlon), 8.0)
    first = data_potential(ReconstructionObjective(data, config), fields)
    changed = {**data, "y": data["y"].copy()}
    changed["y"][data["split"] != 0] = 100000
    second = data_potential(ReconstructionObjective(changed, config), fields)
    torch.testing.assert_close(first, second)


def test_age_likelihood_is_mixture_not_likelihood_of_mean(tiny_problem):
    config, data, _ = tiny_problem
    objective = ReconstructionObjective(data, config)
    fields = torch.stack(
        [
            torch.full((config.grid.nlat, config.grid.nlon), float(i * 10 + 1))
            for i in range(config.n_anchors)
        ]
    )
    mask = objective.masks["train"]
    mu = objective.node_predictions(fields)[mask]
    weights, y, sigma = objective.weights[mask], objective.y[mask], objective.sigma[mask]
    logp = torch.distributions.StudentT(4.0, mu, sigma[:, None]).log_prob(y[:, None])
    logw = torch.where(weights > 0, weights.log(), -torch.inf)
    expected = -torch.logsumexp(logp + logw, dim=1).mean()
    torch.testing.assert_close(objective.observation_metrics(fields)["nll"], expected)
    wrong = -torch.distributions.StudentT(4.0, (mu * weights).sum(1), sigma).log_prob(y).mean()
    assert not np.isclose(float(expected), float(wrong))
