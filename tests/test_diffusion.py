import json

import numpy as np
import pytest
import torch

from norn_earth.models.diffusion import sample_diffusion_prior, train_latent_prior
from norn_earth.models.ensemble import decoder_fingerprint
from norn_earth.models.factory import build_model
from norn_earth.models.temporal import unwhiten_cores, whiten_cores


def test_prior_normalization_gauge_and_observation_guidance(tiny_problem, tmp_path):
    config, data, _ = tiny_problem
    torch.set_num_threads(1)
    torch.manual_seed(20)
    model = build_model(config).eval().requires_grad_(False)
    inputs = torch.tensor(data["inputs"])
    # Validation deliberately differs strongly. Normalization must use train only.
    white = torch.randn(4, *model.cores.shape)
    white[3] += 10
    cores = torch.stack([unwhiten_cores(x, model.prior_cholesky) for x in white])
    ensemble = tmp_path / "ensemble.npz"
    np.savez_compressed(
        ensemble,
        cores=cores.numpy(),
        age_ma=config.anchor_ages,
        history_id=["a", "b", "c", "d"],
        group_id=["a", "b", "c", "d"],
        split=["train", "train", "train", "validation"],
        metadata=np.array(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_semantics": "synthetic",
                    "decoder_sha256": decoder_fingerprint(model, inputs, config),
                }
            )
        ),
    )
    prior = tmp_path / "prior.pt"
    train_latent_prior(ensemble, prior, steps=3, width=8, layers=1, device="cpu")
    body = torch.load(prior, weights_only=True)
    expected = (
        torch.stack([whiten_cores(x, model.prior_cholesky) for x in cores[:3]])
        .flatten(2)
        .mean((0, 1))
    )
    torch.testing.assert_close(body["mean"], expected)

    def potential(core):
        return (core[0, 0, 0] - 3).square()

    unconditioned, _ = sample_diffusion_prior(
        prior, model, inputs, config, potential, draws=2, chains=1, steps=6, guidance=0, seed=21
    )
    conditioned, _ = sample_diffusion_prior(
        prior, model, inputs, config, potential, draws=2, chains=1, steps=6, guidance=0.2, seed=21
    )
    assert (conditioned[:, :, 0, 0, 0] - 3).square().mean() < (
        unconditioned[:, :, 0, 0, 0] - 3
    ).square().mean()
    changed = inputs.clone()
    changed[0, 5] += 0.1
    with pytest.raises(ValueError, match="latent gauge"):
        sample_diffusion_prior(prior, model, changed, config, potential, draws=2, chains=1)
    with pytest.raises(FloatingPointError, match="conditioning energy"):
        sample_diffusion_prior(
            prior,
            model,
            inputs,
            config,
            lambda core: core.sum() * float("nan"),
            draws=2,
            chains=1,
            steps=2,
            guidance=0,
        )
    body["scale"][0] = 0
    invalid = tmp_path / "invalid_prior.pt"
    torch.save(body, invalid)
    with pytest.raises(ValueError, match="normalization"):
        sample_diffusion_prior(invalid, model, inputs, config, potential, draws=2, chains=1)
