import copy
import json

import numpy as np
import pytest
import torch

from norn_earth.inference import NornPredictor, export_prediction
from norn_earth.models.posterior import evaluate_posterior, sample_posterior
from norn_earth.training.trainer import load_checkpoint, train


def test_checkpoint_resume_inference_and_sampling(tiny_problem, tmp_path):
    config, _, dataset = tiny_problem
    train(config, stop_after=1)
    train(config, resume=str(tmp_path / "run" / "last.pt"))
    predictor = NornPredictor(tmp_path / "run" / "best.pt", "cpu")
    predicted = predictor.predict_points(lon=[-1, 359], lat=[0, 0], age=[13.4, 13.4])
    np.testing.assert_allclose(predicted[0], predicted[1])
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    _, manifest = export_prediction(predictor, [0, 13.4], tmp_path / "prediction")
    assert manifest["interval_kind"] == "point_estimate"
    result = sample_posterior(
        str(tmp_path / "run" / "best.pt"),
        dataset,
        tmp_path / "posterior",
        draws=3,
        chains=2,
        burnin=2,
        thin=1,
        seed=7,
        device="cpu",
    )
    assert not result["calibrated_uncertainty"]
    predictive = evaluate_posterior(
        str(tmp_path / "run" / "best.pt"),
        dataset,
        tmp_path / "posterior" / "latent_samples.npz",
        tmp_path / "predictive.json",
        device="cpu",
    )
    assert predictive["interval_kind"] == "posterior_predictive_observation"
    assert np.isfinite(predictive["posterior_predictive_nll"])
    with np.load(tmp_path / "posterior" / "posterior_summary.npz") as b:
        assert np.all(b["thickness_lower_km"] <= b["thickness_upper_km"])
        assert json.loads(str(b["metadata"]))["latitude_order"] == "north_to_south"
    with pytest.raises(FileExistsError):
        sample_posterior(
            str(tmp_path / "run" / "best.pt"),
            dataset,
            tmp_path / "posterior",
            draws=3,
            device="cpu",
        )


def test_legacy_sfno_checkpoint_without_new_config_keys(tiny_problem, tmp_path):
    config, _, _ = tiny_problem
    config = copy.deepcopy(config)
    config.model.backbone = "sfno"
    config.training.latent_gp_weight = 0
    config.training.steps = 1
    config.output_dir = str(tmp_path / "sfno")
    train(config)
    path = tmp_path / "sfno" / "best.pt"
    body = torch.load(path, weights_only=True)
    for key in (
        "spatial_rank",
        "context_rank",
        "latent_lengthscale_ma",
        "latent_amplitude",
        "latent_scale_km",
    ):
        del body["config"]["model"][key]
    for key in ("group_column", "test_age_window_ma", "test_max_age_width_ma"):
        del body["config"]["data"][key]
    del body["config"]["training"]["latent_gp_weight"]
    torch.save(body, path)
    _, restored, model, inputs = load_checkpoint(path)
    assert restored.model.backbone == "sfno"
    assert torch.isfinite(model(inputs[:1], torch.tensor([0.0]))).all()
