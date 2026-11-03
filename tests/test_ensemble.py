import json

import numpy as np
import pytest
import torch

from norn_earth.models.ensemble import (
    decoder_fingerprint,
    encode_ensemble,
    load_latent_ensemble,
    validate_history_split,
)
from norn_earth.models.factory import build_model
from norn_earth.geometry.regrid import gauss_grid
from norn_earth.training.trainer import train


def test_related_windows_cannot_cross_prior_splits():
    with pytest.raises(ValueError, match="parent history"):
        validate_history_split(
            ["a0", "a1", "b0", "c0"],
            ["a ", " a", "b", "c"],
            ["train", "validation", "train", "train"],
        )
    with pytest.raises(ValueError, match="independent training"):
        validate_history_split(
            ["a0", "a1", "b0"], ["a", "a", "b"], ["train", "train", "validation"]
        )


def test_latent_gauge_and_geometry_are_bound_but_target_cores_are_free(tiny_problem):
    config, data, _ = tiny_problem
    model = build_model(config)
    inputs = torch.tensor(data["inputs"])
    original = decoder_fingerprint(model, inputs, config)
    with torch.no_grad():
        model.cores.add_(1)
    assert original == decoder_fingerprint(model, inputs, config)
    changed = inputs.clone()
    changed[1, 5] += 0.1
    assert original != decoder_fingerprint(model, changed, config)
    with torch.no_grad():
        next(model.spatial.parameters()).add_(0.1)
    assert original != decoder_fingerprint(model, inputs, config)


def test_duplicate_trajectories_are_rejected(tmp_path):
    path = tmp_path / "ensemble.npz"
    np.savez_compressed(
        path,
        cores=np.ones((3, 2, 2, 2)),
        age_ma=[0, 1],
        history_id=["a", "b", "c"],
        group_id=["a", "b", "c"],
        split=["train", "train", "validation"],
        metadata=np.array(
            json.dumps(
                {"schema_version": 1, "source_semantics": "synthetic", "decoder_sha256": "fixture"}
            )
        ),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_latent_ensemble(path)


def test_invalid_encoded_ensemble_is_not_published(tiny_problem, tmp_path):
    config, data, _ = tiny_problem
    train(config, stop_after=1)
    lat, lon, _ = gauss_grid(config.grid.nlat, config.grid.nlon)
    values = np.full((3, config.n_anchors, config.grid.nlat, config.grid.nlon), 8.0)
    raw = tmp_path / "duplicate_fields.npz"
    output = tmp_path / "invalid_encoded.npz"
    np.savez_compressed(
        raw,
        thickness_km=values,
        observed_mask=np.ones_like(values, dtype=bool),
        age_ma=config.anchor_ages,
        latitude=lat[::-1],
        longitude=lon,
        history_id=["a", "b", "c"],
        group_id=["a", "b", "c"],
        split=["train", "train", "validation"],
        metadata=np.array(json.dumps({"source_semantics": "synthetic"})),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        encode_ensemble(tmp_path / "run" / "best.pt", raw, output, steps=1, device="cpu")
    assert not output.exists()
