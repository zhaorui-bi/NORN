"""Pure-data paths, deterministic model gradients and validation-only selection."""

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_harmonics")

from norn_earth.config import PhysicsConfig, config_from_dict, load_config
from norn_earth.data.demo import create_demo
from norn_earth.data.preparation import load_dataset, prepare_dataset
from norn_earth.inference import NornPredictor, export_prediction
from norn_earth.losses.objective import ReconstructionObjective
from norn_earth.models.operator import NornSFNO
from norn_earth.training.trainer import anchor_fields, recompute_backward, train, validation_score


@pytest.fixture
def pure(tmp_path):
    c = load_config(
        create_demo(tmp_path, "cpu", nlat=12, nlon=24, width=4, blocks=1, steps=4, anchor_step=30)
    )
    c.physics = PhysicsConfig()
    c.training.objective = "data_only"
    c.training.selection_metric = "validation_mae"
    c.training.observation_huber_weight = 1.0
    c.training.anchor_batch_size = 2
    c.training.num_threads = 2
    c.model.backbone = "gated_sfno"
    c.model.head_mode = "input_residual"
    c.model.time_frequency_bands = 4
    c.data.prepared = str(prepare_dataset(c, tmp_path / "pure.npz"))
    torch.set_num_threads(2)
    return c, load_dataset(c.data.prepared, c)


@pytest.mark.parametrize(
    "payload",
    [
        {"physics": {"constraints": "must-not-be-loaded.npz"}},
        {"physics": {"trajectory_weight": 1}},
        {"physics": {"budget_weight": 1}},
        {"physics": {"source_prior_weight": 1}},
        {"training": {"temporal_smooth_weight": 1}},
    ],
)
def test_data_only_forbids_any_consistency_terms(payload):
    body = copy.deepcopy(payload)
    body.setdefault("training", {})["objective"] = "data_only"
    with pytest.raises(ValueError, match="data_only"):
        config_from_dict(body)


def test_public_defaults_are_pure_ml():
    c = config_from_dict({})
    assert c.training.objective == "data_only"
    assert c.training.selection_metric == "validation_mae"
    assert c.physics.constraints is None


def test_pure_loss_does_not_instantiate_or_call_physics(pure, monkeypatch):
    c, data = pure
    import norn_earth.physics.differentiable as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure baseline must not construct physics")

    monkeypatch.setattr(module, "PhysicsObjective", forbidden)
    objective = ReconstructionObjective(data, c)
    fields = torch.full((3, 12, 24), 30.0, requires_grad=True)
    loss, parts = objective(fields)
    assert objective.physics is None
    assert not any("physics" in key or "temporal" in key for key in parts)
    expected = (
        c.training.observation_weight
        * (
            parts["observation_nll"]
            + c.training.observation_huber_weight * parts["observation_huber"]
        )
        + c.training.modern_weight * parts["modern_nll"]
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(fields.grad).all()


def test_test_labels_cannot_change_pure_training_loss(pure):
    c, data = pure
    modified = {
        key: value.copy() if isinstance(value, np.ndarray) else value for key, value in data.items()
    }
    index = np.flatnonzero(modified["valid"] & (modified["split"] == 0))[0]
    modified["split"][index] = 2
    reference = ReconstructionObjective(modified, c)
    changed = copy.deepcopy(modified)
    changed["y"][index] += 1000
    altered = ReconstructionObjective(changed, c)
    fields = torch.full((3, 12, 24), 30.0)
    torch.testing.assert_close(reference(fields)[0], altered(fields)[0], rtol=0, atol=0)


def test_huber_is_one_age_marginal_error_per_record(pure):
    c, data = pure
    objective = ReconstructionObjective(data, c)
    fields = torch.stack([torch.full((12, 24), float(h)) for h in (25, 35, 50)])
    expected = (objective.node_predictions(fields) * objective.weights).sum(-1)
    mask = objective.masks["train"]
    manual = (
        torch.nn.functional.smooth_l1_loss(
            expected[mask], objective.y[mask], beta=c.training.huber_beta_km
        )
        / c.training.huber_beta_km
    )
    torch.testing.assert_close(objective(fields)[1]["observation_huber"], manual)


@pytest.mark.parametrize("head", ["absolute", "input_residual"])
def test_gated_backbone_two_pass_exact_chain_rule(pure, head):
    c, data = pure
    c.model.head_mode = head
    torch.manual_seed(3)
    full = NornSFNO(c.grid, c.model, c.max_age_ma)
    two = copy.deepcopy(full)
    objective = ReconstructionObjective(data, c)
    inputs = torch.tensor(data["inputs"])
    ages = torch.tensor(c.anchor_ages)
    value, _ = objective(anchor_fields(full, inputs, ages, 2))
    value.backward()
    with torch.no_grad():
        cache = anchor_fields(two, inputs, ages, 2)
    leaves = cache.detach().requires_grad_(True)
    objective(leaves)[0].backward()
    recompute_backward(two, inputs, ages, leaves.grad, 2)
    for a, b in zip(full.parameters(), two.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=2e-7)
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm2d)) for m in two.modules())


def test_residual_head_uses_reference_or_missing_reference_fallback(pure):
    c, data = pure
    model = NornSFNO(c.grid, c.model, c.max_age_ma)
    torch.nn.init.zeros_(model.head[-1].weight)
    torch.nn.init.zeros_(model.head[-1].bias)
    x = torch.tensor(data["inputs"])
    x[1, 3] = 0
    expected = torch.where(x[:, 3] > 0, x[:, 3] * 40, c.model.initial_thickness_km)
    torch.testing.assert_close(
        model(x, torch.tensor(c.anchor_ages))[:, 0], expected, rtol=1e-6, atol=2e-5
    )


def test_selection_uses_validation_mae_not_modern_or_nll(pure):
    c, _ = pure
    metrics = {"nll": 99.0, "age_marginal_mean_mae_km": 7.5}
    assert validation_score(c.training, metrics, {"modern_nll": 10000.0}, 9999.0) == 7.5
    with pytest.raises(ValueError, match="nonempty validation"):
        validation_score(c.training, None, {}, 1.0)


def test_pure_checkpoint_resume_and_standalone_prediction(pure, tmp_path):
    c, _ = pure
    first = train(c, stop_after=2)
    assert not first["completed"]
    final = train(c, resume=Path(c.output_dir) / "last.pt")
    assert final["completed"]
    assert not any("physics" in key or "source_coefficients" in key for key in final)
    path = Path(c.output_dir) / "last.pt"
    checkpoint = torch.load(path, weights_only=True)
    assert "physics_state" not in checkpoint
    assert "physics" not in checkpoint["provenance"]
    assert checkpoint["config"]["training"]["objective"] == "data_only"
    assert (
        checkpoint["best_validation_score"] <= final["final"]["validation_age_marginal_mean_mae_km"]
    )
    changed = copy.deepcopy(c)
    changed.training.observation_huber_weight = 0
    with pytest.raises(ValueError, match="observation_huber_weight"):
        train(changed, resume=path)
    moved = tmp_path / "standalone.pt"
    moved.write_bytes(path.read_bytes())
    Path(c.data.prepared).unlink()
    Path(c.data.observations).unlink()
    predictor = NornPredictor(moved, "cpu")
    values = predictor.predict_points([-120, 240], [20, 20], [13.4, 13.4])
    np.testing.assert_array_equal(values[:1], values[1:])
    assert (values > 0).all()
    maps, metadata = export_prediction(predictor, [0, 13.4, 60], tmp_path / "maps", ("npz",))
    assert metadata["objective"] == "data_only" and "physics" not in metadata
    assert maps["thickness_km"].shape == (3, 180, 360)
    assert (tmp_path / "maps" / "thickness.npz").is_file()
