"""Regression and end-to-end checks for the released reconstruction pipeline."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_harmonics")

from norn_earth.config import config_from_dict, load_config
from norn_earth.data.demo import create_demo, truth
from norn_earth.data.preparation import load_dataset, prepare_dataset
from norn_earth.geometry.regrid import (
    conservative_native_to_gauss,
    gauss_area_weights_km2,
    gauss_grid,
    native_cell_areas_km2,
)
from norn_earth.geometry.sampling import make_query
from norn_earth.inference import NornPredictor, export_prediction
from norn_earth.losses.objective import ReconstructionObjective
from norn_earth.models.operator import NornSFNO
from norn_earth.physics.differentiable import PhysicsObjective
from norn_earth.training.trainer import anchor_fields, recompute_backward, train


@pytest.fixture
def demo(tmp_path):
    config = load_config(
        create_demo(tmp_path, "cpu", nlat=12, nlon=24, width=4, blocks=1, steps=4, anchor_step=30)
    )
    config.training.num_threads = 2
    torch.set_num_threads(2)
    dataset = prepare_dataset(config, tmp_path / "dataset.npz")
    config.data.prepared = str(dataset)
    return config, load_dataset(dataset, config)


def test_periodic_longitudes_and_north_south_orientation():
    lat, lon, _ = gauss_grid(12, 24)
    fields = torch.tensor(
        np.broadcast_to(lat[::-1, None], (3, 12, 24)).copy(),
        dtype=torch.float32,
        requires_grad=True,
    )
    q = make_query([-120, 240, 0, 360], [30, 30, -40, -40], [13.4] * 4, 12, 24, 30, 3)
    actual = q.sample(fields)
    torch.testing.assert_close(actual, torch.tensor([30.0, 30.0, -40.0, -40.0]), atol=1e-5, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(fields.grad).all()
    assert fields.grad.sum().item() == pytest.approx(4)


def test_true_age_interpolation_and_domain_guard():
    fields = torch.stack([torch.full((12, 24), float(a)) for a in (0, 30, 60)])
    query = make_query([0, 0, 0], [0, 0, 0], [0, 13.4, 60], 12, 24, 30, 3)
    torch.testing.assert_close(query.sample(fields), torch.tensor([0.0, 13.4, 60.0]))
    for age in (-0.01, 60.01, float("nan")):
        with pytest.raises(ValueError):
            make_query(0, 0, age, 12, 24, 30, 3)


def test_conservative_remap_preserves_seam_and_quadrature_integral():
    field = np.zeros((180, 360))
    field[:, -1] = 100
    native = (field * native_cell_areas_km2()).sum()
    remapped = conservative_native_to_gauss(field, 24, 48)
    gauss = (remapped * gauss_area_weights_km2(24, 48)).sum()
    assert gauss == pytest.approx(native, rel=2e-13)
    assert remapped[:, 0].max() > 0


def test_configuration_paths_and_unknown_keys(tmp_path):
    config = config_from_dict({"data": {"observations": "observations.csv"}}, tmp_path)
    assert config.data.observations == str(tmp_path / "observations.csv")
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"training": {"stepz": 20}})
    with pytest.raises(ValueError, match="constraints"):
        config_from_dict({"physics": {"budget_weight": 1}})


def test_dataset_split_groups_and_age_weights(demo):
    config, data = demo
    train_groups = set(data["groups"][data["split"] == 0])
    val_groups = set(data["groups"][data["split"] == 1])
    assert train_groups.isdisjoint(val_groups)
    sums = np.bincount(data["node_record"], weights=data["node_weight"], minlength=len(data["y"]))
    np.testing.assert_allclose(sums[data["valid"]], 1, atol=1e-7)
    assert (data["node_lon"] < 0).any()
    assert data["metadata"]["coordinate_mode"] == "present"
    assert data["inputs"].shape == (config.n_anchors, 8, config.grid.nlat, config.grid.nlon)
    np.testing.assert_array_equal(data["feature_ages"], config.anchor_ages)


def test_age_batches_preserve_input_age_pairing_and_backward():
    class PairedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x, age):
            return self.scale * (x[:, :1] + age.reshape(-1, 1, 1, 1))

    model = PairedModel()
    inputs = torch.zeros(4, 8, 4, 8)
    inputs[:, 0] = torch.arange(4)[:, None, None]
    ages = torch.tensor([0.0, 20, 40, 60])
    fields = anchor_fields(model, inputs, ages, 3)
    torch.testing.assert_close(fields[:, 0, 0], torch.tensor([0.0, 42, 84, 126]))
    fields.sum().backward()
    full_gradient = model.scale.grad.clone()
    model.zero_grad()
    recompute_backward(model, inputs, ages, torch.ones_like(fields), 3)
    torch.testing.assert_close(model.scale.grad, full_gradient)


def test_prepared_dataset_rejects_corrupt_feature_age_order(demo, tmp_path):
    config, data = demo
    arrays = {key: value for key, value in data.items() if key != "metadata"}
    for ages in [[0, 30, 30], [0, 30, 61]]:
        meta = copy.deepcopy(data["metadata"])
        meta["feature_ages_ma"] = ages
        arrays["metadata"] = np.array(json.dumps(meta))
        arrays["feature_ages"] = np.asarray(ages, dtype=np.float32)
        path = tmp_path / "corrupt.npz"
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match="age ordering"):
            load_dataset(path, config)
    with pytest.raises(ValueError, match="max_age_ma"):
        config_from_dict({"max_age_ma": 61})


def test_prepared_dataset_rejects_changed_error_scales(demo):
    config, _ = demo
    config.data.modern_sigma_km += 1
    with pytest.raises(ValueError, match="data.modern_sigma_km"):
        load_dataset(config.data.prepared, config)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_objective_two_pass_gradient_equivalence_including_sources(demo, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config, data = demo
    torch.manual_seed(11)
    base = NornSFNO(config.grid, config.model, config.max_age_ma).to(device)
    full, twopass = copy.deepcopy(base), copy.deepcopy(base)
    loss_full = ReconstructionObjective(data, config, device)
    loss_two = ReconstructionObjective(data, config, device)
    inputs = torch.tensor(data["inputs"], device=device)
    ages = torch.tensor(config.anchor_ages, device=device)
    fields = anchor_fields(full, inputs, ages, 2)
    value, _ = loss_full(fields)
    value.backward()
    with torch.no_grad():
        cached = anchor_fields(twopass, inputs, ages, 2)
    leaves = cached.detach().requires_grad_(True)
    recomputed, _ = loss_two(leaves)
    recomputed.backward()
    recompute_backward(twopass, inputs, ages, leaves.grad, 2)
    reference = list(full.parameters()) + list(loss_full.physics.parameters())
    actual = list(twopass.parameters()) + list(loss_two.physics.parameters())
    numerator = sum((a.grad - b.grad).square().sum() for a, b in zip(reference, actual))
    denominator = sum(a.grad.square().sum() for a in reference)
    relative = float(torch.sqrt(numerator / denominator))
    assert relative < 1e-5
    assert torch.isfinite(loss_two.physics.sources.raw.grad).all()


def test_manufactured_forward_time_source_and_volume_sign(demo):
    config, _ = demo
    physics = PhysicsObjective(config.physics.constraints, config)
    lat, lon, _ = gauss_grid(config.grid.nlat, config.grid.nlon)
    lon, lat = np.meshgrid(lon, lat[::-1])
    fields = torch.tensor(
        np.stack([truth(lon, lat, a) for a in config.anchor_ages]),
        dtype=torch.float32,
        requires_grad=True,
    )
    loss, parts = physics(fields)
    assert float(parts["trajectory"]) < 1e-9
    assert float(parts["budget"]) < 1e-9
    assert float(parts["birth"]) < 0.01  # spatial interpolation of analytic field
    loss.backward()
    assert torch.isfinite(fields.grad).all()


def _modify_physics(path, output, mutator):
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    mutator(arrays)
    np.savez_compressed(output, **arrays)


def test_unknown_flux_is_masked_not_filled_with_zero(demo, tmp_path):
    config, _ = demo

    def change(a):
        a["budget_flux_known"][0] = False
        a["budget_relative_normal_velocity_km_myr"][0] = np.nan

    path = tmp_path / "unknown_flux.npz"
    _modify_physics(config.physics.constraints, path, change)
    physics = PhysicsObjective(path, config)
    assert physics.metadata["active_factors"]["budget"] == 1
    loss, _ = physics(torch.full((3, 12, 24), 8.0, requires_grad=True))
    assert torch.isfinite(loss)


def test_rejects_shared_evidence_in_trajectory_and_budget(demo, tmp_path):
    config, _ = demo

    def change(a):
        meta = json.loads(str(a["metadata"]))
        meta["evidence_ids"]["budget"][0] = meta["evidence_ids"]["trajectory"][0]
        a["metadata"] = np.array(json.dumps(meta))

    path = tmp_path / "duplicate.npz"
    _modify_physics(config.physics.constraints, path, change)
    with pytest.raises(ValueError, match="counted in both"):
        PhysicsObjective(path, config)


def test_bounded_sources(demo):
    config, _ = demo
    sources = PhysicsObjective(config.physics.constraints, config).sources
    with torch.no_grad():
        sources.raw.fill_(100)
    assert torch.all(sources() <= sources.upper)
    with torch.no_grad():
        sources.raw.fill_(-100)
    assert torch.all(sources() >= sources.lower)


def test_checkpoint_resume_standalone_inference_and_export(demo, tmp_path):
    config, _ = demo
    report = train(config, stop_after=2)
    assert not report["completed"]
    changed = copy.deepcopy(config)
    changed.training.modern_weight += 1
    with pytest.raises(ValueError, match="training.modern_weight"):
        train(changed, resume=Path(config.output_dir) / "last.pt")
    report = train(config, resume=Path(config.output_dir) / "last.pt")
    assert report["completed"] and report["completed_steps"] == 4
    assert report["final"]["loss"] < report["initial_loss_this_invocation"]
    checkpoint = Path(config.output_dir) / "last.pt"
    uninterrupted = copy.deepcopy(config)
    uninterrupted.output_dir = str(tmp_path / "uninterrupted")
    train(uninterrupted)
    resumed_state = torch.load(checkpoint, weights_only=True)
    full_state = torch.load(Path(uninterrupted.output_dir) / "last.pt", weights_only=True)
    assert resumed_state["completed_steps"] == full_state["completed_steps"] == 4
    assert torch.equal(resumed_state["torch_rng_state"], full_state["torch_rng_state"])
    for key in ("model_state", "physics_state"):
        for name, value in resumed_state[key].items():
            # Parallel CPU kernels can change the last float32 bits between
            # invocations, even with identical RNG and optimizer state.
            torch.testing.assert_close(value, full_state[key][name], rtol=1e-6, atol=1e-8)
    moved = tmp_path / "standalone.pt"
    moved.write_bytes(checkpoint.read_bytes())
    # Inference must need neither original observations nor physics artifact.
    Path(config.data.observations).unlink()
    Path(config.physics.constraints).unlink()
    predictor = NornPredictor(moved, "cpu")
    p = predictor.predict_points([-120, 240], [20, 20], [13.4, 13.4])
    np.testing.assert_array_equal(p[:1], p[1:])
    training_query = make_query([-120, 240], [20, 20], [13.4, 13.4], 12, 24, 30, 3).sample(
        predictor.anchor_fields()
    )
    np.testing.assert_array_equal(p, training_query.numpy())
    result, _ = export_prediction(
        predictor, [0, 13.4, 60], tmp_path / "predictions", ("npz", "dat")
    )
    assert result["thickness_km"].shape == (3, 180, 360)
    assert np.isfinite(result["thickness_km"]).all() and (result["thickness_km"] > 0).all()
    assert np.loadtxt(tmp_path / "predictions/thickness_0Ma.dat").shape == (64800, 3)
