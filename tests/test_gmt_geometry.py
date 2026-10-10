"""Manufactured GMT/GPML registration, paired tags and portable checkpoints."""

import builtins
import copy
import json
from pathlib import Path

import numpy as np
import pytest

pygplates = pytest.importorskip("pygplates")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_harmonics")

from norn_earth.data.dynamic_geometry import material_reference, partition_ids
from norn_earth.data.kinematics_engine import KinematicsEngine
from norn_earth.data.gmt import parse_gmt
from norn_earth.data.gmt_geometry import GMTGeometrySource, snapshot_features
from norn_earth.data.preparation import load_dataset, prepare_dataset
from norn_earth.inference import NornPredictor, export_prediction
from norn_earth.losses.objective import ReconstructionObjective
from norn_earth.training.trainer import train
import test_dynamic_geometry

# Reuse the manufactured fixture, without any local raw/model data dependency.
gpml_plate = test_dynamic_geometry.rotating_plate


def write_export(path, items, age, rotation_name):
    text = [
        "# @VGMT1.0 @GMULTIPOLYGON",
        "# @NANCHOR|TIME|RECONFILE1|PLATEID1|GPGIM_TYPE|NAME|FEATURE_ID",
    ]
    for item in items:
        feature = item.get_feature()
        network = isinstance(item, pygplates.ResolvedTopologicalNetwork)
        kind = "gpml:TopologicalNetwork" if network else "gpml:TopologicalClosedPlateBoundary"
        text += [
            ">",
            f"# @D0|{age:g}|{rotation_name}|{feature.get_reconstruction_plate_id()}|{kind}|example|{feature.get_feature_id().get_string()}",
            "# @P",
        ]
        text += [
            f"{lon:.17g} {lat:.17g}"
            for lat, lon in (
                p.to_lat_lon() for p in item.get_resolved_boundary().get_exterior_ring_points()
            )
        ]
    path.write_text("\n".join(text) + "\n")


@pytest.fixture
def rotating_gmt(gpml_plate, tmp_path):
    config, _, root = gpml_plate
    # Add a moving network outside the continent. Its outline is an input,
    # not a velocity field or a historical thickness observation.
    line = pygplates.Feature()
    line.set_geometry(
        pygplates.PolylineOnSphere([(-10, 35), (-10, 50), (10, 50), (10, 35), (-10, 35)]),
        pygplates.PropertyName.gpml_center_line_of,
    )
    line.set_reconstruction_plate_id(1)
    delegate = pygplates.GpmlPropertyDelegate(
        line.get_feature_id(), pygplates.PropertyName.gpml_center_line_of, pygplates.GmlLineString
    )
    network = pygplates.Feature(pygplates.FeatureType.gpml_topological_network)
    network.set_topological_geometry(
        pygplates.GpmlTopologicalNetwork([pygplates.GpmlTopologicalLineSection(delegate, False)])
    )
    network.set_reconstruction_plate_id(1)
    features = list(pygplates.FeatureCollection(str(root / "topologies.gpml"))) + [line, network]
    pygplates.FeatureCollection(features).write(str(root / "topologies.gpml"))
    config.data.geometry_source = "gmt"
    config.data.plate_gmt_dir = str(root / "plates")
    config.data.deformation_gmt_dir = str(root / "networks")
    Path(config.data.plate_gmt_dir).mkdir()
    Path(config.data.deformation_gmt_dir).mkdir()
    rotation = pygplates.RotationModel(config.data.rotations)
    for age in config.anchor_ages:
        items = []
        pygplates.resolve_topologies(features, rotation, items, float(age))
        plates = [p for p in items if not isinstance(p, pygplates.ResolvedTopologicalNetwork)]
        networks = [p for p in items if isinstance(p, pygplates.ResolvedTopologicalNetwork)]
        assert plates and networks
        for directory, selected in (
            (config.data.plate_gmt_dir, plates),
            (config.data.deformation_gmt_dir, networks),
        ):
            write_export(
                Path(directory) / f"topology_{age:.2f}Ma.gmt", selected, float(age), "rotation.rot"
            )
    config.data.prepared = str(prepare_dataset(config, tmp_path / "gmt.npz"))
    return config, load_dataset(config.data.prepared, config), root, features, rotation


def test_gmt_age_bound_inputs_and_observation_coordinates(rotating_gmt):
    config, data, _, _, _ = rotating_gmt
    assert data["inputs"].shape == (3, 8, 12, 24)
    geometry = data["metadata"]["geometry"]
    assert geometry["source"] == "gmt"
    assert (
        len(
            [
                key
                for key in geometry["source_sha256"]
                if key.startswith(("plate_gmt/", "deformation_gmt/"))
            ]
        )
        == 6
    )
    assert all(item["frame_checked_features"] == 2 for item in geometry["anchors"])
    assert max(item["frame_max_error_km"] for item in geometry["anchors"]) < 1e-8
    assert not np.array_equal(data["inputs"][0, 4], data["inputs"][2, 4])
    assert not np.array_equal(data["inputs"][0, 5], data["inputs"][2, 5])
    assert not np.array_equal(data["inputs"][0, 6], data["inputs"][2, 6])
    assert data["inputs"][:, 6].sum() > 0
    np.testing.assert_allclose(
        (data["node_lon"] - data["present_lon"][data["node_record"]] - 1.5 * data["node_age"] + 180)
        % 360
        - 180,
        0,
        atol=1e-5,
    )
    changed = copy.deepcopy(config)
    changed.data.geometry_source = "gpml"
    with pytest.raises(ValueError, match="geometry_source"):
        load_dataset(config.data.prepared, changed)


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("age", "TIME"),
        ("anchor", "ANCHOR"),
        ("rotation", "rotation reference"),
        ("vertices", "mismatch"),
        ("edges", "edge differs"),
    ],
)
def test_mismatched_or_corrupted_snapshot_fails(rotating_gmt, mutation, diagnostic):
    config, _, _, features, rotation = rotating_gmt
    path = Path(config.data.plate_gmt_dir) / "topology_30.00Ma.gmt"
    text = path.read_text()
    if mutation == "age":
        text = text.replace("# @D0|30|", "# @D0|60|")
    elif mutation == "anchor":
        text = text.replace("# @D0|", "# @D1|")
    elif mutation == "rotation":
        text = text.replace("rotation.rot", "other.rot")
    else:
        lines = text.splitlines()
        at = lines.index("# @P") + 1
        if mutation == "vertices":
            lon, lat = map(float, lines[at].split())
            lines[at] = f"{lon + 1} {lat}"
        else:
            lines[at + 1], lines[at + 2] = lines[at + 2], lines[at + 1]
        text = "\n".join(lines)
    path.write_text(text)
    resolved = []
    pygplates.resolve_topologies(features, rotation, resolved, 30)
    with pytest.raises(ValueError, match=diagnostic):
        GMTGeometrySource(config).at_age(30, resolved, pygplates)


def test_same_filename_different_rotation_frame_is_rejected(rotating_gmt, tmp_path):
    config, _, _, features, _ = rotating_gmt
    wrong = tmp_path / "rotation.rot"  # same name, different absolute frame
    wrong.write_text("1 0 90 0 0 0\n1 60 90 0 100 0\n")
    changed = copy.deepcopy(config)
    changed.data.rotations = str(wrong)
    resolved = []
    pygplates.resolve_topologies(features, pygplates.RotationModel(str(wrong)), resolved, 30)
    with pytest.raises(ValueError, match="reference-frame or geometry mismatch"):
        GMTGeometrySource(changed).at_age(30, resolved, pygplates)


def test_empty_network_export_cannot_hide_resolved_networks(rotating_gmt):
    config, _, _, features, rotation = rotating_gmt
    path = Path(config.data.deformation_gmt_dir) / "topology_30.00Ma.gmt"
    path.write_text("\n".join(path.read_text().splitlines()[:2]))
    resolved = []
    pygplates.resolve_topologies(features, rotation, resolved, 30)
    with pytest.raises(ValueError, match="Incomplete"):
        GMTGeometrySource(config).at_age(30, resolved, pygplates)


def test_native_gmt_parts_holes_and_dateline(tmp_path):
    path = tmp_path / "topology_0.00Ma.gmt"
    path.write_text(
        "# @VGMT1.0 @GMULTIPOLYGON\n# @NPLATEID1|FEATURE_ID\n# @D7|id\n"
        "# @P\n170 -20\n-170 -20\n-170 20\n170 20\n"
        ">\n# @H\n175 -5\n-175 -5\n-175 5\n175 5\n"
        ">\n# @P\n60 -10\n80 -10\n80 10\n60 10\n"
    )
    features, _ = snapshot_features(parse_gmt(path), pygplates)
    points = [
        pygplates.PointOnSphere(lat, lon) for lat, lon in [(0, 180), (10, 180), (0, 70), (0, 0)]
    ]
    np.testing.assert_array_equal(
        partition_ids(features, pygplates.RotationModel([]), points, pygplates), [-1, 7, 7, -1]
    )


def test_zero_area_network_keeps_boundary_but_not_area(tmp_path):
    path = tmp_path / "topology_0.00Ma.gmt"
    path.write_text(
        "# @VGMT1.0 @GMULTIPOLYGON\n# @NPLATEID1|FEATURE_ID\n# @D7|rift\n"
        "# @P\n0 0\n0 5\n0 10\n0 5\n"
    )
    features, polygons = snapshot_features(parse_gmt(path), pygplates)
    assert not features and len(polygons) == 1 and polygons[0].get_area() == 0


def test_missing_plate_circuit_cannot_silently_use_identity(rotating_gmt, monkeypatch):
    config, data, _, _, rotation = rotating_gmt
    engine = KinematicsEngine(config.data.rotations, config.data.static_polygons)
    monkeypatch.setattr(engine, "plate_ids", lambda lon, lat: np.full(len(lon), 999999))
    _, _, supported = engine.positions_at([0], [0], [30])
    assert not supported.any()
    values, supported = material_reference(
        {"H_km": data["modern_H"]},
        [pygplates.PointOnSphere(0, 0)],
        np.array([999999]),
        np.array([False]),
        rotation,
        30,
    )
    assert not supported.any() and values[0] == 0


def priors(path):
    ages = np.array([[60, 30, 0]], np.float32)
    lon = np.array([[90, 45, 0]], np.float32)
    budget_lon = lon[:, :, None] + 2
    meta = {
        "schema_version": 1,
        "time_coordinate": "forward_tau=max_age-age",
        "kind": "manufactured",
        "evidence_ids": {"trajectory": ["t"], "budget": ["b"]},
    }
    np.savez_compressed(
        path,
        metadata=np.array(json.dumps(meta)),
        trajectory_active=np.ones(1, bool),
        trajectory_source_known=np.ones(1, bool),
        trajectory_lon=lon,
        trajectory_lat=np.zeros_like(lon),
        trajectory_age=ages,
        trajectory_divergence_per_myr=np.zeros_like(lon),
        trajectory_source_basis=np.zeros(lon.shape + (0,)),
        trajectory_sigma_km=np.array([10], np.float32),
        budget_active=np.ones(1, bool),
        budget_source_known=np.ones(1, bool),
        budget_flux_known=np.ones(1, bool),
        budget_lon=budget_lon,
        budget_lat=np.zeros_like(budget_lon),
        budget_age=ages,
        budget_area_km2=np.full_like(budget_lon, 1000),
        budget_source_basis=np.zeros(budget_lon.shape + (0,)),
        budget_boundary_lon=budget_lon,
        budget_boundary_lat=np.zeros_like(budget_lon),
        budget_relative_normal_velocity_km_myr=np.zeros_like(budget_lon),
        budget_boundary_length_km=np.ones_like(budget_lon),
        budget_reference_volume_km3=np.array([35000], np.float32),
        budget_sigma_normalized=np.array([0.2], np.float32),
    )
    return path


def test_no_physics_does_not_even_import_physics(rotating_gmt, monkeypatch):
    config, data, _, _, _ = rotating_gmt
    original = builtins.__import__

    def forbidden(name, *args, **kwargs):
        if name.startswith("physics") or ".physics" in name:
            raise AssertionError("no_physics must not import physics modules")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden)
    objective = ReconstructionObjective(data, config.with_tag("no_physics"))
    loss, parts = objective(torch.full((3, 12, 24), 30.0, requires_grad=True))
    loss.backward()
    assert objective.physics is None and "physics_loss" not in parts


def test_physics_terms_are_added_without_changing_data_loss(rotating_gmt, tmp_path):
    config, data, _, _, _ = rotating_gmt
    pure = ReconstructionObjective(data, config.with_tag("no_physics"))
    enabled = config.with_tag("physics", priors(tmp_path / "priors.npz"))
    physical = ReconstructionObjective(data, enabled)
    fields = torch.stack(
        [torch.full((12, 24), value) for value in (25.0, 30.0, 40.0)]
    ).requires_grad_(True)
    base_loss, base_parts = pure(fields)
    loss, parts = physical(fields)
    for key in ("observation_nll", "modern_nll"):
        torch.testing.assert_close(parts[key], base_parts[key], rtol=0, atol=0)
    assert parts["physics_loss"] > 0
    torch.testing.assert_close(loss, base_loss + parts["physics_loss"])
    loss.backward()
    assert torch.isfinite(fields.grad).all()


def test_tagged_training_portable_checkpoints_and_cross_tag_resume_rejected(rotating_gmt, tmp_path):
    config, data, root, _, _ = rotating_gmt
    a = config.with_tag("no_physics")
    b = config.with_tag("physics", priors(tmp_path / "priors.npz"))
    train(a)
    train(b)
    paths = [Path(c.output_dir) / "best.pt" for c in (a, b)]
    checkpoints = [torch.load(path, weights_only=True) for path in paths]
    assert checkpoints[0]["experiment_tag"] == "no_physics"
    assert "physics_state" not in checkpoints[0]
    assert checkpoints[1]["experiment_tag"] == "physics"
    assert "physics_state" in checkpoints[1]
    torch.testing.assert_close(checkpoints[0]["inputs"], checkpoints[1]["inputs"], rtol=0, atol=0)
    torch.testing.assert_close(
        checkpoints[0]["geometry"]["plate_ids"],
        checkpoints[1]["geometry"]["plate_ids"],
        rtol=0,
        atol=0,
    )
    changed = copy.deepcopy(b)
    changed.training.steps += 1
    with pytest.raises(ValueError, match="changed physics"):
        train(changed, resume=paths[0])
    # Remove all source artifacts: inference must use embedded age-bound inputs.
    import shutil

    shutil.rmtree(root)
    Path(config.data.prepared).unlink()
    (tmp_path / "priors.npz").unlink()
    for index, path in enumerate(paths):
        predictor = NornPredictor(path, "cpu")
        result, meta = export_prediction(predictor, [0, 15, 60], tmp_path / f"export_{index}")
        assert meta["geometry_source"] == "gmt"
        assert meta["experiment_tag"] == ("no_physics" if index == 0 else "physics")
        assert (result["thickness_km"] > 0).all()
        assert not np.array_equal(
            result["continental_fraction"][0], result["continental_fraction"][2]
        )


def test_rigid_prior_preparation_rejects_changed_model(rotating_gmt, tmp_path):
    from norn_earth.physics.preparation import prepare_rigid_priors

    config, _, _, _, _ = rotating_gmt
    changed = copy.deepcopy(config)
    path = tmp_path / "wrong.rot"
    path.write_text("1 0 90 0 0 0\n1 60 90 0 100 0\n")
    changed.data.rotations = str(path)
    with pytest.raises(ValueError, match="differs from the frozen dataset model"):
        prepare_rigid_priors(changed, config.data.prepared, tmp_path / "priors.npz")


def test_rigid_priors_cannot_be_reused_from_other_geometry(rotating_gmt, tmp_path):
    config, data, _, _, _ = rotating_gmt
    path = priors(tmp_path / "priors.npz")
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    meta = json.loads(str(arrays["metadata"]))
    meta.update(kind="model_derived_rigid_source_free_prior", geometry_source="gpml")
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="different geometry_source"):
        ReconstructionObjective(data, config.with_tag("physics", path))
