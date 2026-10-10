"""Manufactured rotating plate: geometry, training, standalone export and movie."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pygplates = pytest.importorskip("pygplates")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_harmonics")

from norn_earth.config import Config
from norn_earth.data.preparation import load_dataset, prepare_dataset
from norn_earth.geometry.regrid import native_grid
from norn_earth.inference import NornPredictor, export_prediction
from norn_earth.losses.objective import ReconstructionObjective
from norn_earth.models.operator import NornSFNO
from norn_earth.training.trainer import anchor_fields, load_checkpoint, recompute_backward, train
from norn_earth.visualization import export_animation, plate_edges


@pytest.fixture
def rotating_plate(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "rotation.rot").write_text("1 0 90 0 0 0\n1 60 90 0 90 0\n")
    vertices = [(-20, -20), (-20, 20), (20, 20), (20, -20), (-20, -20)]
    continent = pygplates.Feature(pygplates.FeatureType.gpml_closed_continental_boundary)
    continent.set_geometry(pygplates.PolygonOnSphere(vertices))
    continent.set_reconstruction_plate_id(1)
    pygplates.FeatureCollection([continent]).write(str(root / "continents.gpml"))
    line = pygplates.Feature()
    line.set_geometry(
        pygplates.PolylineOnSphere(vertices), pygplates.PropertyName.gpml_center_line_of
    )
    line.set_reconstruction_plate_id(1)
    delegate = pygplates.GpmlPropertyDelegate(
        line.get_feature_id(), pygplates.PropertyName.gpml_center_line_of, pygplates.GmlLineString
    )
    boundary = pygplates.Feature(pygplates.FeatureType.gpml_topological_closed_plate_boundary)
    boundary.set_topological_geometry(
        pygplates.GpmlTopologicalPolygon([pygplates.GpmlTopologicalLineSection(delegate, False)])
    )
    boundary.set_reconstruction_plate_id(1)
    pygplates.FeatureCollection([line, boundary]).write(str(root / "topologies.gpml"))
    lat, lon = native_grid()
    lon, lat = np.meshgrid(lon, lat)
    h = np.where((np.abs((lon + 180) % 360 - 180) < 20) & (np.abs(lat) < 20), 35, 7)
    np.savetxt(root / "modern.dat", np.column_stack([lon.ravel(), lat.ravel(), -h.ravel()]))
    n = 18
    pd.DataFrame(
        {
            "observation_id": [f"r{i}" for i in range(n)],
            "present_lon": np.linspace(-15, 15, n),
            "present_lat": np.zeros(n),
            "age_lower_ma": np.resize([0, 30, 60], n),
            "age_upper_ma": np.resize([0, 30, 60], n),
            "thickness_km": np.linspace(30, 40, n),
        }
    ).to_csv(root / "observations.csv", index=False)
    c = Config()
    c.grid.nlat, c.grid.nlon = 12, 24
    c.model.width, c.model.blocks, c.model.lmax, c.model.mmax = 4, 1, 4, 4
    c.anchor_step_myr = 30
    c.data.observations = str(root / "observations.csv")
    c.data.modern = str(root / "modern.dat")
    c.data.rotations = str(root / "rotation.rot")
    c.data.static_polygons = c.data.continental_polygons = str(root / "continents.gpml")
    c.data.topologies = str(root / "topologies.gpml")
    c.data.geometry_mode = "dynamic"
    c.data.source_semantics = "synthetic"
    c.training.steps, c.training.num_threads, c.training.anchor_batch_size = 2, 2, 2
    c.training.device = "cpu"
    c.output_dir = str(tmp_path / "run")
    c.validate()
    c.data.prepared = str(prepare_dataset(c, tmp_path / "dynamic.npz"))
    torch.set_num_threads(2)
    return c, load_dataset(c.data.prepared, c), root


def test_age_bound_geometry_and_observation_positions(rotating_plate):
    c, data, _ = rotating_plate
    f = data["inputs"]
    assert f.shape == (3, 8, 12, 24)
    np.testing.assert_array_equal(data["feature_ages"], [0, 30, 60])
    assert f[0, 4, 6, 0] == 1 and f[2, 4, 6, 0] == -1
    assert f[0, 4, 6, 6] == -1 and f[2, 4, 6, 6] == 1
    assert f[0, 3, 6, 0] > 0.5 and f[2, 3, 6, 6] > 0.5
    assert data["plate_ids"][0, 6, 0] == data["plate_ids"][2, 6, 6] == 1
    for channel in [0, 1, 2, 7]:
        np.testing.assert_array_equal(f[0, channel], f[2, channel])
    records = data["node_record"]
    expected = data["present_lon"][records] + 1.5 * data["node_age"]
    difference = (data["node_lon"] - expected + 180) % 360 - 180
    np.testing.assert_allclose(difference, 0, atol=1e-5)
    assert data["metadata"]["geometry"]["mode"] == "dynamic"
    changed = copy.deepcopy(c)
    changed.data.geometry_mode = "static"
    with pytest.raises(ValueError, match="geometry_mode"):
        load_dataset(c.data.prepared, changed)


def test_dynamic_two_pass_gradient_equivalence(rotating_plate):
    c, data, _ = rotating_plate
    torch.manual_seed(2)
    full = NornSFNO(c.grid, c.model, c.max_age_ma)
    two = copy.deepcopy(full)
    inputs = torch.tensor(data["inputs"])
    ages = torch.tensor(c.anchor_ages)
    objective = ReconstructionObjective(data, c)
    loss, _ = objective(anchor_fields(full, inputs, ages, 2))
    loss.backward()
    with torch.no_grad():
        cached = anchor_fields(two, inputs, ages, 2)
    leaf = cached.detach().requires_grad_(True)
    objective(leaf)[0].backward()
    recompute_backward(two, inputs, ages, leaf.grad, 2)
    for a, b in zip(full.parameters(), two.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-7)
    with pytest.raises(ValueError, match="per age"):
        anchor_fields(full, inputs[:1], ages, 2)


@pytest.mark.parametrize("extension", [".gif", ".mp4"])
def test_dynamic_standalone_geometry_export_and_movie(rotating_plate, tmp_path, extension):
    c, data, root = rotating_plate
    train(c)
    checkpoint = Path(c.output_dir) / "best.pt"
    moved = tmp_path / "standalone.pt"
    moved.write_bytes(checkpoint.read_bytes())
    for path in root.iterdir():
        path.unlink()
    predictor = NornPredictor(moved, "cpu")
    result, meta = export_prediction(predictor, [0, 30, 60], tmp_path / "maps", ("npz",))
    assert meta["geometry_mode"] == "dynamic"
    assert result["continental_fraction"].shape == (3, 180, 360)
    assert not np.array_equal(result["continental_fraction"][0], result["continental_fraction"][2])
    assert set(np.unique(predictor.predict_geometry([15])["plate_id"])) <= {-1, 1}
    assert (result["thickness_km"] > 0).all()
    assert plate_edges(np.array([[1, 1, 2, 2]])).tolist() == [[True, False, True, False]]
    pytest.importorskip("matplotlib")
    from PIL import Image
    from matplotlib.animation import FFMpegWriter

    if extension == ".mp4" and not FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg unavailable")
    movie = tmp_path / ("movie" + extension)
    report = export_animation(predictor, movie, [0, 30, 60], fps=2, dpi=40)
    assert report["colour_limits_km"] == [0, 80]
    assert report["ages_ma"] == [0, 30, 60]
    if extension == ".gif":
        with Image.open(movie) as image:
            assert image.n_frames == 3
    else:
        assert movie.stat().st_size > 1000
    assert movie.with_suffix(".png").exists()
    assert json.loads(movie.with_suffix(".json").read_text())["geometry_mode"] == "dynamic"
    with pytest.raises(FileExistsError):
        export_animation(predictor, movie, [0, 30])
    with pytest.raises(ValueError):
        predictor.predict_geometry([-1])


def test_legacy_checkpoint_requires_retraining(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"checkpoint_schema": 1}, path)
    with pytest.raises(ValueError, match="retraining"):
        load_checkpoint(path)
