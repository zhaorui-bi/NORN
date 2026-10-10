"""GMT parsing, diagnostic geometry and experiment tags without optional ML."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from norn_earth.cli import main
from norn_earth.config import Config, PhysicsConfig, config_from_dict, load_config
from norn_earth.data.gmt import parse_gmt, polygon_parts, snapshot_paths
from norn_earth.geometry.rasterize import coverage_fraction, rasterize_features
from norn_earth.geometry.regrid import gauss_grid
from norn_earth.geometry.sphere import (
    distance_to_polygon_km,
    lonlat_to_vectors,
    points_in_polygon,
    polygon_signed_area_steradian,
)


HEADER = (
    "# @VGMT1.0 @GMULTIPOLYGON\n# @NANCHOR|TIME|RECONFILE1|PLATEID1|GPGIM_TYPE|NAME|FEATURE_ID\n"
)


def test_metadata_first_attribute_quotes_parts_and_holes(tmp_path):
    path = tmp_path / "topology_0.00Ma.gmt"
    path.write_text(
        HEADER + '>\n# @D0|0|rotation.rot|1|gpml:TopologicalNetwork|"A | B"|feature-1\n'
        "# @P\n-20 -20\n20 -20\n20 20\n-20 20\n-20 -20\n"
        ">\n# @H\n-5 -5\n5 -5\n5 5\n-5 5\n-5 -5\n"
        ">\n# @P\n70 -10\n90 -10\n90 10\n70 10\n70 -10\n"
    )
    snapshot = parse_gmt(path)
    item = snapshot["features"][0]
    assert snapshot["fields"][0] == "ANCHOR"
    assert item["meta"]["ANCHOR"] == "0"
    assert item["name"] == "A | B"
    assert item["ring_roles"] == ["exterior", "hole", "exterior"]
    assert [len(ring) for ring in item["rings"]] == [4, 4, 4]
    assert [len(holes) for _, holes in polygon_parts(item)] == [1, 0]
    result = rasterize_features([item], 36, 72, 1)[0]["coverage"]
    lat, lon, _ = gauss_grid(36, 72)
    at = np.argmin(abs(lat))
    assert result[at, 0] == 0  # hole
    assert result[at, 3] == 1  # first exterior
    assert result[at, 16] == 1  # second exterior
    assert result[at, 36] == 0  # no antipodal ghost
    assert lon[16] == 80


@pytest.mark.parametrize("coordinates", ["nan 0", "0 91", "broken 0", "0"])
def test_malformed_coordinates_are_not_silently_dropped(tmp_path, coordinates):
    path = tmp_path / "topology_0.00Ma.gmt"
    path.write_text(HEADER + "# @D0|0|rotation.rot|1|x|name|id\n# @P\n" + coordinates)
    with pytest.raises(ValueError, match=r"topology_0.00Ma.gmt:\d+"):
        parse_gmt(path)


def test_exact_snapshot_ages_and_domain(tmp_path):
    for age in (0, 30, 60, 66):
        (tmp_path / f"topology_{age:.2f}Ma.gmt").write_text(HEADER)
    assert list(snapshot_paths(tmp_path, [0, 30, 60])) == [0, 30, 60]
    with pytest.raises(ValueError, match="exactly one"):
        snapshot_paths(tmp_path, [15])
    with pytest.raises(ValueError, match="0–60"):
        snapshot_paths(tmp_path, [66])
    (tmp_path / "other_0.0Ma.gmt").write_text(HEADER)
    with pytest.raises(ValueError, match="Duplicate"):
        snapshot_paths(tmp_path, [0])


def test_unmarked_multiple_rings_are_not_guessed_to_be_holes():
    ring = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]])
    with pytest.raises(ValueError, match="explicit @P"):
        polygon_parts({"rings": [ring, ring + 30]})
    with pytest.raises(ValueError, match="following @H"):
        polygon_parts({"rings": [ring], "ring_roles": ["hole"]})


def test_concavity_orientation_and_antipodal_raster_coverage():
    lon, lat = np.array([-20, 20, 20, 0, 0, -20]), np.array([-20, -20, 0, 0, 20, 20])
    vertices = lonlat_to_vectors(lon, lat)
    area = polygon_signed_area_steradian(vertices)
    assert 0 < area < 1
    assert polygon_signed_area_steradian(vertices[::-1]) == pytest.approx(-area)
    for lo, la in ((lon, lat), (lon[::-1], lat[::-1])):
        np.testing.assert_array_equal(
            points_in_polygon([-10, 10, 170], [10, 10, -10], lo, la), [True, False, False]
        )
    coverage = coverage_fraction([-10, 10, 10, -10], [-10, -10, 10, 10], 36, 72, 1)
    mid = np.argmin(abs(gauss_grid(36, 72)[0]))
    assert coverage[mid, 0] == 1 and coverage[mid, 36] == 0


def test_minor_arc_distance_does_not_extend_short_edges():
    distance = distance_to_polygon_km([80], [0], [-10, 10, 0], [0, 0, -10])[0]
    assert distance == pytest.approx(6371 * np.radians(70), abs=1e-6)


def test_tags_only_change_consistency_terms():
    baseline = Config()
    baseline.output_dir = "runs/common"
    enabled = baseline.with_tag("physics", "priors.npz")
    disabled = enabled.with_tag("no_physics")
    for tag in (enabled, disabled):
        assert tag.model == baseline.model and tag.data == baseline.data
        assert tag.training.seed == baseline.training.seed
        assert tag.training.steps == baseline.training.steps
        for key, value in vars(baseline.training).items():
            if key not in ("objective", "temporal_smooth_weight"):
                assert getattr(tag.training, key) == value
    assert enabled.physics.trajectory_weight == enabled.physics.budget_weight == 0.05
    assert disabled.physics == PhysicsConfig()
    assert enabled.output_dir == "runs/common/physics"
    assert disabled.output_dir == "runs/common/no_physics"
    assert baseline.physics == PhysicsConfig()  # no mutation
    with pytest.raises(ValueError, match="requires"):
        baseline.with_tag("physics")
    with pytest.raises(ValueError, match="requires --tag physics"):
        baseline.with_tag("no_physics", "priors.npz")


def test_cli_tags_dry_run_and_separate_directories(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output_dir": "run"}))
    assert main(["train", "--config", str(path), "--tag", "no_physics", "--dry-run"]) == 0
    pure = json.loads(capsys.readouterr().out)
    assert pure["training"]["objective"] == "data_only"
    assert pure["output_dir"] == str(tmp_path / "run" / "no_physics")
    assert (
        main(
            [
                "train",
                "--config",
                str(path),
                "--tag",
                "physics",
                "--physics-constraints",
                str(tmp_path / "priors.npz"),
                "--dry-run",
            ]
        )
        == 0
    )
    physical = json.loads(capsys.readouterr().out)
    assert physical["data"] == pure["data"] and physical["model"] == pure["model"]
    assert physical["training"]["selection_metric"] == "validation_mae"
    assert main(["train", "--config", str(path), "--tag", "physics", "--dry-run"]) == 1
    assert "requires" in capsys.readouterr().err


def test_gmt_sources_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="both GMT"):
        config_from_dict({"data": {"geometry_source": "gmt"}})
    with pytest.raises(ValueError, match="refusing to ignore"):
        config_from_dict({"data": {"plate_gmt_dir": "plates"}})


def test_public_pair_uses_identical_geometry_backbone_loss_and_selection():
    root = Path(__file__).resolve().parents[1]
    baseline = load_config(root / "configs/a40.json")
    physical = load_config(root / "configs/a40_physics.json")
    assert baseline.data.geometry_source == physical.data.geometry_source == "gmt"
    assert baseline.data == physical.data and baseline.model == physical.model
    a, b = copy.deepcopy(vars(baseline.training)), copy.deepcopy(vars(physical.training))
    a.pop("objective")
    b.pop("objective")
    assert a == b
    assert baseline.experiment_tag == "no_physics" and physical.experiment_tag == "physics"
