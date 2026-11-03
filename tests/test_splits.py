import copy

import numpy as np
import pandas as pd
import pytest

from norn_earth.config import Config
from norn_earth.data.preparation import _split_records, load_dataset, prepare_dataset
from norn_earth.utils.hashing import sha256_file


def test_whole_interval_excluded_even_when_midpoint_outside_window():
    config = Config()
    config.data.test_age_window_ma = [20, 30]
    config.data.validation_fraction = 0
    table = pd.DataFrame(
        {
            "observation_id": list("abcdef"),
            "present_lon": np.arange(6) * 20,
            "present_lat": np.zeros(6),
            "age_lower_ma": [0, 22, 0, 31, 29, 55],
            "age_upper_ma": [5, 23, 21, 45, 10, 60],
        }
    )
    split, _ = _split_records(table, config)
    # Row c's midpoint is 10.5, but its support intersects the test window.
    # Row e also overlaps after normalizing reversed bounds.
    np.testing.assert_array_equal(split, [0, 2, -1, 0, -1, 0])


def test_validation_uses_declared_study_groups():
    config = Config()
    config.data.group_column = "study_id"
    config.data.validation_fraction = 0.5
    table = pd.DataFrame(
        {
            "present_lon": [0, 120, 10, 130],
            "present_lat": [0, 30, 60, -60],
            "study_id": ["a", "a", "b", "b"],
        }
    )
    split, _ = _split_records(table, config)
    assert split[0] == split[1]
    assert split[2] == split[3]
    assert split[0] != split[2]
    with pytest.raises(ValueError, match="group_column"):
        _split_records(table.drop(columns="study_id"), config)


def test_prepared_split_protocol_cannot_be_silently_changed(tiny_problem):
    config, _, path = tiny_problem
    changed = copy.deepcopy(config)
    changed.data.test_age_window_ma = [20, 30]
    with pytest.raises(ValueError, match="test_age_window_ma"):
        load_dataset(path, changed)
    changed = copy.deepcopy(config)
    changed.data.holdout_csv = "unrelated.csv"
    with pytest.raises(ValueError, match="holdout_csv"):
        load_dataset(path, changed)


def test_relocated_holdout_does_not_replace_dataset_hash(tiny_problem, tmp_path):
    config, _, _ = tiny_problem
    first = tmp_path / "holdout.csv"
    second = tmp_path / "moved.csv"
    first.write_text("observation_id,chal_union\ndemo0000,1\n", encoding="utf-8")
    second.write_bytes(first.read_bytes())
    config.data.holdout_csv = str(first)
    dataset = prepare_dataset(config, tmp_path / "held_out.npz")
    config.data.holdout_csv = str(second)
    loaded = load_dataset(dataset, config)
    assert loaded["dataset_sha256"] == sha256_file(dataset)
    assert loaded["dataset_sha256"] != sha256_file(second)
