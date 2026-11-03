import numpy as np
import pytest
import torch

from norn_earth.evaluation.missing import missing_field_metrics, training_query_support
from norn_earth.losses.objective import ReconstructionObjective


def test_missing_score_excludes_supervised_values_and_uses_area():
    truth = np.zeros((2, 2, 2))
    prediction = np.full_like(truth, 100.0)
    support = np.ones_like(truth, dtype=bool)
    support[1, :, 0] = False
    prediction[1, :, 0] = [2, 4]
    result = missing_field_metrics(truth, prediction, support, np.array([[1, 1], [3, 3]]))
    assert result["n_cells"] == 2
    assert result["area_mae_km"] == 3.5
    assert result["area_rmse_km"] == pytest.approx(np.sqrt(13))


def test_query_support_excludes_held_out_records_and_zero_weight_anchors(tiny_problem):
    config, data, _ = tiny_problem
    objective = ReconstructionObjective(data, config)
    support = training_query_support(objective)
    assert support[0].all()  # Modern supervision at 0 Ma.
    objective.masks["train"].fill_(False)
    without_paleo = training_query_support(objective)
    assert without_paleo[0].all()
    assert not without_paleo[1:].any()  # a1 at the modern query has exactly zero time weight.
    assert np.all(without_paleo <= support)
    objective.config.training.modern_weight = 0
    assert not training_query_support(objective).any()
    assert torch.equal(objective.masks["validation"], torch.as_tensor(data["split"] == 1))
