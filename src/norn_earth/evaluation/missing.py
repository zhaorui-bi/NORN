"""Synthetic field scores restricted to cells outside training query support.

This is a conservative grid-cell definition, not a claim that neighboring
cells are statistically independent or that modern conditioning is absent.
"""

import numpy as np
import torch


def training_query_support(objective):
    """Mark only positive-weight corners of training/modern observation queries."""
    c = objective.config
    support = torch.zeros(
        (c.n_anchors, c.grid.nlat, c.grid.nlon), dtype=torch.bool, device=objective.y.device
    )

    def mark(query, selected):
        for anchor, time_weight in ((query.a0, 1 - query.fraction), (query.a1, query.fraction)):
            active = selected[:, None] & (time_weight[:, None] * query.weight > 0)
            age = anchor[:, None].expand_as(query.i)
            support[age[active], query.i[active], query.j[active]] = True

    selected = objective.masks["train"][objective.node_record]
    selected = selected & (c.training.observation_weight > 0)
    selected = selected & (objective.weights[objective.node_record, objective.slots] > 0)
    mark(objective.query, selected)
    if c.training.modern_weight > 0:
        mark(objective.modern_query, torch.ones_like(objective.modern_query.a0, dtype=torch.bool))
    return support.cpu().numpy()


def missing_field_metrics(truth, prediction, observed_support, cell_area):
    """Area-weighted errors on the complement of *frozen* training support."""
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    observed = np.asarray(observed_support)
    areas = np.asarray(cell_area, float)
    if (
        truth.ndim != 3
        or prediction.shape != truth.shape
        or observed.shape != truth.shape
        or areas.shape != truth.shape[1:]
        or not np.isin(observed, [0, 1]).all()
        or not np.isfinite(areas).all()
        or np.any(areas <= 0)
    ):
        raise ValueError(
            "Require matching age/lat/lon fields, binary support and positive cell areas"
        )
    missing = ~observed.astype(bool)
    if not np.isfinite(truth[missing]).all() or not np.isfinite(prediction[missing]).all():
        raise ValueError("Missing-cell truth and prediction must be finite")
    count = int(missing.sum())
    if count == 0:
        return {"n_cells": 0, "area_rmse_km": None, "area_mae_km": None}
    weights = np.broadcast_to(areas, truth.shape)[missing]
    weights = weights / weights.sum()
    error = prediction[missing] - truth[missing]
    return {
        "n_cells": count,
        "area_rmse_km": float(np.sqrt(np.sum(weights * error**2))),
        "area_mae_km": float(np.sum(weights * np.abs(error))),
    }
