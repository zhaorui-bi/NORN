"""Observation batch assembly: records -> age-node tensors (§5.2/§5.5).

For every eligible record:
- nodes: TRUE age quadrature nodes (never rounded to TIME anchors);
- weights: sum to 1 (one record = one evidence unit);
- positions: per-node paleo positions from the trajectory when reliably
  locatable (gap<=5 Myr rule), else present coordinates with a support flag;
- effective scale: sqrt(s_i^2 + s_source^2) from the SourceRegistry;
- c_i: reliable-support mass is recorded, and masking never depends on y.

Outputs are plain numpy arrays ready for any backend.
"""

import numpy as np

from ..losses.chronology import make_age_nodes
from .trajectories import sample_position

DEFAULT_K = {"point": 1, "narrow": 3, "wide": 8}


def _node_count(semantics, k_narrow=3, k_wide=8):
    if semantics.startswith("point"):
        return 1
    if semantics in ("narrow_interval",):
        return k_narrow
    return k_wide


def assemble_observation_batch(
    table,
    traj_lookup,
    registry,
    source_of_record=None,
    k_narrow=3,
    k_wide=8,
    domain=(0.0, 60.0),
    source_sigma_override=None,
):
    """Build ragged-node batch arrays aligned with `table` rows.

    Returns dict of equal-length per-record arrays plus flattened node arrays.
    Rows outside the age domain keep zero nodes (excluded downstream, never
    relabelled).
    """
    n = len(table)
    rec = {
        "y_km": np.asarray(table["thickness_km"], dtype=float),
        "semantics": table["age_semantics"].to_numpy(),
        "site_cluster": np.asarray(table.get("site_cluster", np.arange(n))),
        "support_mass": np.asarray(table.get("trajectory_support_mass", np.zeros(n)), dtype=float),
        "primary_set": np.asarray(table.get("primary_set", np.ones(n, dtype=bool)), dtype=bool)
        if "primary_set" in table
        else np.ones(n, dtype=bool),
    }
    node_ages, node_w, node_lon, node_lat = [], [], [], []
    node_counts = np.zeros(n, dtype=int)
    node_positions_from_trajectory = []
    source_ids = []
    for i in range(n):
        row = table.iloc[i]
        k = _node_count(row["age_semantics"], k_narrow, k_wide)
        nodes, weights, _c = make_age_nodes(
            float(row["age_lower_ma"]), float(row["age_upper_ma"]), k, domain=domain
        )
        count = len(nodes)
        node_counts[i] = count
        traj = traj_lookup.get(i)
        from_traj = []
        for a in nodes:
            pos = sample_position(traj, float(a)) if traj is not None else None
            if pos is None:
                node_lon.append(float(row["present_lon"]))
                node_lat.append(float(row["present_lat"]))
                from_traj.append(False)
            else:
                node_lon.append(pos[0])
                node_lat.append(pos[1])
                from_traj.append(True)
        node_ages.extend(np.asarray(nodes, dtype=float).tolist())
        node_w.extend(np.asarray(weights, dtype=float).tolist())
        node_positions_from_trajectory.extend(from_traj)
        sid = source_of_record(i) if source_of_record else "unverified_proxy"
        source_ids.append(sid)
    if source_sigma_override is not None:
        src_scale = np.asarray(source_sigma_override, dtype=float)
        if len(src_scale) != n:
            raise ValueError("source_sigma_override must be parallel to the table")
    else:
        src_scale = np.array([registry.scales.get(s, np.nan) for s in source_ids], dtype=float)
        if np.isnan(src_scale).any():
            missing = sorted({s for s, v in zip(source_ids, src_scale) if np.isnan(v)})
            raise KeyError(f"unregistered source scale(s): {missing}; calibrate before batching")
    rec.update(
        {
            "node_counts": node_counts,
            "node_ages": np.asarray(node_ages, dtype=float),
            "node_weights": np.asarray(node_w, dtype=float),
            "node_lon": np.asarray(node_lon, dtype=float),
            "node_lat": np.asarray(node_lat, dtype=float),
            "node_from_trajectory": np.asarray(node_positions_from_trajectory, dtype=bool),
            "source_ids": np.asarray(source_ids, dtype=object),
            "record_sigma_km": np.full(n, 4.0, dtype=float),  # placeholder: frozen at G0/G2
            "source_sigma_km": src_scale,
            "effective_sigma_km": rec_full_sigma(np.full(n, 4.0), src_scale),
        }
    )
    return rec


def rec_full_sigma(record_sigma, source_sigma):
    """Effective scale sqrt(s_record^2 + s_source^2) (§5.2)."""
    return np.sqrt(
        np.asarray(record_sigma, dtype=float) ** 2 + np.asarray(source_sigma, dtype=float) ** 2
    )


def batch_slices(node_counts):
    """Per-record [start, end) slices into the flattened node arrays."""
    ends = np.cumsum(node_counts)
    starts = ends - node_counts
    return starts, ends
