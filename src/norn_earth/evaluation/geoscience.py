"""Geoscience-facing evaluation: evidence tiers and budget honesty (§13.2)."""

import numpy as np


def evidence_class(
    observation_distance_km,
    age_width_myr,
    kinematic_coverage=1.0,
    process_reliability=0.0,
):
    """Predefined evidence tiers; NOT the model's own output sigma.

    observation-supported: nearby observation with resolvable age;
    physics-supported: no observation but validated kinematics + process;
    prior-dominated: everything else (report honestly, do not hide).
    """
    near = observation_distance_km <= 200.0
    resolvable = age_width_myr <= 10.0
    if near and resolvable:
        return "observation_supported"
    if kinematic_coverage >= 0.9 and process_reliability >= 0.5:
        return "physics_supported"
    return "prior_dominated"


def classify_grid(obs_dist_km, age_width_myr, kin_cov=None, proc_rel=None):
    kin = np.ones_like(obs_dist_km, dtype=float) if kin_cov is None else kin_cov
    prc = np.zeros_like(obs_dist_km, dtype=float) if proc_rel is None else proc_rel
    out = np.empty(obs_dist_km.shape, dtype=object)
    for idx in np.ndindex(obs_dist_km.shape):
        out[idx] = evidence_class(
            float(obs_dist_km[idx]), float(age_width_myr[idx]), float(kin[idx]), float(prc[idx])
        )
    return out


def region_budget_error_km3(predicted_volume_change_km3, independent_volume_change_km3):
    """Signed budget discrepancy against an INDEPENDENT estimate.

    Agreement with the same physics that generated a prior is not validation.
    """
    return float(predicted_volume_change_km3 - independent_volume_change_km3)


def thickness_increment(field_early, field_late, areas_km2, mask=None):
    """Material-group thickness increment dH = sum(H_late - H_early)*dA over mask."""
    d = (np.asarray(field_late) - np.asarray(field_early)) * np.asarray(areas_km2)
    if mask is not None:
        d = d * np.asarray(mask, dtype=float)
    return float(d.sum())
