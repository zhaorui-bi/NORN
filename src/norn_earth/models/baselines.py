"""Baseline machinery (B0/B1/B2/B6). Fitting on real data is gated.

- B0: robust regional mean + rolling age window (sanity check).
- B1: spherical RBF/low-rank GP basis; must be optimized with the SAME
  age-marginal likelihood as the main model (fair wide-age baseline, §10).
- B2: feature table builder; the representative-age variant is an explicitly
  labeled POINT-AGE baseline, not an equivalent chronology-aware method.
- B6: kinematic backtrack with calibrated low-dimensional sources; vanished
  material is reported as unknown, never reconstructed.
"""

import numpy as np


def robust_center_scale(values, lo=0.05, hi=0.95):
    v = np.asarray(values, dtype=float)
    a, b = np.quantile(v, [lo, hi])
    center = 0.5 * (a + b)
    scale = max(b - a, 1e-6) / 3.29
    return float(center), float(scale)


def b0_predict(train_thickness, train_ages, query_ages, window_myr=10.0):
    """Rolling median in age windows; sanity baseline only."""
    t = np.asarray(train_ages, dtype=float)
    y = np.asarray(train_thickness, dtype=float)
    out = []
    for q in np.atleast_1d(query_ages):
        m = np.abs(t - q) <= window_myr
        out.append(float(np.median(y[m])) if m.sum() else float(np.median(y)))
    return np.array(out)


def spherical_rbf_kernel(lon_a, lat_a, lon_b, lat_b, length_km=800.0):
    """RBF kernel on great-circle distance (km)."""
    from ..geometry.sphere import haversine_km

    d = haversine_km(
        np.asarray(lon_a, dtype=float)[..., None],
        np.asarray(lat_a, dtype=float)[..., None],
        np.asarray(lon_b, dtype=float)[None, ...],
        np.asarray(lat_b, dtype=float)[None, ...],
    )
    return np.exp(-0.5 * (d / float(length_km)) ** 2)


def b1_marginal_nll(y, mu_nodes, weights, kernel_train, sigma_obs, jitter=1e-6):
    """GP-style marginal NLL for the fair B1 baseline (synthetic/self-checks only).

    Uses the same per-record age-mixture structure as the main model: for each
    record the latent mean is the weight-averaged node prediction. Fitting on
    real observations requires training authorization.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu_nodes, dtype=float)
    pred = np.sum(w * mu, axis=1)
    r = y - pred
    K = np.asarray(kernel_train, dtype=float) + (jitter + np.mean(sigma_obs) ** 2) * np.eye(len(y))
    sign, logdet = np.linalg.slogdet(K)
    if sign <= 0:
        raise ValueError("kernel not PD; increase jitter")
    alpha = np.linalg.solve(K, r)
    n = len(y)
    return 0.5 * float(r @ alpha) + 0.5 * float(logdet) + 0.5 * n * np.log(2 * np.pi)


def b2_build_features(observations, point_age=False):
    """Feature rows for B2. point_age=True marks the representative-age variant."""
    rows = []
    for o in observations:
        age = o["age_representative"] if point_age else 0.5 * (o["age_lower"] + o["age_upper"])
        rows.append(
            {
                "lon": o["present_lon"],
                "lat": o["present_lat"],
                "sin_lat": np.sin(np.radians(o["present_lat"])),
                "age": age,
                "age_width": o["age_upper"] - o["age_lower"],
                "thickness_km": o["thickness_km"],
                "variant": "point_age" if point_age else "interval_midpoint",
            }
        )
    return rows


def b6_backtrack(h_present, jacobian_cumulative, jacobian_from, jacobian_to):
    """H(earlier) = H(later) * J(later->now) / J(earlier->now), with J = 1+dilatation.

    Vanished material (no trajectory to the queried age) must be reported as
    unknown by callers; this function never extrapolates beyond coverage.
    """
    if jacobian_from <= 0 or jacobian_to <= 0:
        raise ValueError("cumulative Jacobians must be positive")
    return float(h_present) * jacobian_to / jacobian_from
