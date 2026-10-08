"""Interval evaluation; latent-H vs predictive-y intervals are distinct (§13.1).

Coverage numbers computed here describe the interval kind supplied by the
caller; five-seed ensembles do not certify 90% coverage anywhere, and
uncalibrated products must be labeled ensemble-sensitivity ranges.
"""

import numpy as np


def interval_metrics(y_true, lower, upper, nominal=0.9):
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    alpha = 1 - nominal
    penalty = 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)
    return {
        "nominal": nominal,
        "coverage": float(covered.mean()),
        "mean_width": float(width.mean()),
        "interval_score": float(np.mean(width + penalty)),
        "n": int(len(y)),
    }


def coverage_by_bin(y_true, lower, upper, bins=None, score=None):
    """Empirical coverage stratified by a scalar (e.g. predicted uncertainty)."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    s = np.asarray(score if score is not None else hi - lo, dtype=float)
    if bins is None:
        bins = np.quantile(s, np.linspace(0, 1, 6))
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (s >= a) & (s <= b if a == bins[-2] else s < b)
        if m.sum() == 0:
            continue
        out.append(
            {
                "bin": [float(a), float(b)],
                "coverage": float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()),
                "n": int(m.sum()),
            }
        )
    return out
