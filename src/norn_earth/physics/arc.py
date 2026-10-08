"""Arc magmatism source budget (§6.5).

Identifiability contract: thickness data alone constrains only the retained
production c = eta*Q (or the net source c_net = c_retained - c_loss when an
independent net budget exists). Gross Q, retention eta and losses are NOT
jointly inferred; parameters below are net/retained by explicit declaration.

Units: c in km^2/Myr (volume per unit trench length per time); the across-arc
kernel K (1/km) integrates to 1 over the arc width, so q = c*K is km/Myr.
"""

import numpy as np


def gaussian_kernel(distance_km, width_km):
    d = np.asarray(distance_km, dtype=float)
    w = float(width_km)
    k = np.exp(-0.5 * (d / w) ** 2)
    norm = w * np.sqrt(2 * np.pi)
    return k / norm  # 1/km


def arc_source_km_per_myr(distance_to_trench_km, c_km2_per_myr, width_km=100.0):
    """Areal source q(x) = c * K(r) in km/Myr (uniform along-starter within a segment)."""
    return float(c_km2_per_myr) * gaussian_kernel(distance_to_trench_km, width_km)


def uniform_width_rate_km_per_myr(c_km2_per_myr, width_km):
    """dH/dtau ~ c / w approximation; only valid when the kernel is roughly flat."""
    return float(c_km2_per_myr) / float(width_km)


def segment_time_spline(times_myr, coefficients):
    """Low-dimensional segment/time spline source; returns callable c(tau)."""
    t = np.asarray(times_myr, dtype=float)
    c = np.asarray(coefficients, dtype=float)

    def fn(tau):
        return float(np.interp(tau, t, c, left=c[0], right=c[-1]))

    return fn
