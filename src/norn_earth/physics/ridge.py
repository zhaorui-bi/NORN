"""Ridge birth-thickness prior (§6.4).

Physics: normal oceanic crust attains near-steady thickness shortly after
formation; subsequent conductive cooling thickens the thermal lithosphere and
subsides the seafloor but does NOT keep thickening basaltic crust like sqrt(age).
The 7 km / 2 km numbers are PILOT STARTING VALUES to be recalibrated from
independent data; applicability masks (ultraslow ridges, hotspots, plateaus)
must come from independent crustal-type/age evidence, never from thickness.
"""

import numpy as np

DEFAULT_BIRTH_MU_KM = 7.0
DEFAULT_BIRTH_SCALE_KM = 2.0
MIN_FULL_RATE_KM_PER_MYR = 5.0  # below this H = Q/u is not applicable


def birth_thickness_prior(mu_km=DEFAULT_BIRTH_MU_KM, scale_km=DEFAULT_BIRTH_SCALE_KM):
    return {"kind": "ridge_birth_thickness", "mu_km": float(mu_km), "scale_km": float(scale_km)}


def retained_melt_thickness_km(q_retained_km2_per_myr, u_full_km_per_myr):
    """H_birth = Q_retained / u_full (km^2/Myr / (km/Myr) -> km). Guarded."""
    q = np.asarray(q_retained_km2_per_myr, dtype=float)
    u = np.asarray(u_full_km_per_myr, dtype=float)
    if np.any(u < MIN_FULL_RATE_KM_PER_MYR):
        raise ValueError("full spreading rate below applicability floor; mask instead of dividing")
    return q / u


def nll_normal(x, mu, scale):
    return 0.5 * ((np.asarray(x) - mu) / scale) ** 2 + np.log(scale) + 0.5 * np.log(2 * np.pi)
