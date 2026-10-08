"""Physical constants and unit conversions (single source of truth).

Conventions follow 最终研究方案_NORN.md §6.1/§6.3/§6.8:
- H in km (positive), velocity in km/Myr, source q in km/Myr, volumes in km^3.
- Forward physical time tau = 60 - age_ma.
"""

import math

EARTH_RADIUS_KM = 6371.0
SECONDS_PER_JULIAN_MYR = 31_557_600.0 * 1.0e6  # 365.25-day years
CM_PER_YR_TO_KM_PER_MYR = 10.0
PER_SECOND_TO_PER_MYR = SECONDS_PER_JULIAN_MYR
AGE_DOMAIN_MA = (0.0, 60.0)
PHYSICAL_TAU_OF_AGE = lambda age: 60.0 - age  # noqa: E731


def cm_per_year_to_km_per_myr(v):
    return float(v) * CM_PER_YR_TO_KM_PER_MYR


def per_second_to_per_myr(r):
    return float(r) * PER_SECOND_TO_PER_MYR


def deg_cm_per_year_pole_to_km_per_myr(rate_deg_per_myr):
    """Tangential speed at equator for a stage pole rate (deg/Myr)."""
    return math.radians(float(rate_deg_per_myr)) * EARTH_RADIUS_KM


def check_units(h_km=None, v_km_per_myr=None, q_km_per_myr=None):
    """Lightweight guard used at budget assembly sites."""
    if h_km is not None and h_km < 0:
        raise ValueError("thickness must be positive km")
    if v_km_per_myr is not None and abs(v_km_per_myr) > 2000.0:
        raise ValueError("velocity looks wrong; expected km/Myr (10 cm/yr == 10 km/Myr)")
    if q_km_per_myr is not None and abs(q_km_per_myr) > 10.0:
        raise ValueError("source q unrealistic in km/Myr; check km^2/Myr vs km/Myr")
