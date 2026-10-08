"""Kinematics: plate rotations, material trajectories, divergence diagnostics.

Conventions (最终研究方案_NORN.md §6):
- forward physical time tau = 60 - age_ma; velocities in km/Myr;
- rigid Euler rotations are divergence-free on the sphere;
- internal strain must come from validated topological networks
  (pygplates strain reconstructions), never from network-outline area change.
"""

import numpy as np

from ..geometry.sphere import (
    lonlat_to_vectors,
    pole_to_axis,
    rotation_matrix,
    rotation_velocity_km_per_myr,
    slerp,
    vectors_to_lonlat,
)
from ..utils.units import EARTH_RADIUS_KM


def compose_rotations(r1, r2):
    return np.asarray(r2) @ np.asarray(r1)


def integrate_rotation(lon, lat, pole_lat, pole_lon, rate_deg_per_myr, dtau_myr, steps):
    """Exact rigid transport of points for constant stage rotation."""
    R = rotation_matrix(pole_to_axis(pole_lat, pole_lon), rate_deg_per_myr * dtau_myr)
    Rn = np.linalg.matrix_power(R, int(steps))
    v = lonlat_to_vectors(lon, lat) @ Rn.T
    return vectors_to_lonlat(v)


def advect_points(lon, lat, velocity_fn, dtau_myr, steps, order=2):
    """Generic material advection with RK1/RK2 on the sphere.

    velocity_fn(lons, lats) -> (east_km_per_myr, north_km_per_myr).
    """
    lon = np.array(lon, dtype=float)
    lat = np.array(lat, dtype=float)
    for _ in range(int(steps)):
        u1, v1 = velocity_fn(lon, lat)
        if order == 1:
            lon = lon + np.degrees(
                u1 * dtau_myr / (EARTH_RADIUS_KM * np.cos(np.radians(np.clip(lat, -89.99, 89.99))))
            )
            lat = lat + np.degrees(v1 * dtau_myr / EARTH_RADIUS_KM)
        else:
            lon_m = lon + np.degrees(
                0.5
                * dtau_myr
                * u1
                / (EARTH_RADIUS_KM * np.cos(np.radians(np.clip(lat, -89.99, 89.99))))
            )
            lat_m = lat + np.degrees(0.5 * dtau_myr * v1 / EARTH_RADIUS_KM)
            u2, v2 = velocity_fn(lon_m, lat_m)
            lon = lon + np.degrees(
                dtau_myr
                * u2
                / (EARTH_RADIUS_KM * np.cos(np.radians(np.clip(lat_m, -89.99, 89.99))))
            )
            lat = lat + np.degrees(dtau_myr * v2 / EARTH_RADIUS_KM)
        lon = np.mod(lon + 180.0, 360.0) - 180.0
        lat = np.clip(lat, -90.0, 90.0)
    return lon, lat


def tangent_divergence_per_myr(u_east, v_north, lats, lons):
    """Finite-difference surface divergence div_s(v) [1/Myr] on cell centers.

    u_east, v_north: (nlat, nlon) km/Myr; lats/lons in degrees, uniform dlon.
    Uses centered differences with metric terms; values near poles are unreliable
    and must be masked by callers (polar cap is not a rigid-plate interior).
    """
    dlon = np.radians(lons[1] - lons[0])
    coslat = np.cos(np.radians(lats))[:, None]
    u = np.asarray(u_east, dtype=float)
    v = np.asarray(v_north, dtype=float)
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dlon)
    vc = v * np.cos(np.radians(lats))[:, None]
    dv_dphi = (np.roll(vc, -1, axis=0) - np.roll(vc, 1, axis=0)) / (
        2 * np.radians(lats[1] - lats[0])
    )
    return (du_dx + dv_dphi) / (EARTH_RADIUS_KM * np.maximum(coslat, 1e-3))


def rigid_rotation_divergence(pole_lat, pole_lon, rate_deg_per_myr, lats, lons):
    """Diagnostic: divergence of a rigid rotation field (should be ~0)."""
    lon2, lat2 = np.meshgrid(lons, lats)
    u, v = rotation_velocity_km_per_myr(pole_lat, pole_lon, rate_deg_per_myr, lon2, lat2)
    return tangent_divergence_per_myr(u, v, lats, lons)


def interpolate_trajectory(times_avail, lons_avail, lats_avail, age, max_gap_myr=5.0):
    """Sample a reconstructed trajectory at `age` (Ma) under §5.3 rules.

    - interpolation only between consecutive available times with gap <= max_gap;
    - no extrapolation beyond available range; no crossing of missing samples;
    - returns (lon, lat) or None.
    """
    t = np.asarray(times_avail, dtype=float)
    order = np.argsort(t)
    t = t[order]
    lo = np.asarray(lons_avail, dtype=float)[order]
    la = np.asarray(lats_avail, dtype=float)[order]
    if age < t[0] - 1e-9 or age > t[-1] + 1e-9:
        return None
    k = np.searchsorted(t, age)
    if k == 0:
        if abs(age - t[0]) > 1e-9:
            return None
        return float(lo[0]), float(la[0])
    if t[k - 1] == age:
        return float(lo[k - 1]), float(la[k - 1])
    if k >= len(t):
        return None
    gap = t[k] - t[k - 1]
    if gap > max_gap_myr + 1e-9:
        return None
    f = (age - t[k - 1]) / gap
    a = lonlat_to_vectors(lo[k - 1], la[k - 1])
    b = lonlat_to_vectors(lo[k], la[k])
    v = slerp(a, b, np.float64(f)).reshape(3)
    lon, lat = vectors_to_lonlat(v)
    return float(np.asarray(lon).reshape(())), float(np.asarray(lat).reshape(()))
