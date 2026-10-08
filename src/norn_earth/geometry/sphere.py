"""Spherical geometry primitives: vectors, rotations, distances, polygons.

All longitude/latitude inputs are degrees; internal math uses unit 3-vectors.
Polygon tests use the exact spherical winding-number method
(sum of signed subtended angles), which handles poles and the dateline seam
without special cases (fails only for points exactly on a boundary vertex).
"""

import math

import numpy as np

from ..utils.units import EARTH_RADIUS_KM


def lonlat_to_vectors(lon, lat):
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def vectors_to_lonlat(v):
    v = np.asarray(v, dtype=float)
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    lat = np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))
    return lon, lat


def haversine_km(lon1, lat1, lon2, lat2):
    a, b = lonlat_to_vectors(lon1, lat1), lonlat_to_vectors(lon2, lat2)
    return angle_between(a, b) * EARTH_RADIUS_KM


def angle_between(a, b):
    """Geodesic angle (radians) between unit vectors, vectorized."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.arctan2(cross, np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def slerp(v0, v1, t):
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    t = np.asarray(t, dtype=float)[..., None]
    omega = angle_between(v0, v1)
    near = omega < 1e-12
    safe = np.where(near, 1.0, omega)
    s = np.sin(safe)
    return (np.sin((1 - t) * safe) / np.where(near, 1.0, s)) * v0 + (
        np.sin(t * safe) / np.where(near, 1.0, s)
    ) * v1


def rotation_matrix(axis_unit, angle_deg):
    """Rodrigues rotation about a unit axis (right-handed, degrees)."""
    k = np.asarray(axis_unit, dtype=float)
    th = math.radians(float(angle_deg))
    kx, ky, kz = k
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def pole_to_axis(pole_lat_deg, pole_lon_deg):
    return lonlat_to_vectors(pole_lon_deg, pole_lat_deg)


def rotation_velocity_km_per_myr(pole_lat, pole_lon, rate_deg_per_myr, lon, lat):
    """Tangential velocity v = R (omega x r_hat) in km/Myr, returned as (east, north)."""
    omega = math.radians(float(rate_deg_per_myr)) * pole_to_axis(pole_lat, pole_lon)
    r = lonlat_to_vectors(lon, lat)
    v = EARTH_RADIUS_KM * np.cross(omega, r)
    lonr = np.radians(np.asarray(lon, dtype=float))
    latr = np.radians(np.asarray(lat, dtype=float))
    e = np.stack([-np.sin(lonr), np.cos(lonr), np.zeros_like(lonr)], axis=-1)
    n = np.stack(
        [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
        axis=-1,
    )
    return np.sum(v * e, axis=-1), np.sum(v * n, axis=-1)


def rotate_points(rotation, lon, lat):
    v = lonlat_to_vectors(lon, lat) @ np.asarray(rotation).T
    return vectors_to_lonlat(v)


def unwrap_longitudes(lons):
    """Unwrap to make consecutive differences <= 180 deg (dateline-crossing polygons)."""
    lons = np.asarray(lons, dtype=float)
    out = np.empty_like(lons)
    out[0] = lons[0]
    offset = 0.0
    for i in range(1, len(lons)):
        d = (lons[i] + offset) - out[i - 1]
        while d > 180.0:
            offset -= 360.0
            d -= 360.0
        while d < -180.0:
            offset += 360.0
            d += 360.0
        out[i] = lons[i] + offset
    return out


def polygon_signed_area_steradian(poly_v):
    """Signed spherical excess (steradian) of a closed spherical polygon.

    Vertices connected by great-circle arcs; sign follows vertex orientation.
    """
    poly = np.asarray(poly_v, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 3 or not np.isfinite(poly).all():
        raise ValueError("Polygon vertices must be a finite (N, 3) array with N >= 3")
    # A repeated closing vertex creates zero-length edges. Their undefined
    # tangent directions otherwise make the orientation depend on roundoff.
    keep = np.r_[True, np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-12]
    poly = poly[keep]
    if len(poly) > 1 and np.linalg.norm(poly[-1] - poly[0]) <= 1e-12:
        poly = poly[:-1]
    if len(poly) < 3:
        raise ValueError("A polygon needs at least three distinct consecutive vertices")
    n = len(poly)
    turn = 0.0
    for i in range(n):
        prev, cur, nxt = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        t_in = prev - np.dot(prev, cur) * cur
        t_out = nxt - np.dot(nxt, cur) * cur
        t_in /= max(np.linalg.norm(t_in), 1e-15)
        t_out /= max(np.linalg.norm(t_out), 1e-15)
        # signed turn at the vertex (positive for CCW seen from outside)
        turn += math.atan2(np.dot(cur, np.cross(t_out, t_in)), np.dot(t_in, t_out))
    return turn - math.copysign((n - 2) * math.pi, turn)


def spherical_winding(points_v, poly_v):
    """Winding number of a closed great-circle polygon around points.

    inside = |winding| > 0.5; exact for points not on the boundary.
    points_v: (N,3); poly_v: (E,3) ordered vertices (no duplicate closing vertex).
    """
    pts = np.asarray(points_v, dtype=float)
    poly = np.asarray(poly_v, dtype=float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    total = np.zeros(len(pts))
    for ai, bi in zip(a, b):
        # tangent directions AT each query point toward the edge endpoints:
        # project the endpoint onto the point's tangent plane (subtract along pts!)
        ta = ai - np.dot(pts, ai)[:, None] * pts
        tb = bi - np.dot(pts, bi)[:, None] * pts
        ta /= np.clip(np.linalg.norm(ta, axis=1, keepdims=True), 1e-15, None)
        tb /= np.clip(np.linalg.norm(tb, axis=1, keepdims=True), 1e-15, None)
        cosd = np.clip(np.sum(ta * tb, axis=1), -1.0, 1.0)
        sind = np.sum(pts * np.cross(ta, tb), axis=1)
        total += np.arctan2(sind, cosd)
    return total / (2 * math.pi)


def validate_ring(poly_lon, poly_lat, tol_excess_sr=None):
    """Ring sanity for point-in-polygon/coverage use.

    A usable simple ring has |signed excess| <= 4*pi + eps. Local GMT exports
    may contain zig-zag/double-traced paths whose excess is many times the
    sphere; such rings must be flagged (and replaced by original GPML at G1),
    never silently rasterized.
    """
    poly_v = lonlat_to_vectors(unwrap_longitudes(poly_lon), poly_lat)
    excess = polygon_signed_area_steradian(poly_v)
    ok = abs(excess) <= 4 * math.pi + 1e-6
    return {
        "valid": ok,
        "signed_excess_sr": float(excess),
        "n_vertices": int(len(poly_v)),
        "note": "" if ok else "invalid/self-intersecting ring; needs original GPML (G1)",
    }


def points_in_polygon(lon, lat, poly_lon, poly_lat):
    """Robust spherical point-in-polygon for simple closed rings.

    Rule: inside(P) iff the bearing-winding W(P) has the same sign as the
    ring orientation sigma (signed spherical excess) and |W| > 0.5.
    Reasoning (verified against square / dateline-box / pole-cap tests):
    W(P) = s * ([P inside] - [antipode of P inside]); the antipodal term is
    exactly what breaks naive |W|>0.5 tests, and orientation disambiguates.
    Rings whose |excess| is within eps of 2*pi are flagged ambiguous.
    """
    poly_v = lonlat_to_vectors(unwrap_longitudes(poly_lon), poly_lat)
    excess = polygon_signed_area_steradian(poly_v)
    if abs(abs(excess) - 2 * math.pi) < 1e-6:
        raise ValueError("hemisphere-area ring is orientation-ambiguous")
    sigma = 1.0 if excess > 0 else -1.0
    pts_v = np.asarray(lonlat_to_vectors(lon, lat), dtype=float)
    w_p = spherical_winding(pts_v, poly_v)
    return (w_p * sigma) > 0.5


def distance_to_polygon_km(lon, lat, poly_lon, poly_lat):
    """Approximate distance to the polygon boundary (vertex+arc minimum)."""
    pts = lonlat_to_vectors(lon, lat)
    poly = lonlat_to_vectors(unwrap_longitudes(poly_lon), poly_lat)
    a = poly
    b = np.roll(poly, -1, axis=0)
    best = np.full(len(pts), np.inf)
    for ai, bi in zip(a, b):
        plane = np.cross(ai, bi)
        norm = np.linalg.norm(plane)
        if norm < 1e-15:
            continue
        plane /= norm
        # projection of points onto the great circle plane
        proj = pts - np.outer(np.dot(pts, plane), plane)
        proj /= np.clip(np.linalg.norm(proj, axis=1, keepdims=True), 1e-15, None)
        # inside the minor arc iff both endpoint dot products are positive
        on_arc = (np.dot(proj, ai) >= 0) & (np.dot(proj, bi) >= 0)
        d_arc = angle_between(pts, proj)
        d_end = np.minimum(angle_between(pts, ai[None, :]), angle_between(pts, bi[None, :]))
        best = np.minimum(best, np.where(on_arc, d_arc, d_end))
    return best * EARTH_RADIUS_KM
