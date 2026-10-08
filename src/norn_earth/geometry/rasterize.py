"""Rasterization of GMT polygon snapshots onto analysis grids.

Produces, per feature: fractional coverage on native/Gauss cells and
boundary-distance fields. Uses exact spherical winding (dateline/pole safe).
Every ring is VALIDATED first (validate_ring): zig-zag/double-traced rings in
local GMT exports are flagged invalid and skipped from coverage (they need
original GPML per G1), never silently rasterized. Category masks come from
GMT metadata only (GPGIM_TYPE/TYPE); a polygon label is an identifier, not a
physical property, and never enters thickness labels.
"""

import numpy as np

from .regrid import gauss_grid, native_grid
from .sphere import (
    distance_to_polygon_km,
    lonlat_to_vectors,
    spherical_winding,
    validate_ring,
)


def _cell_centers_for_subsamples(nlat, nlon):
    lats, lons = native_grid() if (nlat, nlon) == (180, 360) else gauss_grid(nlat, nlon)[:2]
    return lats, lons


def coverage_fraction(poly_lon, poly_lat, nlat=180, nlon=360, subsample=2):
    """Fractional cell coverage by 2x2 subsampling of each cell (fast, unbiased).

    Exact winding is used on subsample centers; subsample=2 gives 0.25 quantum.
    """
    lats, lons = _cell_centers_for_subsamples(nlat, nlon)
    offsets = (np.arange(subsample) + 0.5) / subsample - 0.5
    dlat = lats[1] - lats[0] if len(lats) > 1 else 1.0
    dlon = lons[1] - lons[0] if len(lons) > 1 else 1.0
    poly_v = lonlat_to_vectors(poly_lon, poly_lat)
    check = validate_ring(poly_lon, poly_lat)
    if not check["valid"]:
        raise ValueError(
            f"invalid ring (excess {check['signed_excess_sr']:.1f} sr); replace via G1 GPML"
        )
    frac = np.zeros((nlat, nlon))
    for da in offsets:
        for db in offsets:
            la = np.clip(lats + da * dlat, -90.0, 90.0)
            lo = lons + db * dlon
            lon2, lat2 = np.meshgrid(lo, la)
            pts = lonlat_to_vectors(lon2.ravel(), lat2.ravel())
            inside = np.abs(spherical_winding(pts, poly_v)) > 0.5
            frac += inside.reshape(nlat, nlon)
    return frac / (subsample * subsample)


def boundary_distance_km(poly_lon, poly_lat, nlat=180, nlon=360, max_km=3000.0):
    """Distance of each cell center to the polygon boundary (capped at max_km)."""
    lats, lons = _cell_centers_for_subsamples(nlat, nlon)
    lon2, lat2 = np.meshgrid(lons, lats)
    d = distance_to_polygon_km(lon2.ravel(), lat2.ravel(), poly_lon, poly_lat)
    return np.minimum(d.reshape(nlat, nlon), max_km)


def rasterize_features(features, nlat=180, nlon=360, subsample=2, with_distance=False):
    """features: list of dicts with keys lon, lat (arrays), type/name metadata.

    Returns list of dicts {coverage, [distance_km], meta}.
    """
    out = []
    for f in features:
        entry = {
            "meta": {k: f.get(k, "") for k in ("name", "type", "gpgim_type", "plate_id")},
        }
        checks = [
            validate_ring(r[:, 0], r[:, 1])
            for r in f.get("rings", [[np.asarray(f["lon"]), np.asarray(f["lat"])]])
        ]
        entry["ring_checks"] = checks
        if all(c["valid"] for c in checks):
            entry["coverage"] = coverage_fraction(f["lon"], f["lat"], nlat, nlon, subsample)
            if with_distance:
                entry["distance_km"] = boundary_distance_km(f["lon"], f["lat"], nlat, nlon)
        else:
            entry["coverage"] = None  # flagged: do not rasterize an invalid ring
        out.append(entry)
    return out
