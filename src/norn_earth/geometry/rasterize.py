"""Diagnostic GMT coverage using signed spherical winding, parts and holes.

The production dynamic-input path uses registered pyGPlates polygons instead.
Ring area checks here are sanity checks, not certificates of simple geometry.
Metadata categories are identifiers, never thickness or process labels.
"""

import numpy as np

from ..data.gmt import polygon_parts
from .regrid import gauss_grid, native_grid
from .sphere import distance_to_polygon_km, points_in_polygon, validate_ring


def _cell_centers_for_subsamples(nlat, nlon):
    lats, lons = native_grid() if (nlat, nlon) == (180, 360) else gauss_grid(nlat, nlon)[:2]
    return lats, lons


def _coverage_parts(parts, nlat, nlon, subsample):
    if subsample < 1:
        raise ValueError("subsample must be >=1")
    for exterior, holes in parts:
        for ring in [exterior] + holes:
            if not validate_ring(ring[:, 0], ring[:, 1])["valid"]:
                raise ValueError("Invalid ring; cannot compute diagnostic coverage")
    lats, lons = _cell_centers_for_subsamples(nlat, nlon)
    offsets = (np.arange(subsample) + 0.5) / subsample - 0.5
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    fraction = np.zeros((nlat, nlon))
    for da in offsets:
        for db in offsets:
            lon, lat = np.meshgrid(lons + db * dlon, np.clip(lats + da * dlat, -90, 90))
            lon, lat = lon.ravel(), lat.ravel()
            union = np.zeros(len(lon), dtype=bool)
            for exterior, holes in parts:
                inside = points_in_polygon(lon, lat, exterior[:, 0], exterior[:, 1])
                for hole in holes:
                    inside &= ~points_in_polygon(lon, lat, hole[:, 0], hole[:, 1])
                union |= inside
            fraction += union.reshape(nlat, nlon)
    return fraction / subsample**2


def coverage_fraction(poly_lon, poly_lat, nlat=180, nlon=360, subsample=2):
    """Approximate cell coverage by subcell probes; no antipodal ghost polygon."""
    ring = np.column_stack([poly_lon, poly_lat])
    return _coverage_parts([(ring, [])], nlat, nlon, subsample)


def boundary_distance_km(poly_lon, poly_lat, nlat=180, nlon=360, max_km=3000.0):
    """Distance of each cell center to the polygon boundary (capped at max_km)."""
    lats, lons = _cell_centers_for_subsamples(nlat, nlon)
    lon, lat = np.meshgrid(lons, lats)
    distance = distance_to_polygon_km(lon.ravel(), lat.ravel(), poly_lon, poly_lat)
    return np.minimum(distance.reshape(nlat, nlon), max_km)


def rasterize_features(features, nlat=180, nlon=360, subsample=2, with_distance=False):
    """All exterior parts contribute; each hole is subtracted from its own part."""
    out = []
    for feature in features:
        rings = feature.get("rings", [np.column_stack([feature["lon"], feature["lat"]])])
        parts = polygon_parts({**feature, "rings": rings})
        checks = [validate_ring(ring[:, 0], ring[:, 1]) for ring in rings]
        entry = {
            "meta": {
                key: feature.get(key, "") for key in ("name", "type", "gpgim_type", "plate_id")
            },
            "ring_checks": checks,
            "coverage": None,
        }
        if all(check["valid"] for check in checks):
            entry["coverage"] = _coverage_parts(parts, nlat, nlon, subsample)
            if with_distance:
                entry["distance_km"] = np.minimum.reduce(
                    [boundary_distance_km(ring[:, 0], ring[:, 1], nlat, nlon) for ring in rings]
                )
        out.append(entry)
    return out
