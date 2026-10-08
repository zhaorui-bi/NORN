"""Grid definitions and conservative regridding.

Two grids coexist by design (最终研究方案_NORN.md §8.3/§8.5):

- NATIVE: the 1-deg cell-center grid of the modern product, 180x360,
  lon 0.5..359.5 eastward, lat -89.5..89.5 south->north.
- GAUSS: internal 180x360 Legendre-Gauss grid for spectral code. Longitudes
  start at 0 (eastward, periodic); latitudes are Gauss nodes south->north
  (torch-harmonics returns co-latitudes; convert explicitly).

Conservative remap native->gauss preserves area integrals exactly
(lat/lon rectangle overlaps). gauss->native is an interpolation sampler for
export; its representation error belongs to source-error budgets, not to
"verified resolution".
"""

import numpy as np

from ..utils.units import EARTH_RADIUS_KM

NATIVE_NLAT, NATIVE_NLON = 180, 360
DLON_DEG = 1.0


def native_grid():
    lons = 0.5 + np.arange(NATIVE_NLON) * DLON_DEG
    lats = -90.0 + 0.5 + np.arange(NATIVE_NLAT) * DLON_DEG
    return lats, lons


def native_lat_edges():
    return -90.0 + np.arange(NATIVE_NLAT + 1) * DLON_DEG


def native_cell_areas_km2():
    edges = native_lat_edges()
    dlon = np.radians(DLON_DEG)
    band = (
        EARTH_RADIUS_KM**2 * dlon * (np.sin(np.radians(edges[1:])) - np.sin(np.radians(edges[:-1])))
    )
    return np.repeat(band[:, None], NATIVE_NLON, axis=1)


def gauss_grid(nlat=NATIVE_NLAT, nlon=NATIVE_NLON):
    """Legendre-Gauss nodes; latitudes ascending (S->N), longitudes from 0."""
    x, w = np.polynomial.legendre.leggauss(nlat)  # x = sin(lat), ascending
    lats = np.degrees(np.arcsin(x))
    lons = 360.0 / nlon * np.arange(nlon)
    return lats, lons, w


def gauss_lat_bounds(lats):
    # Cell areas must equal the SHT quadrature weights, rather than areas
    # bounded by arithmetic midpoints of the latitude nodes.
    _, weights = np.polynomial.legendre.leggauss(len(lats))
    edges = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    return np.degrees(np.arcsin(np.clip(edges, -1.0, 1.0)))


def gauss_lon_bounds(nlon=NATIVE_NLON):
    step = 360.0 / nlon
    centers = step * np.arange(nlon)
    return centers - 0.5 * step, centers + 0.5 * step  # wrapped at ±180 handled by caller


def gauss_area_weights_km2(nlat=NATIVE_NLAT, nlon=NATIVE_NLON):
    _, _, w = gauss_grid(nlat, nlon)
    dlon = 2 * np.pi / nlon
    return EARTH_RADIUS_KM**2 * dlon * np.repeat(w[:, None], nlon, axis=1)


def total_gauss_area_km2(nlat=NATIVE_NLAT, nlon=NATIVE_NLON):
    """Quadrature estimate of the full sphere area: 4*pi*R^2."""
    _, _, w = gauss_grid(nlat, nlon)
    return EARTH_RADIUS_KM**2 * 2 * np.pi * float(np.sum(w))


def _overlap_matrix(gauss_edges, native_edges):
    """Overlap in transformed edge units (rows: gauss cells, cols: native cells).

    Callers pass edges already mapped through sin(lat) (latitude) or radians
    (longitude) so that products are proportional to true spherical area and
    conservation holds EXACTLY for fully covered cells (see module tests).
    """
    ng = len(gauss_edges) - 1
    nn = len(native_edges) - 1
    M = np.zeros((ng, nn))
    for j in range(nn):
        a, b = native_edges[j], native_edges[j + 1]
        i0 = np.searchsorted(gauss_edges, a, side="right") - 1
        i1 = np.searchsorted(gauss_edges, b, side="left")
        for i in range(max(i0, 0), min(i1, ng)):
            M[i, j] = max(0.0, min(b, gauss_edges[i + 1]) - max(a, gauss_edges[i]))
    return M


def conservative_native_to_gauss(field, nlat=NATIVE_NLAT, nlon=NATIVE_NLON, valid=None):
    """Area-weighted conservative remap native->gauss (matrix form).

    Lat overlaps use sin-edge units and lon overlaps radians, so with
    valid all-True the map preserves the global integral EXACTLY for any
    field; with a mask it returns the masked area-weighted mean where
    coverage exists and NaN elsewhere.
    """
    field = np.asarray(field, dtype=float)
    if field.shape != (NATIVE_NLAT, NATIVE_NLON):
        raise ValueError("field must be native 180x360")
    if valid is None:
        valid = np.ones_like(field, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    sinr = np.radians
    glats, _, _ = gauss_grid(nlat, nlon)
    g_lat_edges = np.sin(sinr(gauss_lat_bounds(glats)))
    n_lat_edges = np.sin(sinr(native_lat_edges()))
    F = _overlap_matrix(g_lat_edges, n_lat_edges)
    glo_lo, glo_hi = gauss_lon_bounds(nlon)
    g_lon_edges = np.radians(np.append(glo_lo, glo_hi[-1]))
    L = _overlap_matrix(g_lon_edges, np.radians(np.arange(NATIVE_NLON + 1) * DLON_DEG))
    # The cell centred on 0 degrees straddles the periodic seam.
    for offset in (-360.0, 360.0):
        L += _overlap_matrix(g_lon_edges, np.radians(np.arange(NATIVE_NLON + 1) + offset))
    num = F @ np.where(valid, field, 0.0) @ L.T
    den = F @ valid.astype(float) @ L.T
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def sample_at_points(field, lats, lons, grid=None):
    """Bilinear sampling of a gauss-grid field (lat S->N, lon from 0, periodic).

    grid: optional (glats, glons) of the field; defaults to the 180x360 gauss grid.
    """
    if grid is None:
        glats, glons, _ = gauss_grid()
    else:
        glats, glons = grid[0], grid[1]
    la = np.asarray(lats, dtype=float)
    lo = np.mod(np.asarray(lons, dtype=float), 360.0)
    fi = np.interp(la, glats, np.arange(len(glats)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(glats) - 2)
    f_lat = fi - i0
    step = glons[1] - glons[0]
    fj = np.mod(lo - glons[0], 360.0) / step
    j0 = np.floor(fj).astype(int) % len(glons)
    j1 = (j0 + 1) % len(glons)
    f_lon = fj - np.floor(fj)
    v00 = field[i0, j0]
    v01 = field[i0, j1]
    v10 = field[i0 + 1, j0]
    v11 = field[i0 + 1, j1]
    return (
        v00 * (1 - f_lat) * (1 - f_lon)
        + v01 * (1 - f_lat) * f_lon
        + v10 * f_lat * (1 - f_lon)
        + v11 * f_lat * f_lon
    )


def gauss_to_native_sample(field):
    """Export sampler: gauss field -> native 64,800 cell centers (non-conservative)."""
    lats, _ = native_grid()
    lon2, lat2 = np.meshgrid(0.5 + np.arange(NATIVE_NLON) * DLON_DEG, lats)
    return sample_at_points(field, lat2, lon2)
