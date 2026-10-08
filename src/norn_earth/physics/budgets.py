"""Finite-volume crustal volume budgets on control grids (§6.7/§6.8).

All terms in km^3. For a fixed-identity control region j integrated over
tau_n -> tau_{n+1}:

    r_jn = V_j^{n+1} - V_j^n + I_flux - I_source,
    I_flux = int int_boundary H (v - v_b).n dl dtau,
    I_source = int int_region q dA dtau.

Ridge birth enters as boundary influx of new material with its birth
thickness; trench death is outward flux of the removed material; neither is
double-counted as a source. Unknown fluxes are masked (never assumed zero).
Residuals are normalized by H_ref * A_ref before entering any objective.
"""

import numpy as np

from ..utils.units import EARTH_RADIUS_KM

H_REF_KM = 35.0
A_REF_KM2 = (111.0**2) * 100.0  # ~10 deg x 10 deg reference region


class ControlGrid:
    """Uniform lat/lon finite-volume grid (default 90x180 = 2 deg cells)."""

    def __init__(self, nlat=90, nlon=180):
        self.nlat, self.nlon = nlat, nlon
        self.lat_edges = -90.0 + 180.0 / nlat * np.arange(nlat + 1)
        self.lat_centers = 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])
        self.lon_edges = 360.0 / nlon * np.arange(nlon + 1)
        self.lon_centers = 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])
        dphi = np.radians(np.diff(self.lat_edges))
        self.areas = (
            (EARTH_RADIUS_KM**2)
            * np.radians(360.0 / nlon)
            * np.repeat(
                (np.sin(np.radians(self.lat_edges[1:])) - np.sin(np.radians(self.lat_edges[:-1])))[
                    :, None
                ],
                nlon,
                axis=1,
            )
        )
        # lon-facing edges depend only on latitude band; lat-facing edges live at
        # lat_edges (pole rows degenerate to zero length, cos(pole)=0)
        self.dlon_edges = EARTH_RADIUS_KM * dphi[:, None] * np.ones((nlat, nlon + 1))
        cos_edges = np.cos(np.radians(self.lat_edges))[:, None]
        self.dlat_edges = (
            EARTH_RADIUS_KM * np.radians(360.0 / nlon) * cos_edges * np.ones((nlat + 1, nlon))
        )

    def total_area_km2(self):
        return float(self.areas.sum())

    def volume_km3(self, h_km):
        return float(np.sum(self.areas * np.asarray(h_km, dtype=float)))

    def cell_net_outflux_per_myr(self, h_km, u_edge_km_per_myr, v_edge_km_per_myr):
        """Per-cell net outward flux (km^3/Myr per cell).

        u_edge: (nlat, nlon+1) on lon edges (edge 0 and nlon are the seam);
        v_edge: (nlat+1, nlon) on lat edges (pole rows degenerate to length 0).
        Interior edges cancel when summed over a region: regional outflow is
        exactly the boundary flux, and the GLOBAL sum vanishes numerically.
        """
        h = np.asarray(h_km, dtype=float)
        u = np.asarray(u_edge_km_per_myr, dtype=float)
        v = np.asarray(v_edge_km_per_myr, dtype=float)
        h_west = 0.5 * (np.concatenate([h[:, -1:], h[:, :-1]], axis=1) + h)
        dl = self.dlon_edges  # (nlat, nlon+1)
        h_east = 0.5 * (h + np.concatenate([h[:, 1:], h[:, :1]], axis=1))
        flux_west = dl[:, :-1] * h_west * u[:, :-1]
        flux_east = dl[:, 1:] * h_east * u[:, 1:]
        h_south = np.vstack([h[0], 0.5 * (h[:-1] + h[1:])])
        h_north = np.vstack([0.5 * (h[:-1] + h[1:]), h[-1]])
        flux_south = self.dlat_edges[:-1] * h_south * v[:-1]
        flux_north = self.dlat_edges[1:] * h_north * v[1:]
        return flux_east - flux_west + flux_north - flux_south

    def flux_out_km3_per_myr(self, h_km, u_edge_km_per_myr, v_edge_km_per_myr, region_mask=None):
        """Regional (default: global) net outward flux in km^3/Myr."""
        net = self.cell_net_outflux_per_myr(h_km, u_edge_km_per_myr, v_edge_km_per_myr)
        if region_mask is None:
            return float(net.sum())
        return float(net[np.asarray(region_mask, dtype=bool)].sum())

    def source_integral_km3_per_myr(self, q_km_per_myr, active_mask=None):
        q = np.asarray(q_km_per_myr, dtype=float)
        if active_mask is None:
            active_mask = np.ones_like(q, dtype=bool)
        return float(np.sum(self.areas * np.where(active_mask, q, 0.0)))

    def budget_residual_km3(
        self, h_start, h_end, u_edge, v_edge, q, dtau_myr, active_mask=None, region_mask=None
    ):
        """Positive residual = unexplained volume loss (module docstring).

        region_mask selects the reporting region; active_mask gates the source
        integral (unknown processes masked, never zero-filled silently).
        """
        region = (
            np.ones(self.areas.shape, dtype=bool)
            if region_mask is None
            else np.asarray(region_mask, dtype=bool)
        )
        v0 = float(np.sum(self.areas[region] * np.asarray(h_start, dtype=float)[region]))
        v1 = float(np.sum(self.areas[region] * np.asarray(h_end, dtype=float)[region]))
        net = self.cell_net_outflux_per_myr(
            0.5 * (np.asarray(h_start, dtype=float) + np.asarray(h_end, dtype=float)),
            u_edge,
            v_edge,
        )
        flux = float(net[region].sum())
        q_arr = np.asarray(q, dtype=float)
        if active_mask is None:
            src = float(np.sum(self.areas[region] * q_arr[region]))
        else:
            src = float(
                np.sum(
                    self.areas[region] * np.where(np.asarray(active_mask, bool), q_arr, 0.0)[region]
                )
            )
        return (v1 - v0) + dtau_myr * (flux - src)

    @staticmethod
    def normalize_residual(residual_km3, h_ref=H_REF_KM, a_ref=A_REF_KM2):
        return float(residual_km3) / (h_ref * a_ref)

    def coverage_report(self, active_mask):
        """Fraction of sphere area with an active budget constraint."""
        active = np.asarray(active_mask, dtype=bool)
        return {
            "active_fraction_of_area": float(self.areas[active].sum() / self.areas.sum()),
            "active_cells": int(active.sum()),
            "total_cells": int(active.size),
        }


def edge_velocity_from_center(u_center, v_center):
    """Cell-center -> staggered edges. Lon edges: (nlat, nlon+1), seam columns
    share the wrapped average; lat edges: (nlat+1, nlon), pole rows zero
    (degenerate edges)."""
    u = np.asarray(u_center, dtype=float)
    v = np.asarray(v_center, dtype=float)
    u_pair = np.concatenate([u[:, -1:], u, u[:, :1]], axis=1)
    u_edge = 0.5 * (u_pair[:, :-1] + u_pair[:, 1:])
    v_edge = np.zeros((v.shape[0] + 1, v.shape[1]))
    v_edge[1:-1] = 0.5 * (v[:-1] + v[1:])
    return u_edge, v_edge
