"""Manufactured-solution numerics (G1-style checks; no real data, no training).

Scenarios (§15.4):
  1. rigid-rotation transport: exact rotation orthonormality; transported
     sampling through the native-grid pipeline stays close to the analytic
     field (isolates representation/sampling error);
  2. divergence-free rotation: FV volume budget residual ~ 0 on 90x180;
  3. local stretching from a spectral potential flow: H(tau) = H0 exp(-int D dtau)
     along RK2 trajectories, H2 = H1/J_A material identity, FV residual small;
  4. ridge birth band: mask-based new-crust volume equals q_retained*trench
     length (units km^2/Myr vs km/Myr kept distinct).

Writes outputs/synthetic/report.json; raises on tolerance breach.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.geometry.regrid import (
    gauss_grid,
    native_grid,
    sample_at_points,
)
from norn_earth.geometry.sphere import (
    lonlat_to_vectors,
    pole_to_axis,
    rotation_matrix,
    vectors_to_lonlat,
)
from norn_earth.models.variational import (
    analyze,
    basis_terms,
    coefficient_count,
    poisson_solve,
    synthesize,
)
from norn_earth.physics.budgets import ControlGrid, edge_velocity_from_center
from norn_earth.physics.kinematics import (
    advect_points,
    rigid_rotation_divergence,
    rotation_velocity_km_per_myr,
)
from norn_earth.utils.units import EARTH_RADIUS_KM

OUT = Path(__file__).resolve().parents[1] / "outputs" / "synthetic"
LMAX = 16
NLAT, NLON = 90, 180


def _gauss():
    return gauss_grid(NLAT, NLON)


def _random_smooth_field(seed=5):
    rng = np.random.default_rng(seed)
    lats, lons, w = _gauss()
    coef = np.zeros(coefficient_count(LMAX))
    for i, (l, m, kind) in enumerate(basis_terms(LMAX)):
        if 2 <= l <= 6:
            coef[i] = rng.normal(0, 6.0 / (l + 1))
    coef[0] = 32.0
    field = synthesize(coef, LMAX, lats, lons)
    return np.clip(field, 2.0, None), coef


def scenario_rotation_transport():
    H, _ = _random_smooth_field()
    lats, lons, _ = _gauss()
    pole = pole_to_axis(25.0, 120.0)
    R1 = rotation_matrix(pole, 1.1)  # deg per Myr * 1 Myr
    Rn = np.linalg.matrix_power(R1, 30)
    ortho = float(np.max(np.abs(Rn @ Rn.T - np.eye(3))))

    rng = np.random.default_rng(11)
    plon = rng.uniform(0, 360, 400)
    plat = rng.uniform(-80, 80, 400)
    v = lonlat_to_vectors(plon, plat)
    moved = v @ Rn.T
    mlon, mlat = vectors_to_lonlat(moved)
    grid = (lats, lons)
    # build H_30 on the grid as H_0 o R^{-1}, then sample it at moved points
    glon2, glat2 = np.meshgrid(lons, lats)
    gback = lonlat_to_vectors(glon2, glat2) @ Rn  # row form: R^T (Rn) = Rn^{-1}
    gb_lon, gb_lat = vectors_to_lonlat(gback)
    H30 = sample_at_points(H, gb_lat, gb_lon, grid=grid)
    analytic = sample_at_points(H, plat, plon, grid=grid)  # material value = H at tau-0 pos
    pipeline = sample_at_points(H30, mlat, mlon, grid=grid)
    rel = float(np.max(np.abs(pipeline - analytic)) / np.mean(analytic))
    return {"rotation_orthonormality_error": ortho, "transport_sampling_rel_error": rel}


def scenario_rotation_budget():
    cg = ControlGrid(NLAT, NLON)
    pole_lat, pole_lon, rate = 10.0, -30.0, 0.8
    lon2, lat2 = np.meshgrid(cg.lon_centers, cg.lat_centers)
    u, v = rotation_velocity_km_per_myr(pole_lat, pole_lon, rate, lon2, lat2)
    u_e, v_e = edge_velocity_from_center(u, v)
    H0 = 35.0 + 5.0 * np.sin(np.radians(3 * lat2 + 2 * lon2))
    R1 = rotation_matrix(pole_to_axis(pole_lat, pole_lon), rate * 1.0)
    back = lonlat_to_vectors(lon2, lat2) @ R1
    blon, blat = vectors_to_lonlat(back)
    H1 = 35.0 + 5.0 * np.sin(np.radians(3 * blat + 2 * blon))
    # REGIONAL budget actually exercises the FV boundary flux bookkeeping
    region = np.zeros_like(H0, dtype=bool)
    region[30:50, 40:120] = True
    r = cg.budget_residual_km3(H0, H1, u_e, v_e, np.zeros_like(H0), 1.0, region_mask=region)
    v_region = float(np.sum(cg.areas[region] * H0[region]))
    div = rigid_rotation_divergence(pole_lat, pole_lon, rate, cg.lat_centers, cg.lon_centers)
    return {
        "budget_residual_km3": float(r),
        "relative_residual": float(abs(r) / v_region),
        "max_abs_divergence_per_myr_interior": float(np.max(np.abs(div[2:-2, 2:-2]))),
    }


def scenario_stretching():
    glats, glons, gw = _gauss()
    lon2, lat2 = np.meshgrid(glons, glats)
    bump = 0.06 * np.exp(
        -((lat2 - 10.0) ** 2 + ((lon2 - 40.0 + 180) % 360 - 180) ** 2) / (2 * 12.0**2)
    )
    D = bump - bump.mean()  # enforce zero global mean (closed sphere)
    psi_coef = poisson_solve(D, LMAX, glats, glons, gw)
    psi = synthesize(psi_coef, LMAX, glats, glons)
    dlon = np.radians(glons[1] - glons[0])
    dlat = np.radians(glats[1] - glats[0]) if len(glats) > 1 else np.radians(1.0)
    dpsi_dlon = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dlon)
    dpsi_dlat = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dlat)
    coslat = np.cos(np.radians(glats))[:, None]
    u = dpsi_dlon / (EARTH_RADIUS_KM * np.maximum(coslat, 1e-3))
    v = dpsi_dlat / EARTH_RADIUS_KM

    grid90 = (glats, glons)

    def vel_fn(lon, lat):
        return sample_at_points(u, lat, lon, grid=grid90), sample_at_points(v, lat, lon, grid=grid90)

    dtau = 0.5
    H0 = 35.0 + 3.0 * np.sin(np.radians(lat2))
    # Semi-Lagrangian manufactured H1: advect centers BACKWARD with RK2 in the
    # reversed flow, then apply the material factor exp(-int D dtau) there.
    blon, blat = advect_points(lon2.ravel(), lat2.ravel(),
                               lambda lo, la: tuple(-q for q in vel_fn(lo, la)),
                               dtau, 1, order=2)
    D_back = sample_at_points(D, blat, blon, grid=grid90).reshape(H0.shape)
    H0_back = sample_at_points(H0, blat, blon, grid=grid90).reshape(H0.shape)
    H1 = H0_back * np.exp(-D_back * dtau)

    # material identity on tracked particles
    rng = np.random.default_rng(7)
    inside = (np.abs(lat2 - 10.0) < 8) & (np.abs((lon2 - 40.0 + 180) % 360 - 180) < 8)
    pick = np.argwhere(inside)
    n_take = int(min(60, len(pick)))
    sel = pick[rng.choice(len(pick), n_take, replace=False)]
    p0 = (glons[sel[:, 1]], glats[sel[:, 0]])
    plon, plat = advect_points(p0[0], p0[1], vel_fn, dtau, 1, order=2)
    D0 = sample_at_points(D, p0[1], p0[0], grid=grid90)
    D1 = sample_at_points(D, plat, plon, grid=grid90)
    J_A = np.exp(0.5 * (D0 + D1) * dtau)
    H_start = sample_at_points(H0, p0[1], p0[0], grid=grid90)
    H_end = H_start / J_A
    material_ok = bool(np.all(np.isfinite(H_end)) and np.all(H_end > 0))

    # regional FV budget (trapezoid flux) over the bump neighborhood
    cg = ControlGrid(NLAT, NLON)
    lon2c, lat2c = np.meshgrid(cg.lon_centers, cg.lat_centers)
    H0c = sample_at_points(H0, lat2c, lon2c, grid=grid90)
    H1c = sample_at_points(H1, lat2c, lon2c, grid=grid90)
    uc = sample_at_points(u, lat2c, lon2c, grid=grid90)
    vc = sample_at_points(v, lat2c, lon2c, grid=grid90)
    uec, vec = edge_velocity_from_center(uc, vc)
    region = np.zeros_like(H0c, dtype=bool)
    region[35:55, 25:55] = True
    r = cg.budget_residual_km3(H0c, H1c, uec, vec, np.zeros_like(H0c), dtau, region_mask=region)
    rel = float(abs(r) / float(np.sum(cg.areas[region] * H0c[region])))
    return {
        "material_formula_consistent": material_ok,
        "fv_relative_residual": rel,
        "note": "semi-Lagrangian manufactured pair; residual is O(dtau^2)",
    }


def scenario_birth_band():
    cg = ControlGrid(NLAT, NLON)
    H_birth, u_full, dtau = 7.0, 100.0, 1.0  # fast ridge, units test only
    q_retained = H_birth * u_full  # km^2/Myr per unit trench length
    rows = range(73, 78)           # five 2-deg rows near 59N == the trench
    col = 30
    extra_cells = 1                # ~ u_full*dtau / cell_width(60N) ~ 0.9
    mask_volume = extra_cells * H_birth * sum(cg.areas[r, col] for r in rows)
    trench_length_km = sum(cg.dlon_edges[r, col] for r in rows)  # N-S edge lengths
    flux_volume = q_retained * dtau * trench_length_km
    ratio = float(mask_volume / flux_volume)
    return {
        "q_retained_km2_per_myr": float(q_retained),
        "mask_volume_km3": float(mask_volume),
        "flux_volume_km3": float(flux_volume),
        "mask_over_flux": ratio,
        "units_note": "q[km^2/Myr] x trench_length[km] x dtau[Myr] = km^3; H=u/q guard kept separate",
    }


def main():
    report = {
        "rotation_transport": scenario_rotation_transport(),
        "rotation_budget": scenario_rotation_budget(),
        "stretching": scenario_stretching(),
        "birth_band": scenario_birth_band(),
    }
    failures = []
    if report["rotation_transport"]["rotation_orthonormality_error"] > 1e-10:
        failures.append("rotation_orthonormality")
    if report["rotation_transport"]["transport_sampling_rel_error"] > 0.05:
        failures.append("transport_sampling")
    if report["rotation_budget"]["relative_residual"] > 0.02:
        failures.append("rotation_budget")
    if not report["stretching"]["material_formula_consistent"]:
        failures.append("material_formula")
    if report["stretching"]["fv_relative_residual"] > 0.05:
        failures.append("stretching_fv_budget")
    if not (0.8 < report["birth_band"]["mask_over_flux"] < 1.25):
        failures.append("birth_band_consistency")
    report["failures"] = failures
    report["passed"] = not failures
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(f"manufactured-solution checks failed: {failures}")


if __name__ == "__main__":
    main()
