"""Synthetic identifiability benchmark generator (§12).

Parameter-family instances (NOT the single Earth history): rigid rotation,
local stretching, ridge birth-band sweep, arc source pulse. Truth fields are
analytic/semi-Lagrangian on the 90x180 gauss grid at 13 anchors (5 Myr);
observations sample the truth with configurable age-interval mixtures and
Student-t noise; a modern endpoint with Gaussian noise closes each instance.

Default pilot: 12 instances (4 scenarios x 3 seeds); --full generates the
280-instance candidate set (development/val/test = 200/40/40).

Running this generator trains nothing and touches no real data.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.batches import batch_slices
from norn_earth.geometry.regrid import gauss_grid, sample_at_points
from norn_earth.models.variational import basis_terms, coefficient_count, synthesize
from norn_earth.utils.hashing import sha256_file, write_manifest

OUT = Path(__file__).resolve().parents[1] / "outputs" / "synthetic_benchmark"
NLAT, NLON, LMAX = 90, 180, 10


def _grid():
    return gauss_grid(NLAT, NLON)


def random_truth(rng, scenario):
    """Return (fields[T+1], meta): fields[t] at age = T - t (tau steps)."""
    lats, lons, w = _grid()
    lon2, lat2 = np.meshgrid(lons, lats)
    T = 60.0
    n_t = 13
    ages = np.linspace(T, 0.0, n_t)
    coef0 = np.zeros(coefficient_count(LMAX))
    terms = basis_terms(LMAX)
    for i, (l, m, kind) in enumerate(terms):
        if 1 <= l <= 5:
            coef0[i] = rng.normal(0, 8.0 / (l + 1))
    coef0[0] = 0.0
    H0 = np.clip(30.0 + synthesize(coef0, LMAX, lats, lons), 3.0, None)
    fields = [H0]
    if scenario == "rotation":
        # pure advection: analytic rotation of the field each step
        pole_lat, pole_lon, rate = rng.uniform(-40, 40), rng.uniform(-180, 180), rng.uniform(0.3, 1.2)
        from norn_earth.geometry.sphere import pole_to_axis, rotation_matrix, lonlat_to_vectors, vectors_to_lonlat

        dtau = T / (n_t - 1)
        R1 = rotation_matrix(pole_to_axis(pole_lat, pole_lon), rate * dtau)
        back = lonlat_to_vectors(lon2, lat2)
        cur = H0.copy()
        for k in range(1, n_t):
            back = back @ R1
            bl, ba = vectors_to_lonlat(back)
            cur = sample_at_points(H0, ba, bl, grid=(lats, lons))
            fields.append(cur)
        meta = {"pole": [pole_lat, pole_lon], "rate_deg_per_myr": rate}
    elif scenario == "stretching":
        amp = rng.uniform(0.02, 0.08)
        c0 = (rng.uniform(-60, 60), rng.uniform(0, 360))
        fields = [H0]
        for k in range(1, n_t):
            tau = k * (T / (n_t - 1))
            D = amp * np.exp(-(((lat2 - c0[0]) ** 2 + (((lon2 - c0[1]) + 180) % 360 - 180) ** 2) / (2 * 15.0**2)))
            fields.append(H0 * np.exp(-D * tau))
        meta = {"amp_per_myr": amp, "center": c0}
    elif scenario == "birth_band":
        # deterministic sweep: gaussian band of new `birth` km crust
        u_full = rng.uniform(30, 80)
        birth = rng.uniform(5, 9)
        width_deg = rng.uniform(4, 10)
        base = rng.uniform(0, 360)
        fields = [H0]
        for k in range(1, n_t):
            tau = k * (T / (n_t - 1))
            center = (base + 40.0 * tau / T * 180.0) % 360
            band = np.exp(-((((lon2 - center) + 180) % 360 - 180) ** 2) / (2 * width_deg**2))
            fields.append(H0 * (1 - 0.5 * band) + birth * band * np.exp(-((lat2) / 60.0) ** 2))
        meta = {"u_full_km_per_myr": u_full, "birth_km": birth, "width_deg": width_deg, "base_lon": base}
    elif scenario == "arc_pulse":
        c0 = (rng.uniform(-50, 50), rng.uniform(0, 360))
        amp = rng.uniform(0.02, 0.06)
        fields = [H0]
        for k in range(1, n_t):
            tau = k * (T / (n_t - 1))
            pulse = np.exp(-((tau - 30.0) ** 2) / (2 * 15.0**2))
            d = np.sqrt((((lon2 - c0[1]) + 180) % 360 - 180) ** 2 * np.cos(np.radians(lat2)) ** 2 + (lat2 - c0[0]) ** 2)
            q = amp * pulse * np.exp(-((d - 8.0) ** 2) / (2 * 6.0**2))
            fields.append(fields[-1] + q * (T / (n_t - 1)))
        meta = {"amp": amp, "center": c0}
    else:
        raise ValueError(scenario)
    return np.asarray(fields), ages, {"scenario": scenario, **meta}


def sample_observations(rng, fields, ages, n_obs=900):
    """Site sampling with mixed age semantics (point/narrow/wide)."""
    lats, lons, _ = _grid()
    lon = rng.uniform(0, 360, n_obs)
    lat = rng.uniform(-80, 80, n_obs)
    kind = rng.choice(["point", "narrow", "wide"], size=n_obs, p=[0.4, 0.35, 0.25])
    y, lo_age, hi_age, node_ages, node_w, node_lon, node_lat, counts = [], [], [], [], [], [], [], []
    for i in range(n_obs):
        true_age = rng.uniform(0, 60)
        if kind[i] == "point":
            lo = hi = true_age
        elif kind[i] == "narrow":
            lo, hi = max(0.0, true_age - 2.5), min(60.0, true_age + 2.5)
        else:
            lo, hi = max(0.0, true_age - rng.uniform(10, 30)), min(60.0, true_age + rng.uniform(10, 30))
        # truth value at true_age via time interpolation of anchors
        t = np.clip(true_age, ages[-1], ages[0])
        H_true = _interp_field(fields, ages, t, lat[i], lon[i], lats, lons)
        noise = rng.standard_t(4) * 3.0
        y.append(float(H_true + noise))
        lo_age.append(lo)
        hi_age.append(hi)
        from norn_earth.losses.chronology import make_age_nodes

        nodes, w, _ = make_age_nodes(lo, hi, 1 if kind[i] == "point" else (3 if kind[i] == "narrow" else 8))
        counts.append(len(nodes))
        node_ages.extend(nodes.tolist())
        node_w.extend(w.tolist())
        node_lon.extend([lon[i]] * len(nodes))
        node_lat.extend([lat[i]] * len(nodes))
    return {
        "y_km": np.asarray(y),
        "age_lower": np.asarray(lo_age),
        "age_upper": np.asarray(hi_age),
        "node_counts": np.asarray(counts, dtype=int),
        "node_ages": np.asarray(node_ages),
        "node_weights": np.asarray(node_w),
        "node_lon": np.asarray(node_lon),
        "node_lat": np.asarray(node_lat),
        "effective_sigma_km": np.full(len(y), np.sqrt(3.0**2 + 4.0**2)),
    }


def _interp_field(fields, ages, age, lat, lon, glats, glons):
    """Time-interpolate stacked anchor fields at one point."""
    ages_asc = ages[::-1]
    idx = np.clip(np.searchsorted(ages_asc, age) - 1, 0, len(ages_asc) - 2)
    f = (age - ages_asc[idx]) / max(ages_asc[idx + 1] - ages_asc[idx], 1e-12)
    H0 = sample_at_points(fields[::-1][idx], np.array([lat]), np.array([lon]), grid=(glats, glons))
    H1 = sample_at_points(fields[::-1][idx + 1], np.array([lat]), np.array([lon]), grid=(glats, glons))
    return float((1 - f) * H0 + f * H1)


def make_modern(fields, ages, rng, sigma=1.5):
    from norn_earth.geometry.regrid import native_cell_areas_km2
    from norn_earth.data.modern import modern_blocks

    H0 = fields[-1]  # youngest anchor == age 0
    return {
        "H_km": (H0 + rng.normal(0, sigma, H0.shape)).astype(np.float32),
        "areas_km2": native_cell_areas_km2().astype(np.float32),
        "sigma_km": np.full(H0.shape, sigma, dtype=np.float32),
        "block_ids": modern_blocks().astype(np.int32),
        "sha256": None,
        "sigma_is_calibrated": True,  # synthetic truth: noise is known
    }


def generate(n_instances=12, seed0=0, out_dir=OUT):
    scenarios = ["rotation", "stretching", "birth_band", "arc_pulse"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"instances": [], "n_instances": int(n_instances)}
    for i in range(n_instances):
        rng = np.random.default_rng(seed0 * 1000 + i)
        scen = scenarios[i % len(scenarios)]
        fields, ages, meta = random_truth(rng, scen)
        obs = sample_observations(rng, fields, ages)
        modern = make_modern(fields, ages, rng)
        name = f"inst_{i:03d}_{scen}"
        np.savez_compressed(
            out_dir / f"{name}.npz",
            truth_fields=fields.astype(np.float32),
            truth_ages=ages.astype(np.float32),
            **{k: v for k, v in obs.items()},
            modern_H=modern["H_km"],
            modern_sigma=modern["sigma_km"],
        )
        manifest["instances"].append({"name": name, "meta": meta,
                                      "n_obs": int(len(obs["y_km"]))})
        print(f"{name}: obs={len(obs['y_km'])}")
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {out_dir / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="12-instance pilot (default)")
    parser.add_argument("--full", action="store_true", help="280-instance candidate set")
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()
    n = 280 if args.full else 12
    generate(n_instances=n, seed0=args.seed0)


if __name__ == "__main__":
    main()
