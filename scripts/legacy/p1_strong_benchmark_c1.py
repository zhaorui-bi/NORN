"""P1 verification: redesigned STRONGLY time-varying benchmark + C1 decision rule.

v1.3 criterion: mixture must beat copied/point by >=10% truth MAE
(bootstrap CI excluding 0) on strongly time-varying truths, else the
chronology-operator claim is downgraded to protocol contribution.

Scenarios (sharp dynamics where age handling provably matters):
  advected_sharp : gaussian bump (8-deg width) rigidly advected ~50 deg
  arc_fast       : localized thickness pulse rising/decaying within ~8 Myr
Wide age intervals (+-20 Myr) + Student-t noise; truth known exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.splits import spatial_block_ids
from norn_earth.geometry.regrid import gauss_area_weights_km2, gauss_grid, sample_at_points
from norn_earth.geometry.sphere import pole_to_axis, rotation_matrix, lonlat_to_vectors, vectors_to_lonlat
from norn_earth.losses.chronology import make_age_nodes
from norn_earth.models.variational_fit import VariationalReconstructor
from norn_earth.utils.hashing import write_manifest

OUT = Path(__file__).resolve().parents[1] / "outputs" / "benchmark_runs"
NLAT, NLON = 90, 180
AGES = np.linspace(60.0, 0.0, 13)  # descending


def bump_field(lon2, lat2, center_lon, center_lat, width_deg=8.0, amp=18.0, base=30.0):
    d2 = (((lon2 - center_lon) + 180) % 360 - 180) ** 2 + (lat2 - center_lat) ** 2
    return base + amp * np.exp(-d2 / (2 * width_deg**2))


def make_truth(rng, scenario):
    lats, lons, _ = gauss_grid(NLAT, NLON)
    lon2, lat2 = np.meshgrid(lons, lats)
    fields = []
    if scenario == "advected_sharp":
        pole_lat, pole_lon, rate = 10.0, rng.uniform(-180, 180), rng.uniform(0.8, 1.4)
        R = rotation_matrix(pole_to_axis(pole_lat, pole_lon), rate * 5.0)  # 5 Myr steps
        base_centers = []
        back = lonlat_to_vectors(lon2, lat2)
        cur = bump_field(lon2, lat2, rng.uniform(0, 360), rng.uniform(-30, 30))
        fields.append(cur.copy())
        for k in range(1, 13):
            back = back @ R
            bl, ba = vectors_to_lonlat(back)
            c = np.unravel_index(np.argmax(cur), cur.shape)
            # advect the BUMP by rotating its center forward
            center = lonlat_to_vectors(np.array([lons[c[1]]]), np.array([lats[c[0]]]))
            moved = center @ np.linalg.matrix_power(R.T, k)
            ml, ma = vectors_to_lonlat(moved)
            cur = bump_field(lon2, lat2, float(ml[0]), float(ma[0]))
            fields.append(cur.copy())
        meta = {"pole": [pole_lat, pole_lon], "rate": rate}
    else:  # arc_fast
        c_lon, c_lat = rng.uniform(0, 360), rng.uniform(-40, 40)
        t_peak, t_width, amp = rng.uniform(20, 40), rng.uniform(3, 6), rng.uniform(12, 20)
        base = bump_field(lon2, lat2, c_lon, c_lat, width_deg=25, amp=5, base=28.0)
        fields.append(base.copy())
        for k in range(1, 13):
            a = AGES[k]
            pulse = amp * np.exp(-((a - t_peak) ** 2) / (2 * t_width**2))
            fields.append(base + bump_field(lon2, lat2, c_lon, c_lat, width_deg=6.0, amp=pulse))
        meta = {"center": [c_lat, c_lon], "t_peak": t_peak, "t_width": t_width, "amp": amp}
    return np.asarray(fields), (lats, lons), meta


def sample_obs(rng, fields, grid, n=600, halfwidth=20.0, noise=3.0):
    lats, lons = grid
    lon = rng.uniform(0, 360, n)
    lat = rng.uniform(-70, 70, n)
    true_age = rng.uniform(0, 60, n)
    y, counts, nages, nw, nlon, nlat = [], [], [], [], [], []
    for i in range(n):
        v = interp_truth(fields, lats, lons, true_age[i], lon[i], lat[i])
        y.append(float(v + rng.standard_t(4) * noise))
        lo = np.clip(true_age[i] - rng.uniform(5, halfwidth), 0, 60)
        hi = np.clip(true_age[i] + rng.uniform(5, halfwidth), 0, 60)
        nodes, w, _ = make_age_nodes(lo, hi, 8)
        counts.append(len(nodes))
        nages.extend(nodes.tolist())
        nw.extend(w.tolist())
        nlon.extend([lon[i]] * len(nodes))
        nlat.extend([lat[i]] * len(nodes))
    return {
        "y_km": np.asarray(y), "node_counts": np.asarray(counts, dtype=int),
        "node_ages": np.asarray(nages), "node_weights": np.asarray(nw),
        "node_lon": np.asarray(nlon), "node_lat": np.asarray(nlat),
        "effective_sigma_km": np.full(n, np.sqrt(noise**2 + 4.0**2)),
    }, true_age, lon, lat


def interp_truth(fields, glats, glons, age, lon, lat):
    asc = AGES[::-1]
    i = int(np.clip(np.searchsorted(asc, age) - 1, 0, len(asc) - 2))
    f = (age - asc[i]) / max(asc[i + 1] - asc[i], 1e-12)
    v0 = sample_at_points(fields[::-1][i], np.array([lat]), np.array([lon]), grid=(glats, glons))
    v1 = sample_at_points(fields[::-1][i + 1], np.array([lat]), np.array([lon]), grid=(glats, glons))
    return float((1 - f) * np.asarray(v0).reshape(-1)[0] + f * np.asarray(v1).reshape(-1)[0])


def modern_of(fields, rng, sigma=1.5):
    glon2, glat2 = np.meshgrid(gauss_grid(NLAT, NLON)[1], gauss_grid(NLAT, NLON)[0])
    return {
        "H_km": (fields[-1] + rng.normal(0, sigma, fields[-1].shape)).astype(np.float32),
        "areas_km2": gauss_area_weights_km2(NLAT, NLON).astype(np.float32),
        "sigma_km": np.full(fields[-1].shape, sigma, dtype=np.float32),
        "block_ids": spatial_block_ids(glon2.ravel(), glat2.ravel()).reshape(NLAT, NLON).astype(np.int32),
    }


def to_point(batch):
    n = len(batch["node_counts"])
    starts = np.concatenate([[0], np.cumsum(batch["node_counts"])])
    mid = np.array([0.5 * (batch["node_ages"][starts[i]] + batch["node_ages"][starts[i + 1] - 1]) for i in range(n)])
    return {"y_km": batch["y_km"], "node_counts": np.ones(n, dtype=int), "node_ages": mid,
            "node_weights": np.ones(n),
            "node_lon": np.array([batch["node_lon"][starts[i]] for i in range(n)]),
            "node_lat": np.array([batch["node_lat"][starts[i]] for i in range(n)]),
            "effective_sigma_km": batch["effective_sigma_km"]}


def main():
    rng0 = np.random.default_rng(11)
    results = []
    for inst in range(8):
        scenario = "advected_sharp" if inst % 2 == 0 else "arc_fast"
        rng = np.random.default_rng(500 + inst)
        fields, grid, meta = make_truth(rng, scenario)
        obs, true_age, lon, lat = sample_obs(rng, fields, grid)
        modern = modern_of(fields, rng)
        truth = np.array([interp_truth(fields, *grid, a, lo, la) for a, lo, la in zip(true_age, lon, lat)])
        row = {"instance": f"{scenario}_{inst}", "n": int(len(truth))}
        for mode, batch in (("mixture", obs), ("copied", obs), ("point", to_point(obs))):
            m = VariationalReconstructor(lmax=16, n_anchors=13, anchor_step=5.0, lam_modern=0.3,
                                         lam_obs=1.0, lam_smooth=1e-3,
                                         loss_mode=("expected_point" if mode == "copied" else "mixture"))
            m.fit(batch, modern, maxiter=90)
            mu, _ = m.predict_nodes(lon, lat, true_age)
            row[mode] = float(np.mean(np.abs(mu - truth)))
        results.append(row)
        print(f"{row['instance']}: mixture={row['mixture']:.3f} copied={row['copied']:.3f} point={row['point']:.3f}")

    # bootstrap CI on paired differences (cluster = instance)
    def boot(a, b, n_boot=2000, seed=0):
        r = np.random.default_rng(seed)
        idx = np.arange(len(results))
        diffs = []
        for _ in range(n_boot):
            take = r.choice(idx, len(idx), replace=True)
            A = np.array([results[i][a] for i in take])
            B = np.array([results[i][b] for i in take])
            diffs.append(np.mean(A - B))
        return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    mc = boot("mixture", "copied")
    mp = boot("mixture", "point")
    gain_vs_copied = 1 - np.mean([r["mixture"] for r in results]) / np.mean([r["copied"] for r in results])
    gain_vs_point = 1 - np.mean([r["mixture"] for r in results]) / np.mean([r["point"] for r in results])
    verdict = {
        "criterion": "mixture beats alternative by >=10% truth MAE with bootstrap CI excluding 0",
        "gain_vs_copied": float(gain_vs_copied), "gain_vs_point": float(gain_vs_point),
        "mixture_minus_copied_ci95": mc, "mixture_minus_point_ci95": mp,
        "rule_passed": bool(gain_vs_copied >= 0.10 and mc[1] > 0 and gain_vs_point >= 0.10 and mp[1] > 0),
    }
    print(json.dumps(verdict, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(OUT / "p1_c1_strongbenchmark.json", {"results": results, "verdict": verdict})


if __name__ == "__main__":
    main()
