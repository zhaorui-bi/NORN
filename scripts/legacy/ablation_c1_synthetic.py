"""C1 ablation on synthetic benchmark: label-copying vs representative-age vs
chronology-aware mixture likelihood (design §14.1 item 1).

Runs B7 (same capacity) under three supervision modes on each instance's
observations with WIDE age uncertainty injected (the regime where the modes
differ), scores against the KNOWN truth at true ages. Synthetic only.

Modes:
  copied     : expected-point objective over all age nodes (label copying)
  point      : single node at the interval midpoint (representative age)
  mixture    : chronology-aware marginal NLL (the proposed operator)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.geometry.regrid import gauss_grid, sample_at_points
from norn_earth.losses.chronology import make_age_nodes
from norn_earth.models.variational_fit import VariationalReconstructor
from norn_earth.utils.hashing import write_manifest

BENCH = Path(__file__).resolve().parents[1] / "outputs" / "synthetic_benchmark"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "benchmark_runs"


def truth_at(z, lon, lat, age):
    fields, ages = z["truth_fields"], z["truth_ages"]
    lats, lons, _ = gauss_grid(90, 180)
    asc = ages[::-1]
    i = int(np.clip(np.searchsorted(asc, age) - 1, 0, len(asc) - 2))
    f = (age - asc[i]) / max(asc[i + 1] - asc[i], 1e-12)
    v0 = sample_at_points(fields[::-1][i], np.array([lat]), np.array([lon]), grid=(lats, lons))
    v1 = sample_at_points(fields[::-1][i + 1], np.array([lat]), np.array([lon]), grid=(lats, lons))
    return float((1 - f) * np.asarray(v0).reshape(-1)[0] + f * np.asarray(v1).reshape(-1)[0])


def wide_age_batch(z, rng, n=600, halfwidth=20.0):
    """Rebuild observations with WIDE intervals and known true ages."""
    n_total = len(z["y_km"])
    take = rng.choice(n_total, min(n, n_total), replace=False)
    true_age = np.clip(z["node_ages"][take], 0, 60)  # point obs: true ages
    lo = np.clip(true_age - rng.uniform(5, halfwidth, len(take)), 0, 60)
    hi = np.clip(true_age + rng.uniform(5, halfwidth, len(take)), 0, 60)
    y, counts, nages, nw, nlon, nlat = [], [], [], [], [], []
    for i in range(len(take)):
        k = take[i]
        yv = z["y_km"][k]
        nodes, w, _ = make_age_nodes(lo[i], hi[i], 8)
        y.append(yv)
        counts.append(len(nodes))
        nages.extend(nodes.tolist())
        nw.extend(w.tolist())
        nlon.extend([z["node_lon"][k]] * len(nodes))
        nlat.extend([z["node_lat"][k]] * len(nodes))
    base = {
        "y_km": np.asarray(y),
        "node_counts": np.asarray(counts, dtype=int),
        "node_ages": np.asarray(nages),
        "node_weights": np.asarray(nw),
        "node_lon": np.asarray(nlon),
        "node_lat": np.asarray(nlat),
        "effective_sigma_km": z["effective_sigma_km"][take],
    }
    return base, true_age, z["node_lon"][take], z["node_lat"][take]


def to_point(batch):
    """Representative-age (midpoint) single-node variant."""
    n = len(batch["node_counts"])
    starts = np.concatenate([[0], np.cumsum(batch["node_counts"])])
    mid = np.array([0.5 * (batch["node_ages"][starts[i]] + batch["node_ages"][starts[i + 1] - 1]) for i in range(n)])
    return {
        "y_km": batch["y_km"],
        "node_counts": np.ones(n, dtype=int),
        "node_ages": mid,
        "node_weights": np.ones(n),
        "node_lon": np.array([batch["node_lon"][starts[i]] for i in range(n)]),
        "node_lat": np.array([batch["node_lat"][starts[i]] for i in range(n)]),
        "effective_sigma_km": batch["effective_sigma_km"],
    }


def run(instance, seed=0):
    z = np.load(BENCH / f"{instance}.npz")
    rng = np.random.default_rng(seed)
    base, true_age, lon, lat = wide_age_batch(z, rng)
    from norn_earth.data.splits import spatial_block_ids
    from norn_earth.geometry.regrid import gauss_area_weights_km2, gauss_grid

    glats, glons, _ = gauss_grid(90, 180)
    glon2, glat2 = np.meshgrid(glons, glats)
    modern = {
        "H_km": z["modern_H"],
        "areas_km2": gauss_area_weights_km2(90, 180).astype(np.float32),
        "sigma_km": z["modern_sigma"],
        "block_ids": spatial_block_ids(glon2.ravel(), glat2.ravel()).reshape(90, 180).astype(np.int32),
    }
    truth = np.array([truth_at(z, lo_, la_, a) for lo_, la_, a in zip(lon, lat, true_age)])
    out = {"instance": instance, "n": int(len(truth))}
    for mode, batch in (("mixture", base), ("copied", base), ("point", to_point(base))):
        model = VariationalReconstructor(
            lmax=8, n_anchors=13, anchor_step=5.0, lam_modern=0.3, lam_obs=1.0,
            lam_smooth=1e-3, loss_mode=("expected_point" if mode == "copied" else "mixture"),
        )
        model.fit(batch, modern, maxiter=100)
        mu, _ = model.predict_nodes(lon, lat, true_age)
        out[mode] = float(np.mean(np.abs(mu - truth)))
    return out


def main():
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    results = [run(inst["name"]) for inst in manifest["instances"][:6]]
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(OUT / "c1_ablation_synthetic.json", {"results": results})
    print(f"{'instance':<22s} {'mixture':>8s} {'copied':>8s} {'point':>8s}")
    for r in results:
        print(f"{r['instance']:<22s} {r['mixture']:8.3f} {r['copied']:8.3f} {r['point']:8.3f}")
    agg = {k: float(np.mean([r[k] for r in results])) for k in ("mixture", "copied", "point")}
    print("MEAN:", {k: round(v, 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
