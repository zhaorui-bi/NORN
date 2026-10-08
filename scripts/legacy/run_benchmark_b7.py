"""Run B7 (variational) + B0 (flat) on synthetic benchmark instances.

Synthetic-only: trains nothing on real data; demonstrates the full
instance-level loop (load -> batch -> fit -> score vs known truth) that the
gated real-data pilot will reuse. Vanished/birth regions score separately
when the instance provides a mask (none in pilot scenarios; reported as n/a).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.geometry.regrid import gauss_grid, sample_at_points
from norn_earth.models.variational_fit import VariationalReconstructor
from norn_earth.utils.hashing import write_manifest

BENCH = Path(__file__).resolve().parents[1] / "outputs" / "synthetic_benchmark"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "benchmark_runs"


def load_instance(name):
    z = np.load(BENCH / f"{name}.npz")
    return z


def truth_at(z, lon, lat, age):
    fields = z["truth_fields"]
    ages = z["truth_ages"]  # descending 60 -> 0
    lats, lons, _ = gauss_grid(90, 180)
    ages_asc = ages[::-1]
    idx = int(np.clip(np.searchsorted(ages_asc, age) - 1, 0, len(ages_asc) - 2))
    f = (age - ages_asc[idx]) / max(ages_asc[idx + 1] - ages_asc[idx], 1e-12)
    v0 = sample_at_points(fields[::-1][idx], np.array([lat]), np.array([lon]), grid=(lats, lons))
    v1 = sample_at_points(fields[::-1][idx + 1], np.array([lat]), np.array([lon]), grid=(lats, lons))
    return float((1 - f) * v0 + f * v1)


def run(name, max_obs=500, maxiter=120):
    z = load_instance(name)
    n = len(z["y_km"])
    take = np.arange(0, n, max(1, n // max_obs))[:max_obs]
    full_counts = z["node_counts"]
    starts = np.concatenate([[0], np.cumsum(full_counts)])
    sel_nodes = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in take])
    obs = {
        "y_km": z["y_km"][take],
        "node_counts": full_counts[take],
        "node_ages": z["node_ages"][sel_nodes],
        "node_weights": z["node_weights"][sel_nodes],
        "node_lon": z["node_lon"][sel_nodes],
        "node_lat": z["node_lat"][sel_nodes],
        "effective_sigma_km": z["effective_sigma_km"][take],
    }
    from norn_earth.data.splits import spatial_block_ids
    from norn_earth.geometry.regrid import gauss_area_weights_km2, gauss_grid

    glats, glons, _ = gauss_grid(90, 180)
    glon2, glat2 = np.meshgrid(glons, glats)
    blocks = spatial_block_ids(glon2.ravel(), glat2.ravel()).reshape(90, 180).astype(np.int32)
    modern = {
        "H_km": z["modern_H"],
        "areas_km2": gauss_area_weights_km2(90, 180).astype(np.float32),
        "sigma_km": z["modern_sigma"],
        "block_ids": blocks,
    }
    model = VariationalReconstructor(lmax=8, n_anchors=13, anchor_step=5.0,
                                     lam_modern=0.3, lam_obs=1.0, lam_smooth=1e-3)
    history = model.fit(obs, modern, maxiter=maxiter)
    mu, _ = model.predict_nodes(obs["node_lon"], obs["node_lat"], obs["node_ages"])
    truth = np.array([truth_at(z, lo, la, a)
                      for lo, la, a in zip(obs["node_lon"], obs["node_lat"], obs["node_ages"])])
    flat = np.full(len(truth), float(np.median(obs["y_km"])))
    return {
        "instance": name,
        "n_obs": int(len(take)),
        "iterations": len(history),
        "b7_mae_vs_truth": float(np.mean(np.abs(mu - truth))),
        "b0_mae_vs_truth": float(np.mean(np.abs(flat - truth))),
        "b7_obs_corr": float(np.corrcoef(mu, truth)[0, 1]),
    }


def main():
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    names = [inst["name"] for inst in manifest["instances"]][:2]
    results = [run(nm) for nm in names]
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(OUT / "b7_synthetic_smoke.json", {"results": results})
    for r in results:
        print(f"{r['instance']}: B7 MAE={r['b7_mae_vs_truth']:.2f} km vs B0 {r['b0_mae_vs_truth']:.2f} km "
              f"(obs corr {r['b7_obs_corr']:.2f}, {r['iterations']} iters)")


if __name__ == "__main__":
    main()
