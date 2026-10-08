"""P3+P5 verification: capacity study + per-domain source scales.

Protocol (v1.3): selection ONLY on the burned development window P-C 40-50
(never on reserved 15-25 / 50-60 / P-A). Grid:
  lmax in {8, 16, 24}  x  source-scale policy in {uniform 8km, per-domain}
Per-domain scales frozen from G0 diagnostics: continental ~9 km,
transitional/oceanic ~26 km (domain from DAT value at the site).
Reports dev-window test-side NLL/MAE; best config frozen into configs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.batches import assemble_observation_batch
from norn_earth.data.modern import assemble_modern_batch
from norn_earth.data.observations import build_observation_table
from norn_earth.data.splits import chronological_holdout
from norn_earth.data.trajectories import build_trajectory_rows, trajectory_lookup
from norn_earth.losses.chronology import mixture_nll
from norn_earth.losses.likelihood import SourceRegistry
from norn_earth.models.variational_fit import VariationalReconstructor
from norn_earth.utils.hashing import sha256_file, write_manifest

OUT = Path(__file__).resolve().parents[1] / "outputs" / "pilot_stat"


def subset(batch, mask):
    idx = np.flatnonzero(mask)
    starts = np.concatenate([[0], np.cumsum(batch["node_counts"])])
    node_cols = ("node_ages", "node_weights", "node_lon", "node_lat", "node_from_trajectory")
    sel = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in idx if batch["node_counts"][i] > 0])
    out = {}
    for k, v in batch.items():
        if not isinstance(v, np.ndarray):
            continue
        out[k] = v[idx] if k in ("y_km", "effective_sigma_km", "node_counts", "record_sigma_km", "source_sigma_km") else (v[sel] if k in node_cols else v[idx])
    return out


def domain_sigma(xlsx_table, dat_path, dom_sigma=(9.0, 26.0), threshold=20.0):
    """Per-record source sigma by crust domain (DAT thickness at the site)."""
    dat = np.loadtxt(dat_path)
    H = -dat[:, 2].reshape(180, 360)

    def look(lon, lat):
        j = int(np.clip(round(lon - 0.5), 0, 359)) % 360
        i = int(np.clip(round(lat + 89.5), 0, 179))
        return H[i, j]
    vals = np.array([look(lo, la) for lo, la in zip(xlsx_table["present_lon"], xlsx_table["present_lat"])])
    return np.where(vals >= threshold, dom_sigma[0], dom_sigma[1]), vals


def score(model, sub):
    counts = sub["node_counts"]
    starts = np.concatenate([[0], np.cumsum(counts)])
    n = len(counts)
    K = int(max(counts.max(), 1))
    pad_mu = np.zeros((n, K)); pad_w = np.zeros((n, K))
    for i in range(n):
        s, e = starts[i], starts[i + 1]
        if e > s:
            mu, _ = model.predict_nodes(sub["node_lon"][s:e], sub["node_lat"][s:e], sub["node_ages"][s:e])
            pad_mu[i, : e - s] = mu
            pad_w[i, : e - s] = sub["node_weights"][s:e]
    keep = counts > 0
    nll = mixture_nll(sub["y_km"][keep], pad_mu[keep], pad_w[keep], sub["effective_sigma_km"][keep])
    rep = np.array([starts[i] for i in range(n) if counts[i] > 0])
    mu0, _ = model.predict_nodes(sub["node_lon"][rep], sub["node_lat"][rep], sub["node_ages"][rep])
    return {"nll": float(np.mean(nll)), "mae": float(np.mean(np.abs(mu0 - sub["y_km"][keep]))),
            "corr": float(np.corrcoef(mu0, sub["y_km"][keep])[0, 1])}


def main():
    xlsx = ROOT / "古地壳厚度数据_新生代_处理后_final.xlsx"
    dat_path = ROOT / "present_crustal_thickness.dat"
    raw = pd.read_excel(xlsx).drop_duplicates().reset_index(drop=True)
    table = build_observation_table(xlsx)
    lookup = trajectory_lookup(build_trajectory_rows(raw)[0])
    train_m, dev_m, _ = chronological_holdout(table, (40.0, 50.0))
    modern = assemble_modern_batch(dat_path, sigma_km=3.0)

    dom_sigma, dom_vals = domain_sigma(table, dat_path)
    print(f"domain split: continental {(dom_sigma==9.0).mean()*100:.1f}% / thin {(dom_sigma==26.0).mean()*100:.1f}%")

    results = {}
    for policy in ("uniform", "per_domain"):
        reg = SourceRegistry()
        if policy == "uniform":
            reg.register("unverified_proxy", 8.0)
            batch = assemble_observation_batch(table, lookup, reg)
        else:
            reg0 = SourceRegistry(); reg0.register("unverified_proxy", 1.0)  # placeholder; overridden below
            batch = assemble_observation_batch(table, lookup, reg0, source_sigma_override=dom_sigma)
        train = subset(batch, train_m)
        dev = subset(batch, dev_m)
        for lmax in (8, 16, 24):
            t0 = time.time()
            m = VariationalReconstructor(lmax=lmax, n_anchors=13, anchor_step=5.0,
                                         lam_modern=0.5, lam_obs=1.0, lam_smooth=1e-3)
            m.fit(train, modern, maxiter=60)
            sc = score(m, dev)
            results[f"{policy}_lmax{lmax}"] = {**sc, "fit_s": time.time() - t0}
            print(f"{policy:>10s} lmax={lmax}: dev NLL={sc['nll']:.4f} MAE={sc['mae']:.3f} corr={sc['corr']:.3f} ({time.time()-t0:.0f}s)")

    best = min(results, key=lambda k: results[k]["nll"])
    print("SELECTED (dev-window NLL):", best)
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(OUT / "p3_p5_capacity_scales.json", {
        "window": "P-C 40-50 (burned development window; selection only)",
        "results": results,
        "selected": best,
        "per_domain_scales_km": {"continental": 9.0, "transitional_oceanic": 26.0, "domain_rule": "DAT>=20km at site"},
    })


if __name__ == "__main__":
    main()
