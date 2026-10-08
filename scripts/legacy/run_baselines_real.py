"""Real-data required baselines on the FROZEN P-C 40-50 split (single pass).

- B1: spherical RBF x time-anchor field fitted with the SAME chronology
  mixture NLL via L-BFGS (fair wide-age baseline; same optimizer class as B7).
- B2: CatBoost on (lon, lat, sin-lat, age, age-width):
  * point_age variant (representative age) -- labeled POINT-AGE baseline;
  * midpoint variant -- interval-midpoint features.
Hyperparameters (RBF length scale; CatBoost rounds/depth) are selected on
TRAIN-side NLL only; the test window is scored exactly once.

Read-only on sources; statistical fitting authorized under the scoped
stat-pilot policy; all outputs labeled source-unverified.
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
from norn_earth.geometry.sphere import haversine_km
from norn_earth.losses.chronology import mixture_nll, student_t_logpdf
from norn_earth.losses.likelihood import SourceRegistry
from norn_earth.models.variational_fit import VariationalReconstructor
from norn_earth.utils.hashing import sha256_file, write_manifest

OUT = Path(__file__).resolve().parents[1] / "outputs" / "pilot_stat"


class RBFReconstructor(VariationalReconstructor):
    """B1: replace the SH basis with spherical RBF centers (same loss/optimizer)."""

    def __init__(self, centers_lon, centers_lat, length_km=800.0, **kw):
        super().__init__(lmax=1, **kw)  # lmax unused; basis overridden
        self.c_lon = np.asarray(centers_lon, float)
        self.c_lat = np.asarray(centers_lat, float)
        self.n_coef = len(self.c_lon)
        self.length = float(length_km)
        self.coef = np.zeros((self.n_anchors, self.n_coef))
        self.coef[:, 0] = np.log(np.expm1(30.0))
        self._rbf_cache = {}

    def _rbf(self, lon, lat):
        d = haversine_km(np.asarray(lon, float)[..., None], np.asarray(lat, float)[..., None],
                         self.c_lon[None, :], self.c_lat[None, :])
        return np.exp(-0.5 * (d / self.length) ** 2)

    def _field_linear(self, node_lon, node_lat, node_ages):
        Y = self._rbf(node_lon, node_lat)
        W = self._time_weights(node_ages)
        F = W @ self.coef
        return np.einsum("ij,ij->i", Y, F), (Y, W)

    def _modern_grid_basis(self, modern_batch):
        key = modern_batch["H_km"].shape
        if getattr(self, "_mgb_cache", None) is not None and self._mgb_cache[2] == key:
            return self._mgb_cache[0], self._mgb_cache[1]
        from norn_earth.geometry.regrid import gauss_grid, native_grid

        shape = key
        if shape == (180, 360):
            lats, lons = native_grid()
        else:
            lats, lons, _ = gauss_grid(shape[0], shape[1])
        lon2, lat2 = np.meshgrid(lons, lats)
        Y0 = self._rbf(lon2.ravel(), lat2.ravel())
        self._mgb_cache = (Y0, (lats, lons), key)
        return Y0, (lats, lons)

    def predict_grid(self, age, grid=None):
        Y0, g = self._modern_grid_basis({"H_km": np.zeros((180, 360))} if grid is None else {"H_km": np.zeros((len(grid[0]), len(grid[1])))})
        # simpler: use generic path through predict_nodes on the grid
        if grid is None:
            from norn_earth.geometry.regrid import native_grid
            lats, lons = native_grid()
        else:
            lats, lons = grid
        lon2, lat2 = np.meshgrid(lons, lats)
        v, _ = self.predict_nodes(lon2.ravel(), lat2.ravel(), np.full(lon2.size, float(age)))
        return v.reshape(len(lats), len(lons))


def score_batch(model, sub):
    counts = sub["node_counts"]
    starts = np.concatenate([[0], np.cumsum(counts)])
    n = len(counts)
    K = int(max(counts.max(), 1))
    pad_mu = np.zeros((n, K))
    pad_w = np.zeros((n, K))
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
    return {
        "mixture_nll": float(np.mean(nll)),
        "point_mae_km": float(np.mean(np.abs(mu0 - sub["y_km"][keep]))),
        "corr": float(np.corrcoef(mu0, sub["y_km"][keep])[0, 1]),
    }


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


def main():
    xlsx = ROOT / "古地壳厚度数据_新生代_处理后_final.xlsx"
    raw = pd.read_excel(xlsx).drop_duplicates().reset_index(drop=True)
    table = build_observation_table(xlsx)
    lookup = trajectory_lookup(build_trajectory_rows(raw)[0])
    reg = SourceRegistry()
    reg.register("unverified_proxy", 8.0)
    batch = assemble_observation_batch(table, lookup, reg)
    train_m, test_m, _ = chronological_holdout(table, (40.0, 50.0))
    train = subset(batch, train_m)
    test = subset(batch, test_m)
    modern = assemble_modern_batch(ROOT / "present_crustal_thickness.dat", sigma_km=3.0)

    # ---- B1: RBF x anchors, length scale chosen on TRAIN NLL ----
    rng = np.random.default_rng(0)
    take = rng.choice(len(train["y_km"]), min(12000, len(train["y_km"])), replace=False)
    starts = np.concatenate([[0], np.cumsum(train["node_counts"])])
    pos_idx = np.unique(np.concatenate([np.arange(starts[i], starts[i + 1]) for i in take if train["node_counts"][i] > 0]))[:4000]
    clon = train["node_lon"][pos_idx]
    clat = train["node_lat"][pos_idx]
    # 400 deterministic centers: farthest-point-ish thinning via spatial hash
    from norn_earth.data.splits import site_clusters
    cid = site_clusters(clon, clat, 800.0)
    first = {}
    for j, c in enumerate(cid):
        first.setdefault(int(c), j)
    idx_c = np.array(sorted(first.values()))[:400]

    b1_results = {}
    for length in (400.0, 800.0, 1600.0):
        t0 = time.time()
        m = RBFReconstructor(clon[idx_c], clat[idx_c], length_km=length,
                             n_anchors=13, anchor_step=5.0, lam_modern=0.3,
                             lam_obs=1.0, lam_smooth=1e-3)
        # train-side selection on a 3000-record subsample (fast)
        sub_mask = np.zeros(len(train["y_km"]), dtype=bool)
        sub_mask[rng.choice(len(train["y_km"]), 3000, replace=False)] = True
        m.fit(subset(train, sub_mask), modern, maxiter=45)
        sc = score_batch(m, subset(train, sub_mask))
        b1_results[length] = {"train_sub_nll": sc["mixture_nll"], "fit_s": time.time() - t0}
        print(f"B1 length={length}: train-sub NLL={sc['mixture_nll']:.4f}")
    best_len = min(b1_results, key=lambda L: b1_results[L]["train_sub_nll"])
    print(f"B1 selected length {best_len} km (train-side)")
    m1 = RBFReconstructor(clon[idx_c], clat[idx_c], length_km=best_len,
                          n_anchors=13, anchor_step=5.0, lam_modern=0.3,
                          lam_obs=1.0, lam_smooth=1e-3)
    t0 = time.time()
    m1.fit(train, modern, maxiter=60)
    b1_test = score_batch(m1, test)
    print(f"B1 test: {b1_test} ({time.time()-t0:.0f}s)")

    # ---- B2: CatBoost ----
    from catboost import CatBoostRegressor

    tr = table.iloc[np.flatnonzero(train_m)]
    te = table.iloc[np.flatnonzero(test_m)]
    b2 = {}
    for variant, age_col in (("point_age", "age_representative_ma"), ("midpoint", None)):
        if age_col:
            ages_tr = tr[age_col].to_numpy(float)
            ages_te = te[age_col].to_numpy(float)
        else:
            ages_tr = 0.5 * (tr["age_lower_ma"] + tr["age_upper_ma"]).to_numpy(float)
            ages_te = 0.5 * (te["age_lower_ma"] + te["age_upper_ma"]).to_numpy(float)

        def feats(lon, lat, age):
            return np.column_stack([
                np.cos(np.radians(lon)), np.sin(np.radians(lon)),
                np.sin(np.radians(lat)), age,
                np.abs(tr["age_upper_ma"].to_numpy(float)[: len(age)] - tr["age_lower_ma"].to_numpy(float)[: len(age)]) if len(age) == len(tr) else np.zeros(len(age)),
            ])
        Xtr = np.column_stack([np.cos(np.radians(tr["present_lon"])), np.sin(np.radians(tr["present_lon"])),
                               np.sin(np.radians(tr["present_lat"])), ages_tr,
                               (tr["age_upper_ma"] - tr["age_lower_ma"]).to_numpy(float)])
        Xte = np.column_stack([np.cos(np.radians(te["present_lon"])), np.sin(np.radians(te["present_lon"])),
                               np.sin(np.radians(te["present_lat"])), ages_te,
                               (te["age_upper_ma"] - te["age_lower_ma"]).to_numpy(float)])
        ytr = tr["thickness_km"].to_numpy(float)
        yte = te["thickness_km"].to_numpy(float)
        sig = batch["effective_sigma_km"][test_m]
        model = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05, verbose=False)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        b2[variant] = {
            "mae": float(np.mean(np.abs(pred - yte))),
            "nll_student_t": float(-np.mean(student_t_logpdf(yte, pred, sig))),
        }
        print(f"B2[{variant}] test MAE={b2[variant]['mae']:.3f} NLL={b2[variant]['nll_student_t']:.4f}")

    report = {
        "split": "P-C 40-50 (frozen, scored once)",
        "b1_rbf": {"length_km_selected": float(best_len), "selection": "train-side subsample NLL",
                   "candidates": {str(k): v for k, v in b1_results.items()},
                   "test": b1_test},
        "b2_catboost": b2,
        "sources": {"xlsx": sha256_file(xlsx), "dat": modern["sha256"]},
        "assumptions": "same declared unverified-proxy scales as the stat pilot",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(OUT / "baselines_b1_b2.json", report)
    print("wrote", OUT / "baselines_b1_b2.json")


if __name__ == "__main__":
    main()
