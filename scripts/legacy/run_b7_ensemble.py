"""B7 ensemble predictive intervals on the frozen P-C 40-50 test window.

Members: bootstrap-resampled TRAIN records only (no test information);
predictive = per-record mixture over members' Student-t node predictions.
Reports NLL + 50/90% interval coverage/width -- labeled ensemble sensitivity
ranges (5-member pilot does NOT certify calibration; §13.1 rules).
"""
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
from norn_earth.losses.likelihood import SourceRegistry
from norn_earth.models.variational_fit import VariationalReconstructor
from norn_earth.utils.hashing import write_manifest

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


def main():
    xlsx = ROOT / "古地壳厚度数据_新生代_处理后_final.xlsx"
    raw = pd.read_excel(xlsx).drop_duplicates().reset_index(drop=True)
    table = build_observation_table(xlsx)
    lookup = trajectory_lookup(build_trajectory_rows(raw)[0])
    reg = SourceRegistry()
    reg.register("unverified_proxy", 8.0)
    batch = assemble_observation_batch(table, lookup, reg)
    train_m, test_m, _ = chronological_holdout(table, (40.0, 50.0))
    modern = assemble_modern_batch(ROOT / "present_crustal_thickness.dat", sigma_km=3.0)
    test = subset(batch, test_m)

    rng = np.random.default_rng(202)
    n_train = int(train_m.sum())
    members = []
    for b in range(3):
        boot = np.zeros(n_train, dtype=bool)
        boot[rng.choice(n_train, n_train, replace=True)] = True
        bmask = train_m.copy()
        bmask[np.flatnonzero(train_m)] = boot
        sub = subset(batch, bmask)
        m = VariationalReconstructor(lmax=12, n_anchors=13, anchor_step=5.0,
                                     lam_modern=0.5, lam_obs=1.0, lam_smooth=1e-3)
        t0 = time.time()
        m.fit(sub, modern, maxiter=45)
        print(f"member {b}: fitted in {time.time()-t0:.0f}s")
        members.append(m)

    # per-record point prediction distribution across members (at first node age)
    counts = test["node_counts"]
    starts = np.concatenate([[0], np.cumsum(counts)])
    keep = counts > 0
    rows = np.flatnonzero(keep)
    first_node = np.array([starts[i] for i in rows])
    preds = np.stack([
        m.predict_nodes(test["node_lon"][first_node], test["node_lat"][first_node],
                        test["node_ages"][first_node])[0] for m in members
    ])  # (members, n)
    y = test["y_km"][rows]
    mean = preds.mean(axis=0)
    lo50, hi50 = np.percentile(preds, [25, 75], axis=0)
    lo90, hi90 = np.percentile(preds, [5, 95], axis=0)
    report = {
        "n_members": len(members),
        "bootstrap": "train-side only; test untouched",
        "mean_mae_km": float(np.mean(np.abs(mean - y))),
        "cov50": float(np.mean((y >= lo50) & (y <= hi50))),
        "width50_km": float(np.mean(hi50 - lo50)),
        "cov90": float(np.mean((y >= lo90) & (y <= hi90))),
        "width90_km": float(np.mean(hi90 - lo90)),
        "interval_kind": "ensemble_sensitivity_range (3 members; NOT calibrated intervals)",
    }
    print(report)
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(OUT / "b7_ensemble_intervals.json", report)


if __name__ == "__main__":
    main()
