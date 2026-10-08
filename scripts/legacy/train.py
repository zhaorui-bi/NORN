"""Training entry point -- REFUSES to train by default (see protocol).

Authorization requires ALL of:
  1. --i-authorize-training (explicit user action in this session);
  2. protocol execution_policy.training_authorized == true in
     研究执行协议_NORN.json (edited by the user, not by scripts);
  3. gates referenced by the chosen profile report PASS in processed/gates.json.

None of these exist today: G0/G1/G2 are OPEN, torch is not installed, and
statistical fitting on real observations also counts as training (§0.4).
--dry-run is always allowed and only reports batch/cost structure.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "研究执行协议_NORN.json"
GATES = Path(__file__).resolve().parents[1] / "processed" / "gates.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def dry_run_report():
    from norn_earth.losses.chronology import make_age_nodes
    from norn_earth.models.sfno import two_pass_step_contract

    nodes, weights, c = make_age_nodes(0.0, 66.0, 8)
    return {
        "two_pass_contract": two_pass_step_contract(),
        "example_wide_interval": {
            "age_interval_ma": [0, 66],
            "nodes": nodes.tolist(),
            "weights_sum": float(weights.sum()),
            "clipped_mass_c_i": c,
        },
        "anchor_fields": {"count": 61, "shape_each": [180, 360], "dtype": "float32"},
        "note": "cost structure only; no forward/backward passes executed",
    }


def run_stat_pilot():
    """Declared-assumption STAT pilot (§0.4 fallback path): B7 variational
    fit on REAL data under explicitly unverified proxy semantics; P-C 40-50
    chronological holdout; scored against B0 rolling-median and flat baselines.
    CPU-only; no physics priors; conclusions labeled source-unverified."""
    import json
    import time

    import numpy as np
    import pandas as pd

    from norn_earth.data.batches import assemble_observation_batch
    from norn_earth.data.modern import assemble_modern_batch
    from norn_earth.data.observations import build_observation_table
    from norn_earth.data.splits import chronological_holdout
    from norn_earth.data.trajectories import build_trajectory_rows, trajectory_lookup
    from norn_earth.losses.chronology import mixture_nll, student_t_logpdf
    from norn_earth.losses.likelihood import SourceRegistry
    from norn_earth.models.variational_fit import VariationalReconstructor
    from norn_earth.utils.hashing import sha256_file, write_manifest

    root = ROOT
    xlsx = root / "古地壳厚度数据_新生代_处理后_final.xlsx"
    raw = pd.read_excel(xlsx).drop_duplicates().reset_index(drop=True)
    table = build_observation_table(xlsx)
    traj_long, _ = build_trajectory_rows(raw)
    lookup = trajectory_lookup(traj_long)
    reg = SourceRegistry()
    reg.register("unverified_proxy", 8.0)  # DECLARED assumption (see gates.json)
    batch = assemble_observation_batch(table, lookup, reg)

    train_m, test_m, buffer_m = chronological_holdout(table, (40.0, 50.0))
    modern = assemble_modern_batch(root / "present_crustal_thickness.dat", sigma_km=3.0)

    def subset(mask):
        idx = np.flatnonzero(mask)
        starts = np.concatenate([[0], np.cumsum(batch["node_counts"])])
        sel = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in idx if batch["node_counts"][i] > 0])
        sub = {k: (v[idx] if k in ("y_km", "effective_sigma_km", "node_counts", "record_sigma_km", "source_sigma_km") else v[sel] if k in ("node_ages", "node_weights", "node_lon", "node_lat", "node_from_trajectory") else v[idx]) for k, v in batch.items() if isinstance(v, np.ndarray)}
        return sub

    train = subset(train_m)
    test = subset(test_m)
    print(f"P-C 40-50: train={len(train['y_km'])} test={len(test['y_km'])} buffer={int(buffer_m.sum())}")

    model = VariationalReconstructor(lmax=12, n_anchors=13, anchor_step=5.0,
                                     lam_modern=0.5, lam_obs=1.0, lam_smooth=1e-3)
    t0 = time.time()
    history = model.fit(train, modern, maxiter=60, verbose=True)
    print(f"fit: {len(history)} evals in {time.time()-t0:.1f}s")

    def score(sub, tag):
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
        rep_idx = np.array([starts[i] for i in range(n) if counts[i] > 0])
        mu0, _ = model.predict_nodes(sub["node_lon"][rep_idx], sub["node_lat"][rep_idx], sub["node_ages"][rep_idx])
        mae = float(np.mean(np.abs(mu0 - sub["y_km"][keep])))
        corr = float(np.corrcoef(mu0, sub["y_km"][keep])[0, 1])
        return {"tag": tag, "n": int(keep.sum()), "mixture_nll": float(np.mean(nll)), "point_mae_km": mae, "corr": corr}

    # B0 baselines on the same test rows
    tr_ages = table["age_representative_ma"].to_numpy(float)[train_m]
    tr_y = table["thickness_km"].to_numpy(float)[train_m]
    te_ages = table["age_representative_ma"].to_numpy(float)[test_m]
    te_y = table["thickness_km"].to_numpy(float)[test_m]
    rolling = np.array([np.median(tr_y[np.abs(tr_ages - q) <= 10.0]) for q in te_ages])
    flat = np.full_like(te_y, np.median(tr_y))
    sig = batch["effective_sigma_km"][test_m]
    b0_nll = -np.mean(student_t_logpdf(te_y, rolling, sig, nu=4.0))
    flat_nll = -np.mean(student_t_logpdf(te_y, flat, sig, nu=4.0))

    report = {
        "profile": "stat_pilot_declared_assumptions",
        "assumptions": "XLSX = unverified proxy obs, source sigma 8 km; no physics; conclusions source-unverified",
        "split": "P-C chronological holdout 40-50 Ma (whole-record support removal)",
        "counts": {"train": int(train_m.sum()), "test": int(test_m.sum()), "buffer": int(buffer_m.sum())},
        "b7": {**score(train, "train"), "history_tail": [float(h) for h in history[-3:]]},
        "b7_test": score(test, "test"),
        "baselines_test": {
            "b0_rolling_median_mae": float(np.mean(np.abs(rolling - te_y))),
            "b0_flat_mae": float(np.mean(np.abs(flat - te_y))),
            "b0_rolling_nll": float(b0_nll),
            "b0_flat_nll": float(flat_nll),
        },
        "sources": {"xlsx": sha256_file(xlsx), "dat": modern["sha256"]},
        "engine": "numpy CPU, L-BFGS, lmax=12, 13 anchors @5 Myr",
    }
    out_dir = root / "norn" / "outputs" / "pilot_stat"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "report.json", report)
    np.savez_compressed(
        out_dir / "fields.npz",
        **{f"age_{a:.0f}": model.predict_grid(float(a)) for a in (0, 20, 40, 60)},
    )
    print(json.dumps({k: report[k] for k in ("counts", "b7_test", "baselines_test")}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="main", choices=["main", "stat", "baseline"])
    parser.add_argument("--i-authorize-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run_report(), ensure_ascii=False, indent=2))
        return 0

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    reasons = []
    if not args.i_authorize_training:
        reasons.append("missing --i-authorize-training")
    if not protocol.get("execution_policy", {}).get("training_authorized", False):
        reasons.append("protocol execution_policy.training_authorized is false")
    if not GATES.is_file():
        reasons.append(f"no gates file at {GATES.relative_to(ROOT)}")
    else:
        gates = json.loads(GATES.read_text(encoding="utf-8"))
        needed = {"main": ["G0_semantics", "G1_kinematics", "G2_experiment_freeze"]}.get(args.profile, [])
        for g in needed:
            if gates.get(g, {}).get("status") != "PASS":
                reasons.append(f"gate {g} not PASS")
    if reasons:
        print("REFUSED: training not authorized.", file=sys.stderr)
        for r in reasons:
            print(f"  - {r}", file=sys.stderr)
        print("Dry-run cost structure is always available via --dry-run.", file=sys.stderr)
        return 1
    if args.profile == "stat":
        return run_stat_pilot()
    print("Authorized training detected for physical-main profile, but the G3", file=sys.stderr)
    print("physical pilot comes after G0 closes (proxy semantics still PARTIAL).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
