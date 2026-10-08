"""Prediction metrics with cluster-aware uncertainty (§11.3)."""

import numpy as np


def point_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_ae": float(np.median(np.abs(err))),
        "bias": float(np.mean(err)),
        "n": int(len(err)),
    }


def macro_point_metrics(y_true, y_pred, group_ids):
    """Macro-average over groups (blocks/regions/studies), then micro."""
    micro = point_metrics(y_true, y_pred)
    per = {}
    for g in np.unique(group_ids):
        m = np.asarray(group_ids) == g
        per[int(g) if isinstance(g, (int, np.integer)) else str(g)] = point_metrics(
            np.asarray(y_true)[m], np.asarray(y_pred)[m]
        )
    macro = {
        k: float(np.mean([v[k] for v in per.values()]))
        for k in ["mae", "rmse", "median_ae", "bias"]
    }
    macro["n_groups"] = len(per)
    return {"micro": micro, "macro": macro, "per_group": per}


def paired_bootstrap_delta(y_true, y_pred_a, y_pred_b, units, n_boot=2000, seed=0):
    """Bootstrap of the paired MAE difference (a - b) resampling CLUSTERS.

    Units are site/study/block identities -- never pixels or seeds. Pointwise
    errors within a unit are correlated by construction.
    """
    rng = np.random.default_rng(seed)
    units = np.asarray(units)
    unique = np.unique(units)
    abs_a = np.abs(np.asarray(y_pred_a) - np.asarray(y_true))
    abs_b = np.abs(np.asarray(y_pred_b) - np.asarray(y_true))
    deltas = []
    by_unit = {u: (abs_a[units == u], abs_b[units == u]) for u in unique}
    for _ in range(n_boot):
        take = rng.choice(unique, size=len(unique), replace=True)
        da, db = [], []
        for u in take:
            da.append(by_unit[u][0])
            db.append(by_unit[u][1])
        da = np.concatenate(da)
        db = np.concatenate(db)
        deltas.append(float(da.mean() - db.mean()))
    deltas = np.array(deltas)
    return {
        "delta_mae_mean": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "effective_clusters": int(len(unique)),
    }


def effective_cluster_count(units):
    return int(len(np.unique(np.asarray(units))))
