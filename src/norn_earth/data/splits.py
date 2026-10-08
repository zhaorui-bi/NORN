from pathlib import Path

"""Split construction and leakage self-checks (§11).

E1 site/study groups: leader clustering on present positions (deterministic
order, minimum separation). Study IDs are UNKNOWN pre-G0, so clusters are a
conservative proxy and are labeled as such.

E2 spatial blocks: equal-area lat bands x adaptive lon sectors (target
~1000 km blocks) with an exclusion buffer around test-block boundaries.

E3 chronological holdout: test = narrow-age records inside the window; the
ENTIRE record is removed from training if its age support [lo, hi] overlaps
the window (deleting TIME columns is not a temporal split).

P-A removes modern-block information entirely (labels AND derived priors or
calibration touching those blocks); P-B keeps modern endpoints by definition.
"""
import numpy as np


def site_clusters(lon, lat, min_sep_km=100.0):
    """Deterministic spatial-hash clustering of present positions.

    Study/sample IDs are UNKNOWN before Gate G0, so this is a conservative
    GEOGRAPHIC proxy for site grouping (documented as such in every split
    manifest); it never claims study-level independence.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    step_deg = max(float(min_sep_km) / 111.32, 1e-6)
    keys = np.stack(
        [np.round(lat / step_deg).astype(np.int64), np.round(lon / step_deg).astype(np.int64)],
        axis=1,
    )
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    return ids


def spatial_block_ids(lon, lat, target_km=1000.0, nlat_bands=20):
    """Equal-area-ish blocks: fixed lat bands, adaptive lon sector counts."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    band_edges = np.linspace(-90, 90, nlat_bands + 1)
    band_centers = 0.5 * (band_edges[:-1] + band_edges[1:])
    ids = np.empty(len(lon), dtype=int)
    base = 0
    sector_maps = []
    for bc in band_centers:
        n_sec = max(1, int(round(2 * np.pi * 6371.0 * np.cos(np.radians(bc)) / target_km)))
        sector_maps.append((base, n_sec))
        base += n_sec
    for i in range(len(lon)):
        b = int(np.clip(np.searchsorted(band_edges, lat[i]) - 1, 0, nlat_bands - 1))
        b0, n_sec = sector_maps[b]
        s = int(np.clip(int((lon[i] + 180.0) / 360.0 * n_sec), 0, n_sec - 1))
        ids[i] = b0 + s
    return ids


def chronological_holdout(table, window=(15.0, 25.0), narrow_max_width=5.0):
    """P-C: remove ENTIRE records whose age support overlaps the window."""
    lo = np.asarray(table["age_lower_ma"], dtype=float)
    hi = np.asarray(table["age_upper_ma"], dtype=float)
    rep = np.asarray(table["age_representative_ma"], dtype=float)
    w0, w1 = window
    width = hi - lo
    overlaps = (hi >= w0) & (lo <= w1)
    test_mask = overlaps & (width <= narrow_max_width) & (rep >= w0) & (rep <= w1)
    train_mask = ~overlaps
    buffer_mask = overlaps & ~test_mask
    return train_mask, test_mask, buffer_mask


def modern_block_holdout(table, block_ids, test_blocks):
    """P-A: test = observations in held-out modern blocks (buffer handled by caller)."""
    in_test = np.isin(np.asarray(block_ids), np.asarray(test_blocks))
    return ~in_test, in_test


def leakage_report(table, train_mask, test_mask, buffer_mask=None):
    """Hard self-checks; a split that fails any check must not be used."""
    obs_ids = table["observation_id"].to_numpy()
    checks = {
        "no_id_overlap": len(set(obs_ids[train_mask]) & set(obs_ids[test_mask])) == 0,
        "test_nonempty": int(test_mask.sum()) > 0,
        "train_nonempty": int(train_mask.sum()) > 0,
    }
    if buffer_mask is not None:
        checks["buffer_disjoint_from_train_test"] = bool(
            (buffer_mask & train_mask).sum() == 0 and (buffer_mask & test_mask).sum() == 0
        )
    n_clusters_train = (
        len(np.unique(table["site_cluster"][train_mask])) if "site_cluster" in table else None
    )
    n_clusters_test = (
        len(np.unique(table["site_cluster"][test_mask])) if "site_cluster" in table else None
    )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {
            "train": int(train_mask.sum()),
            "test": int(test_mask.sum()),
            "buffer": int(buffer_mask.sum()) if buffer_mask is not None else 0,
        },
        "effective_site_clusters": {"train": n_clusters_train, "test": n_clusters_test},
    }


class TestWindowError(RuntimeError):
    """Raised when a reserved test window is about to be scored."""

    __test__ = False


def assert_window_usable(window_key, registry_path):
    """P2 guard (v1.3): reserved windows refuse scoring; burned = selection only.

    window_key like 'P-C_15_25'; registry JSON lists status per window.
    """
    import json

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    entry = registry.get(window_key)
    if entry is None:
        raise TestWindowError(f"unknown test window '{window_key}'; add it to the registry first")
    status = entry["status"]
    if status.startswith("reserved"):
        raise TestWindowError(
            f"window '{window_key}' is {status} ({entry['reason']}); scoring it now would burn the final experiments"
        )
    return status


def build_split_manifest(table, config=None):
    """Freeze E1/E2/E3 + P-A/P-B/P-C masks with leakage self-checks.

    Status is PRELIMINARY until G0 (study IDs / semantics) and G1 (verified
    regions) close: site clusters remain geographic proxies and test-block
    selection follows a fixed coverage rule, never model performance.
    """
    import json

    from ..utils.hashing import sha256_bytes

    cfg = {
        "site_cluster_min_sep_km": 100.0,
        "block_target_km": 1000.0,
        "chronological_windows_ma": [[15.0, 25.0], [40.0, 50.0]],
        "older_tail_window_ma": [50.0, 60.0],
        "p_a_test_block_rule": "two most observation-dense blocks, fixed rule, coverage-based only",
    }
    cfg.update(config or {})

    if "site_cluster" not in table:
        table = table.copy()
        table["site_cluster"] = site_clusters(
            table["present_lon"].to_numpy(float),
            table["present_lat"].to_numpy(float),
            cfg["site_cluster_min_sep_km"],
        )
    if "spatial_block" not in table:
        table = table.copy()
        table["spatial_block"] = spatial_block_ids(
            table["present_lon"].to_numpy(float),
            table["present_lat"].to_numpy(float),
            cfg["block_target_km"],
        )

    manifest = {
        "status": "PRELIMINARY_UNTIL_G0_G1",
        "config": cfg,
        "n_records": int(len(table)),
        "n_site_clusters": int(table["site_cluster"].nunique()),
        "splits": {},
    }

    for w0, w1 in cfg["chronological_windows_ma"] + [cfg["older_tail_window_ma"]]:
        train, test, buffer = chronological_holdout(table, (w0, w1))
        rep = leakage_report(table, train, test, buffer)
        manifest["splits"][f"P-C_{w0:.0f}_{w1:.0f}"] = rep

    counts = table.groupby("spatial_block").size().sort_values(ascending=False)
    test_blocks = [int(counts.index[0]), int(counts.index[1])]
    train, test = modern_block_holdout(table, table["spatial_block"].to_numpy(), test_blocks)
    rep = leakage_report(table, train, test, None)
    rep["test_blocks"] = test_blocks
    manifest["splits"]["P-A_modern_blocks"] = rep

    manifest["sha256_of_counts"] = sha256_bytes(
        json.dumps(manifest["splits"], sort_keys=True).encode("utf-8")
    )
    manifest["all_passed"] = all(v["passed"] for v in manifest["splits"].values())
    return manifest
