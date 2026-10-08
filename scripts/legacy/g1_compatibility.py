"""G1 compatibility test: Müller 2019 rotations vs the XLSX paleo coordinates.

For a sample of XLSX rows: assign present-day plate IDs via the model's
static polygons, reconstruct positions at 0..60 Ma with the plate circuit,
and measure angular distances against the XLSX paleo_lon/paleo_lat columns.

Interpretation guardrails (§4.3):
- t=0 must reproduce present coordinates (exact identity);
- deforming zones use rigid static-polygon reconstruction here, so distance
  quantifies BOTH model mismatch AND network deformation -> the per-time
  distributions (not a single pass/fail) are the compatibility evidence;
- agreement does not identify the exact model version; disagreement rules
  out this model as the XLSX source.

Read-only on sources; writes processed/g1_compatibility.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.gpml import load_gpml
from norn_earth.data.rotations import RotationModel, parse_rotations
from norn_earth.geometry.sphere import haversine_km, lonlat_to_vectors, spherical_winding
from norn_earth.utils.hashing import sha256_file, write_manifest

ROOT = Path(__file__).resolve().parents[2]
EXT = Path(__file__).resolve().parents[1] / "data" / "external" / "muller2019"
OUT = Path(__file__).resolve().parents[1] / "processed"

TIME_COLS = [f"TIME{'' if i == 0 else f'.{i}'}" for i in range(13)]
LON_COLS = [f"paleo_lon{'' if i == 0 else f'.{i}'}" for i in range(13)]
LAT_COLS = [f"paleo_lat{'' if i == 0 else f'.{i}'}" for i in range(13)]


def assign_plate_ids(lon, lat, features, workers_mask_poles=True):
    """Bounding-box prefilter + spherical winding on candidate outer rings."""
    rings = []
    for f in features:
        r = f["rings"][0]
        rings.append((f["plate_id"], r[:, 0], r[:, 1], r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max()))
    lons = np.mod(np.asarray(lon, dtype=float), 360.0)
    lats = np.asarray(lat, dtype=float)
    ids = np.full(len(lons), -1, dtype=int)
    for i in range(len(lons)):
        lo, la = lons[i], lats[i]
        best = None
        for pid, rlon, rlat, lonmin, lonmax, latmin, latmax in rings:
            if not (latmin - 0.5 <= la <= latmax + 0.5):
                continue
            span = ((lonmax - lonmin) % 360) + 1.0
            if ((lo - lonmin) % 360) > span:
                continue
            v = lonlat_to_vectors(np.array([lo]), np.array([la]))
            poly = lonlat_to_vectors(rlon, rlat)
            if abs(spherical_winding(v, poly)[0]) > 0.5:
                if best is None or len(rlon) < best[1]:
                    best = (pid, len(rlon))
        if best is not None:
            ids[i] = best[0]
    return ids


def best_fit_rotation(a_vec, b_vec):
    """Orthogonal Procrustes on the sphere: R minimizing sum |R a - b|^2.

    Returns rotation aligning a -> b (det=+1 enforced).
    """
    M = b_vec.T @ a_vec
    U, S, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def main():
    df = pd.read_excel(ROOT / "古地壳厚度数据_新生代_处理后_final.xlsx").drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(42)
    n_sample = min(2500, len(df))
    idx = rng.choice(len(df), n_sample, replace=False)
    sub = df.iloc[idx]
    lon = sub["经度"].to_numpy(float)
    lat = sub["纬度"].to_numpy(float)

    feats = load_gpml(EXT / "StaticPolygons/Muller_etal_2019_Global_StaticPlatePolygons.gpmlz")
    model = RotationModel(parse_rotations(EXT / "Rotations/Muller_etal_2019_CombinedRotations.rot"))

    print(f"static polygons: {len(feats)}; assigning plate ids to {n_sample} sites ...")
    pids = assign_plate_ids(lon, lat, feats)
    covered = pids >= 0
    print(f"assigned: {covered.sum()}/{n_sample} ({covered.mean()*100:.1f}%)")

    report = {
        "model": {
            "rotations": "Muller_etal_2019_CombinedRotations.rot",
            "static_polygons": "Muller_etal_2019_Global_StaticPlatePolygons.gpmlz",
            "source": "repo.gplates.org/webdav/pmm/muller2019 (public)",
            "rotations_sha256": sha256_file(EXT / "Rotations/Muller_etal_2019_CombinedRotations.rot"),
            "static_sha256": sha256_file(EXT / "StaticPolygons/Muller_etal_2019_Global_StaticPlatePolygons.gpmlz"),
        },
        "sample": {"rows": int(n_sample), "plate_id_assigned": int(covered.sum()), "assignment_rate": float(covered.mean())},
        "times_ma": {},
        "caveats": [
            "rigid static-polygon reconstruction only: distances inside deforming networks mix model mismatch with real network deformation",
            "agreement constrains compatibility, not exact provenance/version identity",
            "after_best_fit_frame_rotation removes ONE global rotation per time: small residuals there mean the RELATIVE plate circuits agree and only the absolute frame differs",
        ],
    }

    for k in range(13):
        t = 5.0 * k
        tc, lc, pc = TIME_COLS[k], LON_COLS[k], LAT_COLS[k]
        m = sub[tc].notna() & sub[lc].notna() & sub[pc].notna() & covered
        xl_lon = sub.loc[m, lc].to_numpy(float)
        xl_lat = sub.loc[m, pc].to_numpy(float)
        p = pids[m.to_numpy()]
        pos_all = np.flatnonzero(m.to_numpy())
        dists, resolved = [], 0
        rec_vecs, xl_vecs = [], []
        for j, (lo, la, pid) in enumerate(zip(xl_lon, xl_lat, p)):
            row_pos = pos_all[j]
            rec = model.reconstruct(lon[row_pos], lat[row_pos], int(pid), t)
            if rec is None:
                continue
            resolved += 1
            rl = float(np.asarray(rec[0]).reshape(-1)[0])
            ra = float(np.asarray(rec[1]).reshape(-1)[0])
            lo_f, la_f = float(lo), float(la)
            d = float(np.atleast_1d(haversine_km(rl, ra, lo_f, la_f))[0])
            dists.append(d)
            rec_vecs.append(np.asarray(lonlat_to_vectors(rl, ra)).reshape(3))
            xl_vecs.append(np.asarray(lonlat_to_vectors(lo_f, la_f)).reshape(3))
        dists = np.array(dists)
        # frame diagnostic: remove ONE best-fit global rotation per time
        aligned_stats = None
        if len(rec_vecs) >= 10:
            A = np.array(rec_vecs)
            B = np.array(xl_vecs)
            from norn_earth.geometry.sphere import angle_between
            from norn_earth.utils.units import EARTH_RADIUS_KM as R_KM
            d_vec = np.asarray(angle_between(A, B)) * R_KM
            if abs(float(np.median(d_vec)) - float(np.median(dists))) > 5.0 + 1e-6:
                raise AssertionError(
                    "internal inconsistency: vector vs haversine medians differ; "
                    "alignment diagnostics would be invalid"
                )
            R = best_fit_rotation(A, B)
            aligned = A @ R.T
            d_al = np.asarray(angle_between(aligned, B)) * R_KM
            aligned_stats = {
                "median_km_after_alignment": float(np.median(d_al)),
                "p90_km_after_alignment": float(np.percentile(d_al, 90)),
                "frac_lt_100km_after_alignment": float(np.mean(d_al < 100)),
                "selfcheck_vector_median_km": float(np.median(d_vec)),
            }
        entry = {
            "n_compared": int(len(dists)),
            "n_unresolved_circuit": int(m.sum() - resolved),
            "median_km": float(np.median(dists)) if len(dists) else None,
            "p75_km": float(np.percentile(dists, 75)) if len(dists) else None,
            "p90_km": float(np.percentile(dists, 90)) if len(dists) else None,
            "max_km": float(dists.max()) if len(dists) else None,
            "frac_lt_100km": float(np.mean(dists < 100)) if len(dists) else None,
            "frac_lt_500km": float(np.mean(dists < 500)) if len(dists) else None,
            "after_best_fit_frame_rotation": aligned_stats,
        }
        report["times_ma"][f"{t:.0f}"] = entry
        print(f"t={t:5.1f} Ma: n={entry['n_compared']:5d} median={entry['median_km'] if entry['median_km'] is not None else float('nan'):8.1f} km  p90={entry['p90_km'] if entry['p90_km'] is not None else float('nan'):8.1f} km  <100km: {entry['frac_lt_100km'] if entry['frac_lt_100km'] is not None else float('nan'):.2f}")
        if aligned_stats:
            print(f"          after frame alignment: median={aligned_stats['median_km_after_alignment']:8.1f} km  p90={aligned_stats['p90_km_after_alignment']:8.1f} km  <100km: {aligned_stats['frac_lt_100km_after_alignment']:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(OUT / "g1_compatibility.json", report)
    print(f"\nwrote {OUT / 'g1_compatibility.json'}")


if __name__ == "__main__":
    main()
