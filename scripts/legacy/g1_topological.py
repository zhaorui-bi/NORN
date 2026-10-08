"""G1 topological decomposition with pygplates (authoritative engine).

Steps:
1. assign present-day plate IDs to XLSX sample sites via PlatePartitioner
   on the model's static polygons (C++ engine);
2. rigid reconstruction with pygplates.RotationModel at 0..60 Ma and
   CROSS-CHECK our pure-python circuit parser against it;
3. at each time, partition reconstructed positions into deforming networks
   vs rigid plates (TopologicalSnapshot) and split XLSX residuals by group:
   rigid-group residual isolates model-version mismatch; the network-group
   residual mixes in real intra-network deformation.

Writes processed/g1_topological.json. Read-only on sources.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXT = Path(__file__).resolve().parents[1] / "data" / "external" / "muller2019"
OUT = Path(__file__).resolve().parents[1] / "processed"
ROOT = Path(__file__).resolve().parents[2]

TIME_COLS = [f"TIME{'' if i == 0 else f'.{i}'}" for i in range(13)]
LON_COLS = [f"paleo_lon{'' if i == 0 else f'.{i}'}" for i in range(13)]
LAT_COLS = [f"paleo_lat{'' if i == 0 else f'.{i}'}" for i in range(13)]


def main():
    import pygplates

    rotation_features = pygplates.FeatureCollection(
        str(EXT / "Rotations" / "Muller_etal_2019_CombinedRotations.rot")
    )
    rm = pygplates.RotationModel(rotation_features)
    static = pygplates.FeatureCollection(
        str(EXT / "StaticPolygons" / "Muller_etal_2019_Global_StaticPlatePolygons.gpmlz")
    )
    topologies = [
        pygplates.FeatureCollection(
            str(EXT / "Topologies" / "Muller_etal_2019_PlateBoundaries_DeformingNetworks.gpmlz")
        )
    ]

    df = pd.read_excel(ROOT / "古地壳厚度数据_新生代_处理后_final.xlsx").drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(42)
    sub = df.iloc[rng.choice(len(df), 2500, replace=False)]
    lon = sub["经度"].to_numpy(float)
    lat = sub["纬度"].to_numpy(float)

    # authoritative plate IDs at present day
    partitioner = pygplates.PlatePartitioner(static, rm)
    pids = []
    for lo, la in zip(lon, lat):
        pid = partitioner.partition_point(pygplates.PointOnSphere(la, lo))
        pids.append(pid.get_feature().get_reconstruction_plate_id() if pid else -1)
    pids = np.array(pids, dtype=int)
    print(f"pygplates plate assignment: {(pids >= 0).mean()*100:.1f}%")

    # our parser cross-check at selected times on resolved points
    from norn_earth.data.rotations import RotationModel, parse_rotations

    ours = RotationModel(parse_rotations(EXT / "Rotations" / "Muller_etal_2019_CombinedRotations.rot"))
    xcheck = {}
    for t in (5.0, 30.0, 60.0):
        diffs = []
        for i in range(len(lon)):
            if pids[i] < 0:
                continue
            anchor = rm.get_rotation(float(t), int(pids[i]))  # plate->anchor at time t
            r = ours.reconstruct(lon[i], lat[i], int(pids[i]), t)
            if r is None:
                continue
            rl, ra = float(np.asarray(r[0]).reshape(-1)[0]), float(np.asarray(r[1]).reshape(-1)[0])
            gp = anchor * pygplates.PointOnSphere(lat[i], lon[i])
            glat, glon = gp.to_lat_lon()
            d = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(
                            np.array([np.cos(np.radians(ra)) * np.cos(np.radians(rl)), np.cos(np.radians(ra)) * np.sin(np.radians(rl)), np.sin(np.radians(ra))]),
                            np.array([np.cos(np.radians(glat)) * np.cos(np.radians(glon)), np.cos(np.radians(glat)) * np.sin(np.radians(glon)), np.sin(np.radians(glat))]),
                        ),
                        -1,
                        1,
                    )
                )
            ) * 6371.0
            diffs.append(d)
        xcheck[f"{t:.0f}"] = {
            "n": len(diffs),
            "median_km": float(np.median(diffs)),
            "p99_km": float(np.percentile(diffs, 99)),
        }
        print(f"parser vs pygplates @ {t:.0f} Ma: median {np.median(diffs):.3f} km, p99 {np.percentile(diffs,99):.2f} km (n={len(diffs)})")

    # residual decomposition by network membership
    report = {"parser_crosscheck_km": xcheck, "times": {}}
    for k in range(13):
        t = 5.0 * k
        m = (sub[TIME_COLS[k]].notna() & sub[LON_COLS[k]].notna() & sub[LAT_COLS[k]].notna() & (pids >= 0)).to_numpy()
        xl_lon = sub.loc[m, LON_COLS[k]].to_numpy(float)
        xl_lat = sub.loc[m, LAT_COLS[k]].to_numpy(float)
        pos = np.flatnonzero(m)
        rec_pts, d_rigid, d_net, pids_sel = [], [], [], []
        for j in range(len(xl_lon)):
            i = pos[j]
            try:
                anchor = rm.get_rotation(float(t), int(pids[i]))
                gp = anchor * pygplates.PointOnSphere(lat[i], lon[i])
            except Exception:
                continue
            rec_pts.append(gp)
            pids_sel.append(int(pids[i]))
            from norn_earth.geometry.sphere import haversine_km

            d = float(np.atleast_1d(haversine_km(gp.to_lat_lon()[1], gp.to_lat_lon()[0], xl_lon[j], xl_lat[j]))[0])
            d_rigid.append(d)
        if not rec_pts:
            continue
        snap = pygplates.TopologicalSnapshot(topologies, rm, t)
        rt = snap.get_resolved_topologies()
        partitioner_t = pygplates.PlatePartitioner(list(rt), rm)
        in_net = []
        for gp in rec_pts:
            resolved = partitioner_t.partition_point(gp)
            in_net.append(resolved is not None and "Network" in type(resolved).__name__)
        in_net = np.array(in_net)
        d_all = np.array(d_rigid)
        entry = {
            "n": int(len(d_all)),
            "median_all_km": float(np.median(d_all)),
            "n_in_network": int(in_net.sum()),
            "median_rigid_km": float(np.median(d_all[~in_net])) if (~in_net).any() else None,
            "median_network_km": float(np.median(d_all[in_net])) if in_net.any() else None,
            "frac_rigid_lt_100km": float(np.mean(d_all[~in_net] < 100)) if (~in_net).any() else None,
        }
        report["times"][f"{t:.0f}"] = entry
        print(f"t={t:5.1f}: all={entry['median_all_km']:7.1f} km | rigid(n={len(d_all)-in_net.sum():4d})={entry['median_rigid_km'] if entry['median_rigid_km'] is not None else float('nan'):7.1f} km | network(n={in_net.sum():3d})={entry['median_network_km'] if entry['median_network_km'] is not None else float('nan'):7.1f} km | rigid<100km: {entry['frac_rigid_lt_100km']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g1_topological.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'g1_topological.json'}")


if __name__ == "__main__":
    main()
