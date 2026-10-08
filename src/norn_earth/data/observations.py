"""Observation table construction (§5.1): one record = one evidence unit.

Reads the source XLSX read-only, removes exact duplicate rows, classifies age
semantics, attaches QC flags and age-integration nodes (weights sum to 1).
Provenance (source/study/sample identity) is UNKNOWN until Gate G0 supplies
it; every output row carries source_unverified=True until then. The builder
never writes thickness labels at TIME columns.
"""

import pandas as pd

from ..losses.chronology import make_age_nodes
from ..losses.likelihood import record_nodes

COL = {
    "lat": "纬度",
    "lon": "经度",
    "age_lo": "年龄_起始 (Ma)",
    "age_hi": "年龄_结束 (Ma)",
    "age_rep": "年龄_代表值 (Ma)",
    "epoch": "时间段",
    "thickness": "平均厚度 (km)",
}

STANDARD_COLUMNS = [
    "observation_id",
    "source_unverified",
    "present_lon",
    "present_lat",
    "age_lower_ma",
    "age_upper_ma",
    "age_representative_ma",
    "age_width_myr",
    "age_semantics",
    "epoch_label",
    "thickness_km",
    "qc_thickness_gt80",
    "qc_thickness_gt100",
    "qc_thickness_lt2",
    "qc_negative_age",
    "qc_coords_out_of_range",
    "qc_age_interval_covers_domain",
    "qc_point_age_outside_domain",
    "age_nodes_k",
    "age_clipped_mass",
    "has_paleo_coordinates",
]


def _age_semantics(lo, hi, domain=(0.0, 60.0)):
    w = hi - lo
    if w == 0:
        inside = domain[0] <= lo <= domain[1]
        return "point_inside" if inside else "point_outside"
    if lo <= domain[0] and hi >= domain[1]:
        return "coarse_full_domain"
    if w >= 30.0:
        return "wide_interval"
    if w <= 5.0:
        return "narrow_interval"
    return "interval"


def build_observation_table(xlsx_path, sheet=0):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dedup = len(df)

    lo = df[[COL["age_lo"], COL["age_hi"]]].min(axis=1).astype(float)
    hi = df[[COL["age_lo"], COL["age_hi"]]].max(axis=1).astype(float)
    rep = pd.to_numeric(df[COL["age_rep"]], errors="coerce").astype(float)
    th = pd.to_numeric(df[COL["thickness"]], errors="coerce").astype(float)

    paleo_cols = [c for c in df.columns if str(c).startswith("paleo_lon")]
    has_paleo = df[paleo_cols].notna().any(axis=1)

    out = pd.DataFrame(
        {
            "observation_id": [f"obs{i:06d}" for i in range(n_dedup)],
            "source_unverified": True,
            "present_lon": df[COL["lon"]].astype(float),
            "present_lat": df[COL["lat"]].astype(float),
            "age_lower_ma": lo,
            "age_upper_ma": hi,
            "age_representative_ma": rep,
            "age_width_myr": hi - lo,
            "age_semantics": [_age_semantics(a, b) for a, b in zip(lo, hi)],
            "epoch_label": df.get(COL["epoch"], pd.Series([""] * n_dedup)).astype(str),
            "thickness_km": th,
            "qc_thickness_gt80": th > 80.0,
            "qc_thickness_gt100": th > 100.0,
            "qc_thickness_lt2": th < 2.0,
            "qc_negative_age": rep < 0.0,
            "qc_coords_out_of_range": ~(
                df[COL["lon"]].between(-180, 360) & df[COL["lat"]].between(-90, 90)
            ),
            "qc_age_interval_covers_domain": (lo <= 0.0) & (hi >= 60.0),
        }
    )
    out["qc_point_age_outside_domain"] = out["age_semantics"].eq("point_outside")
    out["has_paleo_coordinates"] = has_paleo.values

    nodes_k, clipped = [], []
    for row in out.itertuples():
        n, _, _ = make_age_nodes(row.age_lower_ma, row.age_upper_ma, 8)
        nodes_k.append(len(n))
        obs = {
            "age_lower": row.age_lower_ma,
            "age_upper": row.age_upper_ma,
        }
        nodes, weights, c = record_nodes(obs)
        clipped.append(float(c) if len(nodes) else 0.0)
    out["age_nodes_k"] = nodes_k
    out["age_clipped_mass"] = clipped
    out.attrs["n_raw_rows"] = n_raw
    out.attrs["n_dedup_rows"] = n_dedup
    out.attrs["source_sha256"] = None  # filled by prepare.py
    return out


def eligible_primary_set(table, max_qc_flags=0):
    """Primary evidence set: no disqualifying QC flags (auxiliary sets report separately)."""
    disqualifying = [
        "qc_coords_out_of_range",
        "qc_negative_age",
        "qc_thickness_lt2",
        "qc_point_age_outside_domain",
    ]
    ok = ~table[disqualifying].any(axis=1)
    n_flags = table[["qc_thickness_gt80", "qc_thickness_gt100"]].sum(axis=1)
    return ok & (n_flags <= max_qc_flags)
