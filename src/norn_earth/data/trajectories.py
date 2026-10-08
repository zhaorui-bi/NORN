"""Trajectory table (§5.3/§5.5): positions come from reconstruction, not truth.

Each source row carries up to 13 reconstructed coordinate times
(0,5,...,60 Ma). These locate a TRAJECTORY; they are not independent
thickness labels, and a missing coordinate is not evidence of crust
birth/death. Age support c_i = mass of p_i(a) inside reliably locatable
segments (gap<=5 Myr rule) is recorded per observation; records below a
frozen threshold move to the auxiliary set and are never relabelled to the
nearest TIME column.
"""

import numpy as np
import pandas as pd

from ..physics.kinematics import interpolate_trajectory

TIME_COLUMNS = [f"TIME{'' if i == 0 else f'.{i}'}" for i in range(13)]
LON_COLUMNS = [f"paleo_lon{'' if i == 0 else f'.{i}'}" for i in range(13)]
LAT_COLUMNS = [f"paleo_lat{'' if i == 0 else f'.{i}'}" for i in range(13)]
EXPECTED_TIMES = np.arange(13) * 5.0
MAX_GAP_MYR = 5.0
MIN_SUPPORT_MASS = 0.5  # frozen default; changes require re-freeze (G2)


def build_trajectory_rows(df):
    """Long-format rows: source_row, time_ma, paleo_lon, paleo_lat.

    Only rows whose TIME column matches its expected 5-Myr slot are accepted;
    mismatches are dropped and counted (never silently reassigned).
    Vectorized: no per-row Python iteration over the full table.
    """
    pieces = []
    dropped_mismatch = 0
    for k, (tc, lc, pc) in enumerate(zip(TIME_COLUMNS, LON_COLUMNS, LAT_COLUMNS)):
        if tc not in df.columns:
            continue
        t = pd.to_numeric(df[tc], errors="coerce")
        valid = df[[tc, lc, pc]].notna().all(axis=1) & np.isfinite(t)
        mismatch = valid & ((t - EXPECTED_TIMES[k]).abs() > 1e-6)
        dropped_mismatch += int(mismatch.sum())
        keep = valid & ~mismatch
        if not keep.any():
            continue
        pieces.append(
            pd.DataFrame(
                {
                    "source_row": np.flatnonzero(keep.to_numpy()),
                    "time_ma": t[keep].astype(float).to_numpy(),
                    "paleo_lon": pd.to_numeric(df.loc[keep, lc], errors="coerce")
                    .astype(float)
                    .to_numpy(),
                    "paleo_lat": pd.to_numeric(df.loc[keep, pc], errors="coerce")
                    .astype(float)
                    .to_numpy(),
                }
            )
        )
    if not pieces:
        return pd.DataFrame(
            columns=["source_row", "time_ma", "paleo_lon", "paleo_lat"]
        ), dropped_mismatch
    return pd.concat(pieces, ignore_index=True), dropped_mismatch


def trajectory_lookup(traj_rows):
    """dict source_row -> (times, lons, lats) sorted by time."""
    out = {}
    for idx, grp in traj_rows.groupby("source_row"):
        g = grp.sort_values("time_ma")
        out[idx] = (
            g["time_ma"].to_numpy(float),
            g["paleo_lon"].to_numpy(float),
            g["paleo_lat"].to_numpy(float),
        )
    return out


def reliable_segments(times):
    """Contiguous [t_k, t_{k+1}] segments with gap <= MAX_GAP_MYR."""
    segs = []
    for a, b in zip(times[:-1], times[1:]):
        if b - a <= MAX_GAP_MYR + 1e-9:
            segs.append((float(a), float(b)))
    return segs


def support_mass(segs, lo, hi, domain=(0.0, 60.0)):
    """Mass of uniform p(a) on [lo,hi] that lies inside reliable segments.

    For point ages the support is 1 if the point is bracketed/adjacent within
    the gap rule, else 0.
    """
    lo, hi = float(lo), float(hi)
    a, b = max(lo, domain[0]), min(hi, domain[1])
    if b < a:
        return 0.0
    if lo == hi:
        for s0, s1 in segs:
            if s0 - 1e-9 <= lo <= s1 + 1e-9:
                return 1.0
        return 0.0
    covered = 0.0
    for s0, s1 in segs:
        covered += max(0.0, min(b, s1) - max(a, s0))
    return covered / (b - a)


def sample_position(traj, age):
    times, lons, lats = traj
    return interpolate_trajectory(times, lons, lats, age, max_gap_myr=MAX_GAP_MYR)


def summarize_coverage(traj_lookup, observation_table):
    """Per-observation support c_i + availability of true-age positions."""
    lo = observation_table["age_lower_ma"].to_numpy(float)
    hi = observation_table["age_upper_ma"].to_numpy(float)
    rep = observation_table["age_representative_ma"].to_numpy(float)
    supports, point_ok = [], []
    for i in range(len(observation_table)):
        traj = traj_lookup.get(i)
        if traj is None or len(traj[0]) == 0:
            supports.append(0.0)
            point_ok.append(False)
            continue
        segs = reliable_segments(traj[0])
        supports.append(support_mass(segs, lo[i], hi[i]))
        point_ok.append(sample_position(traj, rep[i]) is not None)
    table = observation_table.copy()
    table["trajectory_support_mass"] = supports
    table["point_age_position_available"] = point_ok
    table["primary_set"] = table["trajectory_support_mass"] >= MIN_SUPPORT_MASS
    return table
