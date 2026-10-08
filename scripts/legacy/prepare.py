"""Prepare processed tables from the read-only source files (F1 tooling).

Never mutates originals. Writes:
  processed/observations.csv        standardized observation table (one record = one evidence)
  processed/trajectories_long.csv   reconstructed coordinate times (long format)
  processed/trajectory_summary.json support/coverage diagnostics
  processed/prepare_manifest.json   hashes, counts, gate status

Provenance remains source-unverified until Gate G0; nothing here fits models.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.gmt import parse_time_from_name, parse_gmt
from norn_earth.data.observations import build_observation_table, eligible_primary_set
from norn_earth.data.trajectories import (
    build_trajectory_rows,
    summarize_coverage,
    trajectory_lookup,
)
from norn_earth.utils.hashing import sha256_file, write_manifest

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "古地壳厚度数据_新生代_处理后_final.xlsx"
GMT_DIRS = [
    ROOT / "古板块边界-时间间隔每百万年 - 最终版",
    ROOT / "构造变形区边界-时间间隔每百万年 - 最终版",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=Path(__file__).resolve().parents[1] / "processed")
    parser.add_argument("--xlsx", default=XLSX)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.xlsx)
    n_raw = len(raw)
    table = build_observation_table(args.xlsx)
    table.attrs["source_sha256"] = sha256_file(args.xlsx)

    traj_long, dropped = build_trajectory_rows(raw)
    lookup = trajectory_lookup(traj_long)
    table = summarize_coverage(lookup, table)
    table["eligible_primary"] = eligible_primary_set(table)

    counts = {
        "rows_raw": n_raw,
        "rows_after_exact_dedup": int(table.attrs["n_dedup_rows"]),
        "unique_present_positions": int(table[["present_lon", "present_lat"]].drop_duplicates().shape[0]),
        "age_semantics": table["age_semantics"].value_counts().to_dict(),
        "qc_flag_counts": {
            c: int(table[c].sum())
            for c in table.columns if c.startswith("qc_")
        },
        "trajectory_dropped_time_mismatches": int(dropped),
        "trajectory_primary_support": int(table["primary_set"].sum()),
        "eligible_primary_set": int(table["eligible_primary"].sum()),
    }

    table.to_csv(out_dir / "observations.csv", index=False)
    traj_long.to_csv(out_dir / "trajectories_long.csv", index=False)
    (out_dir / "trajectory_summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    gmt_stats = {}
    for d in GMT_DIRS:
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.gmt"))
        gmt_stats[d.name] = {
            "file_count": len(files),
            "times_ma": [parse_time_from_name(p) for p in files][:5] + ["..."],
            "sample_features": len(parse_gmt(files[0])["features"]) if files else 0,
        }
    write_manifest(
        out_dir / "prepare_manifest.json",
        {
            "source_xlsx": {"path": Path(args.xlsx).name, "sha256": sha256_file(args.xlsx)},
            "gmt_directories": gmt_stats,
            "counts": counts,
            "gates": {"G0_semantics": "OPEN", "G1_kinematics": "OPEN", "G2_experiment_freeze": "OPEN"},
            "notes": "source-unverified; no model fitting performed here",
        },
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"\nWrote tables to {out_dir}")


if __name__ == "__main__":
    main()
