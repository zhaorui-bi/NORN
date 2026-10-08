"""Geological-challenge test set (user-directed, 2026-10-08).

Selects XLSX records from regions/times with KNOWN tectonic stories
(special plate-motion periods). Selection is by geological criteria only
(location + age window) -- never by model performance.

Challenge sets:
  1. India-Eurasia collision (50-35 Ma, 70-100E / 20-40N): crustal thickening
  2. Basin-and-Range extension (20-10 Ma, 105-125W / 30-45N): crustal thinning
  3. Andean arc pulse (25-10 Ma, 65-80W / 5-40S): magmatic addition
  4. Afar/Ethiopian plume (35-25 Ma, 30-50E / 0-20N): flood basalt / uplift
  5. Columbia River LIP (18-15 Ma, 110-125W / 40-50N): flood basalt
  6. Alpine orogeny (30-15 Ma, 5-20E / 40-50N): collision thickening
  7. Izu-Bonin-Mariana arc (30-15 Ma, 130-155E / 10-35N): arc maturation
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.utils.hashing import sha256_file, write_manifest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "processed"

CHALLENGES = {
    "india_eurasia_collision": {
        "lon": (60, 105), "lat": (15, 42), "age": (45, 60),
        "story": "India-Eurasia collision; crustal thickening to 60-80 km",
    },
    "basin_range_extension": {
        "lon": (235, 255), "lat": (28, 48), "age": (8, 25),
        "story": "Basin-and-Range extension; crustal thinning from ~50 to ~30 km",
    },
    "andean_arc_pulse": {
        "lon": (280, 295), "lat": (-45, 0), "age": (8, 30),
        "story": "Andean arc magmatism + crustal thickening",
    },
    "afar_plume": {
        "lon": (30, 55), "lat": (-5, 22), "age": (22, 38),
        "story": "Afar plume / Ethiopian flood basalt; uplift + magmatic underplating",
    },
    "columbia_river_lip": {
        "lon": (235, 250), "lat": (38, 52), "age": (13, 20),
        "story": "Columbia River flood basalts; Yellowstone hotspot",
    },
    "alpine_orogeny": {
        "lon": (3, 22), "lat": (38, 50), "age": (12, 35),
        "story": "Alpine collision orogeny; crustal thickening",
    },
    "ibm_arc_maturation": {
        "lon": (130, 155), "lat": (8, 38), "age": (12, 35),
        "story": "Izu-Bonin-Mariana arc maturation; transition from juvenile to mature arc",
    },
}


def select_challenges(table):
    """Return dict of challenge_name -> boolean mask + metadata."""
    lon = table["present_lon"].to_numpy(float)
    # normalize to 0-360
    lon360 = np.where(lon < 0, lon + 360, lon)
    lat = table["present_lat"].to_numpy(float)
    age = table["age_representative_ma"].to_numpy(float)
    lo = table["age_lower_ma"].to_numpy(float)
    hi = table["age_upper_ma"].to_numpy(float)

    out = {}
    for name, spec in CHALLENGES.items():
        lonmin, lonmax = spec["lon"]
        latmin, latmax = spec["lat"]
        amin, amax = spec["age"]
        # position match (handle both 0-360 and -180 to 180 conventions)
        pos = (
            ((lon360 >= lonmin) & (lon360 <= lonmax))
            | ((lon >= lonmin - 360) & (lon <= lonmax - 360))
            | ((lon >= lonmin + 360) & (lon <= lonmax + 360))
        ) & (lat >= latmin) & (lat <= latmax)
        # age overlap: record's age support [lo, hi] overlaps [amin, amax]
        age_ok = (hi >= amin) & (lo <= amax)
        mask = pos & age_ok
        n = int(mask.sum())
        # unique positions for cluster counting
        if n > 0:
            sub = table[mask]
            npos = len(sub[["present_lon", "present_lat"]].drop_duplicates())
        else:
            npos = 0
        out[name] = {
            "mask": mask,
            "n_records": n,
            "n_unique_positions": npos,
            "story": spec["story"],
            "box": {"lon": list(spec["lon"]), "lat": list(spec["lat"]), "age": list(spec["age"])},
        }
        print(f"  {name:<28s}: {n:5d} records, {npos:4d} unique positions -- {spec['story'][:50]}")
    return out


def main():
    table = pd.read_csv(OUT / "observations.csv")
    print("Selecting geological challenge test sets...")
    challenges = select_challenges(table)

    # union mask (a record can belong to multiple challenge sets; count once)
    union = np.zeros(len(table), dtype=bool)
    for c in challenges.values():
        union |= c["mask"]
    print(f"\nUnion: {int(union.sum())} records, "
          f"{len(table[union][['present_lon','present_lat']].drop_duplicates())} unique positions")

    # per-challenge stats (thickness distribution)
    report = {"challenges": {}, "union_n": int(union.sum())}
    for name, c in challenges.items():
        sub = table[c["mask"]]
        if len(sub) > 0:
            report["challenges"][name] = {
                **{k: v for k, v in c.items() if k != "mask"},
                "thickness_stats": {
                    "median": float(sub["thickness_km"].median()),
                    "p25": float(sub["thickness_km"].quantile(0.25)),
                    "p75": float(sub["thickness_km"].quantile(0.75)),
                    "n": len(sub),
                },
                "age_width_median": float((sub["age_upper_ma"] - sub["age_lower_ma"]).median()),
            }
        else:
            report["challenges"][name] = {k: v for k, v in c.items() if k != "mask"}

    # save the challenge masks as a separate CSV (record indices)
    idx_df = pd.DataFrame({"observation_id": table["observation_id"]})
    for name in CHALLENGES:
        idx_df[f"chal_{name}"] = challenges[name]["mask"]
    idx_df["chal_union"] = union
    idx_df.to_csv(OUT / "challenge_test_set.csv", index=False)

    write_manifest(OUT / "challenge_test_manifest.json", {
        **report,
        "source": {"observations_csv": sha256_file(OUT / "observations.csv")},
        "selection_rule": "geological location + age window only; no model scores involved",
    })
    print(f"\nWrote {OUT / 'challenge_test_set.csv'} and challenge_test_manifest.json")


if __name__ == "__main__":
    main()
