"""Freeze preliminary split manifests (G2 artifacts, PRELIMINARY until G0/G1).

Runs on processed/observations.csv (read-only) and writes
processed/split_manifest.json with leakage self-checks. Selection rules are
coverage-based only; nothing here looks at model performance or thickness
values of test rows.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.splits import build_split_manifest
from norn_earth.utils.hashing import sha256_file, write_manifest

HERE = Path(__file__).resolve().parents[1]


def main():
    table = pd.read_csv(HERE / "processed" / "observations.csv")
    manifest = build_split_manifest(table)
    manifest["source"] = {"observations_csv": sha256_file(HERE / "processed" / "observations.csv")}
    out = HERE / "processed" / "split_manifest.json"
    write_manifest(out, manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "n_records", "n_site_clusters", "all_passed")}, indent=2))
    for name, rep in manifest["splits"].items():
        print(f"{name}: train={rep['counts']['train']} test={rep['counts']['test']} "
              f"buffer={rep['counts']['buffer']} passed={rep['passed']} "
              f"clusters(test)={rep['effective_site_clusters']['test']}")


if __name__ == "__main__":
    main()
