"""Evaluation entry: consumes FROZEN artifacts only (§11.4/§14.2).

Refuses to run when outputs/evaluation/freeze.json is absent: metrics may be
computed only after models/thresholds are frozen and the test set is opened
once. Helper functions for ad-hoc diagnostics live in the library.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
FREEZE = Path(__file__).resolve().parents[1] / "outputs" / "evaluation" / "freeze.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not FREEZE.is_file():
        print(
            "REFUSED: no frozen evaluation manifest. Open the test set only after "
            "model/protocol freeze (P-A/P-B/P-C rules in 最终研究方案_NORN.md §11.4).",
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"freeze": json.loads(FREEZE.read_text(encoding="utf-8"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
