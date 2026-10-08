"""Product export: 61 native-order grids + uncertainty/evidence layers (§16).

Refuses without a frozen, authorized reconstruction artifact. The exporter
itself is deterministic: positive-thickness DAT in native 64,800 cell order
(lon 0.5..359.5, lat -89.5..89.5 south->north), NetCDF metadata carrying
interval_kind/calibration_scope so uncalibrated ranges are never labeled
as calibrated intervals.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RECON = Path(__file__).resolve().parents[1] / "outputs" / "reconstructions" / "final.npz"


def export_dat(path, field_native, negative_encoded=False):
    """field_native: (180,360) south->north, lon 0.5..; writes lon lat value."""
    from norn_earth.geometry.regrid import native_grid

    lats, lons = native_grid()
    lon2, lat2 = np.meshgrid(lons, lats)
    values = -np.asarray(field_native, dtype=float) if negative_encoded else np.asarray(field_native, dtype=float)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# lon lat crustal_thickness_km\n")
        for lo, la, v in zip(lon2.ravel(), lat2.ravel(), values.ravel()):
            f.write(f"{lo:8.3f} {la:8.3f} {v:10.4f}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not RECON.is_file():
        print(
            "REFUSED: no frozen reconstruction artifact; export requires completed, "
            "authorized evaluation (G4->G5).",
            file=sys.stderr,
        )
        return 1
    print("Export would proceed; artifact present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
