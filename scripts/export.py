"""Compatibility wrapper; use norn infer for checkpoint-based exports."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from norn_earth.cli import main


def export_dat(path, field_native, negative_encoded=False):
    from norn_earth.geometry.regrid import native_grid

    lat, lon = native_grid()
    lon, lat = np.meshgrid(lon, lat)
    values = np.asarray(field_native, dtype=float)
    if values.shape != (180, 360) or not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Expected a positive finite 180x360 thickness field")
    if negative_encoded:
        values = -values
    np.savetxt(
        path,
        np.column_stack([lon.ravel(), lat.ravel(), values.ravel()]),
        fmt="%.6f",
        header="lon lat crustal_thickness_km",
    )


if __name__ == "__main__":
    raise SystemExit(main(["infer", *sys.argv[1:]]))
