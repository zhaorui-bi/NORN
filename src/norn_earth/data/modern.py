"""Modern endpoint grid (DAT) loading and batching.

The DAT is an OBSERVATION with its own error, not perfect truth: the modern
source bias is a gauge fixed at zero (§5.4) and its per-cell sigma must be
supplied by calibration config (never learned freely, never assumed zero).
Loader verifies shape/order so the native 64,800-cell contract holds.
"""

import numpy as np

from ..geometry.regrid import native_cell_areas_km2, native_grid
from ..utils.hashing import sha256_file

EXPECTED_SHAPE = (180, 360)
EXPECTED_LONS = 0.5 + np.arange(360)
EXPECTED_LATS = -89.5 + np.arange(180)


def load_modern_grid(path, expect_sha256=None):
    """Return dict with H (180,360 positive km), areas, provenance."""
    if expect_sha256 is not None:
        got = sha256_file(path)
        if got != expect_sha256:
            raise ValueError(f"DAT hash changed: {got} != {expect_sha256}")
    raw = np.loadtxt(path)
    if raw.shape != (64800, 3):
        raise ValueError(f"unexpected DAT shape {raw.shape}")
    if not np.allclose(raw[:, 0].reshape(180, 360)[0], EXPECTED_LONS):
        raise ValueError("DAT longitude ordering mismatch")
    if not np.allclose(raw[:, 1].reshape(180, 360)[:, 0], EXPECTED_LATS):
        raise ValueError("DAT latitude ordering mismatch (expect south->north)")
    values = raw[:, 2].reshape(180, 360)
    if not (values < 0).all():
        raise ValueError("DAT sign convention violated: expect negative-encoded thickness")
    return {
        "H_km": -values,
        "areas_km2": native_cell_areas_km2(),
        "sha256": sha256_file(path),
    }


def modern_blocks(target_km=1000.0, nlat_bands=20):
    """Spatial blocks on the native grid (same rule as observation splits)."""
    from .splits import spatial_block_ids

    lats, _ = native_grid()
    lon2, lat2 = np.meshgrid(0.5 + np.arange(360), lats)
    return spatial_block_ids(lon2.ravel(), lat2.ravel(), target_km, nlat_bands).reshape(180, 360)


def assemble_modern_batch(path, sigma_km=3.0, expect_sha256=None, with_blocks=True):
    """Training-ready modern endpoint batch (numpy). sigma is a CONFIG value,
    calibrated externally; it is never a learned per-pixel quantity."""
    grid = load_modern_grid(path, expect_sha256)
    blocks = modern_blocks() if with_blocks else np.zeros(EXPECTED_SHAPE, dtype=int)
    return {
        "H_km": grid["H_km"].astype(np.float32),
        "areas_km2": grid["areas_km2"].astype(np.float32),
        "sigma_km": np.full(EXPECTED_SHAPE, float(sigma_km), dtype=np.float32),
        "block_ids": blocks.astype(np.int32),
        "sha256": grid["sha256"],
        "sigma_is_calibrated": False,
    }
