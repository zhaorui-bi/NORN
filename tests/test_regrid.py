import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.geometry.regrid import (
    conservative_native_to_gauss,
    gauss_area_weights_km2,
    gauss_grid,
    gauss_to_native_sample,
    native_cell_areas_km2,
    sample_at_points,
    total_gauss_area_km2,
)
from norn_earth.utils.units import EARTH_RADIUS_KM


class TestRegrid(unittest.TestCase):
    def test_total_areas(self):
        self.assertAlmostEqual(total_gauss_area_km2(), 4 * np.pi * EARTH_RADIUS_KM**2, delta=1.0)
        self.assertAlmostEqual(
            native_cell_areas_km2().sum(), 4 * np.pi * EARTH_RADIUS_KM**2, delta=1.0
        )

    def test_gauss_grid_ordering_and_weights(self):
        lats, lons, w = gauss_grid()
        self.assertTrue(np.all(np.diff(lats) > 0))  # south -> north
        self.assertAlmostEqual(lons[0], 0.0)
        self.assertAlmostEqual(w.sum(), 2.0, delta=1e-12)
        aw = gauss_area_weights_km2()
        self.assertAlmostEqual(aw.sum(), 4 * np.pi * EARTH_RADIUS_KM**2, delta=1.0)

    def test_conservative_constant_field(self):
        const = np.full((180, 360), 13.5)
        out = conservative_native_to_gauss(const)
        self.assertTrue(np.allclose(out, 13.5, atol=1e-9))

    def test_conservative_integral_preserved(self):
        rng = np.random.default_rng(1)
        field = rng.uniform(5, 60, size=(180, 360))
        native_integral = float((native_cell_areas_km2() * field).sum())
        out = conservative_native_to_gauss(field)
        gauss_integral = float(np.nansum(gauss_area_weights_km2() * out))
        self.assertAlmostEqual(
            gauss_integral / native_integral,
            1.0,
            delta=5e-3,
            msg="area-weighted integrals must agree up to quadrature error",
        )

    def test_roundtrip_constant_sample(self):
        const = np.full((180, 360), 21.0)
        out = gauss_to_native_sample(const)
        self.assertTrue(np.allclose(out, 21.0, atol=1e-6))

    def test_sampling_locations(self):
        lats, lons, _ = gauss_grid()
        field = np.ones((180, 360))
        v = sample_at_points(field, np.array([lats[90]]), np.array([lons[100]]))
        self.assertAlmostEqual(float(v[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
