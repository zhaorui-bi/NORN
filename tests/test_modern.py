import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.modern import assemble_modern_batch, load_modern_grid


class TestModern(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "modern.dat"
        self.thickness = np.full((180, 360), 30.0)
        self.thickness[122, 85] = 65.0
        self.thickness[90, 0] = 7.5
        lon, lat = np.meshgrid(np.arange(0.5, 360), np.arange(-89.5, 90))
        np.savetxt(
            self.path,
            np.column_stack([lon.ravel(), lat.ravel(), -self.thickness.ravel()]),
        )

    def test_load_native_negative_encoded_dat(self):
        grid = load_modern_grid(self.path)
        self.assertEqual(grid["H_km"].shape, (180, 360))
        self.assertTrue((grid["H_km"] > 0).all())
        np.testing.assert_array_equal(grid["H_km"], self.thickness)

    def test_assemble_batch(self):
        b = assemble_modern_batch(self.path, sigma_km=3.0)
        self.assertEqual(b["H_km"].shape, b["sigma_km"].shape)
        np.testing.assert_array_equal(b["H_km"], self.thickness)
        np.testing.assert_array_equal(b["sigma_km"], np.full((180, 360), 3.0))
        self.assertFalse(b["sigma_is_calibrated"])
        self.assertEqual(int(b["block_ids"].min()), int(b["block_ids"][0, 0]))


if __name__ == "__main__":
    unittest.main()
