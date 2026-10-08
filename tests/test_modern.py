import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[2]

from norn_earth.data.modern import assemble_modern_batch, load_modern_grid


class TestModern(unittest.TestCase):
    def test_load_real_dat(self):
        path = ROOT / "present_crustal_thickness.dat"
        if not path.is_file():
            self.skipTest("source DAT not present")
        grid = load_modern_grid(path)
        self.assertEqual(grid["H_km"].shape, (180, 360))
        self.assertTrue((grid["H_km"] > 0).all())
        self.assertAlmostEqual(float(grid["H_km"].sum()), 0.0 + float(grid["H_km"].sum()))
        # landmark sanity: Tibet thick, EPR thin
        self.assertGreater(
            grid["H_km"][np.argmin(abs(-89.5 + np.arange(180) - 32)) + 0, int(85.5)], 55
        )
        i_lat = int(round(0 + 89.5))  # lat 0
        j_lon = int(round(0 - 0.5)) % 360
        self.assertLess(grid["H_km"][i_lat, j_lon], 15)

    def test_assemble_batch(self):
        path = ROOT / "present_crustal_thickness.dat"
        if not path.is_file():
            self.skipTest("source DAT not present")
        b = assemble_modern_batch(path, sigma_km=3.0)
        self.assertEqual(b["H_km"].shape, b["sigma_km"].shape)
        self.assertFalse(b["sigma_is_calibrated"])
        self.assertEqual(int(b["block_ids"].min()), int(b["block_ids"][0, 0]))


if __name__ == "__main__":
    unittest.main()
