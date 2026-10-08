import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.evaluation.calibration import coverage_by_bin, interval_metrics
from norn_earth.evaluation.geoscience import (
    classify_grid,
    evidence_class,
    region_budget_error_km3,
    thickness_increment,
)
from norn_earth.evaluation.metrics import (
    effective_cluster_count,
    macro_point_metrics,
    paired_bootstrap_delta,
    point_metrics,
)


class TestMetrics(unittest.TestCase):
    def test_point_metrics(self):
        y = np.array([30.0, 40.0, 50.0])
        p = np.array([32.0, 38.0, 55.0])
        m = point_metrics(y, p)
        self.assertAlmostEqual(m["mae"], (2 + 2 + 5) / 3)
        self.assertEqual(m["n"], 3)

    def test_macro_metrics(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(10, 60, 300)
        p = y + rng.normal(0, 4, 300)
        g = rng.integers(0, 5, 300)
        out = macro_point_metrics(y, p, g)
        self.assertEqual(out["macro"]["n_groups"], 5)
        self.assertAlmostEqual(out["micro"]["n"], 300)

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(1)
        n = 300
        units = rng.integers(0, 25, n)
        y = rng.uniform(10, 60, n)
        pa = y + rng.normal(0, 3, n)
        pb = y + rng.normal(0, 6, n)
        res = paired_bootstrap_delta(y, pa, pb, units, n_boot=200)
        self.assertLess(res["ci_low"], res["ci_high"])
        self.assertLess(res["delta_mae_mean"], 0.0)  # a is tighter
        self.assertEqual(effective_cluster_count(units), 25)


class TestCalibration(unittest.TestCase):
    def test_interval_metrics_perfect(self):
        y = np.array([30.0, 40.0])
        out = interval_metrics(y, y - 1, y + 1)
        self.assertEqual(out["coverage"], 1.0)
        self.assertEqual(out["mean_width"], 2.0)

    def test_coverage_bins(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(0, 50, 400)
        half = rng.uniform(2, 20, 400)
        out = coverage_by_bin(y, y - half, y + half)
        self.assertGreater(len(out), 2)
        for row in out:
            self.assertGreaterEqual(row["coverage"], 0.0)
            self.assertLessEqual(row["coverage"], 1.0)


class TestGeoscience(unittest.TestCase):
    def test_evidence_classes(self):
        self.assertEqual(evidence_class(50.0, 4.0), "observation_supported")
        self.assertEqual(
            evidence_class(50.0, 40.0, kinematic_coverage=1.0, process_reliability=0.9),
            "physics_supported",
        )
        self.assertEqual(evidence_class(2000.0, 40.0, 0.5, 0.2), "prior_dominated")

    def test_classify_grid_and_budget(self):
        obs = np.array([[100.0, 5000.0], [5000.0, 5000.0]])
        age = np.array([[5.0, 5.0], [40.0, 40.0]])
        cls = classify_grid(obs, age)
        self.assertEqual(cls[0, 0], "observation_supported")
        self.assertEqual(cls[1, 1], "prior_dominated")
        self.assertAlmostEqual(region_budget_error_km3(100.0, 90.0), 10.0)
        inc = thickness_increment(np.full((4, 4), 30.0), np.full((4, 4), 32.0), np.ones((4, 4)))
        self.assertAlmostEqual(inc, 32.0)


if __name__ == "__main__":
    unittest.main()


class TestBlockConformal(unittest.TestCase):
    def test_coverage_on_exchangeable_blocks(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from norn_earth.evaluation.conformal import (
            block_conformal_calibrate,
            block_conformal_intervals,
            coverage,
            split_by_block,
        )

        rng = np.random.default_rng(3)
        n = 3000
        perm = rng.permutation(n)
        blocks = np.empty(n, dtype=int)
        blocks[perm] = np.repeat(np.arange(60), n // 60)
        pred = rng.uniform(10, 60, n)
        y = pred + rng.normal(0, 5, n) * (1 + 0.5 * (blocks % 3))
        cal_ids, test_ids = split_by_block(blocks, cal_fraction=0.5, seed=1)
        cal = np.isin(blocks, cal_ids)
        test = np.isin(blocks, test_ids)
        q, _ = block_conformal_calibrate(y[cal], pred[cal], blocks[cal], alpha=0.1)
        lo, hi = block_conformal_intervals(pred[test], q)
        cov_point = coverage(y[test], lo, hi)
        self.assertGreater(cov_point, 0.85)
        self.assertLess(cov_point, 0.97)
