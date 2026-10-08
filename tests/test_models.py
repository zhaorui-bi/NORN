import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.geometry.regrid import gauss_grid
from norn_earth.models.variational import (
    analyze,
    anchor_interpolation_weights,
    basis_terms,
    coefficient_count,
    evaluate_basis,
    poisson_solve,
    synthesize,
)
from norn_earth.models.baselines import (
    b0_predict,
    b1_marginal_nll,
    b2_build_features,
    b6_backtrack,
    robust_center_scale,
    spherical_rbf_kernel,
)


class TestVariational(unittest.TestCase):
    def test_coefficient_counts(self):
        self.assertEqual(coefficient_count(0), 1)
        self.assertEqual(coefficient_count(2), 9)
        self.assertEqual(len(basis_terms(4)), 25)

    def test_orthonormality(self):
        lats, lons, w = gauss_grid(64, 128)
        Y = evaluate_basis(4, lats, lons)
        W = np.repeat(w[:, None], 128, axis=1).ravel()
        W = W / W.sum() * 4 * np.pi  # cell weights sum to 4*pi steradian
        G = (Y * W[:, None]).T @ Y
        self.assertTrue(np.allclose(G, np.eye(G.shape[0]), atol=1e-10))

    def test_roundtrip_random_field(self):
        lats, lons, w = gauss_grid(48, 96)
        rng = np.random.default_rng(6)
        coef = rng.normal(size=coefficient_count(4)) * 0.01
        coef[0] = 30.0
        field = synthesize(coef, 4, lats, lons)
        got = analyze(field, 4, lats, lons, w)
        self.assertTrue(np.allclose(got, coef, atol=1e-8))

    def test_constant_field_is_l0(self):
        lats, lons, w = gauss_grid(32, 64)
        field = np.full((32, 64), 17.0)
        coef = analyze(field, 3, lats, lons, w)
        self.assertAlmostEqual(coef[0], 17.0 * np.sqrt(4 * np.pi), delta=1e-9)
        self.assertAlmostEqual(float(np.abs(coef[1:]).max()), 0.0, delta=1e-9)

    def test_poisson_solve_laplacian(self):
        lats, lons, w = gauss_grid(48, 96)
        lon2, lat2 = np.meshgrid(lons, lats)
        D = 0.05 * np.exp(-((lat2 - 20.0) ** 2 + lon2**2) / (2 * 15.0**2))
        D = D - D.mean()
        psi_coef = poisson_solve(D, 6, lats, lons, w)
        psi = synthesize(psi_coef, 6, lats, lons)
        dlon = np.radians(lons[1] - lons[0])
        dlat = np.radians(lats[1] - lats[0])
        lap = (
            (np.roll(psi, -1, axis=1) - 2 * psi + np.roll(psi, 1, axis=1)) / dlon**2
            + (np.roll(psi, -1, axis=0) - 2 * psi + np.roll(psi, 1, axis=0)) / dlat**2
        ) / 6371.0**2
        # coarse check: correlated with target D away from poles
        c = np.corrcoef(lap[10:-10].ravel(), D[10:-10].ravel())[0, 1]
        self.assertGreater(c, 0.5)

    def test_anchor_weights_partition_of_unity(self):
        w = anchor_interpolation_weights(13.4, anchor_step=1.0, n_anchors=61)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(np.count_nonzero(w), 2)
        w_end = anchor_interpolation_weights(60.0)
        self.assertEqual(np.count_nonzero(w_end), 1)


class TestBaselines(unittest.TestCase):
    def test_b0_rolling_median(self):
        ages = np.random.default_rng(0).uniform(0, 60, 500)
        thick = 30 + 0.1 * ages
        pred = b0_predict(thick, ages, [10.0, 40.0])
        self.assertEqual(len(pred), 2)
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_b1_marginal_nll_finite(self):
        rng = np.random.default_rng(1)
        n = 40
        lon = rng.uniform(0, 360, n)
        lat = rng.uniform(-60, 60, n)
        K = spherical_rbf_kernel(lon, lat, lon, lat)
        mu = rng.normal(30, 5, (n, 3))
        w = np.tile(np.array([0.5, 0.3, 0.2]), (n, 1))
        y = rng.normal(30, 5, n)
        nll = b1_marginal_nll(y, mu, w, K, np.full(n, 3.0))
        self.assertTrue(np.isfinite(nll))

    def test_b2_features_and_b6(self):
        obs = [
            {
                "present_lon": 10.0,
                "present_lat": 5.0,
                "age_lower": 20.0,
                "age_upper": 30.0,
                "age_representative": 25.0,
                "thickness_km": 40.0,
            }
        ]
        rows = b2_build_features(obs, point_age=True)
        self.assertEqual(rows[0]["variant"], "point_age")
        self.assertAlmostEqual(b6_backtrack(35.0, 1.2, 1.0, 1.2), 35.0 * 1.2 / 1.0)
        center, scale = robust_center_scale(np.arange(100))
        self.assertGreater(scale, 0)


if __name__ == "__main__":
    unittest.main()
