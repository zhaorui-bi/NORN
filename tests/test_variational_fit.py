import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.models.variational_fit import VariationalReconstructor


def _synthetic_instance(rng, n_obs=150, lmax=6):
    """Small closed instance: SH truth x anchors, noisy point observations."""
    from norn_earth.models.variational import coefficient_count, synthesize
    from norn_earth.geometry.regrid import gauss_grid

    lats, lons, _ = gauss_grid(45, 90)
    K, L = 7, lmax
    coef = np.zeros((K, coefficient_count(L)))
    for m in range(K):
        coef[m, 0] = 30.0 + 2.0 * np.sin(m / K * np.pi)
        coef[m, 3:10] = rng.normal(0, 9.0, 7)
    ages = np.linspace(60.0, 0.0, K)

    def truth_field(k):
        return synthesize(coef[k], L, lats, lons)

    truth = np.stack([truth_field(k) for k in range(K)])

    lon = rng.uniform(0, 360, n_obs)
    lat = rng.uniform(-70, 70, n_obs)
    true_age = rng.uniform(0, 60, n_obs)
    # sample truth by interpolation in age and space (coarse but consistent)
    from norn_earth.geometry.regrid import sample_at_points

    y = []
    for i in range(n_obs):
        f = (60.0 - true_age[i]) / 10.0
        k0 = int(np.clip(np.floor(f), 0, K - 2))
        w = f - k0
        v0 = sample_at_points(truth[k0], np.array([lat[i]]), np.array([lon[i]]), grid=(lats, lons))
        v1 = sample_at_points(
            truth[k0 + 1], np.array([lat[i]]), np.array([lon[i]]), grid=(lats, lons)
        )
        y.append(float(((1 - w) * v0 + w * v1 + rng.normal(0, 2.0)).item()))
    counts = np.ones(n_obs, dtype=int)
    obs = {
        "y_km": np.asarray(y),
        "node_counts": counts,
        "node_ages": true_age.copy(),
        "node_weights": np.ones(n_obs),
        "node_lon": lon,
        "node_lat": lat,
        "effective_sigma_km": np.full(n_obs, np.sqrt(2.0**2 + 3.0**2)),
    }
    modern = {
        "H_km": (truth[-1] + rng.normal(0, 1.5, truth[-1].shape)).astype(np.float32),
        "areas_km2": np.ones_like(truth[-1], dtype=np.float32),  # uniform weights suffice here
        "sigma_km": np.full(truth[-1].shape, 1.5, dtype=np.float32),
        "block_ids": np.zeros(truth[-1].shape, dtype=np.int32),
    }
    return coef, ages, lats, lons, truth, obs, modern


class TestVariationalFit(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        _, ages, lats, lons, truth, obs, modern = _synthetic_instance(rng, n_obs=60)
        model = VariationalReconstructor(
            lmax=4, n_anchors=7, anchor_step=10.0, lam_modern=0.1, lam_obs=1.0, lam_smooth=0.01
        )
        model.coef = rng.normal(0, 0.05, model.coef.shape) + model.coef
        total, G = model.grad(obs, modern)
        # finite differences on a few coordinates
        idx = [(0, 0), (3, 2), (6, 5)]
        for m, c in idx:
            eps = 1e-5
            old = model.coef[m, c]
            model.coef[m, c] = old + eps
            up = model.objective(obs, modern)
            model.coef[m, c] = old - eps
            dn = model.objective(obs, modern)
            model.coef[m, c] = old
            fd = (up - dn) / (2 * eps)
            self.assertAlmostEqual(
                float(G[m, c]),
                fd,
                delta=1e-4,
                msg=f"grad mismatch at ({m},{c}): {G[m, c]} vs fd {fd}",
            )

    def test_fit_improves_over_flat_baseline(self):
        rng = np.random.default_rng(7)
        _, ages, lats, lons, truth, obs, modern = _synthetic_instance(rng, n_obs=250)
        model = VariationalReconstructor(
            lmax=6, n_anchors=7, anchor_step=10.0, lam_modern=0.5, lam_obs=1.0, lam_smooth=1e-4
        )
        before = model.objective(obs, modern)
        model.fit(obs, modern, maxiter=120)
        after = model.objective(obs, modern)
        self.assertLess(after, before)
        # prediction correlation on held-out ages
        mu, _ = model.predict_nodes(obs["node_lon"], obs["node_lat"], obs["node_ages"])
        corr = np.corrcoef(mu, obs["y_km"])[0, 1]
        self.assertGreater(corr, 0.6, "fitted field should track synthetic signal")
        flat = np.full_like(obs["y_km"], np.median(obs["y_km"]))
        mae_fit = float(np.mean(np.abs(mu - obs["y_km"])))
        mae_flat = float(np.mean(np.abs(flat - obs["y_km"])))
        self.assertLess(mae_fit, mae_flat / 3, "B7 must beat B0 flat decisively on synthetic")


if __name__ == "__main__":
    unittest.main()
