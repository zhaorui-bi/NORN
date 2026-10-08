import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.physics.arc import (
    arc_source_km_per_myr,
    gaussian_kernel,
    uniform_width_rate_km_per_myr,
)
from norn_earth.physics.priors import (
    EvidenceGroup,
    merge_duplicates,
    replication_invariant_nll,
    shared_offset_information,
)
from norn_earth.physics.ridge import birth_thickness_prior, retained_melt_thickness_km
from norn_earth.losses.likelihood import SourceRegistry, modern_endpoint_nll
from norn_earth.losses.weak_form import (
    budget_loss,
    choose_trajectory_or_budget,
    mask_unknown_flux,
)


class TestPriorsLedger(unittest.TestCase):
    def test_kernel_normalized(self):
        d = np.linspace(-1000, 1000, 4001)
        k = gaussian_kernel(d, 100.0)
        # discrete integral ~ 1 per km
        self.assertAlmostEqual(__import__("scipy").integrate.trapezoid(k, d), 1.0, delta=1e-3)

    def test_arc_units(self):
        q = arc_source_km_per_myr(0.0, 75.0, width_km=150.0)  # c=75 km^2/Myr
        self.assertGreater(q, 0.0)
        self.assertAlmostEqual(uniform_width_rate_km_per_myr(75.0, 150.0), 0.5)

    def test_ridge_guards(self):
        prior = birth_thickness_prior()
        self.assertEqual(prior["mu_km"], 7.0)
        with self.assertRaises(ValueError):
            retained_melt_thickness_km(30.0, 3.0)  # below applicability floor
        self.assertAlmostEqual(float(retained_melt_thickness_km(40.0, 10.0)), 4.0)

    def test_evidence_group_merge_and_replication(self):
        g = EvidenceGroup("arc_q_asia", ["zhou2025_tableS2"], "net_production", 1.0, [[0.04]])
        merged = merge_duplicates([g, g, g])
        self.assertEqual(len(merged), 1)

        def stat(rep):
            return np.array([1.0])

        nll = replication_invariant_nll(g, 5, stat)
        self.assertTrue(np.isfinite(nll))

    def test_shared_offset_information_bound(self):
        for n in (5, 31, 200):
            info = shared_offset_information(n, 2.0, 4.0)
            expected = n / (2.0**2 + n * 4.0**2)
            self.assertAlmostEqual(info, expected, delta=1e-12)
            self.assertLess(info, 1.0 / 4.0**2 + 1e-9)  # bounded by shared error

    def test_learned_covariance_penalty(self):
        g = EvidenceGroup("x", ["s"], "stat", 0.0, [[1.0]])
        n1 = g.nll(np.array([0.5]))
        n2 = g.nll(np.array([0.5]), learned_scale=2.0)
        # larger learned scale must pay the logdet penalty, not silently reduce loss
        self.assertGreater(n2, n1 - 0.5 * np.log(4.0))


class TestLikelihood(unittest.TestCase):
    def test_source_registry(self):
        reg = SourceRegistry()
        reg.register("modern_dat", 1.5, bias_km=0.0)
        self.assertAlmostEqual(reg.effective_scale(2.0, "modern_dat"), np.sqrt(4 + 2.25))
        with self.assertRaises(KeyError):
            reg.effective_scale(2.0, "unknown")

    def test_modern_endpoint_area_normalized(self):
        rng = np.random.default_rng(5)
        h_pred = rng.uniform(10, 50, 64800).reshape(180, 360)
        from norn_earth.geometry.regrid import native_cell_areas_km2

        areas = native_cell_areas_km2()
        nll = modern_endpoint_nll(h_pred, h_pred.copy(), np.full(64800, 3.0).ravel(), areas.ravel())
        self.assertTrue(np.isfinite(nll))


class TestWeakForm(unittest.TestCase):
    def test_budget_loss_masks(self):
        r = np.array([1.0, 2.0, 100.0])
        active = np.array([True, True, False])
        loss, info = budget_loss(r, active)
        scale = 35.0 * (111.0**2) * 100.0
        expected = np.mean([(1.0 / scale) ** 2, (2.0 / scale) ** 2])
        self.assertAlmostEqual(loss, expected, delta=1e-18)
        self.assertEqual(info["active_domains"], 2)

    def test_role_choice_and_unknown_flux(self):
        roles = choose_trajectory_or_budget(
            {"a": ["trajectory", "budget"], "b": ["trajectory"], "c": []}
        )
        self.assertEqual(roles["a"], "budget")
        self.assertEqual(roles["b"], "trajectory")
        self.assertIsNone(roles["c"])
        known = np.array([True, False])
        active = np.array([True, True])
        self.assertEqual(list(mask_unknown_flux(known, active)), [True, False])


if __name__ == "__main__":
    unittest.main()
