import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.losses.chronology import (
    jensen_gap,
    make_age_nodes,
    mixture_nll,
    replication_invariance_error,
    student_t_logpdf,
)


class TestChronology(unittest.TestCase):
    def test_weights_sum_and_clipped_mass(self):
        nodes, w, c = make_age_nodes(0.0, 66.0, 8)
        self.assertEqual(len(nodes), 8)
        self.assertAlmostEqual(w.sum(), 1.0, delta=1e-12)
        self.assertAlmostEqual(c, 60.0 / 66.0, delta=1e-9)
        n2, w2, c2 = make_age_nodes(13.2, 13.2, 3)
        self.assertEqual(list(n2), [13.2])
        self.assertAlmostEqual(w2.sum(), 1.0)
        self.assertAlmostEqual(c2, 1.0)

    def test_point_age_outside_domain_dropped(self):
        nodes, w, c = make_age_nodes(63.0, 63.0, 1)
        self.assertEqual(len(nodes), 0)
        self.assertEqual(c, 0.0)

    def test_touch_only_intervals_have_zero_domain_mass(self):
        for lower, upper in [(-5, 0), (60, 65)]:
            for distribution in ["uniform", "triangular"]:
                nodes, weights, mass = make_age_nodes(lower, upper, 8, distribution=distribution)
                self.assertEqual(len(nodes), 0)
                self.assertEqual(len(weights), 0)
                self.assertEqual(mass, 0)
        for age in [0, 60]:
            nodes, weights, mass = make_age_nodes(age, age, 1)
            self.assertEqual(list(nodes), [age])
            self.assertEqual(list(weights), [1])
            self.assertEqual(mass, 1)

    def test_narrow_vs_wide_node_counts(self):
        n, w, _ = make_age_nodes(10.0, 12.0, 3)
        self.assertEqual(len(n), 3)

    def test_mixture_nll_and_gradient(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(20, 50, 6)
        mu = rng.uniform(20, 50, (6, 4))
        w = rng.dirichlet(np.ones(4), size=6)
        scale = np.full(6, 4.0)
        nll, grad = mixture_nll(y, mu, w, scale, return_grad=True)
        self.assertEqual(nll.shape, (6,))
        self.assertEqual(grad.shape, (6, 4))
        # finite-difference check on one mean
        eps = 1e-6
        mu2 = mu.copy()
        mu2[3, 2] += eps
        nll2 = mixture_nll(y, mu2, w, scale)
        fd = (nll2[3] - nll[3]) / eps
        self.assertAlmostEqual(fd, grad[3, 2], delta=1e-4)

    def test_replication_invariance(self):
        rng = np.random.default_rng(3)
        y = rng.uniform(20, 50, 5)
        mu = rng.uniform(20, 50, (5, 3))
        w = rng.dirichlet(np.ones(3), size=5)
        scale = np.full(5, 3.5)
        err = replication_invariance_error(y, mu, w, scale)
        self.assertLess(err, 1e-10)

    def test_jensen_gap_sign(self):
        rng = np.random.default_rng(4)
        y = rng.uniform(20, 50, 5)
        mu = rng.uniform(20, 50, (5, 3))
        w = rng.dirichlet(np.ones(3), size=5)
        scale = np.full(5, 3.0)
        # Jensen: mixture NLL <= expected point NLL; strictly smaller when nodes differ
        self.assertLess(jensen_gap(y, mu, w, scale), 0.0)
        # and the two objectives are not interchangeable when any mu differ
        self.assertNotAlmostEqual(jensen_gap(y, mu, w, scale), 0.0, places=6)

    def test_nonpositive_scale_rejected(self):
        with self.assertRaises(ValueError):
            student_t_logpdf(1.0, 0.0, -1.0)

    def test_bad_weights_rejected(self):
        y = np.array([30.0])
        mu = np.array([[30.0, 30.0]])
        w = np.array([[0.7, 0.7]])
        with self.assertRaises(ValueError):
            mixture_nll(y, mu, w, np.array([3.0]))


if __name__ == "__main__":
    unittest.main()
