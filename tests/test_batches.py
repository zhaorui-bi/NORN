import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.batches import assemble_observation_batch, batch_slices
from norn_earth.data.trajectories import trajectory_lookup
from norn_earth.losses.likelihood import SourceRegistry


def _mini_table(rng, n=120):
    import pandas as pd

    lo = rng.uniform(0, 60, n)
    width = rng.choice([0.0, 2.0, 40.0], n)
    table = pd.DataFrame(
        {
            "observation_id": [f"o{i}" for i in range(n)],
            "source_unverified": True,
            "present_lon": rng.uniform(0, 360, n),
            "present_lat": rng.uniform(-80, 80, n),
            "age_lower_ma": lo,
            "age_upper_ma": np.minimum(lo + width, 66),
            "age_representative_ma": lo + width / 2,
            "age_semantics": [
                "point_inside" if w == 0 else ("narrow_interval" if w <= 5 else "wide_interval")
                for w in width
            ],
            "thickness_km": rng.uniform(10, 60, n),
            "trajectory_support_mass": rng.uniform(0, 1, n),
            "primary_set": rng.uniform(size=n) > 0.2,
        }
    )
    return table


def _mini_traj(rng, n=120):
    import pandas as pd

    rows = []
    for i in range(n):
        for k in range(0, 13, 2):  # every 10 Myr: gaps > 5 Myr rule applies
            if rng.uniform() < 0.7:
                rows.append(
                    {
                        "source_row": i,
                        "time_ma": 5.0 * k,
                        "paleo_lon": rng.uniform(0, 360),
                        "paleo_lat": rng.uniform(-80, 80),
                    }
                )
    return trajectory_lookup(pd.DataFrame(rows))


class TestBatches(unittest.TestCase):
    def test_assembly_shapes_and_weights(self):
        rng = np.random.default_rng(0)
        table = _mini_table(rng)
        traj = _mini_traj(rng)
        reg = SourceRegistry()
        reg.register("unverified_proxy", 5.0)
        batch = assemble_observation_batch(table, traj, reg)
        self.assertEqual(len(batch["y_km"]), len(table))
        self.assertEqual(batch["node_ages"].shape, batch["node_weights"].shape)
        self.assertEqual(batch["node_ages"].shape, batch["node_lon"].shape)
        starts, ends = batch_slices(batch["node_counts"])
        for i, (s, e) in enumerate(zip(starts, ends)):
            if e > s:
                self.assertAlmostEqual(batch["node_weights"][s:e].sum(), 1.0, delta=1e-9)
        # effective scale = sqrt(record^2 + source^2)
        expect = np.sqrt(batch["record_sigma_km"] ** 2 + batch["source_sigma_km"] ** 2)
        self.assertTrue(np.allclose(batch["effective_sigma_km"], expect))
        # positions: nodes either from trajectory or flagged present-coords
        self.assertTrue(set(np.unique(batch["node_from_trajectory"])) <= {True, False})

    def test_unregistered_source_rejected(self):
        rng = np.random.default_rng(1)
        table = _mini_table(rng, 20)
        with self.assertRaises(KeyError):
            assemble_observation_batch(table, {}, SourceRegistry())


if __name__ == "__main__":
    unittest.main()
