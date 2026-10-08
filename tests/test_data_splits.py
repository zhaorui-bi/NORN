import sys
import unittest
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.observations import build_observation_table, eligible_primary_set
from norn_earth.data.trajectories import (
    build_trajectory_rows,
    reliable_segments,
    sample_position,
    support_mass,
    trajectory_lookup,
)
from norn_earth.data.splits import (
    TestWindowError,
    assert_window_usable,
    chronological_holdout,
    leakage_report,
    modern_block_holdout,
    site_clusters,
    spatial_block_ids,
)


def make_mini_xlsx(path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    n = int(n)
    cols = {}
    cols["纬度"] = rng.uniform(-80, 80, n)
    cols["经度"] = rng.uniform(0, 360, n)
    age_lo = rng.uniform(0, 66, n)
    age_hi = age_lo + rng.choice([0.0, 2.0, 8.0, 40.0, 66.0], n)
    cols["年龄_起始 (Ma)"] = np.minimum(age_lo, age_hi)
    cols["年龄_结束 (Ma)"] = np.maximum(age_lo, age_hi)
    cols["年龄_代表值 (Ma)"] = 0.5 * (cols["年龄_起始 (Ma)"] + cols["年龄_结束 (Ma)"])
    cols["时间段"] = "Cenozoic"
    cols["平均厚度 (km)"] = rng.uniform(5, 70, n)
    for i in range(13):
        s = "" if i == 0 else f".{i}"
        has = rng.uniform(size=n) < 0.6
        cols[f"TIME{s}"] = np.where(has, 5.0 * i, np.nan)
        cols[f"paleo_lon{s}"] = np.where(has, rng.uniform(0, 360, n), np.nan)
        cols[f"paleo_lat{s}"] = np.where(has, rng.uniform(-80, 80, n), np.nan)
    df = pd.DataFrame(cols)
    df.to_excel(path, index=False)
    return df


_MINI = {}


def _mini():
    """Shared in-memory mini XLSX fixture (fast, isolated)."""
    if not _MINI:
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        xlsx = Path(tmp.name) / "mini.xlsx"
        _MINI["tmp"] = tmp
        _MINI["df"] = make_mini_xlsx(xlsx)
        _MINI["path"] = xlsx
    return _MINI["path"], _MINI["df"]


class TestObservations(unittest.TestCase):
    def test_build_and_semantics(self):
        path, df = _mini()
        table = build_observation_table(path)
        self.assertEqual(len(table), len(df.drop_duplicates()))
        self.assertIn("age_semantics", table.columns)
        kinds = set(table["age_semantics"])
        self.assertTrue(
            kinds
            <= {
                "point_inside",
                "point_outside",
                "narrow_interval",
                "interval",
                "wide_interval",
                "coarse_full_domain",
            }
        )
        self.assertTrue((table["age_nodes_k"] >= 0).all())
        self.assertTrue(table["source_unverified"].all())

    def test_eligible_excludes_bad_qc(self):
        path, _ = _mini()
        table = build_observation_table(path)
        elig = eligible_primary_set(table)
        bad = table["qc_negative_age"] | table["qc_coords_out_of_range"] | table["qc_thickness_lt2"]
        self.assertFalse((elig & bad).any())


class TestTrajectories(unittest.TestCase):
    def test_segments_and_support(self):
        times = np.array([0.0, 5.0, 20.0])
        segs = reliable_segments(times)
        self.assertEqual(segs, [(0.0, 5.0)])
        self.assertAlmostEqual(support_mass(segs, 1.0, 4.0), 1.0)
        self.assertAlmostEqual(support_mass(segs, 0.0, 20.0), 5.0 / 20.0)
        self.assertEqual(support_mass(segs, 3.0, 3.0), 1)  # point inside segment
        self.assertEqual(support_mass(segs, 10.0, 10.0), 0)

    def test_long_rows_and_lookup(self):
        _, raw = _mini()
        rows, dropped = build_trajectory_rows(raw)
        self.assertGreater(len(rows), 0)
        lookup = trajectory_lookup(rows)
        traj = next(iter(lookup.values()))
        pos = sample_position(traj, float(traj[0][0]))
        self.assertIsNotNone(pos)


class TestSplits(unittest.TestCase):
    def test_site_clusters_deterministic(self):
        rng = np.random.default_rng(2)
        lon = rng.uniform(0, 50, 200)
        lat = rng.uniform(0, 50, 200)
        c1 = site_clusters(lon, lat, 100.0)
        c2 = site_clusters(lon, lat, 100.0)
        self.assertTrue((c1 == c2).all())

    def test_spatial_blocks_cover_all(self):
        rng = np.random.default_rng(3)
        ids = spatial_block_ids(rng.uniform(0, 360, 500), rng.uniform(-89, 89, 500))
        self.assertEqual(len(ids), 500)
        self.assertTrue(np.all(ids >= 0))

    def test_chronological_holdout_removes_entire_records(self):
        n = 200
        rng = np.random.default_rng(4)
        table = pd.DataFrame(
            {
                "observation_id": [f"o{i}" for i in range(n)],
                "age_lower_ma": rng.uniform(0, 60, n),
                "age_upper_ma": rng.uniform(0, 66, n),
                "age_representative_ma": rng.uniform(0, 60, n),
                "site_cluster": rng.integers(0, 20, n),
            }
        )
        lo, hi = table["age_lower_ma"], table["age_upper_ma"]
        table["age_lower_ma"], table["age_upper_ma"] = np.minimum(lo, hi), np.maximum(lo, hi)
        train, test, buffer = chronological_holdout(table, (15.0, 25.0))
        overlap = (table["age_upper_ma"] >= 15) & (table["age_lower_ma"] <= 25)
        self.assertFalse((train & overlap).any())  # every overlapping record left training
        rep = np.asarray(table["age_representative_ma"], dtype=float)
        width = np.asarray(table["age_upper_ma"], dtype=float) - np.asarray(
            table["age_lower_ma"], dtype=float
        )
        self.assertTrue(bool(np.all(rep[test] >= 15.0) and np.all(rep[test] <= 25.0)))
        self.assertTrue(bool(np.all(width[test] <= 5.0)))
        report = leakage_report(table, train, test, buffer)
        self.assertTrue(report["passed"])

    def test_modern_holdout(self):
        n = 100
        table = pd.DataFrame({"observation_id": [f"o{i}" for i in range(n)]})
        blocks = np.arange(n) % 10
        train, test = modern_block_holdout(table, blocks, [3, 7])
        self.assertEqual(int(test.sum()), 20)
        self.assertFalse((train & test).any())


if __name__ == "__main__":
    unittest.main()


class TestWindowGuard(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.registry = Path(directory.name) / "registry.json"
        self.registry.write_text(
            json.dumps(
                {
                    "P-C_15_25": {"status": "reserved", "reason": "synthetic final holdout"},
                    "P-A_modern_blocks": {
                        "status": "reserved",
                        "reason": "synthetic spatial holdout",
                    },
                    "P-C_40_50": {"status": "development_burned"},
                }
            ),
            encoding="utf-8",
        )

    def test_reserved_window_refuses(self):
        with self.assertRaises(TestWindowError):
            assert_window_usable("P-C_15_25", self.registry)
        with self.assertRaises(TestWindowError):
            assert_window_usable("P-A_modern_blocks", self.registry)

    def test_burned_window_selectable(self):
        status = assert_window_usable("P-C_40_50", self.registry)
        self.assertEqual(status, "development_burned")

    def test_unknown_window_refuses(self):
        with self.assertRaises(TestWindowError):
            assert_window_usable("P-C_99_99", self.registry)

    def test_missing_registry_refuses(self):
        with self.assertRaises(FileNotFoundError):
            assert_window_usable("P-C_40_50", self.registry.with_name("missing.json"))
