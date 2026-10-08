import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.physics.budgets import ControlGrid, edge_velocity_from_center
from norn_earth.physics.kinematics import (
    advect_points,
    interpolate_trajectory,
    rigid_rotation_divergence,
    rotation_velocity_km_per_myr,
)
from norn_earth.utils.units import CM_PER_YR_TO_KM_PER_MYR


class TestKinematics(unittest.TestCase):
    def test_unit_conversion(self):
        self.assertAlmostEqual(CM_PER_YR_TO_KM_PER_MYR * 1.0, 10.0)

    def test_rigid_rotation_divergence_free(self):
        cg = ControlGrid(90, 180)
        div = rigid_rotation_divergence(15.0, 40.0, 1.0, cg.lat_centers, cg.lon_centers)
        interior = div[5:-5, 5:-5]
        self.assertLess(float(np.max(np.abs(interior))), 2e-4)

    def test_advect_rigid_matches_exact_rotation(self):
        pole_lat, pole_lon, rate = 20.0, -60.0, 0.5

        def vel(lon, lat):
            return rotation_velocity_km_per_myr(pole_lat, pole_lon, rate, lon, lat)

        lon = np.array([10.0, -30.0, 150.0])
        lat = np.array([0.0, 45.0, -20.0])
        steps, dtau = 20, 1.0
        lo_rk, la_rk = advect_points(lon, lat, vel, dtau, steps, order=2)
        from norn_earth.physics.kinematics import integrate_rotation

        lo_ex, la_ex = integrate_rotation(lon, lat, pole_lat, pole_lon, rate, dtau, steps)
        from norn_earth.geometry.sphere import haversine_km

        d = haversine_km(lo_rk, la_rk, lo_ex, la_ex)
        self.assertLess(float(d.max()), 30.0)  # RK2 vs exact over 10 deg

    def test_interpolate_trajectory_rules(self):
        times = np.array([0.0, 5.0, 20.0])
        lons = np.array([10.0, 15.0, 40.0])
        lats = np.array([0.0, 0.0, 0.0])
        got = interpolate_trajectory(times, lons, lats, 2.5)
        self.assertIsNotNone(got)
        self.assertIsNone(interpolate_trajectory(times, lons, lats, 10.0))  # gap 15 > 5
        self.assertIsNone(interpolate_trajectory(times, lons, lats, -1.0))  # no extrapolation


class TestBudgets(unittest.TestCase):
    def test_control_grid_area(self):
        cg = ControlGrid(90, 180)
        from norn_earth.utils.units import EARTH_RADIUS_KM

        self.assertAlmostEqual(cg.total_area_km2(), 4 * np.pi * EARTH_RADIUS_KM**2, delta=1.0)

    def test_no_flow_no_source_budget(self):
        cg = ControlGrid(90, 180)
        H = np.full((90, 180), 35.0)
        zeros_lon = np.zeros((90, 181))
        zeros_lat = np.zeros((91, 180))
        r = cg.budget_residual_km3(H, H.copy(), zeros_lon, zeros_lat, np.zeros_like(H), 1.0)
        self.assertAlmostEqual(r, 0.0, delta=1e-6)

    def test_uniform_outflow_budget(self):
        cg = ControlGrid(90, 180)
        H = np.full((90, 180), 35.0)
        # uniform eastward outflow through periodic edges cancels globally
        u = np.ones((90, 180))
        v = np.zeros((90, 180))
        u_e, v_e = edge_velocity_from_center(u, v)
        r = cg.budget_residual_km3(H, H.copy(), u_e, v_e, np.zeros_like(H), 1.0)
        self.assertAlmostEqual(r, 0.0, delta=1e-6)  # periodicity: no net flux

    def test_source_only_budget(self):
        cg = ControlGrid(90, 180)
        H = np.full((90, 180), 35.0)
        zeros_lon = np.zeros((90, 181))
        zeros_lat = np.zeros((91, 180))
        dtau = 2.0
        q = np.full((90, 180), 0.01)  # km/Myr
        expected_dh = q * dtau
        H1 = H + expected_dh
        r = cg.budget_residual_km3(H, H1, zeros_lon, zeros_lat, q, dtau)
        self.assertAlmostEqual(r, 0.0, delta=1e-3)

    def test_coverage_report(self):
        cg = ControlGrid(45, 90)
        mask = np.zeros((45, 90), dtype=bool)
        mask[:10] = True
        rep = cg.coverage_report(mask)
        self.assertGreater(rep["active_fraction_of_area"], 0.0)
        self.assertLess(rep["active_fraction_of_area"], 0.25)


if __name__ == "__main__":
    unittest.main()
