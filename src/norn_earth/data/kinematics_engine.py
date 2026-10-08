"""P4 hard rule (v1.3): observation node positions from the G1-VERIFIED engine.

Main-model node positions are rebuilt with pygplates + Muller2019 v2.0
(static-polygon partitioning + plate-circuit rotation). The XLSX paleo
columns are VALIDATION-ONLY (kinematics circularity is forbidden).

Degradation: without pygplates, callers must fall back to present-day
coordinates WITH a position-error inflation -- never to XLSX paleo columns
for the main model (stat-pilot legacy behavior is explicitly flagged).
"""

import numpy as np


class KinematicsEngine:
    """Reconstruct present-day positions to age a (Ma) via pygplates."""

    def __init__(self, rotations_path, static_polygons_path):
        import pygplates

        self.pygplates = pygplates
        self.rm = pygplates.RotationModel(str(rotations_path))
        static = pygplates.FeatureCollection(str(static_polygons_path))
        self.partitioner = pygplates.PlatePartitioner(static, self.rm)

    def plate_ids(self, lon, lat):
        ids = []
        for lo, la in zip(np.atleast_1d(lon), np.atleast_1d(lat)):
            resolved = self.partitioner.partition_point(
                self.pygplates.PointOnSphere(float(la), float(lo))
            )
            ids.append(resolved.get_feature().get_reconstruction_plate_id() if resolved else -1)
        return np.array(ids, dtype=int)

    def positions_at(self, lon, lat, ages):
        """Reconstruct paired points to their per-point ages.

        Returns (lon_arr, lat_arr, ok_mask); unresolved points keep their
        present coordinates and are flagged, never silently trusted.
        """
        lon = np.atleast_1d(np.asarray(lon, float))
        lat = np.atleast_1d(np.asarray(lat, float))
        ages = np.atleast_1d(np.asarray(ages, float))
        if not (len(lon) == len(lat) == len(ages)):
            raise ValueError("lon/lat/ages must be parallel arrays")
        # Repeated records and age nodes share modern positions. Resolve each
        # unique position once, then reuse the authoritative plate assignment.
        unique, inverse = np.unique(np.column_stack([lon, lat]), axis=0, return_inverse=True)
        pids = self.plate_ids(unique[:, 0], unique[:, 1])[inverse]
        out_lon, out_lat = lon.copy(), lat.copy()
        ok = np.ones(len(lon), dtype=bool)
        rotations = {}
        for i in range(len(lon)):
            if pids[i] < 0:
                ok[i] = False
                continue
            try:
                key = (float(ages[i]), int(pids[i]))
                if key not in rotations:
                    rotations[key] = self.rm.get_rotation(*key)
                R = rotations[key]
                p = R * self.pygplates.PointOnSphere(float(lat[i]), float(lon[i]))
                out_lat[i], out_lon[i] = p.to_lat_lon()
            except Exception:
                ok[i] = False
        return out_lon, out_lat, ok


def engine_paths(norn_root):
    ext = norn_root / "data" / "external" / "muller2019"
    return (
        ext / "Rotations" / "Muller_etal_2019_CombinedRotations.rot",
        ext / "StaticPolygons" / "Muller_etal_2019_Global_StaticPlatePolygons.gpmlz",
    )


def rebuild_node_positions(engine, node_lon_present, node_lat, node_ages):
    """Clean batch API: engine-rebuilt positions for flat node arrays."""
    return engine.positions_at(node_lon_present, node_lat, node_ages)
