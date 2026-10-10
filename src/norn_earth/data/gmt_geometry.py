"""Use both same-age GMT exports as geometry, after numerical frame registration.

Original GPML topologies are used only to verify export/model compatibility;
plate IDs, boundary distances and network coverage come from the GMT polygons.
A newer rotation file with the same model name is NOT presumed compatible.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.sphere import lonlat_to_vectors
from ..utils.hashing import sha256_file
from .gmt import parse_gmt, polygon_parts, snapshot_paths

FRAME_TOLERANCE_KM = 0.001  # Full-precision exports must match to one metre.
_REQUIRED = {"ANCHOR", "TIME", "RECONFILE1", "PLATEID1", "GPGIM_TYPE", "FEATURE_ID"}


def topology_paths(path):
    """A combined GPML file or a directory of mutually referencing GPML files."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".gpml", ".gpmlz"))
        if not files:
            raise ValueError(f"No GPML topology files in {path}")
        return files
    return [path]


def polygon_vertices(polygon):
    rings = [polygon.get_exterior_ring_points()]
    rings.extend(
        polygon.get_interior_ring_points(i) for i in range(polygon.get_number_of_interior_rings())
    )
    return np.asarray([point.to_xyz() for ring in rings for point in ring])


def snapshot_features(snapshot, pygplates):
    features, polygons = [], []
    for item in snapshot["features"]:
        parts = []
        for outer, holes in polygon_parts(item):
            polygon = pygplates.PolygonOnSphere(
                [(float(lat), float(lon)) for lon, lat in outer],
                [[(float(lat), float(lon)) for lon, lat in hole] for hole in holes],
            )
            area = polygon.get_area()
            if not np.isfinite(area) or not 0 <= area < 4 * np.pi:
                raise ValueError(f"Invalid GMT polygon area: {item['feature_id']}")
            # A resolved incipient rift may trace the same line forward/back
            # and enclose exactly zero area. Retain its boundary, but do not
            # invent area coverage or an all-sphere polygon for containment.
            polygons.append(polygon)
            if area > 1e-15:
                parts.append(polygon)
        if parts:
            feature = pygplates.Feature()
            feature.set_reconstruction_plate_id(int(item["plate_id"]))
            feature.set_geometry(parts)
            features.append(feature)
    return features, polygons


class GMTGeometrySource:
    def __init__(self, config):
        self.config = config
        self.paths = {
            "plate_gmt": snapshot_paths(config.data.plate_gmt_dir, config.anchor_ages),
            "deformation_gmt": snapshot_paths(config.data.deformation_gmt_dir, config.anchor_ages),
        }
        self.hashes = {
            f"{kind}/{age:g}Ma/{path.name}": sha256_file(path)
            for kind, paths in self.paths.items()
            for age, path in paths.items()
        }

    def at_age(self, age, resolved, pygplates):
        """Verify every exported feature against the matching original model."""
        by_id = {}
        for item in resolved:
            key = item.get_feature().get_feature_id().get_string()
            by_id.setdefault(key, []).append(item)
        exports, report = {}, {"frame_max_error_km": 0.0, "frame_checked_features": 0}
        expected_rotation = Path(self.config.data.rotations).name
        for kind, paths in self.paths.items():
            path = paths[float(age)]
            snapshot = parse_gmt(path)
            if snapshot["geometry_type"] not in ("POLYGON", "MULTIPOLYGON"):
                raise ValueError(f"{path}: Expected a GMT polygon export")
            if not _REQUIRED <= set(snapshot["fields"]):
                raise ValueError(f"{path}: Missing GMT time/reference-frame/feature attributes")
            if kind == "plate_gmt" and not snapshot["features"]:
                raise ValueError(f"{path}: Empty plate geometry")
            expected_ids = {
                key
                for key, items in by_id.items()
                if any(
                    hasattr(item, "get_resolved_boundary")
                    and isinstance(item, pygplates.ResolvedTopologicalNetwork)
                    == (kind == "deformation_gmt")
                    for item in items
                )
            }
            exported_ids = {item["feature_id"] for item in snapshot["features"]}
            if expected_ids != exported_ids:
                raise ValueError(
                    f"{path}: Incomplete or different GMT/model feature set: "
                    f"{len(expected_ids - exported_ids)} missing, "
                    f"{len(exported_ids - expected_ids)} unexpected"
                )
            for feature in snapshot["features"]:
                meta, key = feature["meta"], feature["feature_id"]
                if not np.isfinite(float(meta["TIME"])) or abs(float(meta["TIME"]) - age) > 1e-6:
                    raise ValueError(f"{path}: GMT TIME differs from the requested age {age:g} Ma")
                if float(meta["ANCHOR"]) != 0:
                    raise ValueError(f"{path}: GMT ANCHOR must be 0 for this reconstruction frame")
                if meta["RECONFILE1"] != expected_rotation:
                    raise ValueError(
                        f"{path}: GMT rotation reference {meta['RECONFILE1']} differs from "
                        f"{expected_rotation}; use the export's original rotation model"
                    )
                native = by_id.get(key, [])
                if not native:
                    raise ValueError(f"{path}: GMT feature {key} absent from same-age GPML model")
                is_network = kind == "deformation_gmt"
                if (feature["gpgim_type"] == "gpml:TopologicalNetwork") != is_network or any(
                    isinstance(item, pygplates.ResolvedTopologicalNetwork) != is_network
                    for item in native
                ):
                    raise ValueError(
                        f"{path}: Plate/network GMT source roles differ from the model"
                    )
                if any(
                    item.get_feature().get_reconstruction_plate_id() != int(feature["plate_id"])
                    for item in native
                ):
                    raise ValueError(f"{path}: GMT and GPML plate IDs differ for {key}")
                boundaries = [item.get_resolved_boundary() for item in native]
                source_xyz = np.concatenate([polygon_vertices(p) for p in boundaries])
                export_xyz = np.concatenate(
                    [lonlat_to_vectors(ring[:, 0], ring[:, 1]) for ring in feature["rings"]]
                )
                forward = cKDTree(source_xyz).query(export_xyz)[0].max()
                backward = cKDTree(export_xyz).query(source_xyz)[0].max()
                error = float(2 * 6371 * np.arcsin(np.clip(max(forward, backward) / 2, 0, 1)))
                if error > FRAME_TOLERANCE_KM:
                    raise ValueError(
                        f"{path}: GMT/model reference-frame or geometry mismatch for {key}: "
                        f"{error:.6g} km > {FRAME_TOLERANCE_KM} km (not just a model-name check)"
                    )
                # Matching vertex sets alone would accept a scrambled/crossed
                # ring. Check each minor-arc midpoint against original edges.
                for ring in feature["rings"]:
                    xyz = lonlat_to_vectors(ring[:, 0], ring[:, 1])
                    mid = xyz + np.roll(xyz, -1, axis=0)
                    norms = np.linalg.norm(mid, axis=1)
                    if np.any(norms < 1e-12):
                        raise ValueError(f"{path}: Ambiguous antipodal GMT edge for {key}")
                    mid /= norms[:, None]
                    for vector in mid:
                        point = pygplates.PointOnSphere(tuple(vector))
                        if not any(
                            pygplates.GeometryOnSphere.distance(
                                point,
                                polygon,
                                distance_threshold_radians=FRAME_TOLERANCE_KM / 6371,
                                geometry2_is_solid=False,
                            )
                            is not None
                            for polygon in boundaries
                        ):
                            # pyGPlates' threshold shortcut can return None on
                            # degenerate/repeated source edges even at distance
                            # zero. Verify the unthresholded distance, without
                            # relaxing the metre tolerance.
                            distance = min(
                                pygplates.GeometryOnSphere.distance(point, polygon)
                                for polygon in boundaries
                            )
                            if distance * 6371 > FRAME_TOLERANCE_KM:
                                raise ValueError(
                                    f"{path}: GMT edge differs from original GPML for {key}"
                                )
                report["frame_max_error_km"] = max(report["frame_max_error_km"], error)
                report["frame_checked_features"] += 1
            features, polygons = snapshot_features(snapshot, pygplates)
            exports[kind] = (features, polygons)
            report[f"{kind}_features"] = len(snapshot["features"])
            report[f"{kind}_coverage_features"] = len(features)
            report[f"{kind}_polygon_parts"] = len(polygons)
            report[f"{kind}_zero_area_parts"] = sum(p.get_area() <= 1e-15 for p in polygons)
            report[f"{kind}_holes"] = sum(
                f["ring_roles"].count("hole") for f in snapshot["features"]
            )
        return exports, report
