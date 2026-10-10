"""Age-bound geometry from one plate model, on the actual SHT grid.

Continents are reconstructed and boundaries/networks are resolved independently
at every anchor age. A transported modern reference is only a conditioning
feature, NEVER a historical label. Network outlines do not provide strain.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.regrid import gauss_grid, native_grid, sample_at_points
from ..geometry.sphere import lonlat_to_vectors
from ..utils.hashing import sha256_file
from .gmt_geometry import GMTGeometrySource, topology_paths


def geometry_source_hashes(config):
    """Include shapefile attributes/plate IDs and GPlates schema sidecars."""
    hashes = {}
    for key in ("rotations", "static_polygons", "continental_polygons", "topologies"):
        value = getattr(config.data, key)
        if not value:
            continue
        path = Path(value)
        if path.is_dir():
            for source in topology_paths(path):
                hashes[f"{key}/{source.name}"] = sha256_file(source)
        else:
            hashes[key] = sha256_file(path)
        if path.suffix.lower() == ".shp":
            for suffix in (".dbf", ".shx", ".prj", ".shp.gplates.xml"):
                sidecar = path.with_suffix(suffix)
                if sidecar.exists():
                    hashes[f"{key}{suffix}"] = sha256_file(sidecar)
    return hashes


def partition_ids(geometries, rotation_model, points, pygplates):
    if not geometries:
        return np.full(len(points), -1, dtype=np.int32)
    partitioner = pygplates.PlatePartitioner(geometries, rotation_model)
    return np.fromiter(
        (
            -1
            if (plate := partitioner.partition_point(p)) is None
            else plate.get_feature().get_reconstruction_plate_id()
            for p in points
        ),
        dtype=np.int32,
        count=len(points),
    )


def boundary_distance(geometries, xyz, tessellation_degrees=0.5):
    """Nearest tessellated boundary vertex; error <= half the segment length.

    Three-dimensional chord queries are periodic and pole-safe. Distances are
    capped at 3000 km. These are unclassified plate/network boundaries, not
    identified ridges or trenches.
    """
    vertices = []
    for item in geometries:
        polygon = item.get_resolved_boundary() if hasattr(item, "get_resolved_boundary") else item
        polygon = polygon.to_tessellated(np.radians(tessellation_degrees))
        vertices.extend(p.to_xyz() for p in polygon.get_exterior_ring_points())
        for i in range(polygon.get_number_of_interior_rings()):
            vertices.extend(p.to_xyz() for p in polygon.get_interior_ring_points(i))
    if not vertices:
        raise ValueError("No resolved plate boundaries at this age")
    chord, _ = cKDTree(np.asarray(vertices)).query(xyz)
    return np.minimum(2 * 6371 * np.arcsin(np.clip(chord / 2, 0, 1)), 3000)


def material_reference(modern, points, plate_ids, network, rotation_model, age):
    """Back-rotate rigid grid locations to sample the observed modern product.

    Nonzero-age network locations and unresolved points are missing (zero
    feature value); their geometry channels identify this support limitation.
    This rigid transport does not assume past thickness equals modern thickness.
    """
    lon = np.zeros(len(points))
    lat = np.zeros(len(points))
    supported = (plate_ids >= 0) & (~network if age != 0 else True)
    for pid in np.unique(plate_ids[supported]):
        rotation = rotation_model.get_rotation(
            float(age), int(pid), use_identity_for_missing_plate_ids=False
        )
        if rotation is None:
            supported[plate_ids == pid] = False
            continue
        inverse = rotation.get_inverse()
        for i in np.flatnonzero(supported & (plate_ids == pid)):
            lat[i], lon[i] = (inverse * points[i]).to_lat_lon()
    values = np.zeros(len(points))
    lats, lons = native_grid()
    values[supported] = sample_at_points(
        modern["H_km"], lat[supported], lon[supported], (lats, lons)
    )
    return values, supported


def build_dynamic_geometry(config, modern, base_features):
    import pygplates

    d, g = config.data, config.grid
    rotation = pygplates.RotationModel(d.rotations)
    continents = pygplates.FeatureCollection(d.continental_polygons)
    topologies = [
        feature
        for path in topology_paths(d.topologies)
        for feature in pygplates.FeatureCollection(str(path))
    ]
    gmt = GMTGeometrySource(config) if d.geometry_source == "gmt" else None
    glat, glon, _ = gauss_grid(g.nlat, g.nlon)
    lon, lat = np.meshgrid(glon, glat[::-1])
    points = [
        pygplates.PointOnSphere(float(la), float(lo)) for lo, la in zip(lon.ravel(), lat.ravel())
    ]
    xyz = lonlat_to_vectors(lon.ravel(), lat.ravel())
    inputs = np.broadcast_to(base_features, (config.n_anchors,) + base_features.shape).copy()
    plate_ids = np.empty((config.n_anchors, g.nlat, g.nlon), dtype=np.int32)
    report = []
    for index, age in enumerate(config.anchor_ages):
        reconstructed, resolved = [], []
        pygplates.reconstruct(continents, rotation, reconstructed, float(age))
        pygplates.resolve_topologies(topologies, rotation, resolved, float(age))
        if not reconstructed or not resolved:
            raise ValueError(f"Missing reconstructed continents or topologies at {age:g} Ma")
        networks = [p for p in resolved if isinstance(p, pygplates.ResolvedTopologicalNetwork)]
        continental = partition_ids(reconstructed, rotation, points, pygplates) >= 0
        gmt_report = {}
        if gmt is None:
            network = partition_ids(networks, rotation, points, pygplates) >= 0
            pids = partition_ids(resolved, rotation, points, pygplates)
            distance = boundary_distance(resolved, xyz)
        else:
            exports, gmt_report = gmt.at_age(float(age), resolved, pygplates)
            plates, plate_polygons = exports["plate_gmt"]
            network_features, network_polygons = exports["deformation_gmt"]
            # Features already contain age-a coordinates. Partition at time 0
            # (identity reconstruction), never rotate the GMT snapshot again.
            pids = partition_ids(plates, rotation, points, pygplates)
            network_ids = partition_ids(network_features, rotation, points, pygplates)
            network = network_ids >= 0
            pids[network] = network_ids[network]
            distance = boundary_distance(plate_polygons + network_polygons, xyz)
        reference, supported = material_reference(modern, points, pids, network, rotation, age)
        inputs[index, 3] = reference.reshape(lon.shape) / 40
        inputs[index, 4] = 2 * continental.reshape(lon.shape).astype(float) - 1
        inputs[index, 5] = distance.reshape(lon.shape) / 3000
        inputs[index, 6] = network.reshape(lon.shape)
        plate_ids[index] = pids.reshape(lon.shape)
        report.append(
            {
                "age_ma": float(age),
                **gmt_report,
                "continents": len(reconstructed),
                "resolved_topologies": len(resolved),
                "networks": len(networks),
                "unassigned_grid_points": int((pids < 0).sum()),
                "reference_supported_grid_points": int(supported.sum()),
            }
        )
        print(
            f"Geometry {age:g} Ma: {len(resolved)} topologies, "
            f"{int(continental.sum())} continental grid points",
            flush=True,
        )
    metadata = {
        "mode": "dynamic",
        "source": d.geometry_source,
        "binding": "inputs[m] belongs to feature_ages_ma[m]",
        "reconstruction": (
            "pygplates continents + same-age plate/network GMT polygons; GPML frame registration only"
            if gmt is not None
            else "pygplates continents + resolved plate boundaries/networks"
        ),
        "reference_frame": {
            "anchor_plate_id": 0,
            "rotation_file": Path(d.rotations).name,
            "gmt_registration": "every feature, vertex and edge midpoint at every anchor; <=1 m"
            if gmt is not None
            else "not applicable",
        },
        "network_coverage": "all exported/resolved network outlines, including inactive networks; not strain",
        "reference": "rigidly transported modern conditioning feature, not ancient labels",
        "network_reference": "missing at nonzero ages; network outlines are not strain",
        "boundary_distance": "nearest vertex after 0.5 degree tessellation, capped at 3000 km",
        "continental_mask": "reconstructed continental polygons, not a thickness threshold",
        "source_sha256": {**geometry_source_hashes(config), **(gmt.hashes if gmt else {})},
        "anchors": report,
        "fractional_age_policy": "thickness/continuous geometry interpolate between anchors; plate IDs use nearest anchor",
    }
    return inputs.astype(np.float32), plate_ids, metadata
