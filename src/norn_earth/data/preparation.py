"""Reproducible data preparation for the public training and inference APIs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry.regrid import (
    conservative_native_to_gauss,
    gauss_grid,
    native_grid,
    sample_at_points,
)
from ..geometry.sphere import lonlat_to_vectors
from ..losses.chronology import make_age_nodes
from ..utils.hashing import sha256_file, write_manifest
from .kinematics_engine import KinematicsEngine
from .modern import assemble_modern_batch
from .splits import site_clusters

FEATURE_NAMES = [
    "sin_lat",
    "cos_lat_cos_lon",
    "cos_lat_sin_lon",
    "modern_reference_div40",
    "continental_mask",
    "plate_boundary_distance_div3000",
    "deformation_coverage",
    "cos_lat_squared",
]


def _split_records(table, config):
    d = config.data
    lon = (table.present_lon.to_numpy(float) + 180) % 360 - 180
    groups = site_clusters(lon, table.present_lat.to_numpy(float))
    split = np.zeros(len(table), dtype=np.int64)  # 0 train, 1 validation, 2 test
    test = np.zeros(len(table), bool)
    if d.holdout_csv:
        holdout = pd.read_csv(d.holdout_csv)
        if "observation_id" not in holdout or d.holdout_column not in holdout:
            raise ValueError(
                "Holdout CSV requires observation_id and the configured holdout_column"
            )
        if holdout.observation_id.duplicated().any():
            raise ValueError("Duplicate observation_id in holdout CSV")
        mask = holdout[d.holdout_column]
        if mask.dtype != bool:
            if not mask.isin([0, 1]).all():
                raise ValueError("Holdout column must contain booleans or 0/1")
            mask = mask.astype(bool)
        held_ids = set(holdout.loc[mask, "observation_id"])
        if not held_ids.issubset(set(table.observation_id)):
            raise ValueError("Holdout CSV contains unknown observation IDs")
        test = table.observation_id.isin(held_ids).to_numpy()
        # Remove the entire geographic group, not only its challenge-age rows.
        test = np.isin(groups, np.unique(groups[test]))
    split[test] = 2
    available = np.unique(groups[~test])
    rng = np.random.default_rng(config.training.seed)
    rng.shuffle(available)
    nv = int(round(len(available) * d.validation_fraction))
    if d.validation_fraction and len(available) > 1:
        nv = min(len(available) - 1, max(1, nv))
    split[np.isin(groups, available[:nv]) & ~test] = 1
    return split, groups


def _geometry_features(config, modern):
    """Features live on actual SHT nodes, never relabelled native grid cells."""
    g, d = config.grid, config.data
    glat, glon, _ = gauss_grid(g.nlat, g.nlon)
    lon, lat = np.meshgrid(glon, glat[::-1])  # torch-harmonics: north -> south
    lr, pr = np.radians(lon), np.radians(lat)
    h = conservative_native_to_gauss(modern["H_km"], g.nlat, g.nlon)[::-1]
    mask = conservative_native_to_gauss((modern["H_km"] >= 20).astype(float), g.nlat, g.nlon)[::-1]
    edge_distance = np.zeros_like(h)
    support = {"modern": True, "static_plate_edges": False, "deformation_coverage": False}
    if d.static_polygons and d.rotations:
        from scipy.spatial import cKDTree

        engine = KinematicsEngine(d.rotations, d.static_polygons)
        pids = engine.plate_ids(lon.ravel(), lat.ravel()).reshape(h.shape)
        edge = (pids != np.roll(pids, 1, axis=1)) | (pids != np.roll(pids, -1, axis=1))
        edge[1:] |= pids[1:] != pids[:-1]
        edge[:-1] |= pids[:-1] != pids[1:]
        if edge.any():
            xyz = lonlat_to_vectors(lon, lat)
            dist, _ = cKDTree(xyz[edge]).query(xyz.reshape(-1, 3))
            edge_distance = (2 * 6371 * np.arcsin(np.clip(dist / 2, 0, 1))).reshape(h.shape)
            support["static_plate_edges"] = True
    features = np.stack(
        [
            np.sin(pr),
            np.cos(pr) * np.cos(lr),
            np.cos(pr) * np.sin(lr),
            h / 40,
            2 * mask - 1,
            np.minimum(edge_distance / 3000, 1),
            np.zeros_like(h),
            np.cos(pr) ** 2,
        ]
    ).astype(np.float32)
    return features, support


def prepare_dataset(config, output):
    """Write a self-contained, pickle-free NPZ dataset and a provenance manifest."""
    config.validate()
    d, g = config.data, config.grid
    if not d.observations or not d.modern:
        raise ValueError("data.observations and data.modern are required for preparation")
    path = Path(d.observations)
    if path.suffix.lower() == ".xlsx":
        from .observations import build_observation_table

        table = build_observation_table(path)
    else:
        table = pd.read_csv(path)
    required = [
        "observation_id",
        "present_lon",
        "present_lat",
        "age_lower_ma",
        "age_upper_ma",
        "thickness_km",
    ]
    if any(name not in table for name in required):
        raise ValueError(f"Observations require columns: {required}")
    if table.observation_id.duplicated().any():
        raise ValueError("Observation IDs must be unique")
    values = table[required[1:]].to_numpy(float)
    valid = (
        np.isfinite(values).all(axis=1)
        & (np.abs(table.present_lat) <= 90)
        & (table.thickness_km > 0)
    )
    if not d.include_flagged_records:
        for flag in (
            "qc_coords_out_of_range",
            "qc_negative_age",
            "qc_thickness_lt2",
            "qc_point_age_outside_domain",
        ):
            if flag in table:
                if table[flag].dtype != bool:
                    raise ValueError(f"{flag} must be boolean")
                valid &= ~table[flag].to_numpy()
    modern = assemble_modern_batch(d.modern, d.modern_sigma_km)
    features, feature_support = _geometry_features(config, modern)
    if d.geometry_mode == "dynamic":
        from .dynamic_geometry import build_dynamic_geometry

        features, plate_ids, geometry = build_dynamic_geometry(config, modern, features)
        feature_support.update(
            static_plate_edges=False,
            plate_boundaries=True,
            continental_geometry=True,
            deformation_coverage=True,
        )
    else:
        features = np.repeat(features[None], config.n_anchors, axis=0)
        plate_ids = np.full((config.n_anchors, g.nlat, g.nlon), -1, dtype=np.int32)
        geometry = {"mode": "static", "binding": "identical baseline features at every age"}
        feature_support["plate_boundaries"] = feature_support["static_plate_edges"]
        feature_support["continental_geometry"] = False
    split, groups = _split_records(table, config)
    node_age, node_weight, node_record, mass = [], [], [], np.zeros(len(table))
    for i, row in enumerate(table.itertuples()):
        if not valid[i]:
            continue
        width = abs(row.age_upper_ma - row.age_lower_ma)
        k = 1 if width == 0 else (3 if width <= 5 else 8)
        ages, weights, mass[i] = make_age_nodes(
            row.age_lower_ma, row.age_upper_ma, k, domain=(0, config.max_age_ma)
        )
        node_age.extend(ages)
        node_weight.extend(weights)
        node_record.extend([i] * len(ages))
    node_age, node_weight = np.asarray(node_age), np.asarray(node_weight)
    node_record = np.asarray(node_record, dtype=np.int64)
    lon = table.present_lon.to_numpy(float)
    lat = table.present_lat.to_numpy(float)
    node_lon, node_lat = lon[node_record].copy(), lat[node_record].copy()
    position_ok = np.ones(len(node_age), dtype=bool)
    if d.coordinate_mode == "paleo":
        if not d.rotations or not d.static_polygons:
            raise ValueError("Paleo coordinates require data.rotations and data.static_polygons")
        engine = KinematicsEngine(d.rotations, d.static_polygons)
        node_lon, node_lat, position_ok = engine.positions_at(node_lon, node_lat, node_age)
        if not position_ok.all() and d.position_failure == "error":
            raise ValueError(f"Unable to reconstruct {int((~position_ok).sum())} age nodes")
    # Drop an entire record if ANY age node is unlocatable. Do not silently
    # renormalize age weights and change that record's observation model.
    failed = np.unique(node_record[~position_ok])
    keep_nodes = ~np.isin(node_record, failed)
    node_record = node_record[keep_nodes]
    node_age, node_weight = node_age[keep_nodes], node_weight[keep_nodes]
    node_lon, node_lat = node_lon[keep_nodes], node_lat[keep_nodes]
    counts = np.bincount(node_record, minlength=len(table))
    valid &= counts > 0
    # Source error classes use periodic modern-position queries on the native
    # source grid, independent of the paleo position used in the likelihood.
    nlats, nlons = native_grid()
    finite_coords = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
    site_h = np.full(len(table), np.nan)
    site_h[finite_coords] = sample_at_points(
        modern["H_km"], lat[finite_coords], lon[finite_coords], (nlats, nlons)
    )
    src = np.where(site_h >= 20, d.continental_sigma_km, d.thin_crust_sigma_km)
    sigma = np.sqrt(d.record_sigma_km**2 + src**2)
    if not np.any(valid & (split == 0)):
        raise ValueError("No eligible training records remain")
    hashes = {
        name: sha256_file(getattr(d, name))
        for name in ("observations", "modern", "rotations", "static_polygons", "holdout_csv")
        if getattr(d, name)
    }
    metadata = {
        "schema_version": 2,
        "geometry": geometry,
        "input_layout": "age,channel,latitude,longitude",
        "feature_ages_ma": config.anchor_ages.tolist(),
        "grid": {
            "nlat": g.nlat,
            "nlon": g.nlon,
            "kind": "legendre-gauss",
            "latitude_order": "north_to_south",
            "longitude_origin": 0.0,
        },
        "max_age_ma": config.max_age_ma,
        "anchor_step_myr": config.anchor_step_myr,
        "feature_names": FEATURE_NAMES,
        "feature_support": feature_support,
        "source_semantics": d.source_semantics,
        "coordinate_mode": d.coordinate_mode,
        "source_sha256": hashes,
        "preparation_config": config.to_dict(),
        "record_count": len(table),
        "eligible_records": int(valid.sum()),
        "age_node_count": len(node_age),
        "position_failed_records": len(failed),
        "negative_longitude_records": int((lon < 0).sum()),
        "splits": {
            name: int((valid & (split == value)).sum())
            for value, name in enumerate(("train", "validation", "test"))
        },
        "split_policy": "100 km geographic proxy groups; entire test groups removed; modern endpoints allowed (P-B)",
        "uncertainty": "Fixed declared error scales; not calibrated historical confidence intervals",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        inputs=features,
        feature_ages=config.anchor_ages,
        plate_ids=plate_ids,
        modern_H=modern["H_km"],
        modern_sigma=modern["sigma_km"],
        modern_area=modern["areas_km2"],
        modern_blocks=modern["block_ids"],
        y=table.thickness_km.to_numpy(np.float32),
        sigma=sigma.astype(np.float32),
        record_id=table.observation_id.to_numpy(str),
        present_lon=lon,
        present_lat=lat,
        split=split,
        groups=groups,
        valid=valid,
        age_mass=mass,
        counts=counts,
        node_record=node_record,
        node_age=node_age.astype(np.float32),
        node_weight=node_weight.astype(np.float32),
        node_lon=node_lon.astype(np.float32),
        node_lat=node_lat.astype(np.float32),
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    write_manifest(output.with_suffix(".json"), {**metadata, "dataset_sha256": sha256_file(output)})
    return output


def load_dataset(path, config=None):
    with np.load(path, allow_pickle=False) as bundle:
        data = {key: bundle[key] for key in bundle.files}
    meta = json.loads(str(data.pop("metadata")))
    if meta["schema_version"] != 2:
        raise ValueError("Legacy static dataset: prepare a new age-bound dataset (schema 2)")
    ages = np.asarray(meta["feature_ages_ma"], dtype=np.float32)
    if (
        ages.ndim != 1
        or len(ages) < 2
        or not np.isfinite(ages).all()
        or ages[0] != 0
        or np.any(ages > 60)
        or np.any(np.diff(ages) <= 0)
        or not np.array_equal(
            ages, np.arange(len(ages), dtype=np.float32) * meta["anchor_step_myr"]
        )
        or not np.isclose(ages[-1], meta["max_age_ma"])
    ):
        raise ValueError("Invalid geometry age ordering or 0–60 Ma domain")
    grid = meta["grid"]
    shape = (len(ages), 8, grid["nlat"], grid["nlon"])
    if (
        data["inputs"].shape != shape
        or data["plate_ids"].shape != (shape[0], shape[2], shape[3])
        or not np.array_equal(data["feature_ages"], ages)
        or not np.isfinite(data["inputs"]).all()
    ):
        raise ValueError("Invalid age-bound inputs, geometry or feature age ordering")
    if config is not None:
        if shape != (config.n_anchors, 8, config.grid.nlat, config.grid.nlon):
            raise ValueError("Prepared dataset grid does not match the configuration")
        if not np.array_equal(ages, config.anchor_ages):
            raise ValueError("Prepared geometry ages do not match the training anchors")
        if meta["geometry"]["mode"] != config.data.geometry_mode:
            raise ValueError("Prepared geometry_mode differs; prepare a new dataset")
        if meta["geometry"].get("source", "gpml") != config.data.geometry_source:
            raise ValueError("Prepared geometry_source differs; prepare a new dataset")
        if (
            meta["max_age_ma"] != config.max_age_ma
            or meta["anchor_step_myr"] != config.anchor_step_myr
        ):
            raise ValueError("Prepared dataset age domain does not match the configuration")
        if (
            meta["coordinate_mode"] != config.data.coordinate_mode
            or meta["source_semantics"] != config.data.source_semantics
        ):
            raise ValueError(
                "Prepared dataset coordinate mode/source semantics do not match the configuration"
            )
        frozen = meta["preparation_config"]["data"]
        for key in (
            "record_sigma_km",
            "continental_sigma_km",
            "thin_crust_sigma_km",
            "modern_sigma_km",
        ):
            if frozen[key] != getattr(config.data, key):
                raise ValueError(
                    f"Prepared dataset has different data.{key}; prepare a new dataset"
                )
    data["metadata"] = meta
    data["dataset_sha256"] = sha256_file(path)
    return data
