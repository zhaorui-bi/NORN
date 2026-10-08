"""Explicit, model-derived rigid-interior priors, never inferred thickness labels."""

import json
from pathlib import Path

import numpy as np

from ..data.kinematics_engine import KinematicsEngine
from ..data.preparation import load_dataset
from ..geometry.regrid import gauss_grid
from ..utils.hashing import sha256_file, write_manifest


def prepare_rigid_priors(config, dataset_path, output, max_regions=48, min_edge_distance_km=500):
    """Construct a documented source-free rigid-model sensitivity hypothesis.

    These constraints are priors, not verified process observations. Regions
    are chosen without ancient thickness labels. Trajectory and budget roles
    are disjoint per plate, so their common kinematic evidence is counted once.
    """
    d = config.data
    if not d.rotations or not d.static_polygons:
        raise ValueError("Rigid priors require the rotation and static polygon model")
    dataset = load_dataset(dataset_path, config)
    if not dataset["metadata"]["feature_support"]["static_plate_edges"]:
        raise ValueError("No static plate edge support in the prepared dataset")
    inputs = dataset["inputs"]
    glat, glon, _ = gauss_grid(config.grid.nlat, config.grid.nlon)
    lon, lat = np.meshgrid(glon, glat[::-1])
    valid = (inputs[4] > 0.5) & (inputs[5] * 3000 >= min_edge_distance_km) & (np.abs(lat) < 75)
    candidates = np.column_stack([lon[valid], lat[valid]])
    if not len(candidates):
        raise ValueError("No interior candidates meet the declared support policy")
    engine = KinematicsEngine(d.rotations, d.static_polygons)
    pids = engine.plate_ids(candidates[:, 0], candidates[:, 1])
    unique = np.unique(pids[pids >= 0])
    if len(unique) < 2:
        raise ValueError(
            "Need at least two supported plates to assign disjoint trajectory/budget roles"
        )
    # Deterministic balanced selection; geographic spacing reduces redundant
    # quadrature while plate-level evidence grouping fixes the total weight.
    selected = []
    for pid in unique:
        points = candidates[pids == pid]
        stride = max(1, int(np.ceil(len(points) / max(1, max_regions // len(unique)))))
        selected.extend(
            (int(pid), point) for point in points[::stride][: max(1, max_regions // len(unique))]
        )
    selected = selected[:max_regions]
    trajectory_plates = set(unique[::2].tolist())
    ages = np.linspace(config.max_age_ma, 0, 7, dtype=np.float32)
    trajectories = []
    budgets = []
    for pid, point in selected:
        lo, la = point
        if pid in trajectory_plates:
            outlo, outla, ok = engine.positions_at(
                np.full(len(ages), lo), np.full(len(ages), la), ages
            )
            if ok.all():
                trajectories.append((pid, outlo, outla))
        else:
            # Four equal-area quadrature points inside a 1-degree material
            # patch. Rigid rotations preserve each point's represented area.
            plo = lo + np.array([-0.25, 0.25, -0.25, 0.25])
            pla = la + np.array([-0.25, -0.25, 0.25, 0.25])
            if not np.all(engine.plate_ids(plo, pla) == pid):
                continue
            outlo, outla, ok = engine.positions_at(
                np.tile(plo, len(ages)), np.tile(pla, len(ages)), np.repeat(ages, 4)
            )
            if ok.all():
                area = (
                    6371**2
                    * np.radians(1)
                    * (np.sin(np.radians(la + 0.5)) - np.sin(np.radians(la - 0.5)))
                )
                budgets.append(
                    (pid, outlo.reshape(len(ages), 4), outla.reshape(len(ages), 4), area)
                )
    if not trajectories or not budgets:
        raise ValueError("Insufficient supported regions for both disjoint physics roles")
    nt, nb = len(trajectories), len(budgets)
    tlo = np.stack([p[1] for p in trajectories])
    tla = np.stack([p[2] for p in trajectories])
    blo = np.stack([p[1] for p in budgets])
    bla = np.stack([p[2] for p in budgets])
    areas = np.array([p[3] for p in budgets])
    meta = {
        "schema_version": 1,
        "time_coordinate": "forward_tau=max_age-age",
        "kind": "model_derived_rigid_source_free_prior",
        "support_policy": f"modern continental fraction >0.75; static model edge distance >= {min_edge_distance_km} km; |lat|<75; fully reconstructable",
        "assumptions": [
            "Source-free rigid interiors are a sensitivity hypothesis, not verified geological process data",
            "Static polygon interior distance is not a guarantee of no deformation over 60 Myr",
            "Material patches follow the same rigid rotation as their boundary: v-v_b=0 exactly within the selected model",
            "Do not interpret these priors as measured magmatic production, loss or plate-boundary fluxes",
        ],
        "evidence_ids": {
            "trajectory": [f"rigid_plate_{p[0]}" for p in trajectories],
            "budget": [f"rigid_plate_{p[0]}" for p in budgets],
        },
        "source_sha256": {
            "rotations": sha256_file(d.rotations),
            "static_polygons": sha256_file(d.static_polygons),
        },
        "selected_plates": {
            "trajectory": sorted({p[0] for p in trajectories}),
            "budget": sorted({p[0] for p in budgets}),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata=np.array(json.dumps(meta, ensure_ascii=False)),
        trajectory_active=np.ones(nt, bool),
        trajectory_source_known=np.ones(nt, bool),
        trajectory_lon=tlo,
        trajectory_lat=tla,
        trajectory_age=np.broadcast_to(ages, tlo.shape),
        trajectory_divergence_per_myr=np.zeros_like(tlo),
        trajectory_source_basis=np.zeros(tlo.shape + (0,)),
        trajectory_sigma_km=np.full(nt, 10.0),
        budget_active=np.ones(nb, bool),
        budget_source_known=np.ones(nb, bool),
        budget_flux_known=np.ones(nb, bool),
        budget_lon=blo,
        budget_lat=bla,
        budget_age=np.broadcast_to(ages, (nb, len(ages))),
        budget_area_km2=np.broadcast_to(areas[:, None, None] / 4, blo.shape),
        budget_source_basis=np.zeros(blo.shape + (0,)),
        budget_boundary_lon=blo,
        budget_boundary_lat=bla,
        budget_relative_normal_velocity_km_myr=np.zeros_like(blo),
        budget_boundary_length_km=np.ones_like(
            blo
        ),  # immaterial for exactly zero relative velocity
        budget_reference_volume_km3=35 * areas,
        budget_sigma_normalized=np.full(nb, 0.2),
    )
    write_manifest(
        output.with_suffix(".json"),
        {**meta, "trajectory_regions": nt, "budget_regions": nb, "sha256": sha256_file(output)},
    )
    return output
