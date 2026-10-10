"""Small manufactured reconstruction with known source and full physics inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..geometry.regrid import native_grid


def truth(lon, lat, age):
    return (
        7 + 0.8 * np.cos(np.radians(lat)) * np.cos(np.radians(lon)) + 0.02 * (60 - np.asarray(age))
    )


def create_demo(
    output, device="auto", nlat=32, nlon=64, width=8, blocks=2, steps=100, anchor_step=5
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    lats, lons = native_grid()
    lon, lat = np.meshgrid(lons, lats)
    h = truth(lon, lat, 0)
    np.savetxt(
        output / "modern.dat", np.column_stack([lon.ravel(), lat.ravel(), -h.ravel()]), fmt="%.7f"
    )
    n = 240
    lon = rng.uniform(-180, 180, n)
    lat = rng.uniform(-65, 65, n)
    ages = rng.uniform(2, 58, n)
    widths = rng.choice([0.0, 2.0, 8.0], n)
    lower = np.maximum(0, ages - widths / 2)
    upper = np.minimum(60, ages + widths / 2)
    true_age = rng.uniform(lower, upper)
    observations = pd.DataFrame(
        {
            "observation_id": [f"demo{i:04d}" for i in range(n)],
            "present_lon": lon,
            "present_lat": lat,
            "age_lower_ma": lower,
            "age_upper_ma": upper,
            "thickness_km": truth(lon, lat, true_age) + rng.normal(0, 0.15, n),
        }
    )
    observations.to_csv(output / "observations.csv", index=False)
    a = np.array([60.0, 45.0, 30.0, 15.0, 0.0])
    tlo = np.broadcast_to(np.array([10.0, 80.0, 190.0, 280.0])[:, None], (4, len(a)))
    tla = np.broadcast_to(np.array([0.0, 20.0, -20.0, 40.0])[:, None], tlo.shape)
    blo = np.broadcast_to(
        np.array([[0.0, 1.0, 0.0, 1.0], [180.0, 181.0, 180.0, 181.0]])[:, None, :], (2, len(a), 4)
    )
    bla = np.broadcast_to(
        np.array([[14.0, 14.0, 15.0, 15.0], [-16.0, -16.0, -15.0, -15.0]])[:, None, :], blo.shape
    )
    birthlon = np.array([-120.0, -60.0, 60.0, 120.0])
    birthlat = np.array([-30.0, 10.0, 30.0, 50.0])
    birthage = np.array([60.0, 50.0, 40.0, 30.0])
    meta = {
        "schema_version": 1,
        "time_coordinate": "forward_tau=max_age-age",
        "kind": "manufactured_solution",
        "description": "Static material, q=0.02 km/Myr, nonuniform initial thickness; all source and flux terms known",
        "evidence_ids": {
            "trajectory": [f"traj{i}" for i in range(4)],
            "budget": ["budget0", "budget1"],
            "birth": [f"birth{i}" for i in range(4)],
            "source": ["source_prior"],
        },
    }
    np.savez_compressed(
        output / "physics.npz",
        metadata=np.array(json.dumps(meta)),
        source_lower=np.array([-0.1]),
        source_upper=np.array([0.1]),
        source_prior_mean=np.array([0.02]),
        source_prior_cov=np.array([[0.005**2]]),
        trajectory_active=np.ones(4, bool),
        trajectory_source_known=np.ones(4, bool),
        trajectory_lon=tlo,
        trajectory_lat=tla,
        trajectory_age=np.broadcast_to(a, tlo.shape),
        trajectory_divergence_per_myr=np.zeros(tlo.shape),
        trajectory_source_basis=np.ones(tlo.shape + (1,)),
        trajectory_sigma_km=np.full(4, 0.5),
        budget_active=np.ones(2, bool),
        budget_source_known=np.ones(2, bool),
        budget_flux_known=np.ones(2, bool),
        budget_lon=blo,
        budget_lat=bla,
        budget_age=np.broadcast_to(a, (2, len(a))),
        budget_area_km2=np.full(blo.shape, 1000.0),
        budget_source_basis=np.ones(blo.shape + (1,)),
        budget_boundary_lon=blo,
        budget_boundary_lat=bla,
        budget_relative_normal_velocity_km_myr=np.zeros(blo.shape),
        budget_boundary_length_km=np.full(blo.shape, 100.0),
        budget_reference_volume_km3=np.full(2, 35 * 4000),
        budget_sigma_normalized=np.full(2, 0.03),
        birth_active=np.ones(4, bool),
        birth_lon=birthlon,
        birth_lat=birthlat,
        birth_age=birthage,
        birth_retained_production_km2_myr=truth(birthlon, birthlat, birthage) * 80,
        birth_full_spreading_km_myr=np.full(4, 80.0),
        birth_sigma_km=np.full(4, 0.5),
    )
    config = Config()
    config.grid.nlat = nlat
    config.grid.nlon = nlon
    config.model.width = width
    config.model.blocks = blocks
    config.model.lmax = min(16, nlat)
    config.model.mmax = min(16, nlat, nlon // 2 + 1)
    config.model.initial_thickness_km = 10
    config.anchor_step_myr = anchor_step
    config.data.observations = "observations.csv"
    config.data.modern = "modern.dat"
    config.data.coordinate_mode = "present"
    config.data.source_semantics = "synthetic"
    config.data.record_sigma_km = 0.2
    config.data.continental_sigma_km = 0.2
    config.data.thin_crust_sigma_km = 0.2
    config.data.modern_sigma_km = 0.3
    config.physics.constraints = "physics.npz"
    config.physics.trajectory_weight = 0.1
    config.physics.budget_weight = 0.1
    config.physics.birth_weight = 0.1
    config.physics.source_prior_weight = 0.01
    # This manufactured fixture explicitly exercises the legacy extension;
    # ordinary Config() and the production templates default to pure ML.
    config.training.objective = "reconstruction"
    config.training.selection_metric = "joint_nll"
    config.training.steps = steps
    config.training.device = device
    config.training.learning_rate = 0.003
    config.training.warmup_steps = 5
    config.training.anchor_batch_size = 4
    config.training.log_every = 10
    config.output_dir = "run"
    path = output / "config.json"
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return path
