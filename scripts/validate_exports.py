"""Audit exported maps against the saved operator and all output formats."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import xarray as xr

from norn_earth.geometry.sampling import make_query
from norn_earth.inference import NornPredictor
from norn_earth.utils.hashing import sha256_file, write_manifest


def validate(run, device, output):
    run = Path(run).resolve()
    inference = run / "inference"
    checkpoint = run / "best.pt"
    with np.load(inference / "thickness.npz", allow_pickle=False) as bundle:
        fields = bundle["thickness_km"]
        ages, lat, lon = (bundle[k] for k in ("age_ma", "latitude", "longitude"))
        metadata = json.loads(str(bundle["metadata"]))
        geometry_keys = (
            "continental_fraction",
            "plate_boundary_distance_km",
            "deformation_coverage",
            "plate_id",
        )
        geometry = {key: bundle[key] for key in geometry_keys}
    assert fields.shape == (len(ages), 180, 360)
    assert np.isfinite(fields).all() and (fields > 0).all()
    np.testing.assert_array_equal(lat, np.arange(-89.5, 90, 1))
    np.testing.assert_array_equal(lon, np.arange(0.5, 360, 1))
    assert metadata["checkpoint_sha256"] == sha256_file(checkpoint)
    assert metadata["interval_kind"] == "point_estimate"
    dat_error = 0.0
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    for age, field in zip(ages, fields):
        dat = np.loadtxt(inference / f"thickness_{age:g}Ma.dat")
        assert dat.shape == (64800, 3)
        np.testing.assert_array_equal(dat[:, 0], mesh_lon.ravel())
        np.testing.assert_array_equal(dat[:, 1], mesh_lat.ravel())
        error = float(np.max(np.abs(dat[:, 2] - field.ravel())))
        assert error <= 5.1e-7
        dat_error = max(dat_error, error)
    with xr.open_dataset(inference / "thickness.nc", engine="scipy") as nc:
        np.testing.assert_array_equal(nc["crustal_thickness"].values, fields)
        for key in geometry_keys:
            np.testing.assert_array_equal(nc[key].values, geometry[key])
        for key, values in zip(("age_ma", "latitude", "longitude"), (ages, lat, lon)):
            np.testing.assert_array_equal(nc[key].values, values)
    predictor = NornPredictor(checkpoint, device)
    np.testing.assert_array_equal(ages, predictor.config.anchor_ages)
    prediction = predictor.predict_grid(ages)
    repeated = prediction["thickness_km"]
    np.testing.assert_allclose(repeated, fields, rtol=2e-5, atol=2e-5)
    for key in geometry_keys:
        np.testing.assert_array_equal(prediction[key], geometry[key])
    assert predictor.inputs.shape[0] == len(ages)
    assert np.isfinite(geometry["continental_fraction"]).all()
    assert ((geometry["continental_fraction"] >= 0) & (geometry["continental_fraction"] <= 1)).all()
    query_lon = np.array([-120.0, 240.0, 0.5, 359.5])
    query_lat = np.array([20.0, 20.0, -89.5, 89.5])
    query_age = np.array([13.4, 13.4, 0, predictor.config.max_age_ma])
    points = predictor.predict_points(query_lon, query_lat, query_age)
    np.testing.assert_array_equal(points[:1], points[1:2])
    c = predictor.config
    direct = (
        make_query(
            query_lon,
            query_lat,
            query_age,
            c.grid.nlat,
            c.grid.nlon,
            c.anchor_step_myr,
            c.n_anchors,
            predictor.device,
        )
        .sample(predictor.anchor_fields())
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(direct, points)
    with tempfile.TemporaryDirectory(prefix="norn-checkpoint-") as directory:
        copied = Path(directory) / "standalone.pt"
        copied.write_bytes(checkpoint.read_bytes())
        cpu = NornPredictor(copied, "cpu")
        cpu_points = cpu.predict_points(query_lon, query_lat, query_age)
        # CUDA convolution defaults can use TF32. Audit strict float32 parity
        # separately instead of hiding backend rounding by loosening tolerance.
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            strict = NornPredictor(copied, device)
            strict_points = strict.predict_points(query_lon, query_lat, query_age)
            np.testing.assert_allclose(cpu_points, strict_points, rtol=2e-5, atol=2e-5)
        finally:
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    report = {
        "status": "passed",
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(predictor.device),
        "torch": str(torch.__version__),
        "n_ages": len(ages),
        "shape": list(fields.shape),
        "thickness_min_km": float(fields.min()),
        "thickness_max_km": float(fields.max()),
        "all_finite_positive": True,
        "dat_files_checked": len(ages),
        "dat_max_rounding_error_km": dat_error,
        "netcdf_npz_exact": True,
        "geometry_npz_netcdf_operator_exact": True,
        "geometry_mode": metadata["geometry_mode"],
        "age_bound_input_shape": list(predictor.inputs.shape),
        "changed_continental_pixels_first_last": int(
            (geometry["continental_fraction"][0] != geometry["continental_fraction"][-1]).sum()
        ),
        "changed_plate_id_pixels_first_last": int(
            (geometry["plate_id"][0] != geometry["plate_id"][-1]).sum()
        ),
        "operator_export_max_error_km": float(np.max(np.abs(repeated - fields))),
        "fractional_age_ma": 13.4,
        "periodic_longitude_exact": True,
        "training_inference_query_exact": True,
        "standalone_cpu_gpu_default_max_error_km": float(np.max(np.abs(cpu_points - points))),
        "standalone_cpu_gpu_strict_fp32_max_error_km": float(
            np.max(np.abs(cpu_points - strict_points))
        ),
        "default_cudnn_allow_tf32": cudnn_tf32,
        "strict_fp32_tolerance": {"rtol": 2e-5, "atol_km": 2e-5},
    }
    write_manifest(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate(args.run, args.device, args.output or args.run / "export_validation.json")
