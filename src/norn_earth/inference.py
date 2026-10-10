"""Standalone inference from one checkpoint, with the training query operator."""

import json
from pathlib import Path

import numpy as np
import torch

from .geometry.regrid import native_grid
from .geometry.sampling import make_query
from .training.trainer import anchor_fields, load_checkpoint, resolve_device
from .utils.hashing import sha256_file, write_manifest


class NornPredictor:
    def __init__(self, checkpoint, device="auto"):
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint)
        self.checkpoint, self.config, self.model, self.inputs = load_checkpoint(
            checkpoint, self.device
        )
        torch.set_num_threads(self.config.training.num_threads)
        self._fields = None

    @torch.no_grad()
    def anchor_fields(self):
        if self._fields is None:
            ages = torch.as_tensor(self.config.anchor_ages, device=self.device)
            self._fields = anchor_fields(
                self.model, self.inputs, ages, self.config.training.anchor_batch_size
            )
            if not torch.isfinite(self._fields).all() or not torch.all(self._fields > 0):
                raise FloatingPointError("Checkpoint produced nonfinite or nonpositive thickness")
        return self._fields

    @torch.no_grad()
    def predict_points(self, lon, lat, age):
        """Coordinates are at the queried age, in the model reconstruction frame.

        The caller must reconstruct present-day sample coordinates when asking
        for a paleo observation. Negative and 0..360 longitudes are equivalent.
        """
        shape = np.broadcast_shapes(np.shape(lon), np.shape(lat), np.shape(age))
        c = self.config
        query = make_query(
            lon, lat, age, c.grid.nlat, c.grid.nlon, c.anchor_step_myr, c.n_anchors, self.device
        )
        return query.sample(self.anchor_fields()).cpu().numpy().reshape(shape)

    @torch.no_grad()
    def predict_geometry(self, ages):
        """Frozen age-bound geometry; no external plate files needed at inference.

        Continuous channels interpolate like thickness. Categorical plate IDs
        use the nearest anchor and nearest spatial grid node, never interpolation.
        Fractional geometry is a display approximation, not topology resolution.
        """
        ages = np.atleast_1d(np.asarray(ages, dtype=float))
        if ages.ndim != 1 or not len(ages):
            raise ValueError("Provide a nonempty one-dimensional age list")
        c = self.config
        lats, lons = native_grid()
        lon, lat = np.meshgrid(lons, lats)
        channels = {
            "continental_fraction": (self.inputs[:, 4] + 1) / 2,
            "plate_boundary_distance_km": self.inputs[:, 5] * 3000,
            "deformation_coverage": self.inputs[:, 6],
        }
        result = {key: [] for key in channels}
        result["plate_id"] = []
        pids = self.checkpoint["geometry"]["plate_ids"].to(self.device)
        for age in ages:
            q = make_query(
                lon, lat, age, c.grid.nlat, c.grid.nlon, c.anchor_step_myr, c.n_anchors, self.device
            )
            for key, field in channels.items():
                upper = 3000.0 if key == "plate_boundary_distance_km" else 1.0
                result[key].append(q.sample(field).clamp(0, upper).cpu().numpy().reshape(lon.shape))
            corner = q.weight.argmax(dim=1)
            rows = torch.arange(len(corner), device=self.device)
            anchor = q.a0 + (q.fraction >= 0.5).long()
            result["plate_id"].append(
                pids[anchor, q.i[rows, corner], q.j[rows, corner]].cpu().numpy().reshape(lon.shape)
            )
        return {key: np.stack(values) for key, values in result.items()}

    def predict_grid(self, ages):
        ages = np.atleast_1d(np.asarray(ages, dtype=float))
        geometry = self.predict_geometry(ages)
        lats, lons = native_grid()
        lon, lat = np.meshgrid(lons, lats)
        fields = np.stack([self.predict_points(lon, lat, np.full_like(lon, age)) for age in ages])
        return {
            "thickness_km": fields,
            "age_ma": ages,
            "latitude": lats,
            "longitude": lons,
            **geometry,
        }


def export_prediction(predictor, ages, output, formats=("npz",)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = predictor.predict_grid(ages)
    metadata = {
        "units": "km",
        "positive": "downward thickness magnitude",
        "latitude_order": "south_to_north",
        "coordinate_frame": "paleo positions in the checkpoint plate reconstruction reference frame",
        "geometry_mode": predictor.config.data.geometry_mode,
        "geometry": predictor.checkpoint["provenance"]["dataset_metadata"]["geometry"],
        "geometry_interpolation": "continuous channels: linear; plate_id: nearest anchor/node",
        "interval_kind": "point_estimate",
        "calibrated_uncertainty": False,
        "checkpoint_sha256": sha256_file(predictor.checkpoint_path),
        "checkpoint_steps": predictor.checkpoint["completed_steps"],
        "source_semantics": predictor.checkpoint["provenance"]["dataset_metadata"][
            "source_semantics"
        ],
        "objective": predictor.config.training.objective,
        "experiment_tag": predictor.config.experiment_tag,
        "geometry_source": predictor.config.data.geometry_source,
        "ages_ma": result["age_ma"].tolist(),
        "shape": list(result["thickness_km"].shape),
    }
    if "physics" in predictor.checkpoint["provenance"]:
        metadata["physics"] = predictor.checkpoint["provenance"]["physics"]
    if "npz" in formats:
        np.savez_compressed(
            output / "thickness.npz",
            **result,
            metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
        )
    if "dat" in formats:
        lon, lat = np.meshgrid(result["longitude"], result["latitude"])
        for age, field in zip(result["age_ma"], result["thickness_km"]):
            np.savetxt(
                output / f"thickness_{age:g}Ma.dat",
                np.column_stack([lon.ravel(), lat.ravel(), field.ravel()]),
                fmt="%.6f",
                header="lon_deg lat_deg thickness_km (positive)",
            )
    if "netcdf" in formats:
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError("NetCDF export requires norn-earth[export]") from exc
        array = xr.DataArray(
            result["thickness_km"],
            dims=("age_ma", "latitude", "longitude"),
            coords={k: result[k] for k in ("age_ma", "latitude", "longitude")},
            name="crustal_thickness",
            attrs={"units": "km", "long_name": "Reconstructed crustal thickness"},
        )
        dataset = array.to_dataset()
        for key in (
            "continental_fraction",
            "plate_boundary_distance_km",
            "deformation_coverage",
            "plate_id",
        ):
            dataset[key] = (("age_ma", "latitude", "longitude"), result[key])
        dataset["plate_boundary_distance_km"].attrs["units"] = "km"
        dataset["plate_id"].attrs["unassigned_value"] = -1
        dataset.attrs = {"metadata_json": json.dumps(metadata, ensure_ascii=False)}
        dataset.to_netcdf(output / "thickness.nc", engine="scipy")
    write_manifest(output / "inference_manifest.json", metadata)
    return result, metadata
