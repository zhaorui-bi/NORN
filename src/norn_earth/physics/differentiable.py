"""Differentiable Methods constraints with explicit support and evidence IDs.

All time integrals advance toward the present: tau = max_age - age.
Unknown source/flux information disables the entire corresponding factor.
"""

import json

import numpy as np
import torch
from torch import nn

from ..geometry.sampling import make_query


class BoundedSources(nn.Module):
    """A fixed, small set of regional/time basis coefficients in km/Myr."""

    def __init__(self, lower, upper, mean, covariance, device):
        super().__init__()
        lower, upper, mean = [
            torch.as_tensor(x, dtype=torch.float32, device=device) for x in (lower, upper, mean)
        ]
        if lower.ndim != 1 or not (lower.shape == upper.shape == mean.shape) or len(lower) > 128:
            raise ValueError(
                "Sources require <=128 fixed-dimensional coefficients with matching bounds/means"
            )
        if not torch.isfinite(torch.cat([lower, upper, mean])).all() or not torch.all(
            upper > lower
        ):
            raise ValueError("Source bounds must be finite and strictly ordered")
        if not torch.all((mean > lower) & (mean < upper)):
            raise ValueError("Source prior means must be strictly inside their bounds")
        for name, value in (("lower", lower), ("upper", upper), ("mean", mean)):
            self.register_buffer(name, value)
        covariance = torch.as_tensor(covariance, dtype=torch.float32, device=device)
        if covariance.shape != (len(mean), len(mean)) or not torch.allclose(
            covariance, covariance.T
        ):
            raise ValueError("Source prior covariance must be symmetric with shape (C,C)")
        self.register_buffer("cholesky", torch.linalg.cholesky(covariance))
        self.raw = nn.Parameter(
            torch.atanh((2 * (mean - lower) / (upper - lower) - 1).clamp(-0.999, 0.999))
        )

    def forward(self):
        return self.lower + (self.upper - self.lower) * (torch.tanh(self.raw) + 1) / 2

    def prior(self):
        delta = (self() - self.mean).unsqueeze(-1)
        z = torch.linalg.solve_triangular(self.cholesky, delta, upper=False)
        return 0.5 * z.square().sum() + self.cholesky.diagonal().log().sum()


class PhysicsObjective(nn.Module):
    def __init__(self, path, config, device="cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.metadata = {"enabled": False, "active_factors": {}}
        self.sources = None
        self.factors = {}
        if path is None:
            return
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
        if "metadata" not in arrays:
            raise ValueError("Physics artifact requires JSON metadata")
        meta = json.loads(str(arrays.pop("metadata")))
        if (
            meta.get("schema_version") != 1
            or meta.get("time_coordinate") != "forward_tau=max_age-age"
        ):
            raise ValueError(
                "Physics artifact must declare schema_version=1 and forward_tau=max_age-age"
            )
        self.metadata = {**meta, "enabled": True, "active_factors": {}}
        c = len(arrays.get("source_lower", []))
        if c:
            if config.physics.source_prior_weight <= 0:
                raise ValueError("Learned source coefficients require source_prior_weight > 0")
            self.sources = BoundedSources(
                arrays["source_lower"],
                arrays["source_upper"],
                arrays["source_prior_mean"],
                arrays["source_prior_cov"],
                device,
            )
        self.coefficient_count = c
        roles = {}
        evidence = meta.get("evidence_ids", {})
        if c:
            for eid in evidence.get("source", []):
                roles[str(eid)] = "source"
            if not evidence.get("source"):
                raise ValueError("Source priors require evidence IDs")
        for kind in ("trajectory", "budget", "birth"):
            weight = getattr(config.physics, f"{kind}_weight")
            if weight <= 0:
                continue
            key = f"{kind}_active"
            if key not in arrays:
                raise ValueError(f"Missing {key} for an enabled physics loss")
            mask = np.asarray(arrays[key], bool)
            if mask.ndim != 1:
                raise ValueError(f"{key} must be one-dimensional")
            if kind in ("trajectory", "budget"):
                known = np.asarray(arrays[f"{kind}_source_known"], bool)
                if known.shape != mask.shape:
                    raise ValueError("Source coverage mask shape mismatch")
                mask &= known
            if kind == "budget":
                known = np.asarray(arrays["budget_flux_known"], bool)
                if known.shape != mask.shape:
                    raise ValueError("Flux coverage mask shape mismatch")
                mask &= known
            ids = evidence.get(kind, [])
            if len(ids) != len(mask):
                raise ValueError(f"Provide one evidence ID per {kind} factor")
            selected_ids = np.asarray(ids, dtype=str)[mask]
            for eid in np.unique(selected_ids):
                if eid in roles and roles[eid] != kind:
                    raise ValueError(f"Evidence {eid!r} is counted in both {roles[eid]} and {kind}")
                roles[eid] = kind
            selected = {
                k[len(kind) + 1 :]: np.asarray(v)[mask]
                for k, v in arrays.items()
                if k.startswith(kind + "_") and k not in (key,)
            }
            self.metadata["active_factors"][kind] = int(mask.sum())
            if not mask.any():
                raise ValueError(f"Enabled {kind} loss has no supported active factors")
            unique, inverse = np.unique(selected_ids, return_inverse=True)
            selected["group"] = self._tensor(inverse, torch.long)
            selected["n_groups"] = len(unique)
            self._prepare(kind, selected, c)
            self.factors[kind] = selected
        if not self.factors and not c:
            raise ValueError("Physics artifact contains no enabled constraints")
        pairs = np.asarray(arrays.get("source_smooth_pairs", np.empty((0, 2), int)), dtype=int)
        if pairs.shape != (len(pairs), 2) or (pairs.size and (pairs.min() < 0 or pairs.max() >= c)):
            raise ValueError("Invalid source_smooth_pairs")
        self.register_buffer("source_smooth_pairs", self._tensor(pairs, torch.long))

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _query(self, lon, lat, age):
        c = self.config
        return make_query(
            lon, lat, age, c.grid.nlat, c.grid.nlon, c.anchor_step_myr, c.n_anchors, self.device
        )

    def _prepare(self, kind, f, c):
        # Only supported active factors reach this function; NaNs in unknown
        # fluxes are not converted to zero or allowed to poison autograd.
        for name, value in f.items():
            if (
                isinstance(value, np.ndarray)
                and value.dtype.kind in "fiu"
                and not np.isfinite(value).all()
            ):
                raise ValueError(f"Nonfinite values in active {kind}_{name}")
        if kind == "trajectory":
            shape = f["age"].shape
            if (
                len(shape) != 2
                or shape[1] < 2
                or f["lon"].shape != shape
                or f["lat"].shape != shape
            ):
                raise ValueError("Trajectory lon/lat/age must have shape (R,T), T>=2")
            if f["divergence_per_myr"].shape != shape or f["source_basis"].shape != shape + (c,):
                raise ValueError("Trajectory divergence/source_basis shapes do not match")
            dt = -np.diff(f["age"], axis=1)
            if not (dt > 0).all():
                raise ValueError("Trajectory ages must strictly decrease in forward physical time")
            f["shape"] = shape
            f["query"] = self._query(f["lon"], f["lat"], f["age"])
            f["dt"] = self._tensor(dt)
            f["div"] = self._tensor(f["divergence_per_myr"])
            f["basis"] = self._tensor(f["source_basis"])
            f["sigma"] = self._tensor(f["sigma_km"])
        elif kind == "budget":
            ages = f["age"]
            shape = f["lon"].shape
            boundary = f["boundary_lon"].shape
            if len(shape) != 3 or ages.shape != shape[:2] or f["lat"].shape != shape:
                raise ValueError("Budget interiors require (B,T,P) coordinates and (B,T) ages")
            if (
                len(boundary) != 3
                or boundary[:2] != shape[:2]
                or f["boundary_lat"].shape != boundary
            ):
                raise ValueError("Budget boundaries require (B,T,E) coordinates")
            if (
                shape[1] < 2
                or f["area_km2"].shape != shape
                or f["source_basis"].shape != shape + (c,)
            ):
                raise ValueError("Budget area/source basis shapes do not match")
            if (
                f["relative_normal_velocity_km_myr"].shape != boundary
                or f["boundary_length_km"].shape != boundary
            ):
                raise ValueError("Boundary velocities and lengths must match boundary coordinates")
            if (f["area_km2"] <= 0).any() or (f["boundary_length_km"] < 0).any():
                raise ValueError("Interior areas must be positive and boundary lengths nonnegative")
            dt = -np.diff(ages, axis=1)
            if not (dt > 0).all():
                raise ValueError("Budget ages must strictly decrease in forward physical time")
            f["shape"], f["boundary_shape"] = shape, boundary
            f["query"] = self._query(f["lon"], f["lat"], np.broadcast_to(ages[..., None], shape))
            f["boundary_query"] = self._query(
                f["boundary_lon"], f["boundary_lat"], np.broadcast_to(ages[..., None], boundary)
            )
            for name in (
                "area_km2",
                "source_basis",
                "relative_normal_velocity_km_myr",
                "boundary_length_km",
            ):
                f[name] = self._tensor(f[name])
            f["dt"] = self._tensor(dt)
            f["reference"] = self._tensor(f["reference_volume_km3"])
            if not torch.all(f["reference"] > 0):
                raise ValueError("Budget normalization must use a positive fixed reference volume")
            f["sigma"] = self._tensor(f["sigma_normalized"])
        else:
            if (
                not (f["retained_production_km2_myr"] > 0).all()
                or not (f["full_spreading_km_myr"] > 0).all()
            ):
                raise ValueError("Birth production and full spreading rate must be positive")
            f["query"] = self._query(f["lon"], f["lat"], f["age"])
            f["target"] = self._tensor(
                f["retained_production_km2_myr"] / f["full_spreading_km_myr"]
            )
            f["sigma"] = self._tensor(f["sigma_km"])
        if f["sigma"].shape != (
            f["shape"][0] if kind != "birth" else len(f["age"]),
        ) or not torch.all(f["sigma"] > 0):
            raise ValueError("Physics error scales must be positive and parallel to factors")

    @staticmethod
    def _mean_groups(residual, f):
        # Duplicate quadrature/factors with the same evidence ID cannot create
        # additional independent evidence in the objective.
        return (
            torch.stack(
                [residual[f["group"] == i].square().mean() for i in range(f["n_groups"])]
            ).mean()
            * 0.5
        )

    def forward(self, fields):
        p = self.config.physics
        zero = fields.sum() * 0
        total = zero
        parts = {}
        coeff = self.sources() if self.sources is not None else fields.new_empty(0)
        for kind, f in self.factors.items():
            if kind == "trajectory":
                h = f["query"].sample(fields).reshape(f["shape"])
                q = f["basis"] @ coeff
                rate = q - h * f["div"]
                integral = (0.5 * (rate[:, 1:] + rate[:, :-1]) * f["dt"]).sum(-1)
                residual = (h[:, -1] - h[:, 0] - integral) / f["sigma"]
            elif kind == "budget":
                h = f["query"].sample(fields).reshape(f["shape"])
                volume = (h * f["area_km2"]).sum(-1)
                q = f["source_basis"] @ coeff
                source = (q * f["area_km2"]).sum(-1)
                hb = f["boundary_query"].sample(fields).reshape(f["boundary_shape"])
                flux = (hb * f["relative_normal_velocity_km_myr"] * f["boundary_length_km"]).sum(-1)
                net = flux - source
                integral = (0.5 * (net[:, 1:] + net[:, :-1]) * f["dt"]).sum(-1)
                residual = (volume[:, -1] - volume[:, 0] + integral) / f["reference"] / f["sigma"]
            else:
                residual = (f["query"].sample(fields) - f["target"]) / f["sigma"]
            loss = self._mean_groups(residual, f)
            parts[kind] = loss
            total = total + getattr(p, f"{kind}_weight") * loss
        if self.sources is not None:
            prior = self.sources.prior()
            parts["source_prior"] = prior
            total = total + p.source_prior_weight * prior
            if len(self.source_smooth_pairs):
                pair = self.source_smooth_pairs
                smooth = (coeff[pair[:, 1]] - coeff[pair[:, 0]]).square().mean()
                parts["source_smooth"] = smooth
                total = total + p.source_smooth_weight * smooth
        return total, parts
