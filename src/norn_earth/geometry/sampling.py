"""Differentiable periodic space/time queries with explicit grid ordering."""

from dataclasses import dataclass

import numpy as np
import torch

from .regrid import gauss_grid


@dataclass
class FieldQuery:
    a0: torch.Tensor
    a1: torch.Tensor
    fraction: torch.Tensor
    i: torch.Tensor
    j: torch.Tensor
    weight: torch.Tensor

    def sample(self, fields):
        if fields.ndim == 4:
            fields = fields[:, 0]
        lower = (fields[self.a0[:, None], self.i, self.j] * self.weight).sum(-1)
        upper = (fields[self.a1[:, None], self.i, self.j] * self.weight).sum(-1)
        return torch.lerp(lower, upper, self.fraction)


def make_query(lon, lat, age, nlat, nlon, anchor_step, n_anchors, device="cpu"):
    """Query SHT fields (north->south, longitude starts at 0). No extrapolation."""
    lon, lat, age = np.broadcast_arrays(
        np.asarray(lon, float), np.asarray(lat, float), np.asarray(age, float)
    )
    lon, lat, age = lon.ravel(), lat.ravel(), age.ravel()
    if not np.all(np.isfinite(lon) & np.isfinite(lat) & np.isfinite(age)):
        raise ValueError("Query coordinates and ages must be finite")
    if np.any(np.abs(lat) > 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    max_age = anchor_step * (n_anchors - 1)
    if np.any((age < 0) | (age > max_age)):
        raise ValueError(f"Query ages must be inside [0, {max_age}] Ma; extrapolation is forbidden")
    glat, _, _ = gauss_grid(nlat, nlon)
    row = nlat - 1 - np.interp(lat, glat, np.arange(nlat))
    i0 = np.clip(np.floor(row).astype(int), 0, nlat - 2)
    fi = row - i0
    col = np.mod(lon, 360.0) / (360.0 / nlon)
    j0 = np.floor(col).astype(int) % nlon
    fj = col - np.floor(col)
    i = np.stack([i0, i0, i0 + 1, i0 + 1], axis=1)
    j = np.stack([j0, (j0 + 1) % nlon, j0, (j0 + 1) % nlon], axis=1)
    weight = np.stack([(1 - fi) * (1 - fj), (1 - fi) * fj, fi * (1 - fj), fi * fj], axis=1)
    time = age / anchor_step
    a0 = np.clip(np.floor(time).astype(int), 0, n_anchors - 2)
    af = time - a0
    return FieldQuery(
        torch.as_tensor(a0, dtype=torch.long, device=device),
        torch.as_tensor(a0 + 1, dtype=torch.long, device=device),
        torch.as_tensor(af, dtype=torch.float32, device=device),
        torch.as_tensor(i, dtype=torch.long, device=device),
        torch.as_tensor(j, dtype=torch.long, device=device),
        torch.as_tensor(weight, dtype=torch.float32, device=device),
    )
