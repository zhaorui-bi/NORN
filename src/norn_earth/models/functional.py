"""Spherical/context functional Tucker decoder with a time-indexed latent core.

The two modes are the embedded sphere and age-bound geological context,
not independent longitude/latitude axes. Uses the same positive thickness
transform and field-query interpolation as the SFNO controls.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import GridConfig, ModelConfig
from .temporal import gp_cholesky, whiten_cores


class ContinuousFactor(nn.Module):
    def __init__(self, channels, width, rank):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.SiLU(), nn.Conv2d(width, rank - 1, 1), nn.Tanh()
        )

    def forward(self, x):
        return torch.cat([torch.ones_like(x[:, :1]), self.net(x)], dim=1)


class NornFunctionalTucker(nn.Module):
    def __init__(self, grid=None, model=None, max_age_ma=60.0, anchor_step_myr=1.0):
        super().__init__()
        self.grid_config = grid or GridConfig()
        self.model_config = model or ModelConfig(backbone="functional_tucker")
        self.max_age_ma = float(max_age_ma)
        self.anchor_step_myr = float(anchor_step_myr)
        m = self.model_config
        n = round(max_age_ma / anchor_step_myr) + 1
        self.spatial = ContinuousFactor(3, m.width, m.spatial_rank)
        self.context = ContinuousFactor(4, m.width, m.context_rank)
        self.register_buffer(
            "prior_cholesky",
            gp_cholesky(
                torch.arange(n) * anchor_step_myr, m.latent_lengthscale_ma, m.latent_amplitude
            ),
            persistent=False,
        )
        initial = self.prior_cholesky @ torch.randn(n, m.spatial_rank * m.context_rank) * 0.01
        self.cores = nn.Parameter(initial.reshape(n, m.spatial_rank, m.context_rank))

    def encode_factors(self, x):
        if x.ndim != 4 or x.shape[1:] != (8, self.grid_config.nlat, self.grid_config.nlon):
            raise ValueError("Input shape does not match the model configuration")
        m = self.model_config
        if m.head_mode == "input_residual":
            reference = torch.where(x[:, 3:4] > 0, x[:, 3:4] * 40, m.initial_thickness_km)
        else:
            reference = torch.full_like(x[:, :1], m.initial_thickness_km)
        z = (reference - m.epsilon_km).clamp_min(0.01)
        base = z + torch.log(-torch.expm1(-z))
        return self.spatial(x[:, :3]), self.context(x[:, 3:7]), base

    def decode_factors(self, factors, cores):
        spatial, context, base = factors
        if cores.shape != (
            spatial.shape[0],
            self.model_config.spatial_rank,
            self.model_config.context_rank,
        ):
            raise ValueError("One matching Tucker core is required per input age")
        raw = torch.einsum("brhw,brs,bshw->bhw", spatial, cores, context)
        raw = (
            raw
            * self.model_config.latent_scale_km
            / math.sqrt(self.model_config.spatial_rank * self.model_config.context_rank)
        )
        return F.softplus(base + raw[:, None]) + self.model_config.epsilon_km

    def cores_at(self, ages):
        ages = ages.reshape(-1).to(self.cores)
        if not torch.isfinite(ages).all() or torch.any((ages < 0) | (ages > self.max_age_ma)):
            raise ValueError(f"Ages must be inside [0, {self.max_age_ma}] Ma")
        u = ages / self.anchor_step_myr
        lo = u.floor().long().clamp(0, len(self.cores) - 2)
        f = (u - lo)[:, None, None]
        return torch.lerp(self.cores[lo], self.cores[lo + 1], f)

    def forward(self, x, age_scalar):
        if age_scalar.numel() != len(x):
            raise ValueError("One age is required per input field")
        return self.decode_factors(self.encode_factors(x), self.cores_at(age_scalar))

    def latent_prior_nll(self):
        return 0.5 * whiten_cores(self.cores, self.prior_cholesky).square().mean()
