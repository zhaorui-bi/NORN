"""Age-conditioned spherical neural operator on a north-to-south Gauss grid."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import GridConfig, ModelConfig


class SphericalConv(nn.Module):
    def __init__(self, width, nlat, nlon, lmax, mmax):
        super().__init__()
        try:
            from torch_harmonics import InverseRealSHT, RealSHT
        except ImportError as exc:
            raise ImportError(
                "Run uv sync --locked --extra cpu (or --extra cuda) to install the spherical operator"
            ) from exc
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
        # Frequency-dependent channelwise filtering and learned channel mixing.
        self.filter = nn.Parameter(torch.zeros(width, lmax, mmax, 2))
        with torch.no_grad():
            self.filter[..., 0].fill_(1.0)
        self.mix = nn.Conv2d(2 * width, 2 * width, 1, bias=False)

    def forward(self, x):
        c = self.sht(x.float())
        c = c * torch.view_as_complex(self.filter.contiguous())[None]
        z = self.mix(torch.cat([c.real, c.imag], dim=1))
        real, imag = z.chunk(2, dim=1)
        return self.isht(torch.complex(real, imag))


class PeriodicConv(nn.Module):
    """Local convolution wraps longitude and replicates the polar edge."""

    def __init__(self, cin, cout, kernel=3):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(cin, cout, kernel)

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        return self.conv(x)


class SFNOBlock(nn.Module):
    def __init__(self, grid, model):
        super().__init__()
        width = model.width
        self.norm = nn.GroupNorm(1, width)
        self.spectral = SphericalConv(width, grid.nlat, grid.nlon, model.lmax, model.mmax)
        self.local = PeriodicConv(width, width)
        self.film = nn.Linear(model.time_features, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, time):
        gamma, beta = self.film(time).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1 + gamma[..., None, None]) + beta[..., None, None]
        return x + F.silu(self.spectral(h) + self.local(h)) / math.sqrt(2.0)


class NornSFNO(nn.Module):
    """One positive thickness channel. Age is supplied in Ma, normalized once."""

    def __init__(self, grid=None, model=None, max_age_ma=60.0):
        super().__init__()
        self.grid_config = grid or GridConfig()
        self.model_config = model or ModelConfig()
        self.max_age_ma = float(max_age_ma)
        g, m = self.grid_config, self.model_config
        self.embed = nn.Sequential(
            PeriodicConv(m.in_channels, m.width), nn.SiLU(), PeriodicConv(m.width, m.width)
        )
        self.age_embed = nn.Sequential(
            nn.Linear(5, m.time_features), nn.SiLU(), nn.Linear(m.time_features, m.time_features)
        )
        self.blocks = nn.ModuleList([SFNOBlock(g, m) for _ in range(m.blocks)])
        self.head = nn.Sequential(
            PeriodicConv(m.width, max(2, m.width // 2)),
            nn.SiLU(),
            nn.Conv2d(max(2, m.width // 2), 1, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        initial = m.initial_thickness_km - m.epsilon_km
        nn.init.constant_(self.head[-1].bias, initial + math.log(-math.expm1(-initial)))

    def forward(self, x, age_scalar):
        if x.ndim != 4 or x.shape[1:] != (
            self.model_config.in_channels,
            self.grid_config.nlat,
            self.grid_config.nlon,
        ):
            raise ValueError("Input shape does not match the model configuration")
        age = age_scalar.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
        if age.shape[0] != x.shape[0]:
            raise ValueError("One age is required per input field")
        u = age / self.max_age_ma
        time = self.age_embed(
            torch.cat(
                [
                    u,
                    torch.sin(math.pi * u),
                    torch.cos(math.pi * u),
                    torch.sin(2 * math.pi * u),
                    torch.cos(2 * math.pi * u),
                ],
                dim=-1,
            )
        )
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, time)
        return F.softplus(self.head(h)) + self.model_config.epsilon_km
