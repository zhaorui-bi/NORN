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

    def __init__(self, cin, cout, kernel=3, groups=1):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(cin, cout, kernel, groups=groups)

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


class GatedSFNOBlock(nn.Module):
    """Deterministic pre-norm spectral/local mixing + SwiGLU + LayerScale.

    No dropout or batch statistics: cache/recompute must produce identical
    fields for the exact two-pass chain-rule gradient.
    """

    def __init__(self, grid, model):
        super().__init__()
        width = model.width
        hidden = width * model.ffn_multiplier
        self.norm = nn.GroupNorm(1, width)
        self.ffn_norm = nn.GroupNorm(1, width)
        self.spectral = SphericalConv(width, grid.nlat, grid.nlon, model.lmax, model.mmax)
        self.local = nn.Sequential(
            PeriodicConv(width, width, groups=width), nn.Conv2d(width, width, 1)
        )
        self.project = nn.Conv2d(width, width, 1)
        self.ffn_in = nn.Conv2d(width, 2 * hidden, 1)
        self.ffn_out = nn.Conv2d(hidden, width, 1)
        self.film = nn.Linear(model.time_features, 4 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.mixer_scale = nn.Parameter(torch.full((width, 1, 1), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((width, 1, 1), 0.1))

    def forward(self, x, time):
        gamma, beta, f_gamma, f_beta = self.film(time).chunk(4, dim=-1)
        h = self.norm(x) * (1 + gamma[..., None, None]) + beta[..., None, None]
        h = (self.spectral(h) + self.local(h)) / math.sqrt(2.0)
        x = x + self.mixer_scale * self.project(F.silu(h))
        h = self.ffn_norm(x) * (1 + f_gamma[..., None, None]) + f_beta[..., None, None]
        gate, value = self.ffn_in(h).chunk(2, dim=1)
        return x + self.ffn_scale * self.ffn_out(F.silu(gate) * value)


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
            nn.Linear(1 + 2 * m.time_frequency_bands, m.time_features),
            nn.SiLU(),
            nn.Linear(m.time_features, m.time_features),
        )
        block = SFNOBlock if m.backbone == "sfno" else GatedSFNOBlock
        self.blocks = nn.ModuleList([block(g, m) for _ in range(m.blocks)])
        self.head = nn.Sequential(
            PeriodicConv(m.width, max(2, m.width // 2)),
            nn.SiLU(),
            nn.Conv2d(max(2, m.width // 2), 1, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        initial = m.initial_thickness_km - m.epsilon_km
        bias = 0.0 if m.head_mode == "input_residual" else initial + math.log(-math.expm1(-initial))
        nn.init.constant_(self.head[-1].bias, bias)

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
        if not torch.isfinite(age).all() or torch.any((age < 0) | (age > self.max_age_ma)):
            raise ValueError(f"Ages must be inside [0, {self.max_age_ma}] Ma")
        u = age / self.max_age_ma
        frequencies = torch.arange(
            1, self.model_config.time_frequency_bands + 1, device=u.device, dtype=u.dtype
        )
        angles = math.pi * u * frequencies
        features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(1)
        time = self.age_embed(torch.cat([u, features], dim=-1))
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, time)
        raw = self.head(h)
        if self.model_config.head_mode == "input_residual":
            # Residual regression from an available input feature, not a
            # conservation/trajectory constraint or a paleo thickness label.
            reference = torch.where(
                x[:, 3:4] > 0, x[:, 3:4] * 40, self.model_config.initial_thickness_km
            )
            z = (reference - self.model_config.epsilon_km).clamp_min(0.01)
            raw = raw + z + torch.log(-torch.expm1(-z))
        return F.softplus(raw) + self.model_config.epsilon_km
