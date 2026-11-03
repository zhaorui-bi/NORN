"""Experimental GP-whitened core diffusion; an adaptation, not SDIFT reproduction.

The learned prior requires independent histories in a single frozen latent
gauge. Conditional guidance is approximate DPS through the actual nonlinear,
age-marginal observation operator; it does not use SDIFT's linear-Gaussian
closed-form MPDPS or certify posterior calibration.
"""

import copy
import math
from pathlib import Path

import torch
from torch import nn

from .. import __version__
from ..training.trainer import resolve_device
from ..utils.hashing import sha256_file, write_manifest
from .ensemble import decoder_fingerprint, load_latent_ensemble
from .temporal import gp_cholesky, unwhiten_cores, whiten_cores


class CoreDenoiser(nn.Module):
    def __init__(self, channels, width=64, layers=2):
        super().__init__()
        if channels < 1 or width < 8 or width % 4 or layers < 1:
            raise ValueError("Denoiser requires channels>=1, width>=8 divisible by 4, layers>=1")
        self.project = nn.Linear(channels, width)
        self.condition = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        block = nn.TransformerEncoderLayer(
            width,
            4,
            dim_feedforward=width * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, sigma, ages):
        # Standard EDM preconditioning in GP-whitened, standardized coordinates.
        sigma = sigma.reshape(-1, 1, 1)
        age = ages[None, :, None].expand(len(x), -1, -1)
        noise = sigma.clamp_min(1e-8).log().expand(-1, len(ages), -1) / 4
        conditions = torch.cat([age, (math.pi * age).sin(), (math.pi * age).cos(), noise], -1)
        data_sigma = 0.5
        denom = sigma.square() + data_sigma**2
        h = self.project(x / denom.sqrt()) + self.condition(conditions)
        out = self.head(self.temporal(h))
        return data_sigma**2 / denom * x + sigma * data_sigma / denom.sqrt() * out


def train_latent_prior(
    ensemble,
    output,
    *,
    steps=500,
    width=64,
    layers=2,
    batch_size=16,
    learning_rate=0.0003,
    lengthscale_ma=10.0,
    amplitude=1.0,
    seed=42,
    device="auto",
):
    if steps < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Prior training steps, batch size and learning rate must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device = resolve_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    data = load_latent_ensemble(ensemble)
    cores = torch.as_tensor(data["cores"], device=device, dtype=torch.float32)
    ages = torch.as_tensor(data["age_ma"], device=device, dtype=torch.float32)
    chol = gp_cholesky(ages, lengthscale_ma, amplitude)
    white = torch.stack([whiten_cores(x, chol) for x in cores]).flatten(2)
    if not torch.isfinite(white).all():
        raise FloatingPointError("Nonfinite whitened ensemble cores")
    train_mask = torch.as_tensor(data["split"] == "train", device=device)
    val_mask = torch.as_tensor(data["split"] == "validation", device=device)
    # No statistics from validation or test histories enter normalization.
    mean = white[train_mask].mean((0, 1))
    scale = white[train_mask].std((0, 1), unbiased=False).clamp_min(1e-4)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise FloatingPointError("Nonfinite training-history normalization")
    train = (white[train_mask] - mean) / scale
    validation = (white[val_mask] - mean) / scale
    age_condition = (ages - ages[0]) / (ages[-1] - ages[0])
    network = CoreDenoiser(train.shape[-1], width, layers).to(device)
    ema = copy.deepcopy(network).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    val_rng = torch.Generator(device=device).manual_seed(seed + 2)
    val_noise = torch.randn(validation.shape, generator=val_rng, device=device)
    val_sigma = torch.full((len(validation),), 0.5, device=device)

    @torch.no_grad()
    def validation_loss():
        pred = ema(validation + 0.5 * val_noise, val_sigma, age_condition)
        value = (pred - validation).square().mean()
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite held-out denoising loss")
        return float(value)

    initial = validation_loss()
    best, best_state, best_step = initial, copy.deepcopy(ema.state_dict()), 0
    last_loss = None
    for step in range(steps):
        network.train()
        indices = torch.randint(
            len(train), (min(batch_size, len(train)),), device=device, generator=generator
        )
        batch = train[indices]
        sigma = (torch.randn(len(batch), device=device, generator=generator) * 1.2 - 1.2).exp()
        noise = torch.randn(batch.shape, generator=generator, device=device)
        pred = network(batch + noise * sigma[:, None, None], sigma, age_condition)
        weight = (sigma.square() + 0.25) / (sigma * 0.5).square()
        loss = ((pred - batch).square() * weight[:, None, None]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite diffusion loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), network.parameters()):
                target.lerp_(source, 0.05)
        last_loss = float(loss.detach())
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step == steps - 1:
            value = validation_loss()
            if value < best:
                best, best_state, best_step = value, copy.deepcopy(ema.state_dict()), step + 1
            print(
                f"prior step {step + 1}/{steps}: train={last_loss:.5f} validation={value:.5f}",
                flush=True,
            )
    metadata = {
        "prior_schema": 1,
        "method": "gp_whitened_core_edm",
        "decoder_sha256": data["metadata"]["decoder_sha256"],
        "source_semantics": data["metadata"]["source_semantics"],
        "ensemble_sha256": sha256_file(ensemble),
        "train_history_ids": data["history_id"][data["split"] == "train"].tolist(),
        "validation_history_ids": data["history_id"][data["split"] == "validation"].tolist(),
        "train_group_ids": sorted(set(data["group_id"][data["split"] == "train"].tolist())),
        "validation_group_ids": sorted(
            set(data["group_id"][data["split"] == "validation"].tolist())
        ),
        "steps": steps,
        "seed": seed,
        "best_step": best_step,
        "initial_validation_denoising_mse": initial,
        "best_validation_denoising_mse": best,
        "last_training_loss": last_loss,
        "calibrated_uncertainty": False,
        "lengthscale_ma": lengthscale_ma,
        "amplitude": amplitude,
        "normalization": "training histories only",
        "software_version": __version__,
        "runtime": {"device": str(device), "torch": str(torch.__version__)},
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": 0.0001},
        "batch_size": batch_size,
        "network": {"channels": train.shape[-1], "width": width, "layers": layers},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "state_dict": best_state,
            "mean": mean.cpu(),
            "scale": scale.cpu(),
            "ages_ma": ages.cpu(),
            "core_shape": list(cores.shape[1:]),
        },
        output,
    )
    write_manifest(output.with_suffix(".json"), metadata)
    return metadata


def sample_diffusion_prior(
    path, model, inputs, config, potential, draws=32, chains=2, steps=32, guidance=0.1, seed=42
):
    if chains < 1 or draws < 2 or steps < 2 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Require draws>=2, diffusion_steps>=2 and finite nonnegative guidance")
    device = inputs.device
    prior = torch.load(path, map_location=device, weights_only=True)
    if prior.get("prior_schema") != 1:
        raise ValueError("Unsupported learned prior schema")
    if prior["decoder_sha256"] != decoder_fingerprint(model, inputs, config):
        raise ValueError(
            "Prior has a different decoder/geometry/latent gauge; re-encode the ensemble"
        )
    ages = torch.as_tensor(config.anchor_ages, device=device)
    if prior["core_shape"] != list(model.cores.shape) or not torch.equal(prior["ages_ma"], ages):
        raise ValueError("Learned prior ages/core shape differ from the reconstruction")
    if (
        prior["lengthscale_ma"] != config.model.latent_lengthscale_ma
        or prior["amplitude"] != config.model.latent_amplitude
    ):
        raise ValueError("Learned prior GP differs from the reconstruction GP configuration")
    network = CoreDenoiser(**prior["network"]).to(device)
    network.load_state_dict(prior["state_dict"], strict=True)
    network.eval().requires_grad_(False)
    chol = model.prior_cholesky
    mean, scale = prior["mean"], prior["scale"]
    expected = (prior["network"]["channels"],)
    if (
        mean.shape != expected
        or scale.shape != expected
        or not torch.isfinite(mean).all()
        or not torch.isfinite(scale).all()
        or torch.any(scale <= 0)
    ):
        raise ValueError("Invalid learned-prior normalization")
    condition = ages / config.max_age_ma
    schedule = torch.cat(
        [torch.logspace(1, -2, steps, device=device), torch.zeros(1, device=device)]
    )
    rng = torch.Generator(device=device).manual_seed(seed)
    all_samples, energies = [], []
    for sample in range(chains * draws):
        x = (
            torch.randn((1, len(ages), prior["network"]["channels"]), generator=rng, device=device)
            * schedule[0]
        )
        for sigma, next_sigma in zip(schedule[:-1], schedule[1:]):
            with torch.enable_grad():
                x = x.detach().requires_grad_(guidance > 0)
                clean = network(x, sigma[None], condition)
                core = unwhiten_cores((clean[0] * scale + mean).reshape_as(model.cores), chol)
                energy = potential(core)
                if not torch.isfinite(energy):
                    raise FloatingPointError("Nonfinite diffusion conditioning energy")
                gradient = torch.autograd.grad(energy, x)[0] if guidance else None
                updated = x + (next_sigma - sigma) * (x - clean) / sigma
                if gradient is not None:
                    # Bounded heuristic DPS step, explicitly not exact Bayes sampling.
                    updated = updated - guidance * gradient / gradient.norm().clamp_min(1.0)
            x = updated.detach()
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite diffusion conditional sample")
        with torch.no_grad():
            core = unwhiten_cores((x[0] * scale + mean).reshape_as(model.cores), chol)
            final_energy = potential(core)
            if not torch.isfinite(core).all() or not torch.isfinite(final_energy):
                raise FloatingPointError("Nonfinite diffusion posterior core or energy")
            energies.append(float(final_energy))
            all_samples.append(core)
        if sample == 0 or sample == chains * draws - 1:
            print(f"generated {sample + 1}/{chains * draws} latent histories", flush=True)
    cores = torch.stack(all_samples).reshape(chains, draws, *model.cores.shape)
    return cores, {
        "diffusion_steps": steps,
        "guidance": guidance,
        "data_energy_by_draw": energies,
        "mcmc_diagnostics_applicable": False,
        "note": "Approximate DPS through nonlinear age-mixture likelihood; no MH correction or calibration",
    }
