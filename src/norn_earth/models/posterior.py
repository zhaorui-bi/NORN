"""Conditional latent sampling with explicit uncertainty and target semantics."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..data.preparation import load_dataset
from ..geometry.regrid import gauss_grid
from ..losses.objective import ReconstructionObjective, age_mixture_log_prob
from ..training.trainer import load_checkpoint, resolve_device
from ..utils.hashing import sha256_file, write_manifest
from .temporal import unwhiten_cores, whiten_cores


def split_rhat(samples):
    """Basic split R-hat; a diagnostic, not a convergence certificate."""
    chains, draws = samples.shape[:2]
    n = draws // 2
    if chains < 2 or n < 2:
        return None
    x = torch.cat([samples[:, :n], samples[:, -n:]], dim=0).flatten(2).double()
    within = x.var(dim=1, unbiased=True).mean(0)
    between = n * x.mean(1).var(dim=0, unbiased=True)
    variance = (n - 1) / n * within + between / n
    if torch.any(within == 0):
        return None
    value = torch.sqrt(variance / within).max()
    return float(value) if torch.isfinite(value) else None


@torch.no_grad()
def pcn_sample(log_likelihood, initial, draws=32, burnin=100, thin=2, beta=0.1, seed=42):
    """pCN preserves N(0,I); MH correction targets exp(log_likelihood) N(0,I).

    `initial` has shape (chain, ...latent dimensions...). The callable takes
    one chain state. No clipping or adaptation of retained states is used.
    """
    if draws < 2 or burnin < 0 or thin < 1 or not 0 < beta <= 1:
        raise ValueError("Require draws>=2, burnin>=0, thin>=1 and 0<beta<=1")
    if initial.ndim < 2 or not torch.isfinite(initial).all():
        raise ValueError("Provide finite initial states with a leading chain dimension")
    if not initial.is_floating_point() or not len(initial) or initial[0].numel() == 0:
        raise ValueError("pCN requires nonempty floating-point chains and latent states")
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    current = initial.detach().clone()
    values = torch.stack([log_likelihood(x) for x in current])
    if not torch.isfinite(values).all():
        raise ValueError("Initial likelihood must be finite")
    accepted = torch.zeros(len(current), device=current.device)
    samples = []
    total = burnin + draws * thin
    for step in range(total):
        noise = torch.randn(
            current.shape, generator=generator, device=current.device, dtype=current.dtype
        )
        proposal = math.sqrt(1 - beta**2) * current + beta * noise
        proposed_values = torch.stack([log_likelihood(x) for x in proposal])
        if torch.isnan(proposed_values).any() or torch.isposinf(proposed_values).any():
            raise FloatingPointError("Nonfinite pCN likelihood")
        logu = torch.rand(len(current), generator=generator, device=current.device).log()
        accept = logu < proposed_values - values
        accepted += accept
        current[accept] = proposal[accept]
        values = torch.where(accept, proposed_values, values)
        if step >= burnin and (step - burnin + 1) % thin == 0:
            samples.append(current.clone())
    samples = torch.stack(samples, dim=1)
    rhat = split_rhat(samples)
    rates = (accepted / total).cpu().tolist()
    return samples, {
        "acceptance_rate_by_chain": rates,
        "split_rhat_max": rhat,
        "mixing_status": (
            "short_run_diagnostic_passed"
            if rhat is not None and rhat < 1.1 and min(rates) > 0
            else "mixing_not_established"
        ),
        "iterations_per_chain": total,
        "diagnostic_note": "Basic split R-hat and acceptance do not establish convergence",
    }


def data_potential(objective, fields):
    """Normalized data energy; excludes physics, Huber and fitted-core penalty.

    Consequently this defines a generalized posterior, not a product of
    independent per-grid-cell likelihoods. Age-mixture Student-t and modern
    block normalization are exactly the operators used in training.
    """
    metrics = objective.observation_metrics(fields, "train")
    if metrics is None:
        raise ValueError("Conditional sampling requires training observations")
    modern, _ = objective.modern_loss(fields)
    t = objective.config.training
    return t.observation_weight * metrics["nll"] + t.modern_weight * modern


def load_sampling_problem(checkpoint_path, dataset, device):
    checkpoint, config, model, inputs = load_checkpoint(checkpoint_path, device)
    if config.model.backbone != "functional_tucker":
        raise ValueError("Latent sampling requires a functional_tucker checkpoint")
    if sha256_file(dataset) != checkpoint["provenance"]["dataset_sha256"]:
        raise ValueError("Sampling dataset differs from the frozen training dataset")
    torch.set_num_threads(config.training.num_threads)
    # Data likelihood evaluation remains portable without external prior files.
    data_config = copy.deepcopy(config)
    from ..config import PhysicsConfig

    data_config.physics = PhysicsConfig()
    data_config.training.objective = "data_only"
    data_config.training.temporal_smooth_weight = 0.0
    objective = ReconstructionObjective(load_dataset(dataset, data_config), data_config, device)
    model.requires_grad_(False)
    with torch.no_grad():
        factors = tuple(x.detach() for x in model.encode_factors(inputs))
    return checkpoint, config, model, inputs, objective, factors


def save_samples(output, cores, model, factors, config, metadata):
    if (
        cores.ndim != 5
        or cores.shape[0] < 1
        or cores.shape[1] < 2
        or tuple(cores.shape[2:]) != tuple(model.cores.shape)
        or not torch.isfinite(cores).all()
    ):
        raise ValueError("Require finite, nonempty chain/draw/age/rank core samples")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "latent_samples.npz"
    summary_path = output / "posterior_summary.npz"
    if samples_path.exists() or summary_path.exists():
        raise FileExistsError("Sampling outputs already exist; choose a new output directory")
    alpha = metadata["interval_alpha"]
    flattened = cores.flatten(0, 1)
    mean, lower, upper = [], [], []
    with torch.no_grad():
        # Bound memory to one age's draws; never stack draws x all global maps.
        for age in range(config.n_anchors):
            one = tuple(x[age : age + 1] for x in factors)
            fields = torch.stack(
                [model.decode_factors(one, core[age : age + 1])[0, 0] for core in flattened]
            )
            if not torch.isfinite(fields).all() or not torch.all(fields > 0):
                raise FloatingPointError("Posterior decoder produced invalid thickness")
            mean.append(fields.mean(0).cpu().numpy())
            lower.append(torch.quantile(fields, alpha / 2, dim=0).cpu().numpy())
            upper.append(torch.quantile(fields, 1 - alpha / 2, dim=0).cpu().numpy())
    lats, lons, _ = gauss_grid(config.grid.nlat, config.grid.nlon)
    metadata = {
        **metadata,
        "grid": "legendre-gauss",
        "latitude_order": "north_to_south",
        "units": "km",
        "calibrated_uncertainty": False,
        "representation_uncertainty_included": False,
        "plate_model_uncertainty_included": False,
        "modern_endpoints_allowed": True,
    }
    meta = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(
        samples_path, cores=cores.cpu().numpy(), age_ma=config.anchor_ages, metadata=meta
    )
    np.savez_compressed(
        summary_path,
        thickness_mean_km=np.stack(mean),
        thickness_lower_km=np.stack(lower),
        thickness_upper_km=np.stack(upper),
        latitude=lats[::-1],
        longitude=lons,
        age_ma=config.anchor_ages,
        metadata=meta,
    )
    write_manifest(output / "sampling_manifest.json", metadata)
    return metadata


def sample_posterior(
    checkpoint,
    dataset,
    output,
    *,
    draws=32,
    burnin=100,
    thin=2,
    chains=2,
    beta=0.1,
    temperature=1.0,
    alpha=0.1,
    seed=42,
    device="auto",
    prior=None,
    diffusion_steps=32,
    guidance=0.1,
):
    if chains < 1 or not math.isfinite(temperature) or temperature <= 0 or not 0 < alpha < 1:
        raise ValueError("Require chains>=1, positive temperature and 0<alpha<1")
    device = resolve_device(device)
    body, config, model, inputs, objective, factors = load_sampling_problem(
        checkpoint, dataset, device
    )
    output = Path(output)
    if (output / "latent_samples.npz").exists() or (output / "posterior_summary.npz").exists():
        raise FileExistsError("Sampling outputs exist; choose a new output directory")

    def potential(core):
        return data_potential(objective, model.decode_factors(factors, core)[:, 0]) / temperature

    if prior:
        from .diffusion import sample_diffusion_prior

        cores, diagnostics = sample_diffusion_prior(
            prior, model, inputs, config, potential, draws, chains, diffusion_steps, guidance, seed
        )
        kind = "approximate_diffusion_conditional_sensitivity"
        method = "whitened_core_diffusion_guidance"
    else:
        cholesky = model.prior_cholesky
        initial = whiten_cores(model.cores.detach(), cholesky)
        rng = torch.Generator(device=device).manual_seed(seed + 1)
        initial = initial[None].repeat(chains, 1, 1, 1)
        initial = initial + 0.01 * torch.randn(initial.shape, generator=rng, device=device)

        def log_likelihood(white):
            return -potential(unwhiten_cores(white, cholesky))

        white, diagnostics = pcn_sample(log_likelihood, initial, draws, burnin, thin, beta, seed)
        cores = torch.stack(
            [torch.stack([unwhiten_cores(x, cholesky) for x in chain]) for chain in white]
        )
        kind = "conditional_generalized_latent_posterior"
        method = "gp_pcn"
    metadata = {
        "method": method,
        "interval_kind": kind,
        "interval_alpha": alpha,
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": sha256_file(dataset),
        "prior_sha256": sha256_file(prior) if prior else None,
        "source_semantics": body["provenance"]["dataset_metadata"]["source_semantics"],
        "draws_per_chain": draws,
        "chains": chains,
        "seed": seed,
        "temperature": temperature,
        "beta": beta if prior is None else None,
        "burnin": burnin if prior is None else None,
        "thin": thin if prior is None else None,
        "data_energy": "train age-mixture Student-t mean + modern block mean; no test/validation",
        "observation_weight": config.training.observation_weight,
        "modern_weight": config.training.modern_weight,
        "lengthscale_ma": config.model.latent_lengthscale_ma,
        "latent_amplitude": config.model.latent_amplitude,
        "training_latent_gp_weight": config.training.latent_gp_weight,
        "latent_scale_km": config.model.latent_scale_km,
        "training_objective": config.training.objective,
        "physics_in_sampling": False,
        "diagnostics": diagnostics,
    }
    return save_samples(output, cores.detach(), model, factors, config, metadata)


def evaluate_posterior(
    checkpoint,
    dataset,
    samples,
    output,
    *,
    split="validation",
    predictive_draws=256,
    alpha=0.1,
    seed=42,
    device="auto",
):
    """Score held-out *observations*, including age mixture and source noise.

    Interval coverage against noisy proxies is predictive coverage, not
    calibration of the latent historical thickness interval.
    """
    if split not in ("train", "validation", "test") or predictive_draws < 32 or not 0 < alpha < 1:
        raise ValueError("Choose a record split, predictive_draws>=32 and 0<alpha<1")
    device = resolve_device(device)
    body, config, model, _, objective, factors = load_sampling_problem(checkpoint, dataset, device)
    with np.load(samples, allow_pickle=False) as bundle:
        cores = torch.as_tensor(bundle["cores"], device=device)
        meta = json.loads(str(bundle["metadata"]))
        if (
            meta["checkpoint_sha256"] != sha256_file(checkpoint)
            or meta["dataset_sha256"] != sha256_file(dataset)
            or not np.array_equal(bundle["age_ma"], config.anchor_ages)
        ):
            raise ValueError("Posterior samples have a different checkpoint, dataset or ages")
    if (
        cores.ndim != 5
        or cores.shape[0] < 1
        or cores.shape[1] < 2
        or tuple(cores.shape[2:]) != tuple(model.cores.shape)
        or not torch.isfinite(cores).all()
    ):
        raise ValueError("Malformed posterior core samples")
    mask = objective.masks[split]
    if not mask.any():
        raise ValueError(f"No {split} records")
    means, logps = [], []
    with torch.no_grad():
        for core in cores.flatten(0, 1):
            fields = model.decode_factors(factors, core)[:, 0]
            means.append(objective.node_predictions(fields)[mask])
        means = torch.stack(means)
        weights, y, sigma = objective.weights[mask], objective.y[mask], objective.sigma[mask]
        logps = age_mixture_log_prob(means, weights, y, sigma)
        nll = -(torch.logsumexp(logps, dim=0) - math.log(len(means))).mean()
        expected = (means * weights).sum(-1).mean(0)
        # Randomness is scoped so evaluation never changes a caller's training RNG.
        devices = []
        if device.type == "cuda":
            devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices if device.type == "cuda" else []):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            indices = torch.randint(len(means), (predictive_draws,), device=device)
            nodes = torch.multinomial(weights, predictive_draws, replacement=True).T
            records = torch.arange(len(y), device=device)[None, :]
            mu = means[indices[:, None], records, nodes]
            predictions = torch.distributions.StudentT(4.0, mu, sigma[None, :]).sample()
        lower = torch.quantile(predictions, alpha / 2, dim=0)
        upper = torch.quantile(predictions, 1 - alpha / 2, dim=0)
    from ..evaluation.calibration import interval_metrics

    report = {
        "split": split,
        "n_records": int(mask.sum()),
        "posterior_predictive_nll": float(nll),
        "posterior_mean_mae_km": float((expected - y).abs().mean()),
        "predictive_interval": interval_metrics(y.cpu(), lower.cpu(), upper.cpu(), 1 - alpha),
        "interval_kind": "posterior_predictive_observation",
        "observation_noise": "Student-t(df=4), declared source scales; age quadrature mixture",
        "calibrated_uncertainty": False,
        "modern_endpoints_allowed": True,
        "source_semantics": body["provenance"]["dataset_metadata"]["source_semantics"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": sha256_file(dataset),
        "samples_sha256": sha256_file(samples),
        "sampling_method": meta.get("method"),
        "sampling_diagnostics": meta.get("diagnostics", {}),
        "predictive_draws": predictive_draws,
        "seed": seed,
    }
    write_manifest(output, report)
    return report
