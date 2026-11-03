"""Declared temporal GP for low-dimensional cores; all ages are in Ma."""

import math

import torch


def gp_cholesky(ages, lengthscale_ma, amplitude=1.0, jitter=1e-5):
    ages = torch.as_tensor(ages)
    if not ages.is_floating_point():
        ages = ages.to(torch.float32)
    if (
        ages.ndim != 1
        or len(ages) < 2
        or not torch.isfinite(ages).all()
        or not torch.all(ages[1:] > ages[:-1])
    ):
        raise ValueError("GP ages must be finite and strictly increasing")
    if any(not math.isfinite(v) or v <= 0 for v in (lengthscale_ma, amplitude, jitter)):
        raise ValueError("GP lengthscale, amplitude and jitter must be finite and positive")
    distance = (ages[:, None] - ages[None, :]) / lengthscale_ma
    covariance = torch.exp(-0.5 * distance.square())
    covariance = amplitude**2 * (
        covariance + jitter * torch.eye(len(ages), device=ages.device, dtype=ages.dtype)
    )
    return torch.linalg.cholesky(covariance)


def whiten_cores(cores, cholesky):
    """Time is dimension 0; remaining dimensions are independent GP outputs."""
    if cores.shape[0] != cholesky.shape[0]:
        raise ValueError("Core time dimension differs from the GP")
    return torch.linalg.solve_triangular(
        cholesky, cores.reshape(len(cores), -1), upper=False
    ).reshape_as(cores)


def unwhiten_cores(white, cholesky):
    if white.shape[0] != cholesky.shape[0]:
        raise ValueError("Core time dimension differs from the GP")
    return (cholesky @ white.reshape(len(white), -1)).reshape_as(white)
