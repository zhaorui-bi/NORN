"""Compatibility factory and gradient checks for the public spherical operator.

New applications should use models.operator.NornSFNO and training.trainer.
This module remains importable without optional PyTorch dependencies.
"""


class TrainingEnvironmentNotInstalled(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise TrainingEnvironmentNotInstalled(
            "Install norn-earth[sfno] to use the spherical operator"
        ) from exc
    return torch


def two_pass_step_contract():
    return {
        "pass_1": "cache fields without activation graphs; differentiate the complete objective on field leaves",
        "pass_2": "recompute anchors and backward immediately per chunk; update parameters only after all chunks",
        "rules": [
            "parameters fixed between passes",
            "no dropout or running batch statistics",
            "no stale field caches",
            "log the original objective",
            "verify gradient equivalence",
        ],
        "field_memory_mb_float32_61x64800": round(61 * 64800 * 4 / 1024**2, 2),
    }


def make_sfno(lmax=32, mmax=32, width=32, blocks=6, in_channels=8, nlat=180, nlon=360):
    _require_torch()
    from ..config import GridConfig, ModelConfig
    from .operator import NornSFNO

    return NornSFNO(
        GridConfig(nlat, nlon),
        ModelConfig(width=width, blocks=blocks, lmax=lmax, mmax=mmax, in_channels=in_channels),
    )


def two_pass_step(
    model, anchors_inputs, ages, objective_on_fields, optimizer, direct_param_regularizer=None
):
    torch = _require_torch()
    with torch.no_grad():
        fields = torch.stack([model(x, a) for x, a in zip(anchors_inputs, ages)])
    loss, G = objective_on_fields(fields)
    optimizer.zero_grad(set_to_none=True)
    for x, a, g in zip(anchors_inputs, ages, G):
        (model(x, a) * g.detach()).sum().backward()
    if direct_param_regularizer is not None:
        params = [p for p in model.parameters() if p.requires_grad]
        _, grad = direct_param_regularizer(params)
        for p, dg in zip(params, grad):
            p.grad = dg.clone() if p.grad is None else p.grad + dg
    optimizer.step()
    return float(loss)


def check_two_pass_gradients(model, anchors_inputs, ages, objective_on_fields, rel_tol=1e-5):
    import copy

    torch = _require_torch()
    full = copy.deepcopy(model)
    fields = torch.stack([full(x, a) for x, a in zip(anchors_inputs, ages)])
    loss, _ = objective_on_fields(fields)
    loss.backward()
    two = copy.deepcopy(model)
    optimizer = torch.optim.SGD(two.parameters(), lr=0.0)
    two_pass_step(two, anchors_inputs, ages, objective_on_fields, optimizer)
    grads = [
        (a.grad, b.grad) for a, b in zip(full.parameters(), two.parameters()) if a.grad is not None
    ]
    numerator = sum(float((a - b).square().sum()) for a, b in grads) ** 0.5
    denominator = sum(float(a.square().sum()) for a, _ in grads) ** 0.5
    error = numerator / max(denominator, 1e-30)
    return {"relative_error": error, "passed": error <= rel_tol}
