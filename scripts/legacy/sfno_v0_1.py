"""SFNO backbone + two-pass anchor-field gradient recomputation (§8).

Role (§8.6): an INSTANCE-SPECIFIC inverse-problem solver. One small SFNO
provides positive single-channel anchor fields H_theta(a_m); observations
query time-interpolated fields at their TRUE ages. Not an amortized inverse
operator; no one-forward-pass generalization claim.

IMPORTANT STATUS NOTE
---------------------
The torch implementation below is WRITTEN BUT UNTESTED IN THIS REPO: the
training environment (torch / torch-harmonics) is intentionally not installed
(protocol execution_policy). Before any training run:

  1. install the pinned environment (requires authorization);
  2. run models/sfno.py as a script: `python -m norn_earth.models.sfno`
     to execute the built-in smoke + gradient-equivalence checks;
  3. only then bind NornSFNO into scripts/train.py.

The numpy-side contract (anchor fields, partition-of-unity time queries,
two-pass gradients) is fully specified in 最终研究方案_NORN.md §8.4-§8.7 and
mirrored by CPU tests elsewhere in this package.
"""
import textwrap


class TrainingEnvironmentNotInstalled(RuntimeError):
    """Raised when torch-dependent components are requested without torch."""


MESSAGE = textwrap.dedent(
    """
    NORN SFNO components require PyTorch, which is not installed here.

    Per 研究执行协议_NORN.json (execution_policy):
      - package installs / training-environment setup need explicit authorization;
      - no training jobs may start while gates G0/G1/G2 are OPEN.

    After authorization, install the pinned environment (configs/main.json
    'environment') and run `python -m norn_earth.models.sfno` to execute the
    self-checks before binding into scripts/train.py.
    """
).strip()


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise TrainingEnvironmentNotInstalled(MESSAGE) from exc
    return torch


def two_pass_step_contract():
    """Machine-readable contract; see scripts/train.py --dry-run."""
    return {
        "pass_1": "no_grad forward per anchor -> cached fields U_m; full objective on U gives G_m = dL/dU_m",
        "pass_2": "re-forward per anchor with grad; loss = sum_m <stopgrad(G_m), H_m>; backward accumulates dL/dtheta exactly once",
        "rules": [
            "no parameter updates between passes or between anchor blocks",
            "dropout/stateful batchnorm disabled or exactly replayed",
            "stale field/gradient caches must never cross optimizer steps",
            "true objective values (pass 1) are logged, never surrogate sums",
            "full-graph vs two-pass relative gradient error must pass <= 1e-5 before use",
        ],
        "field_memory_mb_float32_61x64800": round(61 * 64800 * 4 / 1024 / 1024, 2),
    }


def make_sfno(lmax=32, mmax=32, width=32, blocks=4, in_channels=8, nlat=180, nlon=360):
    """Build the NORN SFNO: spectral conv blocks + pointwise bypass + FiLM.

    Inputs per anchor (channels, frozen covariates only, §8.2):
      0: plate-boundary distance (normalized)
      1: deforming-network coverage fraction
      2: continental fraction (data-derived mask)
      3: reliable oceanic-age support
      4: process-prior mean (model-derived flag channel 5 gates it)
      5: prior-applicability mask
      6: sin(lat) geometry
      7: cos(lat) geometry
    Time enters via FiLM from low-frequency features of a (never training-
    label-derived) scalar age input.
    """
    torch = _require_torch()
    import torch.nn as nn

    try:
        from torch_harmonics import RealSHT, InverseRealSHT
    except ImportError as exc:  # pragma: no cover
        raise TrainingEnvironmentNotInstalled(
            "torch-harmonics is required (pinned commit; see configs/main.json)"
        ) from exc

    class SpectralBlock(nn.Module):
        """Spectral mixing via complex SHT handled as concatenated real/imag."""

        def __init__(self, width):
            super().__init__()
            self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
            self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss")
            self.lin = nn.Conv2d(2 * width, 2 * width, 1)

        def forward(self, x):
            c = self.sht(x)  # complex (B, C, l, m)
            re, im = c.real, c.imag
            z = self.lin(torch.cat([re, im], dim=1))
            re, im = z[:, : x.shape[1]], z[:, x.shape[1]:]
            return x + self.isht(torch.complex(re, im))

    class NornSFNO(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Conv2d(in_channels, width, 1)
            self.blocks = nn.ModuleList([SpectralBlock(width) for _ in range(blocks)])
            self.bypass = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks)])
            self.film = nn.ModuleList([nn.Linear(8, 2 * width) for _ in range(blocks)])
            self.head = nn.Conv2d(width, 1, 1)
            self.age_feat = nn.Sequential(nn.Linear(1, 8), nn.Tanh())

        def forward(self, x, age_scalar):
            h = self.embed(x)
            t = self.age_feat(age_scalar)
            for blk, byp, film in zip(self.blocks, self.bypass, self.film):
                g, b = film(t)[:, :, None, None].chunk(2, dim=1)
                h = torch.nn.functional.softplus((1 + g) * h + b) * 0.5 + h * 0.5
                h = blk(h) + byp(h)
            return torch.nn.functional.softplus(self.head(h)) + 1e-3

    return NornSFNO()


def two_pass_step(model, anchors_inputs, ages, objective_on_fields, optimizer,
                  direct_param_regularizer=None):
    """One optimizer step implementing the §8.7 contract exactly.

    objective_on_fields(fields_61x180x360) -> (loss_value, grad_wrt_fields)
    must be differentiable-free w.r.t. theta (it operates on cached fields).
    """
    torch = _require_torch()
    params = [p for p in model.parameters() if p.requires_grad]
    # pass 1: fields without graph
    with torch.no_grad():
        fields = []
        for x, a in zip(anchors_inputs, ages):
            fields.append(model(x, a))
        fields = torch.stack(fields)
    loss_val, G = objective_on_fields(fields)  # G: same shape, d loss / d fields
    # pass 2: recompute with graph; surrogate = sum <stopgrad(G), H>
    optimizer.zero_grad(set_to_none=True)
    surrogate = 0.0
    for x, a, g in zip(anchors_inputs, ages, G):
        h = model(x, a)
        surrogate = surrogate + (g.detach() * h).sum()
    surrogate.backward()  # dL/dtheta == d<stopgrad(dL/dU), H>/dtheta
    if direct_param_regularizer is not None:
        reg, reg_grad = direct_param_regularizer(params)
        with torch.no_grad():
            for p, dg in zip(params, reg_grad):
                p.grad = (0 if p.grad is None else p.grad) + dg
    optimizer.step()
    return float(loss_val)


def check_two_pass_gradients(model, anchors_inputs, ages, objective_on_fields,
                             rel_tol=1e-5):
    """Full-graph vs two-pass equivalence test (must pass before training)."""
    torch = _require_torch()
    import copy

    reference = copy.deepcopy(model)
    # full graph
    for p in reference.parameters():
        p.requires_grad_(True)
    fields = torch.stack([reference(x, a) for x, a in zip(anchors_inputs, ages)])
    loss, _ = objective_on_fields(fields)
    loss.backward()
    full = [p.grad.clone() for p in reference.parameters()]
    # two-pass
    clone = copy.deepcopy(model)
    optim = torch.optim.SGD(clone.parameters(), lr=0.0)  # lr=0: gradient read-out
    two_pass_step(clone, anchors_inputs, ages, objective_on_fields, optim)
    two = [p.grad.clone() for p in clone.parameters()]
    num = sum(float((f - t).norm() ** 2) for f, t in zip(full, two)) ** 0.5
    den = sum(float(f.norm() ** 2) for f in full) ** 0.5
    return {"relative_error": num / max(den, 1e-30), "passed": num / max(den, 1e-30) <= rel_tol}


class NornSFNOStub:  # pragma: no cover - torch absent in this environment
    """Importable name that raises the actionable error on construction."""

    def __init__(self, *args, **kwargs):
        _require_torch()
        raise RuntimeError("unreachable: torch present but stub imported")


if __name__ == "__main__":  # pragma: no cover - requires torch
    torch = _require_torch()
    torch.manual_seed(0)
    model = make_sfno(lmax=16, mmax=16, width=8, blocks=2, in_channels=8)
    xs = [torch.randn(2, 8, 180, 360) for _ in range(3)]
    ages = [torch.tensor([[0.0], [60.0]]), torch.tensor([[30.0], [30.0]]), torch.tensor([[60.0], [0.0]])]

    gen = torch.Generator().manual_seed(123)
    target = torch.full((3, 2, 1, 180, 360), 35.0)  # (anchors, batch, channel, lat, lon)
    w = torch.rand(target.shape, generator=gen)

    def objective(fields):
        loss = (w * (fields - target) ** 2).mean()
        g = 2 * w * (fields - target) / fields.numel()
        return loss, g

    res = check_two_pass_gradients(model, xs, ages, objective)
    print(res)
    raise SystemExit(0 if res["passed"] else 1)
