"""P6b: two-pass gradient equivalence with the PRODUCTION mixture loss.

Replaces the quadratic self-check: the objective is the real chronology
mixture NLL + modern endpoint NLL computed on cached anchor fields via
autograd w.r.t. the fields (exact G_m), then compared against full-graph
backward through the SFNO. Must pass <= 1e-5 before any training run.
CPU-small: lmax 8, width 8, 2 blocks, 3 anchors, 64x128 grid.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import torch

    import norn_earth.models.sfno as M
    from norn_earth.losses.chronology import student_t_logpdf

    torch.manual_seed(0)
    np.random.seed(0)
    nlat, nlon = 64, 128
    n_anchors, n_obs = 3, 400
    model = M.make_sfno(lmax=8, mmax=8, width=8, blocks=2, in_channels=4, nlat=nlat, nlon=nlon)

    # synthetic observation set: node ages/weights/positions + targets
    rng = np.random.default_rng(7)
    node_counts = rng.choice([1, 3, 8], size=n_obs, p=[0.5, 0.3, 0.2])
    starts = np.concatenate([[0], np.cumsum(node_counts)]).astype(int)
    n_nodes = starts[-1]
    node_ages = rng.uniform(0, 60, n_nodes)
    node_w = np.concatenate([rng.dirichlet(np.ones(c)) for c in node_counts])
    node_lon = rng.uniform(0, 360, n_nodes)
    node_lat = rng.uniform(-80, 80, n_nodes)
    y = rng.uniform(10, 60, n_obs)
    sig = np.full(n_obs, 4.0)
    grid_lat = np.linspace(-89, 89, nlat)
    grid_lon = np.linspace(0, 360, nlon, endpoint=False)
    anchor_ages = np.array([0.0, 30.0, 60.0])

    # grid indices for each node (nearest cell) for field lookup
    ii = np.clip(np.searchsorted(grid_lat, node_lat), 0, nlat - 1)
    jj = np.clip(np.searchsorted(grid_lon, node_lon) - 1, 0, nlon - 1) % nlon
    anchor_w = np.zeros((n_nodes, n_anchors))
    x = node_ages / 30.0
    i0 = np.clip(np.floor(x).astype(int), 0, n_anchors - 2)
    f = x - i0
    rows = np.arange(n_nodes)
    anchor_w[rows, i0] = 1 - f
    anchor_w[rows, i0 + 1] = f
    anchor_w_t = torch.tensor(anchor_w, dtype=torch.float32)
    node_w_t = torch.tensor(node_w, dtype=torch.float32)
    ii_t = torch.tensor(ii, dtype=torch.long)
    jj_t = torch.tensor(jj, dtype=torch.long)
    y_t = torch.tensor(y, dtype=torch.float32)
    sig_t = torch.tensor(sig, dtype=torch.float32)
    starts_t = torch.tensor(starts, dtype=torch.long)

    xs = [torch.randn(2, 4, nlat, nlon) for _ in range(n_anchors)]
    ages = [torch.full((2, 1), float(a)) for a in anchor_ages]

    def logsumexp_t(z, dim):
        m, _ = z.max(dim=dim, keepdim=True)
        return (m.squeeze(dim) + (z - m).exp().sum(dim=dim).log())

    def objective_on_fields(fields):
        """fields: (A, B, 1, lat, lon) leaf tensor; returns (loss, dL/dfields)."""
        if not fields.requires_grad:
            fields = fields.detach().requires_grad_(True)
        flat = fields.squeeze(2)  # (A, B, lat, lon)
        vals = torch.stack([flat[a, 0, ii_t, jj_t] for a in range(n_anchors)], dim=1)  # (n_nodes, A)
        mu = (vals * anchor_w_t).sum(dim=1)
        # pad per record: ragged -> loop records (small n)
        nlls = []
        for r in range(n_obs):
            s, e = int(starts_t[r]), int(starts_t[r + 1])
            mu_r = mu[s:e]
            w_r = node_w_t[s:e] / node_w_t[s:e].sum()
            lp = student_t_logpdf_t(y_t[r], mu_r, sig_t[r])
            z = (w_r.log() + lp)
            m = z.max()
            nlls.append(-(m + torch.log((z - m).exp().sum())))
        loss = torch.stack(nlls).mean()
        G, = torch.autograd.grad(loss, fields, retain_graph=True)
        return loss, G  # keep graph: callers may backward it

    def student_t_logpdf_t(yv, mu, s, nu=4.0):
        import math
        z = (yv - mu) / s
        c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
        return c - s.log() - (nu + 1) / 2 * torch.log1p(z ** 2 / nu)

    # correctness gate
    res = M.check_two_pass_gradients(model, xs, ages, objective_on_fields)
    print(res)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
