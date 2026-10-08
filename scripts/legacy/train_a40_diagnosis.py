"""Diagnosis training: WITH vs WITHOUT challenge exclusion.

Full-size model (25M, LayerNorm, 6 blocks, 8ch) + temporal smoothness.
Runs twice:
  A) challenge excluded (generalization test -- same as before)
  B) challenge INCLUDED (capability test -- can the model learn at all?)
If B succeeds where A fails, the problem is data exclusion, not architecture.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from norn_earth.data.modern import assemble_modern_batch
from norn_earth.utils.hashing import write_manifest

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda:0")
NLAT, NLON = 180, 360
N_ANCHORS = 61
ANCHOR_STEP = 1.0
IN_CH = 8
WIDTH = 32
LMAX = 32
BLOCKS = 6
STEPS = 600
LAM_SMOOTH_T = 0.01


class SpectralBlock(nn.Module):
    def __init__(self, width, nlat, nlon, lmax):
        super().__init__()
        from torch_harmonics import RealSHT, InverseRealSHT
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid="legendre-gauss")
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid="legendre-gauss")
        self.lin = nn.Conv2d(2 * width, 2 * width, 1)
        self.norm = nn.InstanceNorm2d(width, affine=True)

    def forward(self, x):
        c = self.sht(x)
        z = self.lin(torch.cat([c.real, c.imag], dim=1))
        re, im = z[:, : x.shape[1]], z[:, x.shape[1]:]
        return self.norm(x + self.isht(torch.complex(re, im)))


class NornBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv2d(IN_CH, WIDTH, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.sblocks = nn.ModuleList([SpectralBlock(WIDTH, NLAT, NLON, LMAX) for _ in range(BLOCKS)])
        self.pblocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.films = nn.ModuleList([nn.Linear(32, 2 * WIDTH) for _ in range(BLOCKS)])
        self.age_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Conv2d(WIDTH, WIDTH // 2, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(WIDTH // 2, 1, 1))

    def forward(self, x, age_scalar):
        h = self.embed(x)
        t = self.age_embed(age_scalar)
        for i, (sb, pb, film) in enumerate(zip(self.sblocks, self.pblocks, self.films)):
            g, b = film(t)[:, :, None, None].chunk(2, dim=1)
            h = h + torch.nn.functional.silu(g * h + b)
            h = sb(h) + pb(h)
        return torch.nn.functional.softplus(self.head(h)) + 0.5


def build_data(exclude_chal):
    table = pd.read_csv(ROOT / "norn" / "processed" / "observations.csv")
    chal = pd.read_csv(ROOT / "norn" / "processed" / "challenge_test_set.csv")
    chal_union = chal["chal_union"].to_numpy(bool)

    lo = table["age_lower_ma"].to_numpy(float)
    hi = table["age_upper_ma"].to_numpy(float)
    y = table["thickness_km"].to_numpy(float)
    lon = table["present_lon"].to_numpy(float)
    lat = table["present_lat"].to_numpy(float)

    dat = np.loadtxt(ROOT / "present_crustal_thickness.dat")
    H0 = -dat[:, 2].reshape(NLAT, NLON)
    jj = np.clip(np.round(lon - 0.5).astype(int), 0, NLON - 1) % NLON
    ii = np.clip(np.round(lat + 89.5).astype(int), 0, NLAT - 1)
    sigma_src = np.where(H0[ii, jj] >= 20.0, 9.0, 26.0)
    sigma_eff = np.sqrt(4.0**2 + sigma_src**2)

    from norn_earth.losses.chronology import make_age_nodes
    node_ages, node_w, node_rec, counts = [], [], [], []
    for r in range(len(table)):
        w = hi[r] - lo[r]
        k = 1 if w == 0 else (3 if w <= 5 else 8)
        nodes, weights, _ = make_age_nodes(lo[r], hi[r], k)
        counts.append(len(nodes))
        node_ages.extend(nodes)
        node_w.extend(weights)
        node_rec.extend([r] * len(nodes))
    counts = np.array(counts, dtype=int)
    node_i = ii[node_rec]
    node_j = jj[node_rec]
    x = np.array(node_ages) / ANCHOR_STEP
    a0 = np.clip(np.floor(x).astype(int), 0, N_ANCHORS - 2)
    af = x - a0
    anchor_w = np.zeros((len(node_ages), N_ANCHORS), dtype=np.float32)
    rows = np.arange(len(node_ages))
    anchor_w[rows, a0] = 1 - af
    anchor_w[rows, a0 + 1] = af

    return {
        "y": torch.tensor(y, dtype=torch.float32, device=DEVICE),
        "sigma_eff": torch.tensor(sigma_eff, dtype=torch.float32, device=DEVICE),
        "counts": torch.tensor(counts, dtype=torch.long, device=DEVICE),
        "node_w": torch.tensor(node_w, dtype=torch.float32, device=DEVICE),
        "node_rec": torch.tensor(node_rec, dtype=torch.long, device=DEVICE),
        "node_i": torch.tensor(node_i, dtype=torch.long, device=DEVICE),
        "node_j": torch.tensor(node_j, dtype=torch.long, device=DEVICE),
        "anchor_w": torch.tensor(anchor_w, device=DEVICE),
        "rec_chal": torch.tensor(chal_union, dtype=torch.bool, device=DEVICE),
        "exclude_chal": exclude_chal,
        "n_records": len(table),
        "n_nodes": len(node_ages),
    }


def compute_loss(fields, obs, modern_H, modern_sigma, modern_blocks_t):
    flat = fields.squeeze(1)
    vals = flat[:, obs["node_i"], obs["node_j"]].T
    mu_nodes = (vals * obs["anchor_w"]).sum(dim=1)

    n_rec = obs["n_records"]
    counts = obs["counts"]
    K = int(counts.max().item())
    starts = torch.cumsum(counts, 0) - counts
    pad_mu = torch.zeros(n_rec, K, device=DEVICE)
    pad_w = torch.zeros(n_rec, K, device=DEVICE)
    slots = torch.arange(obs["n_nodes"], device=DEVICE) - starts[obs["node_rec"]]
    pad_mu[obs["node_rec"], slots] = mu_nodes
    pad_w[obs["node_rec"], slots] = obs["node_w"]

    keep = counts > 0
    if obs["exclude_chal"]:
        keep = keep & ~obs["rec_chal"]

    y_k = obs["y"][keep]
    sig_k = obs["sigma_eff"][keep]
    mu_k = pad_mu[keep]
    w_k = pad_w[keep]

    nu = torch.tensor(4.0, device=DEVICE)
    z = (y_k[:, None] - mu_k) / sig_k[:, None]
    log_p = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
             - 0.5 * torch.log(nu * torch.pi) - sig_k.log()[:, None]
             - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))
    log_joint = torch.log(w_k.clamp(min=1e-30)) + log_p
    m = log_joint.max(dim=1, keepdim=True)[0]
    nll = -(m.squeeze(1) + (log_joint - m).exp().sum(dim=1).log()).mean()

    pred_0 = fields[0].squeeze()
    resid2 = ((pred_0 - modern_H) / modern_sigma) ** 2
    nll_mod = (0.5 * (resid2 + torch.log(2 * torch.pi * modern_sigma ** 2)))
    per_blk = [nll_mod[modern_blocks_t == b].mean() for b in torch.unique(modern_blocks_t)]
    loss_modern = 0.5 * torch.stack(per_blk).mean()

    diff = fields[1:] - fields[:-1]
    loss_smooth = LAM_SMOOTH_T * (diff ** 2).mean()

    return nll + loss_modern + loss_smooth, {"obs": nll.item(), "modern": loss_modern.item(), "smooth": loss_smooth.item()}


def train_and_eval(exclude_chal, tag, static_in, obs, modern_H, modern_sigma, modern_blocks_t, ages_t):
    OUT = Path(__file__).resolve().parents[1] / "outputs" / f"a40_diag_{tag}"
    OUT.mkdir(parents=True, exist_ok=True)

    model = NornBig().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training [{tag}] exclude_chal={exclude_chal} params={n_params:,}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=50)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS - 50, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[50])

    best_loss = float("inf")
    for step in range(STEPS):
        model.train()
        optimizer.zero_grad()

        with torch.no_grad():
            fields_cached = torch.cat([model(static_in, a) for a in ages_t], dim=0)
        fields_leaf = fields_cached.detach().requires_grad_(True)
        loss_val, parts = compute_loss(fields_leaf, obs, modern_H, modern_sigma, modern_blocks_t)
        G = torch.autograd.grad(loss_val, fields_leaf)[0]

        surrogate = sum((G[a].detach() * model(static_in, ages_t[a])).sum() for a in range(N_ANCHORS))
        surrogate.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}: total={loss_val.item():.4f} obs={parts['obs']:.4f} "
                  f"modern={parts['modern']:.4f} smooth={parts['smooth']:.5f}")
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            torch.save(model.state_dict(), OUT / "best_model.pt")

    # evaluate
    model.eval()
    with torch.no_grad():
        fields = torch.cat([model(static_in, a) for a in ages_t], dim=0)
    flat = fields.squeeze(1)

    table = pd.read_csv(ROOT / "norn" / "processed" / "observations.csv")
    chal = pd.read_csv(ROOT / "norn" / "processed" / "challenge_test_set.csv")
    results = {}
    for cname in [c for c in chal.columns if c.startswith("chal_") and c != "chal_union"]:
        short = cname.replace("chal_", "")
        mask = chal[cname].to_numpy(bool)
        if mask.sum() == 0:
            continue
        sub = table[mask]
        ages_r = sub["age_representative_ma"].to_numpy(float)
        y_true = sub["thickness_km"].to_numpy(float)
        lon_r = sub["present_lon"].to_numpy(float)
        lat_r = sub["present_lat"].to_numpy(float)
        jj_r = np.clip(np.round(lon_r - 0.5).astype(int), 0, NLON - 1) % NLON
        ii_r = np.clip(np.round(lat_r + 89.5).astype(int), 0, NLAT - 1)
        preds = np.zeros(len(sub))
        for r in range(len(sub)):
            x = ages_r[r] / ANCHOR_STEP
            a0 = int(np.clip(np.floor(x), 0, N_ANCHORS - 2))
            f = x - a0
            preds[r] = (1 - f) * flat[a0, ii_r[r], jj_r[r]].item() + f * flat[a0 + 1, ii_r[r], jj_r[r]].item()
        corr = float(np.corrcoef(preds, y_true)[0, 1]) if len(preds) > 2 else None
        results[short] = {"n": int(mask.sum()), "mae": float(np.mean(np.abs(preds - y_true))),
                          "b0_mae": float(np.mean(np.abs(np.median(y_true) - y_true))),
                          "corr": corr, "pred_med": float(np.median(preds)), "true_med": float(np.median(y_true))}
        print(f"  {short:<28s}: n={results[short]['n']:4d} mae={results[short]['mae']:6.2f} "
              f"corr={results[short]['corr']:.3f} pred_med={results[short]['pred_med']:5.1f} true={results[short]['true_med']:5.1f}")

    np.savez_compressed(OUT / "fields.npz", **{f"age_{a}": fields[a].squeeze().cpu().numpy() for a in (0, 10, 20, 30, 40, 50, 60)})
    write_manifest(OUT / "results.json", {"tag": tag, "exclude_chal": exclude_chal,
                                           "params": n_params, "results": results, "best_loss": best_loss})
    return results


def main():
    static_in = torch.tensor(
        np.load(ROOT / "norn" / "outputs" / "a40_training" / "input_features_8ch.npy"),
        device=DEVICE,
    ).unsqueeze(0)
    modern = assemble_modern_batch(ROOT / "present_crustal_thickness.dat", sigma_km=3.0)
    modern_H = torch.tensor(modern["H_km"], dtype=torch.float32, device=DEVICE)
    modern_sigma = torch.tensor(modern["sigma_km"], dtype=torch.float32, device=DEVICE)
    modern_blocks_t = torch.tensor(modern["block_ids"], dtype=torch.long, device=DEVICE)
    ages_t = [torch.tensor([[float(a)]], device=DEVICE) for a in range(N_ANCHORS)]

    # Run A: without challenge (generalization)
    obs_a = build_data(exclude_chal=True)
    res_a = train_and_eval(True, "excluded", static_in, obs_a, modern_H, modern_sigma, modern_blocks_t, ages_t)

    # Run B: with challenge (capability)
    obs_b = build_data(exclude_chal=False)
    res_b = train_and_eval(False, "included", static_in, obs_b, modern_H, modern_sigma, modern_blocks_t, ages_t)

    # comparison table
    print(f"\n{'='*70}")
    print(f"{'DIAGNOSIS: excluded vs included challenge data':^70}")
    print(f"{'='*70}")
    print(f"{'Challenge':<28s} {'excl MAE':>8s} {'incl MAE':>8s} {'excl corr':>9s} {'incl corr':>9s} {'diagnosis':>12s}")
    print("-" * 70)
    for name in res_a:
        if name not in res_b:
            continue
        a, b = res_a[name], res_b[name]
        diff = b["mae"] - a["mae"]
        if diff > 2.0:
            diag = "data excl"  # including helps a lot -> problem was data exclusion
        elif b["corr"] is not None and a["corr"] is not None and b["corr"] > a["corr"] + 0.15:
            diag = "data excl"
        else:
            diag = "architecture"  # even with data, model can't learn -> architecture issue
        print(f"{name:<28s} {a['mae']:8.1f} {b['mae']:8.1f} {a['corr']:9.3f} {b['corr']:9.3f} {diag:>12s}")

    write_manifest(Path(__file__).resolve().parents[1] / "outputs" / "a40_diagnosis.json",
                   {"excluded": res_a, "included": res_b})


if __name__ == "__main__":
    main()
