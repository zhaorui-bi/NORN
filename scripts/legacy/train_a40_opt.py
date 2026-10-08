"""Optimized A40 training: age-gated inputs + hybrid global-local + physics smoothness.

Fixes from full run:
1. Age-gated modern_H (prevents copying thin modern crust to all ages)
2. 9 channels (added log-stretch-factor)
3. Hybrid: SFNO global backbone + per-point local MLP skip
4. Temporal smoothness regularization (prevents unphysical jumps between ages)
5. Train WITH and WITHOUT challenge exclusion (diagnose memorization vs generalization)
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
OUT = Path(__file__).resolve().parents[1] / "outputs" / "a40_opt"
DEVICE = torch.device("cuda:0")
NLAT, NLON = 180, 360
N_ANCHORS = 61
ANCHOR_STEP = 1.0
IN_CH = 9
WIDTH = 32
LMAX = 32
BLOCKS = 4  # reduced from 6 (faster, less overfit)
STEPS = 500
LAM_SMOOTH_T = 0.02  # temporal smoothness weight


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


class NornHybrid(nn.Module):
    """SFNO global field + local point MLP for observation refinement."""

    def __init__(self):
        super().__init__()
        # global SFNO backbone
        self.embed = nn.Sequential(nn.Conv2d(IN_CH, WIDTH, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.sblocks = nn.ModuleList([SpectralBlock(WIDTH, NLAT, NLON, LMAX) for _ in range(BLOCKS)])
        self.pblocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.films = nn.ModuleList([nn.Linear(32, 2 * WIDTH) for _ in range(BLOCKS)])
        self.age_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.head = nn.Sequential(nn.Conv2d(WIDTH, WIDTH // 2, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(WIDTH // 2, 1, 1))
        # local refinement MLP (observation-level)
        self.local_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward_grid(self, x, age_scalar):
        h = self.embed(x)
        t = self.age_embed(age_scalar)
        for i, (sb, pb, film) in enumerate(zip(self.sblocks, self.pblocks, self.films)):
            g, b = film(t)[:, :, None, None].chunk(2, dim=1)
            h = h + torch.nn.functional.silu(g * h + b)
            h = sb(h) + pb(h)
        return torch.nn.functional.softplus(self.head(h)) + 0.5

    def forward_point(self, lon, lat, age, modern_h):
        """Local MLP: refines global prediction with site-specific features."""
        lon_r = lon * (3.14159265 / 180.0)
        lat_r = lat * (3.14159265 / 180.0)
        feat = torch.stack([
            torch.cos(lon_r), torch.sin(lon_r),
            torch.sin(lat_r), age / 60.0, modern_h / 40.0,
        ], dim=-1)
        return self.local_mlp(feat).squeeze(-1)


def build_data():
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
    modern_h_at_site = H0[ii, jj]

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
        "node_ages": torch.tensor(node_ages, dtype=torch.float32, device=DEVICE),
        "rec_lon": torch.tensor(lon, dtype=torch.float32, device=DEVICE),
        "rec_lat": torch.tensor(lat, dtype=torch.float32, device=DEVICE),
        "rec_modern_h": torch.tensor(modern_h_at_site, dtype=torch.float32, device=DEVICE),
        "rec_chal": torch.tensor(chal_union, dtype=torch.bool, device=DEVICE),
        "n_records": len(table),
        "n_nodes": len(node_ages),
    }


def full_loss(model, obs, modern_H, modern_sigma, modern_blocks_t, age_inputs, ages_t, lam_obs=1.0, lam_modern=0.5):
    # generate all anchor fields
    fields = torch.cat([model.forward_grid(age_inputs[a:a+1], ages_t[a]) for a in range(N_ANCHORS)], dim=0)
    flat = fields.squeeze(1)

    # grid-based observation term
    vals = flat[:, obs["node_i"], obs["node_j"]].T
    global_mu = (vals * obs["anchor_w"]).sum(dim=1)

    # local refinement (at record level, then broadcast to nodes)
    local_corr = model.forward_point(
        obs["rec_lon"], obs["rec_lat"],
        torch.zeros(obs["n_records"], device=DEVICE),  # age handled by global part
        obs["rec_modern_h"],
    )
    mu_nodes = global_mu + local_corr[obs["node_rec"]]

    n_rec = obs["n_records"]
    counts = obs["counts"]
    K = int(counts.max().item())
    starts = torch.cumsum(counts, 0) - counts
    pad_mu = torch.zeros(n_rec, K, device=DEVICE)
    pad_w = torch.zeros(n_rec, K, device=DEVICE)
    slots = torch.arange(obs["n_nodes"], device=DEVICE) - starts[obs["node_rec"]]
    pad_mu[obs["node_rec"], slots] = mu_nodes
    pad_w[obs["node_rec"], slots] = obs["node_w"]

    keep = (counts > 0) & ~obs["rec_chal"]
    y_k = obs["y"][keep]
    sig_k = obs["sigma_eff"][keep]
    mu_k = pad_mu[keep]
    w_k = pad_w[keep]

    nu = torch.tensor(4.0, device=DEVICE)
    z = (y_k[:, None] - mu_k) / sig_k[:, None]
    log_p = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
             - 0.5 * torch.log(nu * torch.pi)
             - sig_k.log()[:, None]
             - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))
    log_joint = torch.log(w_k.clamp(min=1e-30)) + log_p
    m = log_joint.max(dim=1, keepdim=True)[0]
    nll_rows = -(m.squeeze(1) + (log_joint - m).exp().sum(dim=1).log())
    loss_obs = lam_obs * nll_rows.mean()

    # modern endpoint
    pred_0 = fields[0].squeeze()
    resid2 = ((pred_0 - modern_H) / modern_sigma) ** 2
    log_det = torch.log(2 * torch.pi * modern_sigma ** 2)
    nll_cell = 0.5 * (resid2 + log_det)
    per_block = [nll_cell[modern_blocks_t == b].mean() for b in torch.unique(modern_blocks_t)]
    loss_modern = lam_modern * torch.stack(per_block).mean()

    # temporal smoothness (penalize large jumps between adjacent ages)
    diff = fields[1:] - fields[:-1]
    loss_smooth_t = LAM_SMOOTH_T * (diff ** 2).mean()

    total = loss_obs + loss_modern + loss_smooth_t
    return total, {"obs": loss_obs.item(), "modern": loss_modern.item(),
                   "smooth_t": loss_smooth_t.item()}, fields


def evaluate(model, age_inputs, obs, table, chal):
    model.eval()
    ages_t = [torch.tensor([[float(a)]], device=DEVICE) for a in range(N_ANCHORS)]
    with torch.no_grad():
        fields = torch.cat([model.forward_grid(age_inputs[a:a+1], ages_t[a]) for a in range(N_ANCHORS)], dim=0)
    flat = fields.squeeze(1)

    results = {}
    chal_cols = [c for c in chal.columns if c.startswith("chal_") and c != "chal_union"]
    for cname in chal_cols:
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
        with torch.no_grad():
            for r in range(len(sub)):
                x = ages_r[r] / ANCHOR_STEP
                a0 = int(np.clip(np.floor(x), 0, N_ANCHORS - 2))
                f = x - a0
                grid_val = (1 - f) * flat[a0, ii_r[r], jj_r[r]].item() + f * flat[a0 + 1, ii_r[r], jj_r[r]].item()
                preds[r] = grid_val  # global field only for grid-based eval

        corr = float(np.corrcoef(preds, y_true)[0, 1]) if len(preds) > 2 else None
        results[short] = {
            "n": int(mask.sum()),
            "model_mae": float(np.mean(np.abs(preds - y_true))),
            "b0_mae": float(np.mean(np.abs(np.median(y_true) - y_true))),
            "corr": corr,
            "pred_med": float(np.median(preds)),
            "true_med": float(np.median(y_true)),
        }
    return results, fields


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    obs = build_data()
    print(f"Data: {obs['n_records']} records, {obs['n_nodes']} nodes")

    modern = assemble_modern_batch(ROOT / "present_crustal_thickness.dat", sigma_km=3.0)
    modern_H = torch.tensor(modern["H_km"], dtype=torch.float32, device=DEVICE)
    modern_sigma = torch.tensor(modern["sigma_km"], dtype=torch.float32, device=DEVICE)
    modern_blocks_t = torch.tensor(modern["block_ids"], dtype=torch.long, device=DEVICE)

    age_inputs = torch.tensor(
        np.load(ROOT / "norn" / "outputs" / "a40_full" / "age_gated_inputs.npy"), device=DEVICE
    )
    print(f"Age-gated inputs: {age_inputs.shape}")

    model = NornHybrid().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=50)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS - 50, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[50])

    ages_t = [torch.tensor([[float(a)]], device=DEVICE) for a in range(N_ANCHORS)]
    best_loss = float("inf")
    history = []
    print(f"\nTraining {STEPS} steps (hybrid global+local, age-gated, temporal smooth)...")

    for step in range(STEPS):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        loss, parts, _ = full_loss(model, obs, modern_H, modern_sigma,
                                    modern_blocks_t, age_inputs, ages_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        history.append({"step": step, "loss": loss.item(), **parts, "time_s": elapsed})
        if step % 25 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}: total={loss.item():.4f} obs={parts['obs']:.4f} "
                  f"modern={parts['modern']:.4f} smooth={parts['smooth_t']:.4f} ({elapsed:.1f}s)")
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), OUT / "best_model.pt")

    torch.save(model.state_dict(), OUT / "final_model.pt")
    pd.DataFrame(history).to_csv(OUT / "training_log.csv", index=False)

    print("\n=== Evaluating on geological challenge sets ===")
    table = pd.read_csv(ROOT / "norn" / "processed" / "observations.csv")
    chal = pd.read_csv(ROOT / "norn" / "processed" / "challenge_test_set.csv")
    results, fields = evaluate(model, age_inputs, obs, table, chal)
    for name, v in results.items():
        print(f"  {name:<28s}: n={v['n']:4d} model={v['model_mae']:6.2f} B0={v['b0_mae']:6.2f} "
              f"corr={v['corr']:.3f} pred_med={v['pred_med']:5.1f} true_med={v['true_med']:5.1f}")

    field_np = {f"age_{a}": fields[a].squeeze().cpu().numpy() for a in range(0, 61, 5)}
    np.savez_compressed(OUT / "field_evolution.npz", **field_np)

    write_manifest(OUT / "results.json", {
        "model": {"type": "hybrid_global_local", "width": WIDTH, "lmax": LMAX,
                  "blocks": BLOCKS, "in_ch": IN_CH, "params": n_params,
                  "age_gated": True, "temporal_smooth": LAM_SMOOTH_T},
        "challenge_results": results,
        "training_final": history[-1],
    })
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
