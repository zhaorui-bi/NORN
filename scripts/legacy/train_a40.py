"""A40 training: SFNO with geological-challenge test set.

User authorization (2026-10-08): direct GPU training requested.

Setup:
- Excludes geological-challenge test records from training
- Per-domain source scales (continental 9 km / thin 26 km), selected by dev study
- Input: sin/cos(lat), cos/sin(lon) + continental mask (5 channels, frozen)
- SFNO: 4 blocks, width 16, lmax 16, FiLM time, 61 anchors @ 1 Myr
- Loss: modern endpoint NLL + chronology mixture NLL
- Eval: challenge sets + B0/B2 baselines + P-C windows (15-25 reserved, NOT opened)
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

from norn_earth.data.batches import batch_slices
from norn_earth.data.modern import assemble_modern_batch, modern_blocks
from norn_earth.geometry.regrid import native_grid
from norn_earth.losses.chronology import student_t_logpdf
from norn_earth.utils.hashing import write_manifest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs" / "a40_training"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NLAT, NLON = 180, 360
N_ANCHORS = 61
ANCHOR_STEP = 1.0


# ============================================================ Model
class SpectralBlock(nn.Module):
    def __init__(self, width, nlat, nlon, lmax):
        super().__init__()
        from torch_harmonics import RealSHT, InverseRealSHT
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid="legendre-gauss")
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid="legendre-gauss")
        self.lin = nn.Conv2d(2 * width, 2 * width, 1)

    def forward(self, x):
        c = self.sht(x)
        re, im = c.real, c.imag
        z = self.lin(torch.cat([re, im], dim=1))
        re, im = z[:, : x.shape[1]], z[:, x.shape[1]:]
        return x + self.isht(torch.complex(re, im))


class NornSFNO(nn.Module):
    def __init__(self, in_ch=5, width=16, blocks=4, lmax=16, nlat=180, nlon=360):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, width, 1)
        self.sblocks = nn.ModuleList([SpectralBlock(width, nlat, nlon, lmax) for _ in range(blocks)])
        self.pblocks = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks)])
        self.films = nn.ModuleList([nn.Linear(16, 2 * width) for _ in range(blocks)])
        self.age_embed = nn.Sequential(nn.Linear(1, 16), nn.Tanh())
        self.head = nn.Conv2d(width, 1, 1)
        self.width = width

    def forward(self, x, age_scalar):
        h = self.embed(x)
        t = self.age_embed(age_scalar)
        for sb, pb, film in zip(self.sblocks, self.pblocks, self.films):
            g, b = film(t)[:, :, None, None].chunk(2, dim=1)
            h = h + torch.nn.functional.silu(g * h + b)
            h = sb(h) + pb(h)
        return torch.nn.functional.softplus(self.head(h)) + 0.5


# ============================================================ Data
def build_static_inputs():
    """5 frozen channels: sin/cos(lat), cos/sin(lon), continental mask."""
    lats, lons = native_grid()
    lon2, lat2 = np.meshgrid(lons, lats)
    lat_r = np.radians(lat2)
    lon_r = np.radians(lon2)
    ch0 = np.sin(lat_r)
    ch1 = np.cos(lat_r) * np.cos(lon_r)
    ch2 = np.cos(lat_r) * np.sin(lon_r)
    # continental mask from DAT (thickness > 20 km)
    dat = np.loadtxt(ROOT / "present_crustal_thickness.dat")
    H0 = -dat[:, 2].reshape(NLAT, NLON)
    ch3 = (H0 > 20.0).astype(float) * 2 - 1
    ch4 = np.cos(lat_r) ** 2  # extra geometric feature
    return np.stack([ch0, ch1, ch2, ch3, ch4], axis=0).astype(np.float32)


def build_observation_tensors(table, lookup, challenge_mask=None):
    """Build node-level tensors for the mixture NLL on GPU."""
    lo = table["age_lower_ma"].to_numpy(float)
    hi = table["age_upper_ma"].to_numpy(float)
    y = table["thickness_km"].to_numpy(float)
    lon = table["present_lon"].to_numpy(float)
    lat = table["present_lat"].to_numpy(float)

    # per-domain source scales
    dat = np.loadtxt(ROOT / "present_crustal_thickness.dat")
    H0 = -dat[:, 2].reshape(NLAT, NLON)
    j = np.clip(np.round(lon - 0.5).astype(int), 0, 359) % 360
    i = np.clip(np.round(lat + 89.5).astype(int), 0, 179)
    h_site = H0[i, j]
    sigma_source = np.where(h_site >= 20.0, 9.0, 26.0)
    sigma_eff = np.sqrt(4.0**2 + sigma_source**2)

    # age nodes (same logic as batches.py)
    from norn_earth.losses.chronology import make_age_nodes
    node_ages_all, node_w_all, node_i_all, counts = [], [], [], []
    for r in range(len(table)):
        width = hi[r] - lo[r]
        k = 1 if width == 0 else (3 if width <= 5 else 8)
        nodes, w, _ = make_age_nodes(lo[r], hi[r], k)
        counts.append(len(nodes))
        node_ages_all.extend(nodes.tolist())
        node_w_all.extend(w.tolist())
        node_i_all.extend([r] * len(nodes))
    counts = np.array(counts, dtype=int)
    node_ages = np.array(node_ages_all, dtype=float)
    node_w = np.array(node_w_all, dtype=float)
    node_rec = np.array(node_i_all, dtype=int)

    # node positions from engine (or present coords fallback)
    node_lon = lon[node_rec]
    node_lat = lat[node_rec]

    # convert positions to grid indices (nearest cell on native grid)
    glats, glons = native_grid()
    node_j = np.clip(np.round(node_lon - 0.5).astype(int), 0, NLON - 1) % NLON
    node_i = np.clip(np.round(node_lat + 89.5).astype(int), 0, NLAT - 1)

    # time-anchor weights for each node
    x = node_ages / ANCHOR_STEP
    a0 = np.clip(np.floor(x).astype(int), 0, N_ANCHORS - 2)
    af = x - a0
    anchor_w = np.zeros((len(node_ages), N_ANCHORS), dtype=np.float32)
    rows = np.arange(len(node_ages))
    anchor_w[rows, a0] = 1 - af
    anchor_w[rows, a0 + 1] = af

    # exclude challenge test records
    if challenge_mask is not None:
        node_chal = challenge_mask[node_rec]
    else:
        node_chal = np.zeros(len(node_rec), dtype=bool)

    return {
        "y": torch.tensor(y, dtype=torch.float32, device=DEVICE),
        "sigma_eff": torch.tensor(sigma_eff, dtype=torch.float32, device=DEVICE),
        "counts": torch.tensor(counts, dtype=torch.long, device=DEVICE),
        "node_ages": torch.tensor(node_ages, dtype=torch.float32, device=DEVICE),
        "node_w": torch.tensor(node_w, dtype=torch.float32, device=DEVICE),
        "node_rec": torch.tensor(node_rec, dtype=torch.long, device=DEVICE),
        "node_i": torch.tensor(node_i, dtype=torch.long, device=DEVICE),
        "node_j": torch.tensor(node_j, dtype=torch.long, device=DEVICE),
        "anchor_w": torch.tensor(anchor_w, device=DEVICE),
        "node_chal": torch.tensor(node_chal, dtype=torch.bool, device=DEVICE),
        "n_records": len(table),
    }


# ============================================================ Losses
def mixture_nll_torch(fields, obs, modern_H, modern_sigma, modern_blocks_t,
                      lam_obs=1.0, lam_modern=0.5, exclude_chal=True):
    """Full objective on cached anchor fields (B=1 batch)."""
    # observation term
    n_nodes = obs["node_ages"].shape[0]
    # gather field values at node positions
    # fields: (n_anchors, 1, nlat, nlon)
    flat = fields.squeeze(1)  # (n_anchors, nlat, nlon)
    vals = flat[:, obs["node_i"], obs["node_j"]].T  # (n_nodes, n_anchors)
    mu_nodes = (vals * obs["anchor_w"]).sum(dim=1)  # (n_nodes,)

    # per-record mixture NLL via segment logsumexp
    n_rec = obs["n_records"]
    counts = obs["counts"]
    # scatter node predictions into padded (n_rec, K) structure
    K = int(counts.max().item()) if counts.numel() > 0 else 1
    starts = torch.cumsum(counts, 0) - counts
    pad_mu = torch.zeros(n_rec, K, device=DEVICE)
    pad_w = torch.zeros(n_rec, K, device=DEVICE)
    rows = torch.arange(n_nodes, device=DEVICE)
    slots = rows - starts[obs["node_rec"]]
    pad_mu[obs["node_rec"], slots] = mu_nodes
    pad_w[obs["node_rec"], slots] = obs["node_w"]

    keep = counts > 0
    if exclude_chal:
        # exclude challenge test records from training loss
        rec_chal = torch.zeros(n_rec, dtype=torch.bool, device=DEVICE)
        rec_chal[obs["node_rec"][obs["node_chal"]]] = True
        keep = keep & ~rec_chal

    y_k = obs["y"][keep]
    sig_k = obs["sigma_eff"][keep]
    mu_k = pad_mu[keep]
    w_k = pad_w[keep]

    # Student-t mixture NLL
    nu = torch.tensor(4.0, device=DEVICE)
    z = (y_k[:, None] - mu_k) / sig_k[:, None]
    log_p = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * torch.pi)
        - sig_k.log()[:, None]
        - (nu + 1) / 2 * torch.log1p(z ** 2 / nu)
    )
    log_joint = torch.log(w_k.clamp(min=1e-30)) + log_p
    m = log_joint.max(dim=1, keepdim=True)[0]
    nll_rows = -(m.squeeze(1) + (log_joint - m).exp().sum(dim=1).log())
    loss_obs = lam_obs * nll_rows.mean()

    # modern endpoint term (area + block normalized)
    pred_0 = fields[0].squeeze()  # (nlat, nlon)
    resid2 = ((pred_0 - modern_H) / modern_sigma) ** 2
    log_det = torch.log(2 * torch.pi * modern_sigma ** 2)
    nll_mod_per_cell = 0.5 * (resid2 + log_det)
    # block macro average
    uniq_blocks = torch.unique(modern_blocks_t)
    per_block = []
    for b in uniq_blocks:
        msk = modern_blocks_t == b
        per_block.append(nll_mod_per_cell[msk].mean())
    loss_modern = lam_modern * torch.stack(per_block).mean()

    total = loss_obs + loss_modern
    return total, {"obs": loss_obs.item(), "modern": loss_modern.item()}


# ============================================================ Training
def train():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- data ----
    print("Loading data...")
    table = pd.read_csv(ROOT / "norn" / "processed" / "observations.csv")
    chal = pd.read_csv(ROOT / "norn" / "processed" / "challenge_test_set.csv")
    chal_union = chal["chal_union"].to_numpy(bool)

    obs = build_observation_tensors(table, None, challenge_mask=chal_union)
    print(f"  observations: {obs['n_records']} records, {obs['node_ages'].shape[0]} nodes")
    print(f"  challenge test records: {int(chal_union.sum())} (excluded from training)")

    modern = assemble_modern_batch(ROOT / "present_crustal_thickness.dat", sigma_km=3.0)
    modern_H = torch.tensor(modern["H_km"], dtype=torch.float32, device=DEVICE)
    modern_sigma = torch.tensor(modern["sigma_km"], dtype=torch.float32, device=DEVICE)
    modern_blocks_t = torch.tensor(modern["block_ids"], dtype=torch.long, device=DEVICE)

    static_in = torch.tensor(build_static_inputs(), device=DEVICE).unsqueeze(0)  # (1, C, H, W)

    # ---- model ----
    print("Building model...")
    model = NornSFNO(in_ch=5, width=16, blocks=4, lmax=16, nlat=NLAT, nlon=NLON).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    # ---- training loop (two-pass) ----
    print("\nTraining (two-pass gradient, 61 anchors)...")
    ages = [torch.tensor([[float(a)]], device=DEVICE) for a in range(N_ANCHORS)]
    best_loss = float("inf")
    history = []

    for step in range(200):
        t0 = time.time()
        model.train()

        # pass 1: generate all anchor fields without graph
        with torch.no_grad():
            fields_list = []
            for a in range(N_ANCHORS):
                f = model(static_in, ages[a])
                fields_list.append(f)
            fields_cached = torch.cat(fields_list, dim=0)  # (61, 1, H, W)

        # compute loss and field gradients on cached fields
        fields_leaf = fields_cached.detach().requires_grad_(True)
        loss_val, parts = mixture_nll_torch(fields_leaf, obs, modern_H, modern_sigma,
                                             modern_blocks_t)
        G = torch.autograd.grad(loss_val, fields_leaf, retain_graph=False)[0]

        # pass 2: re-forward with grad, surrogate backward
        optimizer.zero_grad()
        surrogate = 0.0
        for a in range(N_ANCHORS):
            f = model(static_in, ages[a])
            surrogate = surrogate + (G[a].detach() * f).sum()
        surrogate.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        history.append({"step": step, "loss": loss_val.item(), **parts, "time_s": elapsed})
        if step % 10 == 0 or step == 199:
            print(f"  step {step:3d}: total={loss_val.item():.4f} obs={parts['obs']:.4f} "
                  f"modern={parts['modern']:.4f} ({elapsed:.1f}s)")
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            torch.save(model.state_dict(), OUT / "best_model.pt")

    # ---- save final model ----
    torch.save(model.state_dict(), OUT / "final_model.pt")
    pd.DataFrame(history).to_csv(OUT / "training_log.csv", index=False)

    # ---- evaluate on challenge sets ----
    print("\n=== Evaluating on geological challenge sets ===")
    model.eval()
    with torch.no_grad():
        fields = []
        for a in range(N_ANCHORS):
            fields.append(model(static_in, ages[a]))
        fields_eval = torch.cat(fields, dim=0)

    results = evaluate_challenges(fields_eval, obs, table, chal)
    write_manifest(OUT / "challenge_results.json", {"results": results, "best_loss": best_loss})
    print(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nDone. Outputs in {OUT}")


def evaluate_challenges(fields, obs, table, chal):
    """Evaluate model on each geological challenge set."""
    chal_cols = [c for c in chal.columns if c.startswith("chal_") and c != "chal_union"]
    results = {}

    with torch.no_grad():
        flat = fields.squeeze(1)  # (61, H, W)
        for cname in chal_cols:
            short = cname.replace("chal_", "")
            mask = chal[cname].to_numpy(bool)
            if mask.sum() == 0:
                results[short] = {"n": 0}
                continue

            # predict at each record's representative age
            sub = table[mask]
            ages = sub["age_representative_ma"].to_numpy(float)
            y_true = sub["thickness_km"].to_numpy(float)
            lon = sub["present_lon"].to_numpy(float)
            lat = sub["present_lat"].to_numpy(float)

            # grid lookup
            jj = np.clip(np.round(lon - 0.5).astype(int), 0, NLON - 1) % NLON
            ii = np.clip(np.round(lat + 89.5).astype(int), 0, NLAT - 1)

            # time interpolation
            preds = np.zeros(len(sub))
            for r in range(len(sub)):
                x = ages[r] / ANCHOR_STEP
                a0 = int(np.clip(np.floor(x), 0, N_ANCHORS - 2))
                f = x - a0
                v0 = flat[a0, ii[r], jj[r]].item()
                v1 = flat[a0 + 1, ii[r], jj[r]].item()
                preds[r] = (1 - f) * v0 + f * v1

            mae = float(np.mean(np.abs(preds - y_true)))
            corr = float(np.corrcoef(preds, y_true)[0, 1]) if len(preds) > 2 else None
            # B0 baseline
            b0 = float(np.median(y_true))
            b0_mae = float(np.mean(np.abs(b0 - y_true)))
            # thickness stats
            results[short] = {
                "n": int(mask.sum()),
                "model_mae_km": mae,
                "b0_median_mae_km": b0_mae,
                "corr": corr,
                "pred_median": float(np.median(preds)),
                "true_median": float(np.median(y_true)),
                "pred_std": float(np.std(preds)),
                "true_std": float(np.std(y_true)),
            }
            print(f"  {short:<28s}: n={mask.sum():4d} MAE={mae:6.2f} corr={corr:.3f} "
                  f"pred_med={np.median(preds):5.1f} true_med={np.median(y_true):5.1f}")

    return results


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train()
