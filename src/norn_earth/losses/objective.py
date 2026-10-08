"""The single observation operator shared by training and evaluation."""

import math

import numpy as np
import torch

from ..geometry.regrid import native_grid
from ..geometry.sampling import make_query
from ..physics.differentiable import PhysicsObjective


class ReconstructionObjective:
    def __init__(self, data, config, device="cpu"):
        self.config = config
        self.data = data
        self.device = device
        c = config
        self.y = torch.as_tensor(data["y"], dtype=torch.float32, device=device)
        self.sigma = torch.as_tensor(data["sigma"], dtype=torch.float32, device=device)
        self.node_record = torch.as_tensor(data["node_record"], dtype=torch.long, device=device)
        counts = data["counts"]
        starts = np.cumsum(counts) - counts
        slots = np.arange(len(data["node_record"])) - starts[data["node_record"]]
        self.slots = torch.as_tensor(slots, dtype=torch.long, device=device)
        self.max_nodes = int(max(counts.max(), 1))
        self.weights = torch.zeros(len(counts), self.max_nodes, device=device)
        self.weights[self.node_record, self.slots] = torch.as_tensor(
            data["node_weight"], device=device
        )
        self.query = make_query(
            data["node_lon"],
            data["node_lat"],
            data["node_age"],
            c.grid.nlat,
            c.grid.nlon,
            c.anchor_step_myr,
            c.n_anchors,
            device,
        )
        self.masks = {
            name: torch.as_tensor(data["valid"] & (data["split"] == i), device=device)
            for i, name in enumerate(("train", "validation", "test"))
        }
        lats, lons = native_grid()
        lon, lat = np.meshgrid(lons, lats)
        self.modern_query = make_query(
            lon,
            lat,
            np.zeros_like(lon),
            c.grid.nlat,
            c.grid.nlon,
            c.anchor_step_myr,
            c.n_anchors,
            device,
        )
        self.modern_h = torch.as_tensor(data["modern_H"].ravel(), device=device)
        self.modern_sigma = torch.as_tensor(data["modern_sigma"].ravel(), device=device)
        _, inverse = np.unique(data["modern_blocks"].ravel(), return_inverse=True)
        self.modern_block = torch.as_tensor(inverse, dtype=torch.long, device=device)
        self.block_counts = torch.bincount(self.modern_block).float()
        self.area = torch.as_tensor(data["modern_area"].ravel(), device=device)
        self.physics = PhysicsObjective(c.physics.constraints, c, device)

    def node_predictions(self, fields):
        values = self.query.sample(fields)
        padded = fields.new_zeros(len(self.y), self.max_nodes)
        padded[self.node_record, self.slots] = values
        return padded

    def observation_metrics(self, fields, split="train"):
        mask = self.masks[split]
        if not mask.any():
            return None
        mu = self.node_predictions(fields)[mask]
        y = self.y[mask, None]
        sigma = self.sigma[mask, None]
        weights = self.weights[mask]
        nu = 4.0
        logp = (
            math.lgamma((nu + 1) / 2)
            - math.lgamma(nu / 2)
            - 0.5 * math.log(nu * math.pi)
            - torch.log(sigma)
        )
        logp = logp - (nu + 1) / 2 * torch.log1p(((y - mu) / sigma).square() / nu)
        logw = torch.where(weights > 0, weights.log(), -torch.inf)
        nll = -torch.logsumexp(logw + logp, dim=1).mean()
        expected = (mu * weights).sum(-1)
        mae = (expected - self.y[mask]).abs().mean()
        return {"nll": nll, "age_marginal_mean_mae_km": mae, "n_records": int(mask.sum())}

    def modern_loss(self, fields):
        pred = self.modern_query.sample(fields)
        cell = (
            0.5 * ((pred - self.modern_h) / self.modern_sigma).square()
            + torch.log(self.modern_sigma)
            + 0.5 * math.log(2 * math.pi)
        )
        sums = cell.new_zeros(len(self.block_counts)).scatter_add_(0, self.modern_block, cell)
        return (sums / self.block_counts).mean(), torch.sqrt(
            ((pred - self.modern_h).square() * self.area).sum() / self.area.sum()
        )

    def __call__(self, fields):
        t = self.config.training
        obs = self.observation_metrics(fields, "train")
        if obs is None:
            raise ValueError("No training observations")
        modern, rmse = self.modern_loss(fields)
        physical, physics_parts = self.physics(fields)
        smooth = ((fields[1:] - fields[:-1]) / self.config.anchor_step_myr).square().mean()
        total = (
            t.observation_weight * obs["nll"]
            + t.modern_weight * modern
            + physical
            + t.temporal_smooth_weight * smooth
        )
        parts = {
            "observation_nll": obs["nll"],
            "modern_nll": modern,
            "modern_area_rmse_km": rmse,
            "temporal_smoothness": smooth,
            "physics_loss": physical,
            **{f"physics_{k}": v for k, v in physics_parts.items()},
        }
        return total, parts
