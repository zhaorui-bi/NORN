"""B7: chronology-aware variational reconstruction (CPU, analytic gradients).

Model: H(a, x) = sum_m b_m(a) * Y_lm(x)^T c_m -- low-order real spherical
harmonics x time anchors with partition-of-unity time weights; optional
softplus positivity (default on).

Objective (instance-specific solver, §8.6):
  L = lam_obs   * mean mixture NLL over records (one evidence unit each)
    + lam_modern* modern endpoint NLL (area + block normalized)
    + lam_smooth* temporal coefficient roughness

Gradients are analytic and mirror the two-pass contract: node predictions are
linear in coefficients; mixture gradients use responsibilities; the modern
term touches anchor 0 only. A finite-difference check is enforced in tests.
Fitting REAL data is gated (statistical fitting counts as training);
synthetic self-tests are allowed.
"""

import numpy as np

from ..geometry.regrid import native_grid
from ..losses.chronology import mixture_nll
from ..losses.likelihood import modern_endpoint_nll
from ..models.variational import (
    anchor_interpolation_weights,
    coefficient_count,
    evaluate_basis_points,
)

_SOFTPLUS = lambda x: np.logaddexp(0.0, x)  # noqa: E731
_DSOFTPLUS = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731


class VariationalReconstructor:
    def __init__(
        self,
        lmax=8,
        n_anchors=61,
        anchor_step=1.0,
        positive=True,
        lam_modern=1.0,
        lam_obs=1.0,
        lam_smooth=0.05,
        loss_mode="mixture",
    ):
        """
        loss_mode:
          "mixture"       - chronology-aware marginal NLL (default, C1)
          "expected_point" - sum_k w_k * point NLL (label-copying objective,
                            the ABLATION variant; never the main model)
        """
        self.lmax = lmax
        self.n_anchors = n_anchors
        self.anchor_step = anchor_step
        self.positive = positive
        self.lam_modern = lam_modern
        self.lam_obs = lam_obs
        self.lam_smooth = lam_smooth
        self.loss_mode = loss_mode
        self.n_coef = coefficient_count(lmax)
        self.coef = np.zeros((n_anchors, self.n_coef))
        # start near a ~30 km mean field under softplus
        self.coef[:, 0] = np.log(np.expm1(30.0))

    # ------------------------------------------------------------------ field
    def _time_weights(self, ages):
        return np.stack(
            [
                anchor_interpolation_weights(float(a), self.anchor_step, self.n_anchors)
                for a in np.atleast_1d(ages)
            ]
        )

    def _field_linear(self, node_lon, node_lat, node_ages):
        """z (pre-activation), plus cache (Y, W) for gradients (paired points)."""
        Y = evaluate_basis_points(
            self.lmax, np.asarray(node_lat, float), np.asarray(node_lon, float)
        )
        W = self._time_weights(node_ages)  # (n, K)
        F = W @ self.coef  # (n, n_coef)
        return np.einsum("ij,ij->i", Y, F), (Y, W)

    def predict_nodes(self, node_lon, node_lat, node_ages):
        z, cache = self._field_linear(node_lon, node_lat, node_ages)
        return (_SOFTPLUS(z) if self.positive else z), (cache, z)

    def predict_grid(self, age, grid=None):
        """Field on a full grid; default native 180x360, or (lats, lons) pair."""
        if grid is None:
            lats, lons = native_grid()
        else:
            lats, lons = grid
        lon2, lat2 = np.meshgrid(lons, lats)
        v, _ = self.predict_nodes(lon2.ravel(), lat2.ravel(), np.full(lon2.size, float(age)))
        return v.reshape(len(lats), len(lons))

    def _modern_grid_basis(self, modern_batch):
        """Basis on the modern batch's grid (cached); supports native or gauss."""
        if (
            getattr(self, "_mgb_cache", None) is not None
            and self._mgb_cache[0].shape == modern_batch["H_km"].shape
        ):
            return self._mgb_cache
        shape = modern_batch["H_km"].shape
        if shape == (180, 360):
            lats, lons = native_grid()
        else:
            from ..geometry.regrid import gauss_grid

            lats, lons, _ = gauss_grid(shape[0], shape[1])
        lon2, lat2 = np.meshgrid(lons, lats)
        Y0 = evaluate_basis_points(self.lmax, lat2.ravel(), lon2.ravel())
        self._mgb_cache = (Y0, (lats, lons))
        return self._mgb_cache

    # ----------------------------------------------------------------- batch
    @staticmethod
    def _node_to_record(counts):
        idx = np.repeat(np.arange(len(counts)), counts)
        return idx

    def objective(self, obs_batch, modern_batch, return_parts=False):
        counts = obs_batch["node_counts"]
        n = len(counts)
        node2rec = self._node_to_record(counts)
        if len(obs_batch["node_ages"]):
            mu_all, ((Yall, Wall), z_all) = self.predict_nodes(
                obs_batch["node_lon"], obs_batch["node_lat"], obs_batch["node_ages"]
            )
        else:
            mu_all = np.zeros(0)
            Yall = Wall = np.zeros((0, self.n_coef)), np.zeros((0, self.n_anchors))
            z_all = np.zeros(0)
        K = int(max(counts.max(), 1)) if len(counts) else 1
        pad_mu = np.zeros((n, K))
        pad_w = np.zeros((n, K))
        if len(mu_all):
            starts = np.concatenate([[0], np.cumsum(counts)])
            for i in range(n):
                s, e = starts[i], starts[i + 1]
                pad_mu[i, : e - s] = mu_all[s:e]
                pad_w[i, : e - s] = obs_batch["node_weights"][s:e]
        keep = counts > 0
        y = obs_batch["y_km"][keep]
        sig = obs_batch["effective_sigma_km"][keep]
        if keep.any():
            if self.loss_mode == "expected_point":
                from ..losses.chronology import student_t_logpdf

                lp = student_t_logpdf(y[:, None], pad_mu[keep], sig[:, None])
                nll_rows = -np.sum(pad_w[keep] * lp, axis=1)
                nu = 4.0
                dl_dmu = (
                    (nu + 1)
                    * (pad_mu[keep] - y[:, None])
                    / (nu * sig[:, None] ** 2 + (y[:, None] - pad_mu[keep]) ** 2)
                )
                dmu = pad_w[keep] * dl_dmu
                mean_obs = float(np.mean(nll_rows))
            else:
                nll_rows, dmu = mixture_nll(y, pad_mu[keep], pad_w[keep], sig, return_grad=True)
                mean_obs = float(np.mean(nll_rows))
        else:
            mean_obs, dmu = 0.0, np.zeros_like(pad_mu[keep])
        Y0, grid = self._modern_grid_basis(modern_batch)
        nll_mod = modern_endpoint_nll(
            modern_batch["H_km"],
            self.predict_grid(0.0, grid=grid),
            modern_batch["sigma_km"],
            modern_batch["areas_km2"],
            modern_batch["block_ids"],
        )
        smooth = float(np.sum(np.diff(self.coef, axis=0) ** 2)) if self.n_anchors > 1 else 0.0
        total = self.lam_obs * mean_obs + self.lam_modern * nll_mod + self.lam_smooth * smooth
        if return_parts:
            # scatter padded dmu back onto flat nodes
            g_mu_flat = np.zeros(len(mu_all))
            if keep.any():
                starts = np.concatenate([[0], np.cumsum(counts)])
                pos = 0
                for i in range(n):
                    c = counts[i]
                    if c == 0:
                        continue
                    g_mu_flat[starts[i] : starts[i] + c] = dmu[pos, :c]
                    pos += 1
            return total, {
                "obs": mean_obs,
                "modern": nll_mod,
                "smooth": smooth,
                "g_mu_flat": g_mu_flat,
                "Yall": Yall,
                "Wall": Wall,
                "z_all": z_all,
                "node2rec": node2rec,
                "n_records": n,
            }
        return total

    # ------------------------------------------------------------ gradients
    def grad(self, obs_batch, modern_batch):
        """Exact analytic gradient (finite-diff verified); fully vectorized."""
        total, parts = self.objective(obs_batch, modern_batch, return_parts=True)
        G = np.zeros_like(self.coef)

        # observation term: flat over all nodes (mean over records folded in)
        if len(parts["g_mu_flat"]):
            g_z = parts["g_mu_flat"] * self.lam_obs / max(parts["n_records"], 1)
            if self.positive:
                g_z = g_z * _DSOFTPLUS(parts["z_all"])
            G += (parts["Wall"] * g_z[:, None]).T @ parts["Yall"]

        # modern term: only anchor 0 receives gradient (age-0 field)
        Y0, grid = self._modern_grid_basis(modern_batch)
        z0 = Y0 @ self.coef[0]
        h = (_SOFTPLUS(z0) if self.positive else z0).ravel()
        sig = modern_batch["sigma_km"].ravel()
        areas = modern_batch["areas_km2"].ravel()
        blocks = np.asarray(modern_batch["block_ids"]).ravel()
        hobs = modern_batch["H_km"].ravel()
        g_h = np.zeros_like(h)
        uniq = np.unique(blocks)
        for b in uniq:
            msk = blocks == b
            w = areas[msk] / areas[msk].sum()
            g_h[msk] = self.lam_modern * w * (h[msk] - hobs[msk]) / sig[msk] ** 2 / len(uniq)
        g_z0 = g_h * (_DSOFTPLUS(z0) if self.positive else 1.0)
        G[0] += Y0.T @ g_z0

        # temporal roughness: L = sum_m ||C_m - C_{m-1}||^2
        if self.n_anchors > 1:
            D = np.zeros_like(self.coef)
            D[0] = 2 * (self.coef[0] - self.coef[1])
            D[-1] = 2 * (self.coef[-1] - self.coef[-2])
            if self.n_anchors > 2:
                D[1:-1] = 2 * (2 * self.coef[1:-1] - self.coef[:-2] - self.coef[2:])
            G += self.lam_smooth * D
        return total, G

    # ---------------------------------------------------------------- fitting
    def fit(self, obs_batch, modern_batch, maxiter=100, verbose=False):
        """L-BFGS with exact analytic gradients; synthetic-safe, gated for real.

        The optimizer sees only the SAME information the loss uses; no test
        data, no y-dependent masks (§5.5).
        """
        from scipy.optimize import minimize

        history = []

        def fun(x):
            self.coef = x.reshape(self.coef.shape)
            total, G = self.grad(obs_batch, modern_batch)
            history.append(total)
            return total, G.ravel()

        res = minimize(
            fun,
            self.coef.ravel().copy(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": int(maxiter), "maxfun": 4 * int(maxiter)},
        )
        self.coef = res.x.reshape(self.coef.shape)
        if verbose:
            print(f"  L-BFGS: nit={res.nit} fun={res.fun:.5f} ({res.message})")
        return history
