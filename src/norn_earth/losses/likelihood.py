"""Observation likelihood assembly: source scales, biases, modern endpoint.

Guards from the design:
- effective scale = sqrt(s_i^2 + s_source^2) with s_source calibrated
  independently or bounded low-dimensional; NO free per-pixel noise head;
- modern source bias is fixed at 0 as a gauge choice (the modern product is
  an observation with its own error, not perfect truth);
- if the ancient-proxy teacher used modern labels related to the modern
  product, the two error families are correlated; independence must be
  verified, never assumed, before using proxies as independent validation.
"""

import numpy as np

from .chronology import make_age_nodes

MODERN_GAUGE_BIAS = 0.0


class SourceRegistry:
    def __init__(self):
        self.scales = {}  # source_id -> sigma_source (km), calibrated/bounded
        self.biases = {}  # source_id -> prior mean bias (km)

    def register(self, source_id, scale_km, bias_km=0.0):
        if scale_km is not None and float(scale_km) <= 0:
            raise ValueError("source scale must be positive")
        self.scales[source_id] = float(scale_km) if scale_km is not None else None
        self.biases[source_id] = float(bias_km)

    def effective_scale(self, record_sigma, source_id):
        s_src = self.scales.get(source_id)
        if s_src is None:
            raise KeyError(f"unregistered source {source_id}; calibrate before use")
        return np.sqrt(np.asarray(record_sigma) ** 2 + s_src**2)

    def bias(self, source_id):
        return self.biases.get(source_id, 0.0)


def modern_endpoint_nll(h_pred_native, h_obs_native, cell_sigma_km, areas_km2, block_ids=None):
    """Area- and block-normalized modern endpoint likelihood.

    The 64,800 grid cells are not 64,800 independent maps: the loss is
    normalized by total area and (optionally) macro-averaged over blocks.
    """
    h_pred = np.asarray(h_pred_native, dtype=float).ravel()
    h_obs = np.asarray(h_obs_native, dtype=float).ravel()
    sig = np.asarray(cell_sigma_km, dtype=float).ravel()
    areas = np.asarray(areas_km2, dtype=float).ravel()
    bids = np.asarray(block_ids).ravel() if block_ids is not None else None
    if bids is None:
        w = areas / areas.sum()
        resid2 = w * ((h_pred - h_obs) / sig) ** 2
        return 0.5 * float(resid2.sum() + np.sum(w * np.log(2 * np.pi * sig**2)))
    per_block = []
    for b in np.unique(bids):
        m = bids == b
        w = areas[m] / areas[m].sum()
        z2 = ((h_pred[m] - h_obs[m]) / sig[m]) ** 2 + np.log(2 * np.pi * sig[m] ** 2)
        per_block.append(0.5 * float(np.sum(w * z2)))
    return float(np.mean(per_block))


def record_nodes(observation, k_narrow=3, k_wide=8, domain=(0.0, 60.0)):
    """Age nodes for one observation row, following §5.2 defaults."""
    lo, hi = observation["age_lower"], observation["age_upper"]
    width = hi - lo
    if width <= 0:
        k = 1
    elif width <= 5.0:
        k = min(k_narrow, 3)
    else:
        k = k_wide
    return make_age_nodes(lo, hi, k, domain=domain)
