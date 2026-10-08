"""Block conformal prediction for y-intervals (P6, v1.3).

Design: calibration/test split BY SPATIAL BLOCK (blocks are the
exchangeable units used to prevent correlated leakage between calibration
and deployment), but scores are PER-POINT absolute residuals -- the
guarantee is marginal point coverage under block exchangeability.
Latent-H intervals remain sensitivity reports until independent H data.
"""

import numpy as np


def block_conformal_calibrate(y_cal, pred_cal, blocks_cal, alpha=0.1):
    """Split-conformal quantile of per-point |residual| over calibration blocks.

    Returns (q, point_scores) with the finite-sample correction
    q = ceil((n+1)(1-alpha))/n quantile of calibration point scores.
    """
    y_cal = np.asarray(y_cal, float)
    pred_cal = np.asarray(pred_cal, float)
    scores = np.abs(pred_cal - y_cal)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(scores, level))
    return q, scores


def block_conformal_intervals(pred, q):
    pred = np.asarray(pred, float)
    return pred - q, pred + q


def coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def split_by_block(blocks, cal_fraction=0.5, seed=0):
    """Deterministic block-level split: returns (cal_block_ids, test_block_ids)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(np.asarray(blocks))
    uniq = rng.permutation(uniq)
    n_cal = max(1, int(round(len(uniq) * cal_fraction)))
    return uniq[:n_cal], uniq[n_cal:]
