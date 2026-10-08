"""Weak-form budget loss with coverage honesty (§6.7/§6.8/§9).

- residuals normalized by H_ref*A_ref (km^3 -> dimensionless), one budget per
  process/region/time window; more control volumes do not mean bigger loss;
- unknown fluxes are masked, never zero-filled;
- a region uses EITHER the trajectory loss OR the weak budget for the same
  information (double counting is rejected, not averaged);
- per-domain loss budget cap prevents unbounded compensation.
"""

import numpy as np

from ..physics.budgets import A_REF_KM2, H_REF_KM


def normalized_budget_residuals(residuals_km3, h_ref=H_REF_KM, a_ref=A_REF_KM2):
    r = np.asarray(residuals_km3, dtype=float)
    return r / (h_ref * a_ref)


def budget_loss(residuals_km3, active_mask=None, tolerance=None, max_total=None):
    """Mean squared normalized residual over ACTIVE domains only."""
    r = normalized_budget_residuals(residuals_km3)
    if active_mask is None:
        active_mask = np.ones(len(r), dtype=bool)
    active = np.asarray(active_mask, dtype=bool)
    if active.sum() == 0:
        return 0.0, {"active_domains": 0}
    quad = float(np.mean(r[active] ** 2))
    info = {"active_domains": int(active.sum()), "rms_normalized": float(np.sqrt(quad))}
    if tolerance is not None:
        info["within_tolerance"] = bool(np.all(np.abs(r[active]) <= tolerance))
    if max_total is not None and quad > max_total:
        info["capped"] = True
        quad = max_total
    return quad, info


def choose_trajectory_or_budget(domains, policy="fixed_per_domain"):
    """Resolve overlap between trajectory constraints and volume budgets.

    Returns a per-domain role array; the SAME material information may enter
    exactly one of the two losses (§6.8 anti-double-count rule).
    """
    roles = {}
    for domain_id, kinds in domains.items():
        kinds = set(kinds)
        if kinds == {"trajectory", "budget"}:
            roles[domain_id] = "budget" if policy == "fixed_per_domain" else "trajectory"
        elif kinds:
            roles[domain_id] = kinds.pop()
        else:
            roles[domain_id] = None
    return roles


def mask_unknown_flux(flux_known_mask, residual_active_mask):
    """Domains with any unknown boundary flux drop out of the hard budget set."""
    known = np.asarray(flux_known_mask, dtype=bool)
    active = np.asarray(residual_active_mask, dtype=bool)
    return active & known
