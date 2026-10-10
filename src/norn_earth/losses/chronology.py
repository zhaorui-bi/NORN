"""Chronology-aware observation likelihood (§5.2/§5.5/§5.6).

One record contributes evidence exactly once:

    NLL_i = -logsumexp_k [ log w_ik + log p(y_i | H(x_i(a_ik), a_ik)) ],

with sum_k w_ik = 1. Adding quadrature nodes improves numerics only; it never
increases the record's total evidence. The mixture NLL is NOT the expected
point loss (Jensen); both objectives exist and are compared as distinct
variants in ablations. Age-node/trajectory masks may never depend on y or on
test errors.
"""

import numpy as np

DEFAULT_NU = 4.0


def student_t_logpdf(y, mu, scale, nu=DEFAULT_NU):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("scale must be positive; free-noise inflation is forbidden")
    from math import lgamma, log, pi

    z = (y - mu) / scale
    const = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi)
    return const - np.log(scale) - (nu + 1) / 2 * np.log1p(z**2 / nu)


def make_age_nodes(age_lower, age_upper, k, domain=(0.0, 60.0), distribution="uniform"):
    """Equal-mass stratified nodes on [max(lo,d0), min(hi,d1)].

    Returns nodes, weights (sum=1), clipped_mass c_i (mass of p(a) inside the
    reconstruction domain; conditioning on the domain must be recorded).
    Point ages (lo==hi) get a single node at the true age, never at TIME.
    """
    lo, hi = float(age_lower), float(age_upper)
    d0, d1 = domain
    if hi < lo:
        lo, hi = hi, lo
    if hi < d0 or lo > d1:
        return np.array([]), np.array([]), 0.0
    if lo == hi:
        inside = d0 - 1e-12 <= lo <= d1 + 1e-12
        if not inside:
            return np.array([]), np.array([]), 0.0
        return np.array([lo]), np.array([1.0]), 1.0
    a, b = max(lo, d0), min(hi, d1)
    # A continuous age interval touching only a domain endpoint has zero
    # probability mass inside the domain, not a point-age observation.
    if b <= a:
        return np.array([]), np.array([]), 0.0
    if distribution == "uniform":
        total = hi - lo
        c = (b - a) / total
        edges = np.linspace(a, b, int(k) + 1)
        nodes = 0.5 * (edges[:-1] + edges[1:])
        weights = np.diff(edges) / max(b - a, 1e-12)
    elif distribution == "triangular":
        m = 0.5 * (lo + hi)
        xs = np.linspace(a, b, 4001)
        dens = np.where(xs <= m, (xs - lo) / max(m - lo, 1e-12), (hi - xs) / max(hi - m, 1e-12))
        dens = np.clip(dens, 0.0, None)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xs))])
        total_mass = cdf[-1]
        c = total_mass
        qs = np.linspace(0, total_mass, int(k) + 1)
        nodes = np.interp(qs, cdf, xs)[1:-1]
        edges_v = np.interp(qs, cdf, xs)
        nodes = 0.5 * (edges_v[:-1] + edges_v[1:])
        weights = np.diff(qs) / max(total_mass, 1e-12)
    else:
        raise ValueError(distribution)
    weights = weights / weights.sum()
    return nodes, weights, float(c)


def mixture_nll(y, mu_nodes, weights, scale, nu=DEFAULT_NU, return_grad=False):
    """Per-record mixture NLL (mean over records when arrays are 2-D).

    y: (R,) targets; mu_nodes: (R,K); weights: (R,K) rows sum to 1;
    scale: (R,) effective scale sqrt(s_i^2 + s_source^2).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu_nodes, dtype=float)
    w = np.asarray(weights, dtype=float)
    sc = np.asarray(scale, dtype=float)
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("age weights must sum to one (one record, one evidence unit)")
    logp = np.log(np.clip(w, 1e-300, None)) + student_t_logpdf(y[:, None], mu, sc[:, None], nu)
    m = logp.max(axis=1)
    nll = -(m + np.log(np.exp(logp - m[:, None]).sum(axis=1)))
    if not return_grad:
        return nll
    resp = np.exp(logp - (-nll)[:, None])  # responsibilities p(k | y)
    dmu = resp * (nu + 1) * (mu - y[:, None]) / (nu * sc[:, None] ** 2 + (mu - y[:, None]) ** 2)
    return nll, dmu


def replication_invariance_error(y, mu, weights, scale, nu=DEFAULT_NU):
    """Max |dNLL| and gradient-folding error when nodes are duplicated 1x->5x/20x."""
    base_nll, base_grad = mixture_nll(y, mu, weights, scale, nu, return_grad=True)
    worst = 0.0
    for factor in (5, 20):
        mu2 = np.repeat(mu, factor, axis=1)
        w2 = np.repeat(weights / factor, factor, axis=1)
        nll2, grad2 = mixture_nll(y, mu2, w2, scale, nu, return_grad=True)
        folded = grad2.reshape(len(y), mu.shape[1], factor).sum(axis=2)
        worst = max(
            worst,
            float(np.max(np.abs(nll2 - base_nll))),
            float(np.max(np.abs(folded - base_grad))),
        )
    return worst


def jensen_gap(y, mu, weights, scale, nu=DEFAULT_NU):
    """mixture_NLL - (-E_w[log p]) <= 0 by Jensen: log E[p] >= E[log p].

    The mixture NLL is the smaller (optimistic) objective; the two are
    distinct targets and are compared explicitly in ablations. Never average
    one and report it as the other.
    """
    mix = mixture_nll(y, mu, weights, scale, nu)
    point = -np.sum(weights * student_t_logpdf(y[:, None], mu, scale[:, None], nu), axis=1)
    return float(np.mean(mix - point))
