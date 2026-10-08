"""Variational reconstruction basis (B7 machinery): real Y_lm x time anchors.

Complete orthonormal real spherical-harmonic basis (cos+sin) on the Gauss
grid with quadrature weights, plus one coefficient set per 1-Myr time anchor.
Partition-of-unity time queries are provided for callers. Fitting coefficients
to REAL data is model training and is gated; this module is linear algebra.
"""

import numpy as np
from math import factorial


def associated_legendre(lmax, lat_deg):
    """P_lm(sin lat), Condon-Shortley, standard unnormalized recursion.

    P_m^m = -(2m-1) s P_{m-1}^{m-1}; P_{m+1}^m = x(2m+1) P_m^m;
    P_l^m = ((2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m) / (l-m).
    Returns (n_lat, l+1, m+1), m <= l.
    """
    x = np.sin(np.radians(np.asarray(lat_deg, dtype=float)))
    P = np.zeros((len(x), lmax + 1, lmax + 1))
    P[:, 0, 0] = 1.0
    s = np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0))
    for degree in range(1, lmax + 1):
        P[:, degree, 0] = (
            (2 * degree - 1) * x * P[:, degree - 1, 0] - (degree - 1) * P[:, degree - 2, 0]
        ) / degree
    P[:, 1, 1] = -s
    for m in range(2, lmax + 1):
        P[:, m, m] = -(2 * m - 1) * s * P[:, m - 1, m - 1]
    for m in range(1, lmax + 1):
        if m + 1 <= lmax:
            P[:, m + 1, m] = x * (2 * m + 1) * P[:, m, m]
        for degree in range(m + 2, lmax + 1):
            P[:, degree, m] = (
                (2 * degree - 1) * x * P[:, degree - 1, m] - (degree + m - 1) * P[:, degree - 2, m]
            ) / (degree - m)
    return P


def _norm(degree, m):
    """N_lm = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!); m>0 real pairs get sqrt(2)
    in evaluate_basis (standard orthonormal real SH convention)."""
    return float(
        np.sqrt((2 * degree + 1) / (4 * np.pi) * (factorial(degree - m) / factorial(degree + m)))
    )


def basis_terms(lmax):
    """Ordered [(l, m, kind)] with kind in {'cos','sin'}; (lmax+1)^2 terms."""
    terms = [(0, 0, "cos")]
    for degree in range(1, lmax + 1):
        terms.append((degree, 0, "cos"))
        for m in range(1, degree + 1):
            terms.append((degree, m, "cos"))
            terms.append((degree, m, "sin"))
    return terms


def coefficient_count(lmax):
    return (lmax + 1) ** 2


def evaluate_basis(lmax, lats, lons):
    """Real Y_lm on a lat x lon grid; returns (nlat*nlon, n_coef).

    lats/lons are 1-D degree arrays; row-major over (lat, lon).
    """
    P = associated_legendre(lmax, lats)  # (nlat, l+1, m+1)
    lon_r = np.radians(np.asarray(lons, dtype=float))
    cols = []
    for degree, m, kind in basis_terms(lmax):
        nrm = _norm(degree, m) * (np.sqrt(2.0) if m > 0 else 1.0)
        ang = np.cos(m * lon_r) if kind == "cos" else np.sin(m * lon_r)
        cols.append((nrm * P[:, degree, m])[:, None] * ang[None, :])  # (nlat, nlon)
    return np.stack([c.ravel() for c in cols], axis=1)


def evaluate_basis_points(lmax, lats, lons):
    """Real Y_lm evaluated at PAIRED points; returns (n_points, n_coef)."""
    P = associated_legendre(lmax, lats)  # (n, l+1, m+1)
    lon_r = np.radians(np.asarray(lons, dtype=float))
    cols = []
    for degree, m, kind in basis_terms(lmax):
        nrm = _norm(degree, m) * (np.sqrt(2.0) if m > 0 else 1.0)
        ang = np.cos(m * lon_r) if kind == "cos" else np.sin(m * lon_r)
        cols.append(nrm * P[:, degree, m] * ang)
    return np.stack(cols, axis=1)


def design_matrix(lmax, lats, lons, quad_weights):
    """Sqrt-quadrature-weighted basis on a Gauss grid for LSQ analysis."""
    lats = np.atleast_1d(lats)
    lons = np.atleast_1d(lons)
    Y = evaluate_basis(lmax, lats, lons)
    nlon = len(lons)
    w = np.repeat(np.asarray(quad_weights, dtype=float)[:, None], nlon, axis=1).ravel()
    w = w / w.sum() * 4 * np.pi
    return Y * np.sqrt(w)[:, None]


def analyze(field, lmax, lats, lons, quad_weights):
    """Quadrature LSQ analysis: solves min || sqrt(w) (Y c - f) ||^2."""
    Phi = design_matrix(lmax, lats, lons, quad_weights)
    nlon = len(np.atleast_1d(lons))
    w = np.repeat(np.asarray(quad_weights, dtype=float)[:, None], nlon, axis=1).ravel()
    w = w / w.sum() * 4 * np.pi
    f = np.asarray(field, dtype=float).ravel() * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Phi, f, rcond=None)
    return coef


def synthesize(coef, lmax, lats, lons):
    lats = np.atleast_1d(lats)
    lons = np.atleast_1d(lons)
    Y = evaluate_basis(lmax, lats, lons)
    return (Y @ np.asarray(coef, dtype=float)).reshape(len(lats), len(lons))


def poisson_solve(divergence, lmax, lats, lons, quad_weights):
    """Solve Delta_s psi = D - mean(D) spectrally (l>=1 only).

    Returns psi coefficients; used ONLY for manufactured solutions and
    diagnostics, never to invent kinematics for real reconstructions.
    """
    coef = analyze(divergence, lmax, lats, lons, quad_weights)
    terms = basis_terms(lmax)
    out = np.zeros_like(coef)
    for i, (degree, m, kind) in enumerate(terms):
        if degree == 0:
            continue
        out[i] = coef[i] / (-(degree * (degree + 1)))
    return out


def time_anchored_coefficients(n_anchors, lmax):
    return np.zeros((n_anchors, coefficient_count(lmax)))


def anchor_interpolation_weights(age, anchor_step=1.0, n_anchors=61):
    """Nonnegative partition-of-unity weights over adjacent time anchors.

    Actual observation ages are never rounded; the MODEL history is finitely
    parameterized at anchors and queried with these weights. Topology events
    must add anchors or mask support (one-sided at domain ends).
    """
    x = float(age) / float(anchor_step)
    i0 = int(np.floor(x))
    f = x - i0
    w = np.zeros(n_anchors)
    if i0 + 1 < n_anchors:
        w[i0], w[i0 + 1] = 1.0 - f, f
    elif 0 <= i0 < n_anchors:
        w[i0] = 1.0
    else:
        raise ValueError("age outside anchor range; extend anchors or mask support")
    if w.sum() > 0:
        w = w / w.sum()
    return w
