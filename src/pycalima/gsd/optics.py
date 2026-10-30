"""Per-gram optics of PL2M bins as functions of their in-bin state.

A bin's (N, M) fix its power-law profile (``pl2m``), so a grain-averaged
quantity per gram of the bin's dust depends on the cell only through the
normalised mean mass

    s = (r - 1) / (eta - 1),    r = M / (N m_L),

which runs from 0 (all grains at a_L) to 1 (all at a_R). For a per-gram
property k(a) = C(a) / m(a) of single grains, the bin value is the
mass-weighted mean over the profile,

    kappa(s) = int k(u) u^(lam + 3) du / int u^(lam + 3) du,   u = a / a_L,

with lam = 3 beta(s) + 2. With k linear in u between fixed nodes, this is exact
for any slope, however steep: ``profile_weights`` returns the matrix W(s) with
kappa(s) = W(s) @ k(u_nodes), the same for every quantity (C_abs, C_sca, C_rp,
a Planck mean at given T, ...). Bins beyond the slope cap are the spikes
``BinProfile`` makes of them, at u = r^(1/3).

RAMSES tabulates kappa per bin on that bin's own s grid and interpolates
ln kappa linearly in s per cell (``lookup``); the linear consumers (opacity,
shielding, radiation pressure) are then kappa(s_i) M_i, without the number
density. A uniform s grid does not work everywhere: in the smallest THEMIS
bin the near-IR kappa grows 300-fold between s = 0 and s = 1/16, because the
sub-nm a-C grains at the bin's lower edge have band gaps above 0.8 eV. The grids
are therefore refined by bisection (``refine_s``).
"""

from __future__ import annotations

import numpy as np

from .pl2m import beta_max, int_pow, mean_mass_ratio, solve_beta


def state_s(grid, N, M):
    """In-bin state s of every bin; empty bins get 0.5 (they carry no mass)."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    filled = N > 0
    eta = grid.etas
    r = np.where(filled, M / np.where(filled, N, 1.0) / grid.m_edges[:-1], 0.5 * (1 + eta))
    return np.clip((r - 1.0) / (eta - 1.0), 0.0, 1.0)


def node_u(eta, n_nodes):
    """Nodes u = a / a_L of k(u), uniform in u over the bin (n_nodes intervals)."""
    return np.linspace(1.0, eta ** (1.0 / 3.0), n_nodes + 1)


def _hat(u, ud):
    """Weights of linear interpolation at ud between the nodes u."""
    w = np.zeros(u.size)
    j = int(np.clip(np.searchsorted(u, ud, side="right") - 1, 0, u.size - 2))
    t = (ud - u[j]) / (u[j + 1] - u[j])
    w[j], w[j + 1] = 1.0 - t, t
    return w


def profile_weights(eta, s, u):
    """W[len(s), len(u)] with kappa(s) = W @ k(u), for k linear in u between
    the nodes ``u`` (from ``node_u``); rows sum to 1."""
    s = np.atleast_1d(np.asarray(s, float))
    r = 1.0 + s * (eta - 1.0)
    bm = beta_max(eta)
    r_lo, r_hi = mean_mass_ratio(np.array([-bm, bm]), eta)
    W = np.zeros((s.size, u.size))
    u0, u1, h = u[:-1], u[1:], np.diff(u)
    for n, rn in enumerate(r):
        if rn <= r_lo or rn >= r_hi:
            W[n] = _hat(u, np.clip(rn, 1.0, eta) ** (1.0 / 3.0))
            continue
        p = 3.0 * solve_beta(rn, eta) + 5.0          # mass weight u^(lam + 3)
        i0 = int_pow(p, u0, u1)
        i1 = int_pow(p + 1.0, u0, u1)
        left = (u1 * i0 - i1) / h                     # int (u1 - u)/h u^p du
        right = (i1 - u0 * i0) / h                    # int (u - u0)/h u^p du
        W[n, :-1] += left
        W[n, 1:] += right
        W[n] /= i0.sum()
    return W


def kappa_tables(grid, s, n_nodes, per_gram):
    """Per-gram bin tables kappa[bin, s, ...] on the s grid ``s``.

    ``per_gram(a_cm)`` returns k at radii ``a_cm`` (shape (len(a), ...)), per
    gram of grains of that radius; it is called once per bin, on that bin's
    nodes.
    """
    a_lo = grid.a_edges[:-1]
    out = []
    for i, eta in enumerate(grid.etas):
        u = node_u(eta, n_nodes)
        out.append(np.tensordot(profile_weights(eta, s, u), np.asarray(per_gram(a_lo[i] * u)),
                                axes=(1, 0)))
    return np.stack(out)


def lookup(s_tab, tab, s):
    """tab[s_tab, ...] at the points ``s``: linear in s on ln(tab), the
    interpolation RAMSES applies to the (log10) tables."""
    s = np.atleast_1d(np.asarray(s, float))
    j = np.clip(np.searchsorted(s_tab, s, side="right") - 1, 0, s_tab.size - 2)
    t = ((s - s_tab[j]) / (s_tab[j + 1] - s_tab[j])).reshape((-1,) + (1,) * (tab.ndim - 1))
    lt = np.log(np.maximum(tab, 1e-300))
    return np.exp((1.0 - t) * lt[j] + t * lt[j + 1])


def refine_s(kappa_of_s, tol, n_start=9, min_step=2.0 ** -16):
    """Bisect an s grid until ``lookup`` meets ``tol`` at every midpoint.

    ``kappa_of_s(s)`` returns positive values of shape (len(s), ...); an
    interval is split while ln-linear interpolation misses the value at its
    midpoint by more than ``tol`` (relative, over all trailing entries) and it
    is wider than ``min_step``. Returns (s, values at s).
    """
    s = np.linspace(0.0, 1.0, n_start)
    vals = np.asarray(kappa_of_s(s), float)
    while True:
        mid = 0.5 * (s[:-1] + s[1:])
        v_mid = np.asarray(kappa_of_s(mid), float)
        lv = np.log(np.maximum(vals, 1e-300))
        guess = np.exp(0.5 * (lv[:-1] + lv[1:]))
        err = np.abs(guess / np.maximum(v_mid, 1e-300) - 1.0).reshape(mid.size, -1).max(axis=1)
        split = (err > tol) & (np.diff(s) > min_step)
        if not split.any():
            return s, vals
        order = np.argsort(np.concatenate([s, mid[split]]))
        s = np.concatenate([s, mid[split]])[order]
        vals = np.concatenate([vals, v_mid[split]])[order]


def adaptive_kappa_tables(grid, n_nodes, per_gram, tol):
    """Per-bin tables on their own s grids: a list of (s, kappa[s, ...]),
    refined by ``refine_s`` to ``tol``."""
    a_lo = grid.a_edges[:-1]
    out = []
    for i, eta in enumerate(grid.etas):
        u = node_u(eta, n_nodes)
        K = np.asarray(per_gram(a_lo[i] * u))
        out.append(refine_s(lambda s, K=K, eta=eta, u=u: np.tensordot(profile_weights(eta, s, u), K,
                                                                      axes=(1, 0)), tol))
    return out
