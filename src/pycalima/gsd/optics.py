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


def group_reducer(lam_um, groups_ev, source_t_k=(1e4, 3e4, 1e5), planck_t_k=np.geomspace(3.0, 3000.0, 12)):
    """The per-wavelength quantities RAMSES derives from a bin's table, as a function
    for the ``reduce`` argument of ``adaptive_kappa_tables_x``.

    ``groups_ev`` is a sequence of (E_lo, E_hi) [eV] photon groups, any number and
    limits. Each group mean is the photon-number-weighted mean of kappa over the
    group for blackbody sources of ``source_t_k`` (RAMSES averages over its own
    groups at start-up with the run's spectrum); the Planck means at ``planck_t_k``
    over all wavelengths stand for the dust emission and the IR group. Input k has
    shape (..., n_lam, n_q); output (..., n_groups * n_sources + n_planck, n_q).
    """
    lam = np.asarray(lam_um, float)
    hc_ev_um = 1.23984198
    dlam = np.gradient(lam)
    rows = []
    for t in source_t_k:
        x = hc_ev_um / lam / (8.617333262e-5 * t)
        n_lam = lam ** -4 / np.expm1(np.minimum(x, 700.0))          # photons per unit wavelength
        for e_lo, e_hi in groups_ev:
            band = (hc_ev_um / lam >= e_lo) & (hc_ev_um / lam < e_hi)
            if not band.any():                                       # narrower than the grid: nearest point
                band = np.zeros(lam.size, bool)
                band[np.argmin(abs(hc_ev_um / lam - 0.5 * (e_lo + e_hi)))] = True
            w = np.where(band, n_lam * dlam, 0.0)
            rows.append(w / w.sum())
    for t in planck_t_k:
        x = 14387.7688 / (lam * t)
        b = lam ** -5 / np.expm1(np.minimum(x, 700.0)) * dlam
        rows.append(b / b.sum())
    W = np.array(rows)

    def reduce(k):
        return np.einsum("rl,...lq->...rq", W, np.asarray(k, float))

    return reduce


def adaptive_kappa_tables_x(grid, n_nodes, per_gram_x, x_range, tol, n_x_start=5, n_s_probe=9,
                            min_dx=1e-4, reduce=None, x_start=None):
    """Per-bin tables on an s grid and a grid in a tracked composition x (the H/C of
    bins carrying M_H): a list of (s, x, kappa[s, x, ...]).

    ``per_gram_x(a_cm, x)`` returns k at radii ``a_cm`` for composition x;
    ``x_range(i)`` gives (x_lo, x_hi) of bin i. The x grid is bisected until ln kappa,
    linear in x, meets ``tol`` at every midpoint for ``n_s_probe`` states s; the s grid
    is the union of each x node's ``refine_s`` grid; ``x_start(i)``, if given, replaces the
    ``n_x_start`` uniform starting nodes (e.g. the band gaps of the optical tables). With ``reduce`` (e.g.
    ``group_reducer``) the tolerance applies to reduce(kappa), the group and Planck means
    RAMSES derives, while the tables keep every wavelength. RAMSES interpolates ln kappa
    bilinearly in (s, x) (``lookup_x``).
    """
    red = (lambda k: k) if reduce is None else reduce
    a_lo = grid.a_edges[:-1]
    s_probe = np.linspace(0.0, 1.0, n_s_probe)
    out = []
    for i, eta in enumerate(grid.etas):
        u = node_u(eta, n_nodes)
        a = a_lo[i] * u
        K = {}

        def k_of(x):
            if x not in K:
                K[x] = np.asarray(per_gram_x(a, x))
            return K[x]

        def kap(s, x):
            return np.tensordot(profile_weights(eta, s, u), k_of(x), axes=(1, 0))

        x = list(np.linspace(*x_range(i), n_x_start)) if x_start is None else list(x_start(i))
        while True:
            new = []
            for x0, x1 in zip(x[:-1], x[1:]):
                if x1 - x0 <= min_dx:
                    continue
                xm = 0.5 * (x0 + x1)
                r0, r1, rm = (red(kap(s_probe, xx)) for xx in (x0, x1, xm))
                guess = np.exp(0.5 * (np.log(np.maximum(r0, 1e-300)) + np.log(np.maximum(r1, 1e-300))))
                if np.max(np.abs(guess / np.maximum(rm, 1e-300) - 1.0)) > tol:
                    new.append(xm)
            if not new:
                break
            x = sorted(x + new)
        s = np.unique(np.concatenate([refine_s(lambda s, x0=x0: red(kap(s, x0)), tol)[0] for x0 in x]))
        out.append((s, np.array(x), np.stack([kap(s, x0) for x0 in x], axis=1)))
    return out


def lookup_x(s_tab, x_tab, tab, s, x):
    """tab[s_tab, x_tab, ...] at one point (s, x): bilinear in (s, x) on ln(tab)."""
    js = int(np.clip(np.searchsorted(s_tab, s, side="right") - 1, 0, s_tab.size - 2))
    jx = int(np.clip(np.searchsorted(x_tab, x, side="right") - 1, 0, x_tab.size - 2))
    ts = (s - s_tab[js]) / (s_tab[js + 1] - s_tab[js])
    tx = (x - x_tab[jx]) / (x_tab[jx + 1] - x_tab[jx])
    lt = np.log(np.maximum(tab[js:js + 2, jx:jx + 2], 1e-300))
    return np.exp((1 - ts) * ((1 - tx) * lt[0, 0] + tx * lt[0, 1]) + ts * ((1 - tx) * lt[1, 0] + tx * lt[1, 1]))
