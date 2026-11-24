"""Per-gram optics of the GSD bins as functions of their in-bin state.

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

The closure eaf (``closures``) blends four components per bin, each a
function of s alone: the PL2M profile above (base), the two fronts and a spike.
``component_kappa_tables`` tabulates them and ``combine_components`` weights them
with the fractions that a bin's neighbours fix (``closures.component_fractions``);
``kappa_direct`` integrates any method's pieces directly.

Node tables (``node_table``, ``node_average``) are the closure-independent alternative, and the way the default
closure, ea, is served: per bin, a per-gram quantity is stored at a few radii (uniform or geometric in u, the cheapest
layout per bin whose bin averages meet a tolerance over power-law profiles and truncated fronts of every slope), and
a cell's per-gram value is the exact integral of that piecewise-linear k(u) over the cell's own reconstruction. For
the A2 carbon grid this costs about as many stored values as the PL2M s tables, and it serves pl2m, ea, eaf, affine
and exp alike (``diagnostics/gsd/check_node_tables.py``).
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


def profile_weights(eta, s, u, method=None):
    """W[len(s), len(u)] with kappa(s) = W @ k(u), for k linear in u between
    the nodes ``u`` (from ``node_u``); rows sum to 1.

    For pl2m, ea and eaf (``closures``) this is the PL2M profile, the base
    component of ea/eaf (their front components: ``component_weights``); for
    the local closures affine and exp, their own profile."""
    from .closures import local_pieces, resolve, state_weights

    meth = resolve(method)
    if meth.closure in ("affine", "exp"):
        s = np.atleast_1d(np.asarray(s, float))
        return state_weights(eta, s, u, *local_pieces(eta, s, meth))
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


def kappa_tables(grid, s, n_nodes, per_gram, method=None):
    """Per-gram bin tables kappa[bin, s, ...] on the s grid ``s``.

    ``per_gram(a_cm)`` returns k at radii ``a_cm`` (shape (len(a), ...)), per
    gram of grains of that radius; it is called once per bin, on that bin's
    nodes. The profile is ``profile_weights``'s for ``method`` (the base
    component for ea/eaf; the front components: ``component_kappa_tables``).
    """
    a_lo = grid.a_edges[:-1]
    out = []
    for i, eta in enumerate(grid.etas):
        u = node_u(eta, n_nodes)
        out.append(np.tensordot(profile_weights(eta, s, u, method), np.asarray(per_gram(a_lo[i] * u)),
                                axes=(1, 0)))
    return np.stack(out)


def component_weights(eta, s, u, component, method=None):
    """W[len(s), len(u)] of one component of the eaf closure ('base' = PL2M, 'right' = u^front_exponent on
    [1, u_c], 'left' = on [u_c, u_R], 'spike' = all grains at the mean mass) at in-bin states s: every component is
    a function of s alone, so RAMSES can tabulate it like the PL2M profile. A front that cannot reach s is the base
    profile there, as in the closure."""
    from .closures import component_pieces, resolve, state_weights

    meth = resolve(method)
    if component == "base":
        return profile_weights(eta, s, u, meth)
    s = np.atleast_1d(np.asarray(s, float))
    return state_weights(eta, s, u, *component_pieces(eta, s, component, meth))


def component_kappa_tables(grid, s, n_nodes, per_gram, method=None):
    """Per-gram tables of every eaf component: {component: kappa[bin, s, ...]} on the s grid ``s``
    (``component_weights``). A bin's value is ``combine_components`` of these with the bin's component
    fractions (``closures.component_fractions``, from its neighbours' N)."""
    from .closures import COMPONENTS

    a_lo = grid.a_edges[:-1]
    out = {c: [] for c in COMPONENTS}
    for i, eta in enumerate(grid.etas):
        u = node_u(eta, n_nodes)
        K = np.asarray(per_gram(a_lo[i] * u))
        for c in COMPONENTS:
            out[c].append(np.tensordot(component_weights(eta, s, u, c, method), K, axes=(1, 0)))
    return {c: np.stack(v) for c, v in out.items()}


def combine_components(tables, s_tab, s, fractions):
    """kappa per bin for in-bin states s[nb] and component fractions[nb, 4] (``closures.component_fractions``):
    sum_c f_c kappa_c(s), each component looked up with ``lookup`` (every component carries the bin's mean mass,
    so number and mass fractions coincide)."""
    from .closures import COMPONENTS

    s = np.atleast_1d(np.asarray(s, float))
    out = None
    for ci, c in enumerate(COMPONENTS):
        vals = np.stack([lookup(s_tab, tables[c][i], s[i:i + 1])[0] for i in range(s.size)])
        term = np.asarray(fractions)[:, ci].reshape((-1,) + (1,) * (vals.ndim - 1)) * vals
        out = term if out is None else out + term
    return out


def kappa_direct(grid, N, M, n_nodes, per_gram, method=None):
    """Per-gram kappa[bin, ...] of the state (N, M) by direct integration over each bin's pieces under any method
    (``closures.pieces``; ea needs this, its front exponents vary per cell). Empty bins get the base profile at
    s = 1/2 (they carry no mass)."""
    from .closures import bin_weights, pieces

    N = np.asarray(N, float)
    M = np.asarray(M, float)
    P = pieces(grid, N, M, method)                 # the true state: empty bins stay empty for their neighbours
    a_lo = grid.a_edges[:-1]
    rows = []
    for i, eta in enumerate(grid.etas):
        u = node_u(eta, n_nodes)
        if N[i] > 0:
            sub = _one_bin(P, i)
            W = bin_weights(sub.grid, sub, u[None, :], "mass")[0]
        else:
            W = profile_weights(eta, [0.5], u, "pl2m")[0]
        rows.append(np.tensordot(W, np.asarray(per_gram(a_lo[i] * u)), axes=(0, 0)))
    return np.stack(rows)


def _one_bin(P, i):
    """Bin i's pieces as a one-bin Pieces (for per-bin weights on that bin's own nodes)."""
    from .closures import Pieces
    from .grid import MassGrid

    g1 = MassGrid(P.grid.m_edges[i:i + 2].copy(), P.grid.rho)
    return Pieces(g1, P.n[i:i + 1], P.A[i:i + 1], P.lam[i:i + 1], P.u1[i:i + 1], P.u2[i:i + 1],
                  P.spike[i:i + 1])


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


# ---------------------------------------------------------------------------------------------------------------------
# RAMSES's RT optics: pseudo-bins (decided 2026-11-18)
# ---------------------------------------------------------------------------------------------------------------------
# The RT cross-sections of the cool step ignore EA's fronts: every GSD bin is exported as n pseudo-bins, the PL2M
# per-gram values at fixed in-bin states, which RAMSES treats as ordinary dust bins (SED table, star polling, IR
# means). A cell blends its bin's two bracketing pseudo-bins linearly in s. That is exact bookkeeping because the
# group average, the star weighting and the blend are all linear in the cross-section. With 5 pseudo-bins the A0 group
# totals are within 4-6 % of the exact EA average (diagnostics/gsd/check_rt_group_optics.py).

def pseudo_states(n=5):
    """The in-bin states of the n pseudo-bins, s_k = (1 - cos(pi k / (n - 1))) / 2: 0, 0.15, 0.5, 0.85, 1 for n = 5."""
    return (1.0 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2.0


def pseudo_bin_tables(grid, per_gram, n=5, n_nodes=64, gap_nodes=None):
    """Per-gram values of every bin's pseudo-bins, [bin, n, ...]: the PL2M profile at ``pseudo_states(n)``
    (``kappa_tables``; ``per_gram(a_cm)`` as there). With ``gap_nodes`` (band gaps [eV] of the covalent bins),
    ``per_gram(a_cm, e_gap)`` and [bin, n, n_gap, ...]."""
    s = pseudo_states(n)
    if gap_nodes is None:
        return kappa_tables(grid, s, n_nodes, per_gram, "pl2m")
    return np.stack([kappa_tables(grid, s, n_nodes, lambda a, e=e: per_gram(a, e), "pl2m") for e in gap_nodes],
                    axis=2)


def pseudo_blend(s, n=5):
    """Bracket k and weight t of in-bin states s between n pseudo-bins: value = (1 - t) P[k] + t P[k + 1]."""
    sk = pseudo_states(n)
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    k = np.clip(np.searchsorted(sk, s, side="right") - 1, 0, n - 2)
    return k, (s - sk[k]) / (sk[k + 1] - sk[k])


def pseudo_lookup(tables, s, gap=None, gap_nodes=None):
    """Per-gram value of every bin at its in-bin state s[bin], from ``pseudo_bin_tables``: linear in s between the
    pseudo-bins, as the RAMSES cool step blends them. With ``gap`` (the bins' band gaps [eV]) and ``gap_nodes``,
    tables [bin, n, n_gap, ...] and also linear in the band gap. Returns [bin, ...]."""
    nb, n = tables.shape[:2]
    i = np.arange(nb)
    k, t = pseudo_blend(np.broadcast_to(s, (nb,)), n)
    if gap is None:
        sh = (nb,) + (1,) * (tables.ndim - 2)
        return (1.0 - t).reshape(sh) * tables[i, k] + t.reshape(sh) * tables[i, k + 1]
    g = np.asarray(gap_nodes, float)
    x = np.clip(np.broadcast_to(np.asarray(gap, float), (nb,)), g[0], g[-1])
    j = np.clip(np.searchsorted(g, x, side="right") - 1, 0, g.size - 2)
    v = (x - g[j]) / (g[j + 1] - g[j])
    sh = (nb,) + (1,) * (tables.ndim - 3)
    t, v = t.reshape(sh), v.reshape(sh)
    return ((1.0 - t) * ((1.0 - v) * tables[i, k, j] + v * tables[i, k, j + 1])
            + t * ((1.0 - v) * tables[i, k + 1, j] + v * tables[i, k + 1, j + 1]))


# ---------------------------------------------------------------------------------------------------------------------
# per-radius node tables: one table per bin for any closure
# ---------------------------------------------------------------------------------------------------------------------

NODE_COUNTS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
FRONT_EXPONENTS = (-6.0, -4.5, -3.5, -2.5, -1.5, 0.0, 1.0, 2.0, 3.0)


class NodeTable:
    """A per-gram quantity at a few radii per bin: ``u[i]`` (u = a / a_L,i, from 1 to u_R) and ``K[i]``
    [len(u[i]), ...]. ``node_average`` integrates it over any closure's reconstruction."""

    def __init__(self, grid, u, K, layout):
        self.grid = grid
        self.u = [np.asarray(x, float) for x in u]
        self.K = [np.asarray(x, float) for x in K]
        self.layout = list(layout)

    @property
    def sizes(self):
        return [x.size for x in self.u]


def probe_profiles(eta, n_s=33, exponents=FRONT_EXPONENTS, n_cut=9):
    """One-bin profiles any closure builds, as pieces (n, lam, u1, u2, spike) of shape [n_profiles, 1]: the PL2M
    power law at ``n_s`` in-bin states (spikes at the extremes), and truncated power laws u^lam (``exponents``) on
    [1, u_c] and [u_c, u_R] with u_c at ``n_cut`` positions (the fronts of ea/eaf)."""
    from .closures import component_pieces

    s = np.linspace(0.0, 1.0, n_s)
    n0, l0, a0, b0, sp0 = component_pieces(eta, s, "base", "pl2m")
    uR = eta ** (1.0 / 3.0)
    cuts = uR ** np.linspace(0.1, 0.9, n_cut)
    rows = []
    for lam in exponents:
        for uc in cuts:
            rows.append((lam, 1.0, uc))
            rows.append((lam, uc, uR))
    lam = np.array([r[0] for r in rows])[:, None]
    u1 = np.array([r[1] for r in rows])[:, None]
    u2 = np.array([r[2] for r in rows])[:, None]
    n = np.ones_like(lam)
    sp = np.zeros(lam.shape, bool)
    cat = lambda x, y: np.concatenate([np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1)])   # noqa: E731
    return (cat(n0, n), cat(l0, lam), cat(a0, u1), cat(b0, u2), cat(sp0, sp))


def _profile_weights_over(u, prof):
    """W[n_profiles, len(u)] of the one-bin test profiles ``prof`` (``probe_profiles``) over nodes u (mass measure)."""
    from .closures import _piece_weights

    n, lam, u1, u2, sp = prof
    U = np.ascontiguousarray(np.repeat(np.asarray(u, float)[None, :], n.shape[0], axis=0))
    out = np.zeros(U.shape)
    _piece_weights(U, np.ascontiguousarray(lam, float), np.ascontiguousarray(u1, float),
                   np.ascontiguousarray(u2, float), np.ascontiguousarray(sp, np.bool_),
                   np.ascontiguousarray(n, float), 3.0, out)
    return out


def node_table(grid, per_gram, tol=3e-3, counts=NODE_COUNTS, n_ref=512, layouts=("geometric", "uniform")):
    """The cheapest per-radius node table per bin: for each bin, the first node count of ``counts`` (and the first
    of ``layouts``: nodes geometric or uniform in u) whose averages over ``probe_profiles`` match those of ``n_ref``
    uniform nodes within ``tol`` (relative, over every trailing entry of ``per_gram``). ``per_gram(a_cm)`` returns k
    at radii a_cm, shape (len(a), ...). Bins that never meet tol get the reference nodes."""
    a_lo = grid.a_edges[:-1]
    U, K, lay = [], [], []
    for i, eta in enumerate(grid.etas):
        prof = probe_profiles(eta)
        u_ref = node_u(eta, n_ref)
        ref = np.tensordot(_profile_weights_over(u_ref, prof), np.asarray(per_gram(a_lo[i] * u_ref)), axes=(1, 0))
        uR = eta ** (1.0 / 3.0)
        chosen = None
        for n in counts:
            for name in layouts:
                u = np.geomspace(1.0, uR, n + 1) if name == "geometric" else node_u(eta, n)
                k = np.asarray(per_gram(a_lo[i] * u))
                got = np.tensordot(_profile_weights_over(u, prof), k, axes=(1, 0))
                if np.max(np.abs(got / ref - 1.0)) <= tol:
                    chosen = (u, k, f"{name} {n}")
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = (u_ref, np.asarray(per_gram(a_lo[i] * u_ref)), f"uniform {n_ref}")
        U.append(chosen[0])
        K.append(chosen[1])
        lay.append(chosen[2])
    return NodeTable(grid, U, K, lay)


def node_average(grid, N, M, table, method=None):
    """Per-gram bin values [nb, ...] of the state (N, M) under ``method``'s closure, from a ``NodeTable``: the exact
    integral of k (linear in u between the table's nodes) over each bin's pieces. Empty bins get the PL2M profile at
    s = 1/2 (they carry no mass)."""
    from .closures import bin_weights, pieces

    N = np.asarray(N, float)
    P = pieces(grid, N, np.asarray(M, float), method)
    rows = []
    for i, eta in enumerate(grid.etas):
        u = table.u[i]
        if N[i] > 0:
            W = bin_weights(_one_bin(P, i).grid, _one_bin(P, i), u[None, :], "mass")[0]
        else:
            W = profile_weights(eta, [0.5], u, "pl2m")[0]
        rows.append(np.tensordot(W, table.K[i], axes=(0, 0)))
    return np.stack(rows)
