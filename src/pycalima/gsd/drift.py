"""Per-moment dust drift (TVA): the drift of a bin's number and mass, and the pyCALIMA mirror of RAMSES's TVA step.

A bin's grains of radius a drift relative to the gas at w(a): the terminal velocity, e.g. proportional to a for
Epstein drag in a pressure gradient. If the whole bin moves at one velocity, N and M move together. The in-bin state
s then never changes, and the bin's big and small grains cannot separate.

Diallo et al. (2026, RAMSES-MCR, Sect. 2.2.5, after Hopkins 2023) give the number and energy moments of a cosmic-ray
momentum bin their own coefficients. For dust (decided 2026-11-17) the bin's moments move at
    w^N = int w dN / N,    w^M = int w m dN / M,
over the bin's PL2M power law dN/du ~ u^lambda, u = a / a_L, lambda = 3 beta(s) + 2 (``pl2m``).

Inside the bin, w(a) is a law through the drift at fixed sizes (``law_radii``; ``LAWS``, the argument ``law``):
- 'linear': piecewise linear in a through the bin's edges and its geometric centre (2 n_bins + 1 values per cell);
- 'quadratic': one quadratic in a through the same three;
- 'cubic': one cubic in a through the edges and the geometric thirds (3 n_bins + 1 values per cell).
All are exact for w ~ a (Epstein drag under one acceleration). Each is split where it changes sign, so it holds a
direction change inside the bin and a maximum inside it. With THEMIS radiation pressure against settling (D1 check
3), one power law through two sizes is off by up to 19 % of max|w| in w^N and w^M over the in-bin states; the
piecewise-linear law by 4.3 %, the quadratic by 1.4 % and the cubic by 0.08 % (the literature review in
CALIMA_GSD_dev, studies/drift_20261127/lit). The drift CFL is on the law's own maximum (``law_max_abs``), which for
the quadratic and the cubic can lie inside the bin, above every node.
A state closer to an edge than the power law can reach (|beta| > beta_max) is a spike at its mean mass (as
``pl2m.BinProfile``), and its grains all move at w there.
``gauss_velocities`` is the alternative: w at the two Gauss nodes in radius (``closures.nodes``, rule 'gauss'), with
no assumption on its shape.

``tva_step`` follows one direction of RAMSES's TVA step (``calima/dust_dynamics.f90``, ``calculate_pure_drag_fluxes``;
Lebreuilly et al. 2019) on a 1-D grid of cells, with each grain crossing a face at its own speed. Per bin:
1. TVD slopes of N and of M (minmod for slope_type 1, MC for 2);
2. each cell's two face states: N plus or minus half its slope, and M plus or minus half its slope clipped so that
   both faces' mean masses stay within the cell's and its neighbours' range (``_reconstruct``). The two face states
   then average to the cell in N and in M, and lie inside the bin (clamped into it against round-off);
3. the face's drift law: its node values, the mean of the two cells' (RAMSES's conservative choice: both grids
   sharing a face build the same flux);
4. the flux, grain by grain: the right-moving grains of the left cell and the left-moving grains of the right cell,
   each at its own w(a) with the MUSCL-Hancock predictor of linear advection, f_R - (c w / 2) (f_R - f_L) from the
   donor's face states f_L, f_R (dN/da of their PL2M profiles). In closed form this needs int w and int w^2 over each
   face state's profile, for dN and for m dN. A direction change inside the bin is split at the radius where w = 0;
5. positivity: no face takes more than 1/2 of a moment from its donor (1/(2 ndim) in 1-D). N and M of one packet are
   scaled alike;
6. the update, and a count of the cells whose mean mass left their bin (clipped, N moved to hold it at the edge). A
   cell whose M is below 1e-292 (tiny / eps: the step's products of it turn subnormal) is set to zero, as a Fortran
   build that flushes subnormals to zero: the bound test has no precision there.

For every grain size a, steps 2 and 4 write a cell's new population as its two face states and the inflowing
packets, each with a non-negative weight (1/2 - p + p^2/2 - q^2/2 for the face state that leaves at p = c w+, with
q = c |w-| at the other face). The weights are non-negative while c max|w| <= 1/2 over the law (``law_max_abs``):
the drift CFL on the fastest grains (w(a_R) for w ~ a, the A4 rule). The mean of two laws' node values is the law
through the mean nodes (its coefficients are linear in them), so the face's law is a law of the same kind. The new population is then non-negative
at every a: the cell stays realisable, with no clip, and N and M are conserved exactly.

``tva_fluxes`` returns the face fluxes of steps 1-5, for an AMR level's refluxing; ``tva_step`` adds the update.
``prolong`` and ``restrict`` are the AMR level operations ("parent s": the children keep their parent's in-bin state).

``tva_step_velocities`` is the same step with given cell velocities per moment (one per bin and cell, upwind by the
sign of each moment's face mean, Hancock per moment), for the alternatives (bin centre, Gauss nodes). Its packets
are not pieces of a profile, so it can leave the bin.

N and M are densities (per unit volume), as RAMSES's ``uold``; c = dt/dx, so a face flux is in density units.
"""
from __future__ import annotations

import itertools

import numpy as np

from .grid import MassGrid
from .pl2m import beta_max, int_pow, mean_mass_ratio, solve_beta

_M_FLUSH = np.finfo(float).tiny / np.finfo(float).eps


def _profile(grid, N, M, tables=None):
    """The PL2M profile of each bin state ([..., bin]): (lambda, spike, u_spike). lambda = 3 beta + 2 of its power law;
    with ``tables`` (one (q, beta) per bin, ``beta_tables``) beta is interpolated linearly in the log mean mass
    q = ln(m_bar/m_L)/ln(eta), as the RAMSES mirror does, else solved. A state closer to an edge than a power law with |beta| <= beta_max(eta) can reach is a spike:
    all its grains at u_spike = (m_bar / m_L)^(1/3), as ``pl2m.BinProfile``. Empty states get the mid-bin profile."""
    eta = grid.etas
    filled = N > 0
    r = np.where(filled, M / np.where(filled, N, 1.0) / grid.m_edges[:-1], 0.5 * (1.0 + eta))
    bm = beta_max(eta)
    r_lo, r_hi = mean_mass_ratio(-bm, eta), mean_mass_ratio(bm, eta)
    spike = filled & ((r <= r_lo) | (r >= r_hi))
    u_spike = np.clip(r, 1.0, eta) ** (1.0 / 3.0)
    r = np.clip(r, r_lo, r_hi)
    if tables is not None:
        q = np.log(r) / np.log(eta)
        beta = np.stack([np.interp(q[..., i], *tables[i]) for i in range(grid.n_bins)], axis=-1)
    else:
        beta = solve_beta(r.reshape(-1), np.broadcast_to(eta, r.shape).reshape(-1)).reshape(r.shape)
    return 3.0 * beta + 2.0, spike, u_spike


def _lambda(grid, N, M, tables=None):
    """lambda of each bin's PL2M power law from its mean mass ([..., bin]; ``_profile``)."""
    return _profile(grid, N, M, tables)[0]


LAWS = ("linear", "quadratic", "cubic")


def _law_t(law):
    """The law's nodes as fractions t of the bin in ln a (u = (a_R / a_L)^t), and its pieces (first node, last node):
    piecewise linear through the edges and the geometric centre; one quadratic through the same three; one cubic
    through the edges and the geometric thirds."""
    if law == "linear":
        return (0.0, 0.5, 1.0), ((0, 1), (1, 2))
    if law == "quadratic":
        return (0.0, 0.5, 1.0), ((0, 2),)
    if law == "cubic":
        return (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0), ((0, 3),)
    raise ValueError(f"unknown drift law {law!r} (one of {LAWS})")


def law_radii(grid: MassGrid, law="linear"):
    """[bin, K] radii [cm] where the drift law is sampled (``_law_t``): the bin's edges and its geometric centre
    (K = 3: 'linear', 'quadratic'), or the edges and its geometric thirds (K = 4: 'cubic'). The edges are shared with
    the neighbouring bins: a cell needs 2 n_bins + 1 values (3 n_bins + 1 for 'cubic')."""
    t, _ = _law_t(law)
    return grid.a_edges[:-1, None] * (grid.a_edges[1:] / grid.a_edges[:-1])[:, None] ** np.asarray(t)[None, :]


def _law_pieces(grid, w_nodes, law):
    """[(ua, ub, c)]: the law's polynomial pieces in u = a / a_L, each on [ua, ub] [bin] with coefficients
    c [..., bin, deg + 1] of u^0 .. u^deg (Lagrange through the piece's nodes: linear in w_nodes)."""
    t, pieces = _law_t(law)
    u = (grid.a_edges[1:] / grid.a_edges[:-1])[:, None] ** np.asarray(t)[None, :]       # [bin, K]
    out = []
    for j0, j1 in pieces:
        un = u[:, j0:j1 + 1]                                                              # [bin, deg + 1]
        V = un[:, :, None] ** np.arange(j1 - j0 + 1)[None, None, :]                        # Vandermonde [bin, k, j]
        Vinv = np.linalg.inv(V)                                                            # [bin, j, k]
        c = np.einsum("bjk,...bk->...bj", Vinv, w_nodes[..., j0:j1 + 1])
        out.append((un[:, 0], un[:, -1], c))
    return out


def _poly(c, u):
    """sum_j c[..., j] u^j (Horner)."""
    v = np.zeros(np.broadcast_shapes(c.shape[:-1], np.shape(u)))
    for j in range(c.shape[-1] - 1, -1, -1):
        v = v * u + c[..., j]
    return v


def _critical_points(c, ua, ub):
    """The polynomial's critical points inside (ua, ub), as two arrays (NaN where none): degree <= 3."""
    nan = np.full(c.shape[:-1], np.nan)
    deg = c.shape[-1] - 1
    if deg <= 1:
        return nan, nan
    if deg == 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(c[..., 2] != 0.0, -c[..., 1] / (2.0 * c[..., 2]), np.nan)
        return np.where((r > ua) & (r < ub), r, np.nan), nan
    a, b, cc = 3.0 * c[..., 3], 2.0 * c[..., 2], c[..., 1]                                # p' = a u^2 + b u + cc
    disc = b * b - 4.0 * a * cc
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
        lin = np.where(b != 0.0, -cc / np.where(b != 0.0, b, 1.0), np.nan)                 # a = 0: p' linear
        r1 = np.where(a != 0.0, (-b - sq) / (2.0 * a), lin)
        r2 = np.where(a != 0.0, (-b + sq) / (2.0 * a), np.nan)
    ok = lambda r: np.where((r > ua) & (r < ub), r, np.nan)                                # noqa: E731
    return ok(np.fmin(r1, r2)), ok(np.fmax(r1, r2))


def _sign_intervals(c, ua, ub):
    """The intervals of [ua, ub] on which the polynomial has one sign: [(u1, u2)], up to 6 (zero-length ones where
    none). Between its critical points it is monotone, and each monotone part holds at most one zero (bisection)."""
    r1, r2 = _critical_points(c, ua, ub)
    ua, ub = np.broadcast_to(ua, r1.shape), np.broadcast_to(ub, r1.shape)
    k1 = np.where(np.isnan(r1), ua, r1)
    k2 = np.where(np.isnan(r2), k1, r2)
    knots = [ua, k1, k2, ub]
    out = []
    for x0, x1 in itertools.pairwise(knots):
        p0, p1 = _poly(c, x0), _poly(c, x1)
        has = p0 * p1 < 0.0
        lo, hi = x0.copy(), x1.copy()
        for _ in range(60):                                                                 # bisection to round-off
            mid = 0.5 * (lo + hi)
            pm = _poly(c, mid)
            left = pm * p0 > 0.0
            lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
        root = np.where(has, 0.5 * (lo + hi), x1)
        out += [(x0, root), (root, x1)]
    return out


def law_values(grid: MassGrid, w_nodes, a, law="linear"):
    """The drift law (module doc) through w_nodes [..., bin, K] at ``law_radii``, evaluated at radii a [..., bin]."""
    w_nodes = np.asarray(w_nodes, float)
    u = np.asarray(a, float) / grid.a_edges[:-1]
    w = np.zeros(np.broadcast_shapes(w_nodes.shape[:-1], u.shape))
    for ua, ub, c in _law_pieces(grid, w_nodes, law):
        w = np.where((u >= ua * (1 - 1e-12)) & (u <= ub * (1 + 1e-12)), _poly(c, u), w)
    return w


def law_max_abs(grid: MassGrid, w_nodes, law="linear"):
    """max |w| of the law over each bin ([..., bin]): at its pieces' ends and critical points. The drift CFL is on it
    (c max|w| <= 1/2, module doc)."""
    w_nodes = np.asarray(w_nodes, float)
    m = np.zeros(w_nodes.shape[:-1])
    for ua, ub, c in _law_pieces(grid, w_nodes, law):
        for x in (ua, ub, *_critical_points(c, ua, ub)):
            v = np.abs(_poly(c, np.where(np.isnan(x), ua, x)))
            m = np.maximum(m, v)
    return m


def _flux_moments(prof, grid, w_nodes, law="linear"):
    """int w dN and int w^2 dN of a bin state's right-moving (w > 0) and left-moving (w < 0) grains, per unit N, and the
    same with m dN per unit M, for the law through w_nodes [..., bin, K] (module doc) over the state's profile
    (``_profile``): (P1N, P1M, P2N, P2M) for w > 0 and (Q1N, Q1M, Q2N, Q2M) for w < 0 (Q1 <= 0 <= Q2)."""
    lam, spike, u_spike = prof
    one = np.ones(np.shape(lam))
    uR = grid.a_edges[1:] / grid.a_edges[:-1]
    u_spike = np.clip(u_spike, 1.0, uR)                                # eta^(1/3) and a_R / a_L differ by round-off
    norm = (int_pow(lam, one, uR * one), int_pow(lam + 3.0, one, uR * one))
    P = [np.zeros(np.shape(lam)) for _ in range(4)]
    Q = [np.zeros(np.shape(lam)) for _ in range(4)]
    w_s = np.zeros(np.shape(lam))
    for ua, ub, c in _law_pieces(grid, np.broadcast_to(w_nodes, (*np.shape(lam), np.shape(w_nodes)[-1])), law):
        cc = np.zeros((*c.shape[:-1], 2 * c.shape[-1] - 1))            # the coefficients of w^2
        for i in range(c.shape[-1]):
            for j in range(c.shape[-1]):
                cc[..., i + j] += c[..., i] * c[..., j]
        for u1, u2 in _sign_intervals(c, ua * one, ub * one):
            sign = _poly(c, 0.5 * (u1 + u2))
            for m, p in enumerate((lam, lam + 3.0)):
                ints = [int_pow(p + j, u1, u2) for j in range(cc.shape[-1])]
                f1 = sum(c[..., j] * ints[j] for j in range(c.shape[-1])) / norm[m]
                f2 = sum(cc[..., j] * ints[j] for j in range(cc.shape[-1])) / norm[m]
                P[m] += np.where(sign > 0.0, np.maximum(f1, 0.0), 0.0)
                P[m + 2] += np.where(sign > 0.0, np.maximum(f2, 0.0), 0.0)
                Q[m] += np.where(sign < 0.0, np.minimum(f1, 0.0), 0.0)
                Q[m + 2] += np.where(sign < 0.0, np.maximum(f2, 0.0), 0.0)
        inside = (u_spike >= ua) & (u_spike <= ub)
        w_s = np.where(inside, _poly(c, u_spike), w_s)
    if np.any(spike):                                     # every grain at u_spike: the moments are powers of w there
        wp, wn = np.maximum(w_s, 0.0), np.minimum(w_s, 0.0)
        P = [np.where(spike, v, x) for v, x in zip((wp, wp, wp * wp, wp * wp), P, strict=True)]
        Q = [np.where(spike, v, x) for v, x in zip((wn, wn, wn * wn, wn * wn), Q, strict=True)]
    return tuple(P), tuple(Q)


def moment_velocities(grid: MassGrid, N, M, w_nodes, tables=None, law="linear"):
    """(w^N, w^M) of every bin ([..., bin]) for the drift w_nodes [..., bin, K] at each bin's ``law_radii(grid, law)``
    (e.g. one row per cell): the module doc's law over the bin's PL2M profile. ``tables``: the bins' beta(s) tables
    (``beta_tables``), as RAMSES interpolates them; None solves beta."""
    N, M = np.asarray(N, float), np.asarray(M, float)
    P, Q = _flux_moments(_profile(grid, N, M, tables), grid, np.asarray(w_nodes, float), law)
    return P[0] + Q[0], P[1] + Q[1]


def gauss_velocities(grid: MassGrid, N, M, w_of_a, method=None):
    """(w^N, w^M) per bin from w at the bin's two Gauss nodes in radius (``closures.nodes``, rule 'gauss'): exact in
    N for w up to cubic in a. ``w_of_a(a)`` is vectorised over radii [cm]. One cell's N, M [bin] at a time."""
    from .closures import nodes

    m, w, _, _ = nodes(grid, np.asarray(N, float), np.asarray(M, float), method, rule="gauss")
    v = np.asarray(w_of_a(grid.radius(m)), float)
    Nn = np.maximum(w.sum(axis=1), 1e-300)
    Mn = np.maximum((w * m).sum(axis=1), 1e-300)
    return (w * v).sum(axis=1) / Nn, (w * m * v).sum(axis=1) / Mn


def beta_table(eta, n=257):
    """(q, beta) on n points for a bin of mass ratio eta: beta(q) of the PL2M power law whose mean mass is
    m_L eta^q, for the RAMSES mirror (Diallo et al. tabulate q(e/n) likewise). q at Chebyshev points, denser at the
    edges, where beta varies fastest. The log mean mass, not the linear s = (m_bar/m_L - 1)/(eta - 1): beta + 3/2 is
    odd about q = 1/2 in every bin (dN/dm ~ m^beta maps to m^(-3-beta) under m -> eta/m), whereas in s the whole
    lower half of a wide bin falls inside the first interval (eta = 1.25e5: q < 0.149, where the s table put lambda
    at -28.6 for an exact -4.9; audit studies/drift_audit_20261127)."""
    q = 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / (n - 1)))
    r = np.clip(eta ** q, 1.0 + 1e-12, eta * (1.0 - 1e-12))
    return q, np.asarray(solve_beta(r, eta), float)


def beta_tables(grid: MassGrid, n=257):
    """``beta_table`` of every bin of ``grid`` (graded grids have one eta per bin)."""
    return [beta_table(float(e), n) for e in grid.etas]


def _limited_slope(q, slope_type, periodic):
    """TVD slope per cell [cell, ...] (RAMSES's: 0 none, 1 minmod, 2 MC; theta 1 or 2), on a periodic or zero-gradient
    grid."""
    if slope_type == 0:
        return np.zeros_like(q)
    if periodic:
        ql, qr = np.roll(q, 1, axis=0), np.roll(q, -1, axis=0)
    else:
        ql = np.concatenate([q[:1], q[:-1]], axis=0)
        qr = np.concatenate([q[1:], q[-1:]], axis=0)
    dl, dr = q - ql, qr - q
    dc = 0.5 * (dl + dr)
    theta = 2.0 if slope_type == 2 else 1.0
    lim = np.sign(dc) * np.minimum(theta * np.minimum(np.abs(dl), np.abs(dr)), np.abs(dc))
    return np.where(dl * dr <= 0.0, 0.0, lim)


def _left_cells(n, periodic):
    """The cell left of each face (face f between cells f-1 and f)."""
    return np.roll(np.arange(n), 1) if periodic else np.concatenate([[0], np.arange(n - 1)])


def _mean_mass_range(grid, N, M, periodic):
    """The range [lo, hi] of the mean masses M/N of each cell and its two neighbours [cell, bin], over the filled ones
    (the bin's edges for an empty cell; inside the bin)."""
    filled = N > 0
    mbar = np.where(filled, M / np.where(filled, N, 1.0), np.nan)
    if periodic:
        left, right = np.roll(mbar, 1, axis=0), np.roll(mbar, -1, axis=0)
    else:
        left = np.concatenate([mbar[:1], mbar[:-1]], axis=0)
        right = np.concatenate([mbar[1:], mbar[-1:]], axis=0)
    trio = np.stack([left, mbar, right])
    mL, mR = np.broadcast_to(grid.m_edges[:-1], N.shape), np.broadcast_to(grid.m_edges[1:], N.shape)
    lo = np.min(np.where(np.isnan(trio), np.inf, trio), axis=0)
    hi = np.max(np.where(np.isnan(trio), -np.inf, trio), axis=0)
    return np.where(filled, np.fmax(lo, mL), mL), np.where(filled, np.fmin(hi, mR), mR)


def _reconstruct(grid, N, M, slope_type, periodic):
    """Each cell's face states (step 2): ((N, M) at its left face, (N, M) at its right face). N takes its full limited
    slope. M's half-slope h_M is its own limited one, clipped so that both faces' mean masses stay within the range
    [lo, hi] of the cell's and its neighbours' (``_mean_mass_range``):
        max(lo N_R - M, M - hi N_L) <= h_M <= min(hi N_R - M, M - lo N_L).
    The interval is never empty, since N_L + N_R = 2N (a TVD slope keeps both faces' N >= 0) and lo <= M/N <= hi; for
    round-off crossings its midpoint is taken. A uniform mean mass gives h_M = m_bar h_N, and at an extremum of the
    mean mass only M's slope is clipped, so the faces stay second order. Bounded by the bin's edges only, a front
    carried the face's mean mass to the edge (the user's audit, studies/drift_audit_20261127, bug 2; this
    construction is the user's). The face states average to the cell in N and M, and are clamped into the bin against
    round-off."""
    dN, dM = _limited_slope(N, slope_type, periodic), _limited_slope(M, slope_type, periodic)
    mLb, mRb = grid.m_edges[:-1], grid.m_edges[1:]
    lo, hi = _mean_mass_range(grid, N, M, periodic)
    hN = 0.5 * dN
    NL, NR = N - hN, N + hN
    hmin = np.maximum(lo * NR - M, M - hi * NL)
    hmax = np.minimum(hi * NR - M, M - lo * NL)
    hM = np.where(hmin <= hmax, np.clip(0.5 * dM, hmin, hmax), 0.5 * (hmin + hmax))
    hM = np.where(N > 0, hM, 0.0)
    NL, NR = np.maximum(NL, 0.0), np.maximum(NR, 0.0)
    ML, MR = np.clip(M - hM, mLb * NL, mRb * NL), np.clip(M + hM, mLb * NR, mRb * NR)
    return (NL, ML), (NR, MR)


def _limit(N, M, FN_p, FM_p, FN_n, FM_n, idx_l, periodic):
    """Positivity per packet (step 5): the face fluxes FN, FM [face, bin]."""
    donL_N, donL_M = N[idx_l], M[idx_l]
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.minimum(np.where(FN_p > 0.5 * donL_N, 0.5 * donL_N / np.where(FN_p > 0, FN_p, 1.0), 1.0),
                        np.where(FM_p > 0.5 * donL_M, 0.5 * donL_M / np.where(FM_p > 0, FM_p, 1.0), 1.0))
        kn = np.minimum(np.where(-FN_n > 0.5 * N, 0.5 * N / np.where(FN_n < 0, -FN_n, 1.0), 1.0),
                        np.where(-FM_n > 0.5 * M, 0.5 * M / np.where(FM_n < 0, -FM_n, 1.0), 1.0))
    FN = kp * FN_p + kn * FN_n
    FM = kp * FM_p + kn * FM_n
    if not periodic:
        FN[0], FM[0] = 0.0, 0.0
    return FN, FM


def _update(grid, N, M, FN, FM, periodic):
    """The update from the face fluxes, and the realisability count (step 6)."""
    if periodic:                                          # N + (in - out): a uniform state stays exact
        N1, M1 = N + (FN - np.roll(FN, -1, axis=0)), M + (FM - np.roll(FM, -1, axis=0))
    else:
        zero = np.zeros_like(FN[:1])
        N1 = N + (FN - np.concatenate([FN[1:], zero], axis=0))
        M1 = M + (FM - np.concatenate([FM[1:], zero], axis=0))
    sub = M1 < _M_FLUSH                                   # near-subnormal mass: flushed to zero (module doc)
    N1, M1 = np.where(sub, 0.0, np.maximum(N1, 0.0)), np.where(sub, 0.0, M1)
    lo, hi = N1 * grid.m_edges[:-1], N1 * grid.m_edges[1:]
    clip = (M1 > 0) & ((M1 < lo * (1 - 1e-12)) | (M1 > hi * (1 + 1e-12)))
    N1 = np.where(clip, np.where(M1 < lo, M1 / grid.m_edges[:-1], M1 / grid.m_edges[1:]), N1)
    return N1, M1, int(clip.sum())


def tva_fluxes(grid: MassGrid, N, M, w_nodes, c, slope_type=1, periodic=True, tables=None, law="linear"):
    """The face fluxes (FN, FM) [face, bin] of one drift step of N, M [cell, bin] (module doc, steps 1-5), with each
    cell's drift w_nodes [cell, bin, K] at its bins' ``law_radii(grid, law)``, and c = dt/dx. Face f lies between cells
    f-1 and f, and a positive flux moves dust to larger f. Periodic: as many faces as cells. Otherwise the slopes have
    zero gradient at the ends, face 0 is closed and there is no face after the last cell."""
    N, M = np.asarray(N, float), np.asarray(M, float)
    w_nodes = np.broadcast_to(np.asarray(w_nodes, float), (*N.shape, len(_law_t(law)[0])))
    (NLs, MLs), (NRs, MRs) = _reconstruct(grid, N, M, slope_type, periodic)
    idx_l = _left_cells(N.shape[0], periodic)
    w_f = 0.5 * (w_nodes[idx_l] + w_nodes)
    pL, pR = _profile(grid, NLs, MLs, tables), _profile(grid, NRs, MRs, tables)
    P_R, _ = _flux_moments(tuple(q[idx_l] for q in pR), grid, w_f, law)   # the left cell's face states, w > 0
    P_L, _ = _flux_moments(tuple(q[idx_l] for q in pL), grid, w_f, law)
    _, Q_L = _flux_moments(pL, grid, w_f, law)                            # the right cell's face states, w < 0
    _, Q_R = _flux_moments(pR, grid, w_f, law)
    h = 0.5 * c
    FN_p = c * (NRs[idx_l] * (P_R[0] - h * P_R[2]) + h * NLs[idx_l] * P_L[2])
    FM_p = c * (MRs[idx_l] * (P_R[1] - h * P_R[3]) + h * MLs[idx_l] * P_L[3])
    FN_n = c * (NLs * (Q_L[0] + h * Q_L[2]) - h * NRs * Q_R[2])
    FM_n = c * (MLs * (Q_L[1] + h * Q_L[3]) - h * MRs * Q_R[3])
    return _limit(N, M, FN_p, FM_p, FN_n, FM_n, idx_l, periodic)


def tva_step(grid: MassGrid, N, M, w_nodes, c, slope_type=1, periodic=True, tables=None, law="linear"):
    """One drift step of N, M [cell, bin] (module doc, steps 1-6; arguments as ``tva_fluxes``).
    Returns (N, M, n_clipped)."""
    N, M = np.asarray(N, float), np.asarray(M, float)
    FN, FM = tva_fluxes(grid, N, M, w_nodes, c, slope_type, periodic, tables, law)
    return _update(grid, N, M, FN, FM, periodic)


def tva_step_velocities(grid: MassGrid, N, M, wN, wM, c, slope_type=1, periodic=True):
    """One drift step with given cell velocities per moment wN, wM [cell, bin]: the face states of ``tva_step``, and
    each moment upwind at the mean of its two cells' velocities, with the Hancock predictor. For the alternatives
    (bin centre, Gauss nodes); the packets' mean mass is not bounded here. Returns (N, M, n_clipped)."""
    N, M = np.asarray(N, float), np.asarray(M, float)
    wN, wM = np.asarray(wN, float), np.asarray(wM, float)
    (NLs, MLs), (NRs, MRs) = _reconstruct(grid, N, M, slope_type, periodic)
    idx_l = _left_cells(N.shape[0], periodic)
    out = []
    for w, qL, qR in ((wN, NLs, NRs), (wM, MLs, MRs)):
        wf = 0.5 * (w[idx_l] + w)
        right = qR[idx_l] - 0.5 * c * wf * (qR[idx_l] - qL[idx_l])     # the left cell's right face, predicted
        left = qL - 0.5 * c * wf * (qR - qL)                           # the right cell's left face, predicted
        out.append((c * np.maximum(wf, 0.0) * right, c * np.minimum(wf, 0.0) * left))
    FN, FM = _limit(N, M, out[0][0], out[1][0], out[0][1], out[1][1], idx_l, periodic)
    return _update(grid, N, M, FN, FM, periodic)


def prolong(N, M, dM=None):
    """AMR prolongation of a parent into its two children along one direction (left, right; axis 0), as RAMSES's
    interpol_hydro: M linear across the parent with the limited slope dM (per cell width; injection when None), and
    both children at the parent's in-bin state s, so N follows M. Conservative (the children's mean is the parent)
    and exactly realisable. A limited slope keeps both children non-negative."""
    N, M = np.asarray(N, float), np.asarray(M, float)
    if dM is None:
        return np.stack([N, N]), np.stack([M, M])
    Mc = np.stack([M - 0.25 * dM, M + 0.25 * dM])
    ratio = np.divide(N, M, out=np.zeros_like(N), where=M > 0)
    return Mc * ratio, Mc


def restrict(N_children, M_children):
    """AMR restriction (RAMSES's upload_fine): the parent's densities are the mean of its children's (axis 0).
    Conservative, and the parent's mean mass is a number-weighted mean of the children's, so it stays inside the
    bin."""
    return np.asarray(N_children, float).mean(axis=0), np.asarray(M_children, float).mean(axis=0)
