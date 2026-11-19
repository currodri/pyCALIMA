"""Experimental variants of the PL2M grain-size closure (two scalars per bin, N and M), for benchmarking only.

Every variant stores exactly what PL2M stores, (N_i, M_i) per bin, and differs only in how the in-bin number
distribution is reconstructed from them (the closure) or in how the collision operator samples it (the nodes). A
closure returns, per bin, a few *pieces*: power laws dN/du = A u^lam on a support [u1, u2] inside the bin
(u = a / a_L), or spikes (all of a piece's number at one radius). Pieces of one bin sum exactly to the bin's N and M,
so the analytic remap of ``gsd._kernels`` (number along characteristics, image mass integrated exactly) applies to
every closure with one kernel (``remap_pieces``), and conservation is that of PL2M by construction.

Closures (``make_closure``):

* ``pl2m``: the production closure (``gsd.pl2m.BinProfile``), one power law over the whole bin, or a spike.
* ``tc:L``: truncated-cap power law, local. As PL2M while the radius exponent lam = 3 beta + 2 lies in [-L, L];
  beyond, lam stays at the bound and the support is cut instead ([1, u_c] or [u_c, u_R], with u_c from M/N). A
  function of the in-bin state s only, so optics/rate tables in s keep working. L = inf is PL2M.
* ``ea:eps:L``: edge-aware, non-local. A bin whose right (left) neighbour is empty -- mean dn/da below eps times its
  own -- is a front bin: its profile is the neighbour-side bin's exponent continued and truncated ([1, u_c] or
  [u_c, u_R]); a bin with both neighbours empty is a spike at its mean mass (moving pivot). Weights e = max(0, 1 -
  q / eps) blend the components continuously (a convex mix of profiles with the same N and M keeps N and M); the
  fallback, and the source of the neighbour exponents, is ``tc:L``. The grid ends count as empty neighbours.
* ``aff``: affine in a (dn/da = c0 + c1 a), switched to a triangle with partial support where a positive line cannot
  reach M/N (local; in the spirit of the two-size-moment method of Laurent et al. 2016, whose realisability
  treatment was not read: this is our own design).
* ``exp``: dN/dm ~ exp(-kappa m), the maximum-entropy profile for given N and M with a uniform measure in m (local).
  Represented by ``N_SUB`` power-law sub-pieces fitted to its exact sub-moments (shape error < 1e-4).

Collision nodes (``node_rule``): ``median`` (PL2M: number halves at their mean masses), ``gauss`` (the 2-point
Gauss rule of the in-bin measure in radius: matches a^0..a^3, so N, the surface and M exactly; nodes inside the
support, positive weights) or ``gaussm`` (the same in mass, m^0..m^3). ``ka:n`` averages the collision kernel over n equal-number sub-slices of each node
(``kernel_average``) while products stay at the nodes.

``patched(closure, nodes=..., ka=...)`` is a context manager that routes ``gsd.remap`` users (exchange, sputtering)
and ``gsd.collisions`` through a variant, without editing the production modules.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from pycalima.collapse._jit import njit
from pycalima.gsd import _kernels as K
from pycalima.gsd.collisions import _bounded_hw, _deposit_segment, _fragments, _node_lo
from pycalima.gsd.pl2m import BinProfile
from pycalima.gsd.remap import realisable

N_SUB = 8          # power-law sub-pieces of the non-power-law local closures (exp)
KP_EA = 4          # pieces per bin of the edge-aware closure: base, right-cut, left-cut, spike
NARROW = 1e-7     # truncated supports narrower than this (relative) are spikes


# ---------------------------------------------------------------------------------------------------------------------
# numba primitives: power-law integrals in a stable form
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _lphi(z):
    """log(expm1(z) / z), stable for any z."""
    if abs(z) < 1e-8:
        return 0.5 * z
    if z > 0.0:
        return z + math.log(-math.expm1(-z)) - math.log(z)
    return math.log(-math.expm1(z)) - math.log(-z)


@njit(cache=True)
def _ip(lam, u1, u2):
    """Integral of u^lam du over [u1, u2] (0 if empty)."""
    if u2 <= u1:
        return 0.0
    L = math.log(u2 / u1)
    p = lam + 1.0
    return math.exp(p * math.log(u1) + _lphi(p * L)) * L


@njit(cache=True)
def _mom(lam, u1, u2, k):
    """<u^k> of u^lam on [u1, u2]."""
    if u2 <= u1 * (1.0 + 1e-14):
        return u1 ** k
    L = math.log(u2 / u1)
    p = lam + 1.0
    return u1 ** k * math.exp(_lphi((p + k) * L) - _lphi(p * L))


@njit(cache=True)
def _solve_lam_full(r, uR, lo, hi):
    """lam in [lo, hi] with <u^3> = r over [1, uR] (monotone increasing in lam); Illinois."""
    a, b = lo, hi
    fa = _mom(a, 1.0, uR, 3.0) - r
    fb = _mom(b, 1.0, uR, 3.0) - r
    if fa >= 0.0:
        return a
    if fb <= 0.0:
        return b
    side = 0
    c = a
    for _ in range(200):
        c = (a * fb - b * fa) / (fb - fa)
        fc = _mom(c, 1.0, uR, 3.0) - r
        if fc == 0.0 or b - a < 1e-11 * (1.0 + abs(c)) or abs(fc) < 1e-15 * r:
            break
        if fc > 0.0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
    return c


@njit(cache=True)
def _cut_f(L, r, lam, uR, right):
    if right:
        return _mom(lam, 1.0, math.exp(L), 3.0) - r          # increasing in L
    return r - _mom(lam, uR * math.exp(-L), uR, 3.0)         # increasing in L


@njit(cache=True)
def _solve_cut(r, lam, uR, right):
    """Support cut u_c of a truncated power law u^lam with <u^3> = r: right=True -> support [1, u_c] (r below the
    full-bin mean), else [u_c, uR]. Illinois (regula falsi) in L = ln of the support width, as PL2M's exponent
    solve."""
    Lmax = math.log(uR)
    a, b = 0.0, Lmax
    fa = _cut_f(a, r, lam, uR, right)
    fb = _cut_f(b, r, lam, uR, right)
    if fa >= 0.0:
        return 1.0 if right else uR
    if fb <= 0.0:
        return uR if right else 1.0
    side = 0
    c = a
    for _ in range(200):
        c = (a * fb - b * fa) / (fb - fa)
        fc = _cut_f(c, r, lam, uR, right)
        if fc == 0.0 or b - a < 1e-15 * Lmax or abs(fc) < 1e-15 * r:
            break
        if fc > 0.0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
    return math.exp(c) if right else uR * math.exp(-c)


# ---------------------------------------------------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------------------------------------------------

class Pieces:
    """Power-law pieces of every bin: arrays [nb, KP]. Piece k of bin i: dN/du = A u^lam on [u1, u2], u = a / a_L,i,
    number n (A = n / int u^lam); or, if ``spike``, all its number n at u1 (= u2). Unused pieces have n = 0."""

    def __init__(self, grid, n, A, lam, u1, u2, spike):
        self.grid = grid
        self.n, self.A, self.lam, self.u1, self.u2, self.spike = n, A, lam, u1, u2, spike
        self.aL = grid.a_edges[:-1].copy()
        self.mL = grid.m_edges[:-1].copy()

    @property
    def kp(self):
        return self.n.shape[1]


@njit(cache=True)
def _amplitudes(n, lam, u1, u2, spike):
    A = np.zeros(n.shape)
    for i in range(n.shape[0]):
        for k in range(n.shape[1]):
            if n[i, k] != 0.0 and not spike[i, k]:
                A[i, k] = n[i, k] / _ip(lam[i, k], u1[i, k], u2[i, k])
    return A


def _pieces(grid, n, lam, u1, u2, spike, A=None):
    if A is None:
        A = _amplitudes(n, lam, u1, u2, spike)
    return Pieces(grid, n, A, lam, u1, u2, spike)


def _state(grid, N, M):
    """Mean-mass ratio r = M / (N m_L), per-bin u_R and the filled mask."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    eta = grid.etas
    filled = N > 0
    r = np.where(filled, M / np.where(filled, N, 1.0) / grid.m_edges[:-1], 0.5 * (1 + eta))
    r = np.clip(r, 1.0, eta)
    return N, M, r, eta ** (1.0 / 3.0), filled


# --- PL2M (production) --------------------------------------------------------------------------------------------

def pieces_pl2m(grid, N, M):
    prof = BinProfile(grid, N, M)
    nb = grid.n_bins
    n = np.where(prof.N > 0, prof.N, 0.0)[:, None].copy()
    sp = prof.delta[:, None].copy()
    u1 = np.where(prof.delta, prof.u_delta, 1.0)[:, None].copy()
    u2 = np.where(prof.delta, prof.u_delta, prof.uR_bins)[:, None].copy()
    lam = prof.lam[:, None].copy()
    A = np.where(prof.delta, 0.0, prof.A)[:, None].copy()
    assert n.shape == (nb, 1)
    return Pieces(grid, n, A, lam, u1, u2, sp)


@njit(cache=True)
def _pl2m_nb(N, r, uR, cap_lo, cap_hi):
    """PL2M in compiled form (for timing): lam in [cap_lo, cap_hi] or a spike."""
    nb = N.size
    lam = np.zeros(nb)
    spike = np.zeros(nb, dtype=np.bool_)
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        if r[i] <= _mom(cap_lo[i], 1.0, uR[i], 3.0) or r[i] >= _mom(cap_hi[i], 1.0, uR[i], 3.0):
            spike[i] = True
            continue
        lam[i] = _solve_lam_full(r[i], uR[i], cap_lo[i], cap_hi[i])
    return lam, spike


# --- TC: truncated cap ----------------------------------------------------------------------------------------------

@njit(cache=True)
def _tc_nb(N, r, uR, Lb):
    """Per bin: lam, u1, u2 of the truncated-cap closure (window [-Lb[i], Lb[i]] in the radius exponent)."""
    nb = N.size
    lam = np.zeros(nb)
    u1 = np.ones(nb)
    u2 = uR.copy()
    spike = np.zeros(nb, dtype=np.bool_)
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        ri = r[i]
        if ri <= 1.0 + 1e-13 or ri >= uR[i] ** 3 * (1.0 - 1e-13):
            spike[i] = True
            u1[i] = u2[i] = min(max(ri, 1.0), uR[i] ** 3) ** (1.0 / 3.0)
            continue
        L = Lb[i]
        r_lo = _mom(-L, 1.0, uR[i], 3.0)
        r_hi = _mom(L, 1.0, uR[i], 3.0)
        if ri < r_lo:
            lam[i] = -L
            u2[i] = _solve_cut(ri, -L, uR[i], True)
        elif ri > r_hi:
            lam[i] = L
            u1[i] = _solve_cut(ri, L, uR[i], False)
        else:
            lam[i] = _solve_lam_full(ri, uR[i], -L, L)
        if u2[i] < u1[i] * (1.0 + NARROW):
            # a support this narrow is a spike (round-off would otherwise dominate the remap)
            spike[i] = True
            u1[i] = u2[i] = ri ** (1.0 / 3.0)
    return lam, u1, u2, spike


def window(grid, L):
    """Per-bin exponent window: min(L, PL2M's own cap 3 beta_max(eta)), so u^lam stays below e^300."""
    from pycalima.gsd.pl2m import beta_max
    return np.minimum(float(L), 3.0 * beta_max(grid.etas)).astype(float)


def pieces_tc(grid, N, M, L):
    N, M, r, uR, filled = _state(grid, N, M)
    lam, u1, u2, sp = _tc_nb(N, r, np.ascontiguousarray(uR, float), window(grid, L))
    n = np.where(filled, N, 0.0)[:, None]
    return _pieces(grid, n.copy(), lam[:, None].copy(), u1[:, None].copy(), u2[:, None].copy(), sp[:, None].copy())


# --- EA: edge-aware mixture ----------------------------------------------------------------------------------------

@njit(cache=True)
def _ea_nb(N, r, uR, da, Lb, eps, lam0, fixed=False):
    """Edge-aware closure: [nb, 4] pieces (base TC, right-cut, left-cut, spike).

    Front weights e_R, e_L = max(0, 1 - q / eps), q the neighbour's mean dn/da over the bin's own (grid ends count
    as empty). A right-front bin continues its left neighbour's exponent, weighted by how far that neighbour is
    interior, (1 - e_L of the neighbour) lam_nb + e_L lam0, and cuts the support at u_c; mirror for left fronts."""
    nb = N.size
    lam_b, u1_b, u2_b, sp_b = _tc_nb(N, r, uR, Lb)
    eR_ = np.zeros(nb)
    eL_ = np.zeros(nb)
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        dens = N[i] / da[i]
        if not dens > 0.0:            # subnormal N: no front detection
            continue
        qR = 0.0 if i == nb - 1 else (N[i + 1] / da[i + 1]) / dens
        qL = 0.0 if i == 0 else (N[i - 1] / da[i - 1]) / dens
        eR_[i] = max(0.0, 1.0 - qR / eps)
        eL_[i] = max(0.0, 1.0 - qL / eps)
    n = np.zeros((nb, 4))
    lam = np.zeros((nb, 4))
    u1 = np.ones((nb, 4))
    u2 = np.ones((nb, 4))
    sp = np.zeros((nb, 4), dtype=np.bool_)
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        eR, eL = eR_[i], eL_[i]
        w = ((1.0 - eL) * (1.0 - eR), eR * (1.0 - eL), eL * (1.0 - eR), eL * eR)
        lam[i, 0], u1[i, 0], u2[i, 0], sp[i, 0] = lam_b[i], u1_b[i], u2_b[i], sp_b[i]
        n[i, 0] = w[0] * N[i]
        ri = r[i]
        # right front: the left neighbour's exponent continued, support [1, u_c]
        if w[1] > 0.0:
            if i > 0 and N[i - 1] > 0.0 and not sp_b[i - 1] and not fixed:
                lr = (1.0 - eL_[i - 1]) * lam_b[i - 1] + eL_[i - 1] * lam0
            else:
                lr = lam0
            if 1.0 + 1e-13 < ri <= _mom(lr, 1.0, uR[i], 3.0):
                lam[i, 1], u1[i, 1], u2[i, 1] = lr, 1.0, _solve_cut(ri, lr, uR[i], True)
                if u2[i, 1] < 1.0 + NARROW:
                    sp[i, 1] = True
                    u1[i, 1] = u2[i, 1] = ri ** (1.0 / 3.0)
                n[i, 1] = w[1] * N[i]
            else:
                n[i, 0] += w[1] * N[i]
        # left front: the right neighbour's exponent continued, support [u_c, uR]
        if w[2] > 0.0:
            if i < nb - 1 and N[i + 1] > 0.0 and not sp_b[i + 1] and not fixed:
                ll = (1.0 - eR_[i + 1]) * lam_b[i + 1] + eR_[i + 1] * lam0
            else:
                ll = lam0
            if _mom(ll, 1.0, uR[i], 3.0) <= ri < uR[i] ** 3 * (1.0 - 1e-13):
                lam[i, 2], u1[i, 2], u2[i, 2] = ll, _solve_cut(ri, ll, uR[i], False), uR[i]
                if u2[i, 2] < u1[i, 2] * (1.0 + NARROW):
                    sp[i, 2] = True
                    u1[i, 2] = u2[i, 2] = ri ** (1.0 / 3.0)
                n[i, 2] = w[2] * N[i]
            else:
                n[i, 0] += w[2] * N[i]
        # isolated: a spike at the mean mass
        if w[3] > 0.0:
            us = ri ** (1.0 / 3.0)
            u1[i, 3] = u2[i, 3] = us
            sp[i, 3] = True
            n[i, 3] = w[3] * N[i]
    return n, lam, u1, u2, sp


def pieces_ea(grid, N, M, eps, L, lam0=0.0, fixed=False):
    N, M, r, uR, filled = _state(grid, N, M)
    da = np.diff(grid.a_edges)
    n, lam, u1, u2, sp = _ea_nb(N, r, np.ascontiguousarray(uR, float), da, window(grid, L), float(eps),
                                float(lam0), bool(fixed))
    return _pieces(grid, n, lam, u1, u2, sp)


# --- AFF: affine in a with a triangle switch -----------------------------------------------------------------------

@njit(cache=True)
def _aff_profile(Ni, ri, U):
    """Affine closure of one bin in t = u - 1: kind 0, f = c0 + c1 t on [1, U] (positive there); kind 1, the
    triangle c0 (hi - u) on [1, hi]; kind 2, c0 (u - lo) on [lo, U]. Polynomial forms, free of cancellation."""
    D = U - 1.0
    mean_lo = 1.0 + D + D * D / 2.0 + D ** 3 / 10.0          # line with its zero at U
    mean_hi = 1.0 + 2.0 * D + 1.5 * D * D + 0.4 * D ** 3       # line with its zero at 1
    if ri < mean_lo:
        a, b = 0.0, D
        for _ in range(200):
            d = 0.5 * (a + b)
            if 1.0 + d + d * d / 2.0 + d ** 3 / 10.0 > ri:
                b = d
            else:
                a = d
        d = 0.5 * (a + b)
        if d < NARROW:
            return 3, 0.0, 0.0, 1.0, 1.0 + d
        return 1, Ni / (d * d / 2.0), 0.0, 1.0, 1.0 + d
    if ri > mean_hi:
        a, b = 0.0, D                       # D' = U - z: the mean decreases with D'
        for _ in range(200):
            dz = 0.5 * (a + b)
            z = U - dz
            if z ** 3 + 2.0 * z * z * dz + 1.5 * z * dz * dz + 0.4 * dz ** 3 > ri:
                a = dz
            else:
                b = dz
        dz = 0.5 * (a + b)
        if dz < NARROW * U:
            return 3, 0.0, 0.0, U - dz, U
        return 2, Ni / (dz * dz / 2.0), 0.0, U - dz, U
    J0 = D + 1.5 * D * D + D ** 3 + D ** 4 / 4.0
    J1 = D * D / 2.0 + D ** 3 + 0.75 * D ** 4 + D ** 5 / 5.0
    det = D * J1 - D * D / 2.0 * J0
    c0 = (Ni * J1 - D * D / 2.0 * Ni * ri) / det
    c1 = (D * Ni * ri - J0 * Ni) / det
    return 0, c0, c1, 1.0, U


@njit(cache=True)
def _aff_f(kind, c0, c1, lo, hi, u):
    if kind == 0:
        return max(c0 + c1 * (u - 1.0), 0.0)
    if kind == 1:
        return c0 * (hi - u)
    return c0 * (u - lo)


@njit(cache=True)
def _aff_nb(N, r, uR, n_sub):
    """Affine closure as n_sub positive power-law sub-pieces (equal in ln u over the support), each fitted to the
    exact sub-moments (3-point Gauss-Legendre, exact for the degree-4 integrand)."""
    nb = N.size
    n = np.zeros((nb, n_sub))
    lam = np.zeros((nb, n_sub))
    u1 = np.ones((nb, n_sub))
    u2 = np.ones((nb, n_sub))
    sp = np.zeros((nb, n_sub), dtype=np.bool_)
    gx = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
    gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        U = uR[i]
        ri = r[i]
        if ri <= 1.0 + 1e-10 * (U ** 3 - 1.0) or ri >= U ** 3 - 1e-10 * (U ** 3 - 1.0):
            sp[i, 0] = True
            u1[i, 0] = u2[i, 0] = min(max(ri, 1.0), U ** 3) ** (1.0 / 3.0)
            n[i, 0] = N[i]
            continue
        kind, c0, c1, lo, hi = _aff_profile(N[i], ri, U)
        if kind == 3 or hi < lo * (1.0 + NARROW):
            sp[i, 0] = True
            u1[i, 0] = u2[i, 0] = ri ** (1.0 / 3.0)
            n[i, 0] = N[i]
            continue
        ue = np.exp(np.linspace(math.log(lo), math.log(hi), n_sub + 1))
        nk = np.zeros(n_sub)
        mk = np.zeros(n_sub)
        for j in range(n_sub):
            h = 0.5 * (ue[j + 1] - ue[j])
            c = 0.5 * (ue[j + 1] + ue[j])
            for q in range(3):
                u = c + h * gx[q]
                f = _aff_f(kind, c0, c1, lo, hi, u) * gw[q] * h
                nk[j] += f
                mk[j] += f * u ** 3
        tot = nk.sum()
        if not tot > 0.0:
            sp[i, 0] = True
            u1[i, 0] = u2[i, 0] = ri ** (1.0 / 3.0)
            n[i, 0] = N[i]
            continue
        scale = N[i] / tot
        for j in range(n_sub):
            if nk[j] <= 0.0:
                continue
            lj, a1, a2, s_ = _fit_piece(nk[j] * scale, mk[j] / nk[j], ue[j], ue[j + 1])
            n[i, j], lam[i, j], u1[i, j], u2[i, j], sp[i, j] = nk[j] * scale, lj, a1, a2, s_
    return n, lam, u1, u2, sp


def pieces_aff(grid, N, M, n_sub=N_SUB):
    N, M, r, uR, filled = _state(grid, N, M)
    n, lam, u1, u2, sp = _aff_nb(N, r, np.ascontiguousarray(uR, float), int(n_sub))
    return _fix_mass(_pieces(grid, n, lam, u1, u2, sp), M)


# --- EXP: maximum entropy, dN/dm ~ exp(-kappa m) ------------------------------------------------------------------

@njit(cache=True)
def _fit_piece(nk, rk, ua, ub):
    """Power law on [ua, ub] with number nk and <u^3> = rk (a spike if outside the reachable range). The exponent is
    bounded so that u^lam stays within e^(+-280) on the piece (steeper pieces are spikes, as PL2M's cap)."""
    lim = min(600.0, 280.0 / max(math.log(ub), 1e-12))
    lo, hi = -lim, lim
    if rk <= _mom(lo, ua, ub, 3.0) or rk >= _mom(hi, ua, ub, 3.0):
        u = min(max(rk, ua ** 3), ub ** 3) ** (1.0 / 3.0)
        return 0.0, u, u, True
    a, b = lo, hi
    fa = _mom(a, ua, ub, 3.0) - rk
    fb = _mom(b, ua, ub, 3.0) - rk
    side = 0
    c = a
    for _ in range(300):
        c = (a * fb - b * fa) / (fb - fa)
        fc = _mom(c, ua, ub, 3.0) - rk
        if fc == 0.0 or b - a < 1e-12 or abs(fc) < 1e-15 * rk:
            break
        if fc > 0.0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
    return c, ua, ub, False


@njit(cache=True)
def _exp_mean(k, W):
    """Mean of y under exp(-k y) on [0, W]: W g(kW), g(z) = (e^z - 1 - z) / (z (e^z - 1)), stable for all z."""
    z = k * W
    if abs(z) < 0.5:
        # series of e^z - 1 - z and of e^z - 1
        num = 0.0
        term = z * z / 2.0
        em1 = z
        for j in range(3, 30):
            num += term
            em1 += term
            term *= z / j
        if z == 0.0:
            return W / 2.0
        return W * num / (z * em1)
    if z > 700.0:
        return W / z
    em1 = math.expm1(z)
    return W * (em1 - z) / (z * em1)


@njit(cache=True)
def _exp_nb(N, r, uR, n_sub):
    """Max-entropy closure, n_sub power-law sub-pieces per bin (equal in ln u)."""
    nb = N.size
    n = np.zeros((nb, n_sub))
    lam = np.zeros((nb, n_sub))
    u1 = np.ones((nb, n_sub))
    u2 = np.ones((nb, n_sub))
    sp = np.zeros((nb, n_sub), dtype=np.bool_)
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        U = uR[i]
        W = U ** 3 - 1.0
        y = r[i] - 1.0
        if y <= 1e-10 * W or y >= W * (1.0 - 1e-10):
            sp[i, 0] = True
            u1[i, 0] = u2[i, 0] = min(max(r[i], 1.0), U ** 3) ** (1.0 / 3.0)
            n[i, 0] = N[i]
            continue
        # kappa (per m_L) from the mean: decreasing in k; bisection on asinh(k)
        a, b = -60.0, 60.0
        for _ in range(200):
            c = 0.5 * (a + b)
            if _exp_mean(math.sinh(c), W) > y:
                a = c
            else:
                b = c
        k = math.sinh(0.5 * (a + b))
        # sub-moments in x = m / m_L - 1 on [x_j, x_j+1]: int e^(-k x), int x e^(-k x)  (scaled by e^(-k x_j))
        tot_n = 0.0
        ns = np.zeros(n_sub)
        rs = np.zeros(n_sub)
        ue = np.exp(np.linspace(0.0, math.log(U), n_sub + 1))
        for j in range(n_sub):
            x0, x1 = ue[j] ** 3 - 1.0, ue[j + 1] ** 3 - 1.0
            h = x1 - x0
            # e^{-k x0} * int_0^h e^{-k t} dt and mean of t
            if abs(k * h) < 1e-4:
                li0 = math.log(h) - 0.5 * k * h + (k * h) ** 2 / 24.0
            elif k > 0.0:
                li0 = math.log(-math.expm1(-k * h)) - math.log(k)
            else:
                li0 = -k * h + math.log(-math.expm1(k * h)) - math.log(-k)
            tm = _exp_mean(k, h)
            # weight relative to the bin's lowest point, kept in logs for large |k|
            ns[j] = li0 - k * x0
            rs[j] = 1.0 + x0 + tm                 # mean m / m_L of the sub-interval
        mx = ns.max()
        for j in range(n_sub):
            ns[j] = math.exp(ns[j] - mx)
            tot_n += ns[j]
        for j in range(n_sub):
            nk = N[i] * ns[j] / tot_n
            if nk <= 1e-300:
                continue
            lj, a1, a2, s_ = _fit_piece(nk, rs[j], ue[j], ue[j + 1])
            n[i, j], lam[i, j], u1[i, j], u2[i, j], sp[i, j] = nk, lj, a1, a2, s_
    return n, lam, u1, u2, sp


def pieces_exp(grid, N, M, n_sub=N_SUB):
    N, M, r, uR, filled = _state(grid, N, M)
    n, lam, u1, u2, sp = _exp_nb(N, r, np.ascontiguousarray(uR, float), int(n_sub))
    P = _pieces(grid, n, lam, u1, u2, sp)
    return _fix_mass(P, M)


def _fix_mass(P, M):
    """Sub-piece closures: the sub-means are exact, so the bin mass is exact to round-off; check it."""
    m = bin_moments(P, 3)
    N, _M, r, _uR, filled = _state(P.grid, bin_moments(P, 0), M)
    ref = N * r                          # the realisable mass / m_L (M/N clipped into the bin, as PL2M does)
    filled = filled & (N > 1e-200 * N.max())   # bins this empty underflow inside the sub-pieces: not checkable
    if np.any(np.abs(m[filled] / ref[filled] - 1.0) > 1e-8):
        raise AssertionError("sub-piece closure lost mass: %g" % np.max(np.abs(m[filled] / ref[filled] - 1.0)))
    return P


# ---------------------------------------------------------------------------------------------------------------------
# closure registry
# ---------------------------------------------------------------------------------------------------------------------

class Closure:
    """A named closure: ``pieces(grid, N, M)`` and whether it is local (a function of each bin's (N, M) alone)."""

    def __init__(self, name, fn, local, params=""):
        self.name, self.fn, self.local, self.params = name, fn, local, params

    def pieces(self, grid, N, M):
        return self.fn(grid, np.asarray(N, float), np.asarray(M, float))

    def __repr__(self):
        return f"Closure({self.name})"


def make_closure(spec):
    """'pl2m', 'tc:L', 'ea:eps[:L[:lam0]]' (L = 'inf' for PL2M's own range), 'eaf:eps[:lam_front]', 'aff',
    'exp'."""
    parts = spec.split(":")
    kind = parts[0]
    if kind == "pl2m":
        return Closure(spec, pieces_pl2m, True)
    if kind == "tc":
        L = float(parts[1])
        return Closure(spec, lambda g, N, M: pieces_tc(g, N, M, L), True, f"L={L:g}")
    if kind == "ea":
        eps = float(parts[1])
        L = float(parts[2]) if len(parts) > 2 else math.inf
        lam0 = float(parts[3]) if len(parts) > 3 else 0.0
        return Closure(spec, lambda g, N, M: pieces_ea(g, N, M, eps, L, lam0), False,
                       f"eps={eps:g}, L={L:g}, lam0={lam0:g}")
    if kind == "eaf":
        # edge-aware with a fixed front exponent lamf (no neighbour exponent): front profiles are functions of s
        eps = float(parts[1])
        lamf = float(parts[2]) if len(parts) > 2 else -3.5
        return Closure(spec, lambda g, N, M: pieces_ea(g, N, M, eps, math.inf, lamf, True), False,
                       f"eps={eps:g}, lam_front={lamf:g}")
    if kind == "aff":
        return Closure(spec, pieces_aff, True)
    if kind == "exp":
        return Closure(spec, pieces_exp, True)
    raise ValueError(spec)


# ---------------------------------------------------------------------------------------------------------------------
# operations on pieces
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _bin_moments(n, A, lam, u1, u2, spike, k):
    nb, kp = n.shape
    out = np.zeros(nb)
    for i in range(nb):
        for j in range(kp):
            if n[i, j] == 0.0:
                continue
            if spike[i, j]:
                out[i] += n[i, j] * u1[i, j] ** k
            else:
                out[i] += A[i, j] * _ip(lam[i, j] + k, u1[i, j], u2[i, j])
    return out


def bin_moments(P, k):
    """Integral of u^k dN per bin (u = a / a_L): k = 0 number, k = 3 mass / m_L."""
    return _bin_moments(P.n, P.A, P.lam, P.u1, P.u2, P.spike, float(k))


def radius_moments(P, k):
    """Integral of a^k dN per bin [cm^k]."""
    return bin_moments(P, k) * P.aL ** k


@njit(cache=True)
def _remap_pieces(a_e, rho, aL, n, A, lam, spike, nodes, X):
    """Remap of power-law pieces along characteristics: nodes[i, j, :] are radii spanning piece (i, j)'s support,
    X their images. As ``gsd._kernels.remap_profiles``, but per piece."""
    nb = a_e.size - 1
    mfac = 4.0 / 3.0 * math.pi * rho
    N_new = np.zeros(nb)
    M_new = np.zeros(nb)
    lost = 0.0
    oN = 0.0
    oM = 0.0
    kp = n.shape[1]
    npc = nodes.shape[2] - 1
    for i in range(nb):
        a0 = aL[i]
        for jj in range(kp):
            if n[i, jj] == 0.0:
                continue
            if spike[i, jj]:
                Xd = X[i, jj, 0]
                j = K._bin_of(a_e, Xd)
                if j < 0 or Xd <= 0.0:
                    lost += n[i, jj]
                elif j >= nb:
                    oN += n[i, jj]
                    oM += n[i, jj] * mfac * Xd ** 3
                else:
                    N_new[j] += n[i, jj]
                    M_new[j] += n[i, jj] * mfac * Xd ** 3
                continue
            Ai = A[i, jj]
            li = lam[i, jj]
            for k in range(npc):
                p, q = nodes[i, jj, k], nodes[i, jj, k + 1]
                if q <= p:
                    continue
                Xp, Xq = X[i, jj, k], X[i, jj, k + 1]
                up_, uq_ = p / a0, q / a0
                if Xq - Xp <= 1e-14 * max(Xq, 1e-300):
                    n_piece = Ai * _ip(li, up_, uq_)
                    j = K._bin_of(a_e, Xp)
                    if j < 0 or Xp <= 0.0:
                        lost += n_piece
                    elif j >= nb:
                        oN += n_piece
                        oM += n_piece * mfac * Xp ** 3
                    else:
                        N_new[j] += n_piece
                        M_new[j] += n_piece * mfac * Xp ** 3
                    continue
                s = (Xq - Xp) / (q - p)
                d1 = s * a0
                c0 = Xp - s * p
                if Xp < a_e[0]:
                    x_hi = min(Xq, a_e[0])
                    u_hi = (p + (x_hi - Xp) / s) / a0
                    lost += Ai * _ip(li, up_, u_hi)
                if Xq > a_e[nb]:
                    x_lo = max(Xp, a_e[nb])
                    u_lo = (p + (x_lo - Xp) / s) / a0
                    oN += Ai * _ip(li, u_lo, uq_)
                    oM += mfac * _cubic(Ai, li, c0, d1, u_lo, uq_)
                j0 = max(K._bin_of(a_e, Xp), 0)
                j1 = min(K._bin_of(a_e, Xq), nb - 1)
                for j in range(j0, j1 + 1):
                    x1 = max(Xp, a_e[j])
                    x2 = min(Xq, a_e[j + 1])
                    if x2 <= x1:
                        continue
                    u1_ = (p + (x1 - Xp) / s) / a0
                    u2_ = (p + (x2 - Xp) / s) / a0
                    N_new[j] += Ai * _ip(li, u1_, u2_)
                    M_new[j] += mfac * _cubic(Ai, li, c0, d1, u1_, u2_)
    return N_new, M_new, lost, oN, oM


@njit(cache=True)
def _cubic(A, lam, d0, d1, u1, u2):
    return A * (d0 ** 3 * _ip(lam, u1, u2) + 3.0 * d0 ** 2 * d1 * _ip(lam + 1.0, u1, u2)
                + 3.0 * d0 * d1 ** 2 * _ip(lam + 2.0, u1, u2) + d1 ** 3 * _ip(lam + 3.0, u1, u2))


def remap(grid, N, M, image, closure, n_pieces=4, P=None):
    """Remap (N, M) along the flow map ``image`` with a variant closure. Returns (N, M, lost_N, overflow_N,
    overflow_M) like ``gsd.remap.remap``."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    if P is None:
        P = closure.pieces(grid, N, M)
    t = np.linspace(0.0, 1.0, n_pieces + 1)
    lo = P.aL[:, None] * P.u1
    hi = P.aL[:, None] * P.u2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lo > 0, hi / np.where(lo > 0, lo, 1.0), 1.0)
    nodes = lo[:, :, None] * ratio[:, :, None] ** t[None, None, :]
    X = np.asarray(image(nodes.ravel()), float).reshape(nodes.shape)
    N1, M1, lost, oN, oM = _remap_pieces(grid.a_edges, float(grid.rho), P.aL, P.n, P.A, P.lam, P.spike, nodes, X)
    return (*realisable(grid, N1, M1, tiny=1e-30 * max(N.sum(), 1e-300)), lost, oN, oM)


def samples(grid, N, M, closure, step=0.01, P=None):
    """Quadrature nodes (a, dN) of the reconstruction (steps of ``step`` in ln a, trapezoid), as
    ``check_bin_convergence.bin_samples``."""
    if P is None:
        P = closure.pieces(grid, N, M)
    a_out, w_out = [], []
    for i in range(grid.n_bins):
        for j in range(P.kp):
            nij = P.n[i, j]
            if nij == 0.0:
                continue
            if P.spike[i, j]:
                a_out.append([P.aL[i] * P.u1[i, j]])
                w_out.append([nij])
                continue
            Lw = math.log(P.u2[i, j] / P.u1[i, j])
            if Lw <= 0:
                continue
            n = max(12, int(math.ceil(Lw / step)))
            u = np.geomspace(P.u1[i, j], P.u2[i, j], n + 1)
            w = np.full(n + 1, Lw / n)
            w[[0, -1]] *= 0.5
            a_out.append(P.aL[i] * u)
            w_out.append(P.A[i, j] * u ** (P.lam[i, j] + 1.0) * w)
    if not a_out:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(a_out), np.concatenate(w_out)


def dnda_curve(grid, N, M, closure, n_pts=200):
    """(a, dn/da) of the reconstruction for plotting: per bin on a fine grid, NaN between bins; spikes are drawn
    as narrow peaks of their number over 1% of the bin."""
    P = closure.pieces(grid, N, M)
    xs, ys = [], []
    for i in range(grid.n_bins):
        aL, aR = grid.a_edges[i], grid.a_edges[i + 1]
        x = np.geomspace(aL, aR, n_pts)
        u = x / aL
        y = np.zeros(n_pts)
        for j in range(P.kp):
            if P.n[i, j] == 0.0:
                continue
            if P.spike[i, j]:
                k = np.argmin(np.abs(u - P.u1[i, j]))
                wid = max(x[min(k + 1, n_pts - 1)] - x[max(k - 1, 0)], 1e-300)
                y[k] += P.n[i, j] / wid
                continue
            inside = (u >= P.u1[i, j] * (1 - 1e-12)) & (u <= P.u2[i, j] * (1 + 1e-12))
            y += np.where(inside, P.A[i, j] * u ** P.lam[i, j] / aL, 0.0)
        if N[i] <= 0:
            y[:] = np.nan
        xs += [x, [np.nan]]
        ys += [y, [np.nan]]
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------------------------------------------------
# collision nodes
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _cum(n, A, lam, u1, u2, spike, i, u):
    """Number and mass/m_L of bin i's grains below u (spikes at u count as below)."""
    c = 0.0
    m = 0.0
    for j in range(n.shape[1]):
        if n[i, j] == 0.0:
            continue
        if spike[i, j]:
            if u1[i, j] <= u:
                c += n[i, j]
                m += n[i, j] * u1[i, j] ** 3
            continue
        hi = min(u, u2[i, j])
        if hi > u1[i, j]:
            c += A[i, j] * _ip(lam[i, j], u1[i, j], hi)
            m += A[i, j] * _ip(lam[i, j] + 3.0, u1[i, j], hi)
    return c, m


@njit(cache=True)
def _mass_of_lowest(n, A, lam, u1, u2, spike, i, uR, t):
    """(u*, mass/m_L of the t lowest grains of bin i) by bisection on the cumulative number."""
    lo, hi = 0.0, math.log(uR)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c, _m = _cum(n, A, lam, u1, u2, spike, i, math.exp(mid))
        if c >= t:
            hi = mid
        else:
            lo = mid
    ulo, uhi = math.exp(lo), math.exp(hi)
    c, m = _cum(n, A, lam, u1, u2, spike, i, ulo)
    if c > t:          # t below everything (only at t ~ 0)
        return ulo, t * ulo ** 3
    return uhi, m + (t - c) * uhi ** 3


@njit(cache=True)
def _nodes_nb(N, M, mL, mR, uR, n, A, lam, u1, u2, spike, gauss, n_sub):
    """Two nodes per bin (masses m, weights w, interval widths wid) and n_sub equal-number sub-slices per node
    (sub-masses ms[nb, 2, n_sub]). gauss: 2-point Gauss rule of the measure in m, else the median split."""
    nb = N.size
    m = np.zeros((nb, 2))
    w = np.zeros((nb, 2))
    wid = np.zeros((nb, 2))
    ms = np.zeros((nb, 2, n_sub))
    for i in range(nb):
        mc = math.sqrt(mL[i] * mR[i])
        m[i, 0] = m[i, 1] = mc
        if N[i] <= 0.0:
            for h in range(2):
                for k in range(n_sub):
                    ms[i, h, k] = mc
            continue
        mbar = M[i] / N[i]
        if N[i] < 1e-300:
            # (sub)normal underflow: no splitting (halves would round to zero), both nodes at the mean mass
            mb = min(max(mbar, mL[i]), mR[i])
            m[i, 0] = m[i, 1] = mb
            w[i, 0] = w[i, 1] = 0.5 * N[i]
            for h in range(2):
                for k in range(n_sub):
                    ms[i, h, k] = mb
            continue
        # pure spike bins (every piece a spike at one point)
        allspike = True
        for j in range(n.shape[1]):
            if n[i, j] != 0.0 and not spike[i, j]:
                allspike = False
        if gauss > 0:
            # 2-point Gauss rule of the in-bin measure: in radius (gauss = 1: moments u^0..u^3, so N, the
            # surface and M exactly) or in mass (gauss = 2: moments m^0..m^3)
            q = 1.0 if gauss == 1 else 3.0
            mu = np.zeros(4)
            for j in range(n.shape[1]):
                if n[i, j] == 0.0:
                    continue
                for k in range(4):
                    # number x normalised moment: no amplitude, so no underflow for nearly empty bins
                    if spike[i, j]:
                        mu[k] += n[i, j] * u1[i, j] ** (q * k)
                    else:
                        mu[k] += n[i, j] * _mom(lam[i, j], u1[i, j], u2[i, j], q * k)
            mu0 = mu[0]
            if not mu0 > 0.0:
                mu0 = 1.0
                allspike = True
            e1 = mu[1] / mu0
            c2 = mu[2] / mu0 - e1 * e1
            c3 = mu[3] / mu0 - 3.0 * e1 * mu[2] / mu0 + 2.0 * e1 ** 3
            if c2 <= 1e-14 * e1 * e1 or allspike:
                p1 = 0.5
                x1 = x2 = mbar
            else:
                sg = math.sqrt(c2)
                g = c3 / sg ** 3
                z1 = 0.5 * (g - math.sqrt(g * g + 4.0))
                z2 = 0.5 * (g + math.sqrt(g * g + 4.0))
                p1 = z2 / (z2 - z1)
                x1 = (e1 + sg * z1) ** (3.0 / q) * mL[i]
                x2 = (e1 + sg * z2) ** (3.0 / q) * mL[i]
                if p1 < 1e-10 or p1 > 1.0 - 1e-10:
                    # one node carries (almost) no weight: a near-degenerate measure, both at the mean
                    p1 = 0.5
                    x1 = x2 = mbar
            w[i, 0] = p1 * N[i]
            w[i, 1] = (1.0 - p1) * N[i]
            # exact mean: x2 from x1
            x1 = min(max(x1, mL[i]), mR[i])
            if w[i, 0] > 0.0 and w[i, 1] > 0.0:
                x2 = (M[i] - w[i, 0] * x1) / w[i, 1]
            else:                     # subnormal N: the weights round to zero
                x1 = x2 = mbar
            x2 = min(max(x2, mL[i]), mR[i])
            m[i, 0], m[i, 1] = x1, x2
            tsplit = w[i, 0]
        else:
            w[i, 0] = w[i, 1] = 0.5 * N[i]
            tsplit = 0.5 * N[i]
            if allspike:
                m[i, 0] = m[i, 1] = mbar
            else:
                us, mlow = _mass_of_lowest(n, A, lam, u1, u2, spike, i, uR[i], tsplit)
                x1 = min(max(mL[i] * mlow / tsplit, mL[i]), mR[i])
                x2 = min(max(2.0 * mbar - x1, mL[i]), mR[i])
                m[i, 0], m[i, 1] = x1, x2
        if allspike or not (w[i, 0] > 0.0 and w[i, 1] > 0.0):
            for h in range(2):
                for k in range(n_sub):
                    ms[i, h, k] = mbar
            continue
        us, _ = _mass_of_lowest(n, A, lam, u1, u2, spike, i, uR[i], tsplit)
        mq = mL[i] * us ** 3
        wid[i, 0] = mq - mL[i]
        wid[i, 1] = mR[i] - mq
        # sub-slices of each node: equal number, mean mass of each slice
        for h in range(2):
            t0 = 0.0 if h == 0 else tsplit
            tw = w[i, h]
            prev = 0.0
            if h == 1:
                _u, prev = _mass_of_lowest(n, A, lam, u1, u2, spike, i, uR[i], t0)
            for k in range(n_sub):
                t1 = t0 + tw * (k + 1) / n_sub
                if h == 1 and k == n_sub - 1:
                    cur = M[i] / mL[i]
                else:
                    _u, cur = _mass_of_lowest(n, A, lam, u1, u2, spike, i, uR[i], t1)
                ms[i, h, k] = min(max(mL[i] * (cur - prev) / (tw / n_sub), mL[i]), mR[i])
                prev = cur
    return m, w, wid, ms


def nodes(grid, N, M, closure, rule="median", n_sub=1, P=None):
    """(m[nb, 2], w[nb, 2], wid[nb, 2], m_sub[nb, 2, n_sub]) of a variant: the closure's pieces, the node rule
    ('median'; 'gauss', the 2-point Gauss rule in radius; 'gaussm', in mass) and n_sub equal-number sub-slices of
    each node."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    if P is None:
        P = closure.pieces(grid, N, M)
    uR = grid.etas ** (1.0 / 3.0)
    g = {"median": 0, "gauss": 1, "gaussm": 2}[rule]
    return _nodes_nb(N, M, grid.m_edges[:-1].copy(), grid.m_edges[1:].copy(), uR, P.n, P.A, P.lam, P.u1, P.u2,
                     P.spike, g, int(n_sub))


# ---------------------------------------------------------------------------------------------------------------------
# routing the production operators through a variant
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _pair_rows_eff(t, T, P, m, w, wid, K, rem, frag, a_hi, a_lo, mpe, mqe, m_edges, a_edges, GM, GN, eM, eN, mfac,
                   keep_small, F, L, H, D):
    """``collisions._pair_rows`` with the donor split of each pair's products taken from per-pair effective
    masses (mpe, mqe: rate-weighted mean masses of the colliding sub-populations of nodes p and q) instead of the
    node masses; deposits use the production routines."""
    geo_m = np.array([math.log(m_edges[0]), 1.0 / math.log(m_edges[1] / m_edges[0])])
    geo_a = np.array([math.log(a_edges[0]), 1.0 / math.log(a_edges[1] / a_edges[0])])
    for c in range(K.shape[0]):
        for p in range(t, P, T):
            i, ia = p // 2, p % 2
            wp = w[i, ia]
            if wp <= 0.0:
                continue
            for q in range(p, P):
                j, jb = q // 2, q % 2
                wq = w[j, jb]
                if wq <= 0.0:
                    continue
                r = (0.5 if q == p else 1.0) * K[c, p, q] * wp * wq
                if r <= 0.0:
                    continue
                mp = mpe[c, p, q]
                mq = mqe[c, p, q]
                fp = mp / (mp + mq)
                fq = 1.0 - fp
                L[i] += r
                L[j] += r
                for st in range(2):
                    mr = rem[st, c, p, q]
                    if mr > 0.0:
                        lop, loq = _node_lo(m_edges, wid, i, ia), _node_lo(m_edges, wid, j, jb)
                        if st == 0:
                            m_own, w_own, l_own, m_oth, w_oth, l_oth = mp, wid[i, ia], lop, mq, wid[j, jb], loq
                        else:
                            m_own, w_own, l_own, m_oth, w_oth, l_oth = mq, wid[j, jb], loq, mp, wid[i, ia], lop
                        s_own = min(mr / m_own, 1.0)
                        s_oth = max(0.0, min((mr - m_own) / m_oth, 1.0))
                        hw = _bounded_hw(0.5 * (w_own * s_own + w_oth * s_oth), mr,
                                         l_own * s_own + l_oth * s_oth,
                                         (l_own + w_own) * s_own + (l_oth + w_oth) * s_oth)
                        _deposit_segment(F, H, i, j, fp, fq, r, mr, hw, m_edges, keep_small, geo_m)
                    mf = frag[st, c, p, q]
                    if mf > 0.0:
                        _fragments(F, H, D, i, j, fp, fq, r, mf, a_lo[st, c, p, q], a_hi[st, c, p, q], m_edges,
                                   a_edges, GM, GN, eM, eN, mfac, keep_small, geo_a)


def _kernel_averaged_terms(model, n_sub, closure, rule, products=True):
    """A replacement for ``collisions._terms`` that averages each node pair's rate coefficients over the
    n_sub x n_sub sub-slices of the two nodes. With ``products``, the products of each pair (remnant and fragment
    masses, fragment ranges) are the rate-weighted means over the same sub-pairs, so the colliding sub-population's
    own masses are deposited; otherwise they stay at the nodes. Channel models only."""
    from pycalima.gsd import collisions as Cm

    def terms(grid, N, M, model_):
        m, w, wid, ms = nodes(grid, N, M, closure, rule, n_sub)
        mf = m.reshape(-1)
        a = grid.radius(mf)
        chans = model_.channels(a, mf)
        msub = ms.reshape(-1)                         # node p's slices at p * n_sub + k
        asub = grid.radius(msub)
        sub = model_.channels(asub, msub)
        P = mf.size
        Ks = []
        sets = [[[], [], [], []], [[], [], [], []]]
        for ch, cs in zip(chans, sub):
            Kf = np.asarray(cs.K, float).reshape(P, n_sub, P, n_sub)
            Kb = Kf.mean(axis=(1, 3))
            Ks.append(Kb)
            for st in (0, 1):
                for k in range(4):
                    if products:
                        x = np.broadcast_to(np.asarray(cs.sets[st][k], float), (P * n_sub, P * n_sub))
                        x = x.reshape(P, n_sub, P, n_sub)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            avg = np.where(Kb > 0, (Kf * x).mean(axis=(1, 3)) / np.where(Kb > 0, Kb, 1.0),
                                           np.broadcast_to(np.asarray(ch.sets[st][k], float), (P, P)))
                        sets[st][k].append(avg)
                    else:
                        sets[st][k].append(np.broadcast_to(np.asarray(ch.sets[st][k], float), (P, P)))
        Kst = np.ascontiguousarray(np.array(Ks, float))
        stack = [np.ascontiguousarray(np.array([sets[st][k] for st in (0, 1)], float)) for k in range(4)]
        if not products:
            return Cm._pair_terms(m, w, wid, Kst, *stack, grid.m_edges, grid.a_edges, float(grid.rho),
                                  float(model_.alpha_f), model_.collector == "smallest")
        # rate-weighted masses of the colliding sub-populations, per pair and channel
        mp_s = msub.reshape(P, n_sub)
        mpe, mqe = [], []
        for cs in sub:
            Kf = np.asarray(cs.K, float).reshape(P, n_sub, P, n_sub)
            Kb = Kf.sum(axis=(1, 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                ep = np.where(Kb > 0, np.einsum("ikjl,ik->ij", Kf, mp_s) / np.where(Kb > 0, Kb, 1.0),
                              mf[:, None] + 0 * mf[None, :])
                eq = np.where(Kb > 0, np.einsum("ikjl,jl->ij", Kf, mp_s) / np.where(Kb > 0, Kb, 1.0),
                              mf[None, :] + 0 * mf[:, None])
            mpe.append(ep)
            mqe.append(eq)
        mpe = np.ascontiguousarray(np.array(mpe, float))
        mqe = np.ascontiguousarray(np.array(mqe, float))
        alpha = float(model_.alpha_f)
        eM, eN = 4.0 - alpha, 1.0 - alpha
        ae = grid.a_edges
        args = (P, m, w, wid, Kst, *stack, mpe, mqe, grid.m_edges, ae, ae ** eM, ae ** eN, eM, eN,
                4.0 / 3.0 * math.pi * float(grid.rho), model_.collector == "smallest")
        work = Kst.shape[0] * P * (P + 1) // 2
        return Cm.run_rows(_pair_rows_eff, args, grid.n_bins, ae, float(grid.rho), alpha, Cm.threads_for(work))
    return terms


@contextlib.contextmanager
def patched(closure=None, rule="median", ka=1, ka_products=True):
    """Route the production operators through a variant for the duration of the block:
    ``gsd.remap.remap`` as used by exchange (accretion) and sputtering -> ``remap`` with ``closure``;
    ``collisions.bin_nodes`` -> ``nodes`` (closure, rule); with ka > 1 ``collisions._terms`` averages the kernels
    (channel models only). ``closure=None`` keeps PL2M's remap."""
    from pycalima.gsd import collisions as Cm
    from pycalima.gsd import exchange as Ex
    from pycalima.gsd import sputtering as Sp

    saved = (Ex.remap, Sp.remap, Cm.bin_nodes, Cm._terms)
    cl = closure if closure is not None else make_closure("pl2m")

    def remap_v(grid, N, M, image, n_pieces=4, profile=None):
        return remap(grid, N, M, image, cl, n_pieces=n_pieces)

    def bin_nodes_v(grid, N, M, widths=False):
        m, w, wid, _ = nodes(grid, N, M, cl, rule, 1)
        return (m, w, wid) if widths else (m, w)

    try:
        if closure is not None:
            Ex.remap = remap_v
            Sp.remap = remap_v
        if closure is not None or rule != "median":
            Cm.bin_nodes = bin_nodes_v
        if ka > 1:
            Cm._terms = _kernel_averaged_terms(None, int(ka), cl, rule, ka_products)
        yield
    finally:
        Ex.remap, Sp.remap, Cm.bin_nodes, Cm._terms = saved
