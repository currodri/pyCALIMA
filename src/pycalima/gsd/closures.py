"""In-bin closures of the two-moment GSD and the solver method that selects them.

Every method stores the same two scalars per bin, N and M; they differ in how the number distribution inside a
bin is rebuilt from them (the *closure*) and in how the collision operator samples it (the *nodes*). A closure gives
each bin a few *pieces*: power laws dN/du = A u^lam on a support [u1, u2] inside the bin (u = a / a_L) or spikes
(all of a piece's number at one radius). The pieces of a bin carry exactly its N and M, so the analytic remap
along characteristics (``remap``) and every integral over the profile are exact for all closures, positivity and
realisability (m_L <= M/N <= m_R) hold by construction, and number and mass are conserved as with PL2M.

Closures (``GSDMethod.closure``):

``pl2m``
    one power law over the whole bin, its exponent fixed by M/N (``pl2m.BinProfile``; spikes beyond the slope cap).
``ea`` (default)
    edge-aware: a front continues the neighbour's own PL2M exponent (to the extent that the neighbour is not itself
    a front; else ``EA_DEFAULT_EXPONENT``), so any power-law segment with sharp cut-offs is exact. Otherwise as
    ``eaf`` below. Its front shapes depend on the neighbour's exponent as well as on s, so per-gram bin averages are
    taken from per-radius node tables integrated over the pieces (``optics.node_average``), not from s tables.
``eaf``
    edge-aware with a fixed front exponent. A bin whose right (left) neighbour is empty -- its mean dn/da below
    ``eps`` times the bin's own -- is a front: its grains follow u^front_exponent on [1, u_c] ([u_c, u_R]), u_c
    from M/N; a bin with both neighbours empty is a spike at its mean mass (moving pivot). Front weights
    e = max(0, 1 - q / eps) blend these with the PL2M profile continuously (a convex mix of profiles with the same
    N and M keeps N and M). The grid ends count as empty neighbours. Every component is a function of the in-bin
    state s alone, so optics and rate tables stay one-dimensional in s: one table per component (base = PL2M,
    right front, left front, spike) and the component fractions from the neighbours (``optics``).
``affine``
    dn/da linear in a while positive, a triangle with partial support beyond (local); stored as ``N_SUB``
    power-law sub-pieces fitted to its exact sub-moments.
``exp``
    dN/dm ~ exp(-kappa m), the maximum-entropy profile for given N and M (local); ``N_SUB`` sub-pieces likewise.

Nodes (``GSDMethod.nodes`` for collisions, ``GSDMethod.rate_nodes`` for rates such as ``charging.destroy``):
``median`` (the number halves at their mean masses, as PL2M) or ``gauss`` (the 2-point Gauss rule of the bin's
measure in radius: matches the integrals of a^0..a^3, so N, the surface and M exactly; positive weights, nodes
inside the support). ``kernel_average`` = n > 1 averages each node pair's collision kernel and products over n x n
equal-number sub-slices of the two nodes (``collisions``; channel models).

Drift velocities per moment (``moment_velocities``) always use the local base closure (PL2M for pl2m/ea/eaf):
the isolated-bin spike of ea/eaf would freeze the segregation of a bin's grains.

The benchmarks behind the choice are in results/gsd_overview/variants/ and diagnostics/gsd/bench_pl2m_variants.py
(EAF was the default on 2026-11-18; EA from 2026-11-19, after EAF's fixed front slope showed up as edge peaks on flat
profiles).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pycalima.collapse._jit import njit

from ._kernels import _bin_of
from .pl2m import BinProfile

CLOSURES = ("pl2m", "ea", "eaf", "affine", "exp")
NODE_RULES = ("median", "gauss")
N_SUB = 8                    # power-law sub-pieces of the affine and exp closures
NARROW = 1e-7                # truncated supports narrower than this (relative) are spikes
EA_DEFAULT_EXPONENT = 0.0    # ea: front exponent when the neighbour is itself a front (flat dn/da)
COMPONENTS = ("base", "right", "left", "spike")


@dataclass(frozen=True)
class GSDMethod:
    """How the GSD operators rebuild and sample the in-bin distribution.

    - ``closure``: 'ea' (default), 'eaf', 'pl2m', 'affine' or 'exp'
    - ``eps``: front threshold of ea/eaf: a neighbour whose mean dn/da is below eps times the bin's counts as
      empty (default 1e-6; larger values flag steep smooth slopes on wide bins as fronts)
    - ``front_exponent``: radius exponent of eaf's front profiles, dN/da ~ a^front_exponent (default -3.5, MRN)
    - ``nodes``: collision nodes, 'median' (default) or 'gauss'
    - ``rate_nodes``: nodes for rates evaluated at nodes (``charging.destroy``), 'gauss' (default) or 'median'
    - ``kernel_average``: n >= 1; n > 1 averages collision kernels and products over n x n node sub-slices
    """

    closure: str = "ea"
    eps: float = 1e-6
    front_exponent: float = -3.5
    nodes: str = "median"
    rate_nodes: str = "gauss"
    kernel_average: int = 1

    def __post_init__(self):
        if self.closure not in CLOSURES:
            raise ValueError(f"unknown closure {self.closure!r}; choose from {CLOSURES}")
        for name in ("nodes", "rate_nodes"):
            if getattr(self, name) not in NODE_RULES:
                raise ValueError(f"unknown {name} {getattr(self, name)!r}; choose from {NODE_RULES}")
        if not (np.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be positive and finite, got {self.eps!r}")
        if not np.isfinite(self.front_exponent):
            raise ValueError(f"front_exponent must be finite, got {self.front_exponent!r}")
        if int(self.kernel_average) != self.kernel_average or self.kernel_average < 1:
            raise ValueError(f"kernel_average must be an integer >= 1, got {self.kernel_average!r}")

    @property
    def local(self) -> bool:
        """True if each bin's profile depends on its own (N, M) only."""
        return self.closure in ("pl2m", "affine", "exp")


#: named methods for ``method=<name>``; 'pl2m' is the method before 2026-11-18 (median nodes everywhere)
METHODS = {
    "ea": GSDMethod(),
    "eaf": GSDMethod(closure="eaf"),
    "pl2m": GSDMethod(closure="pl2m", rate_nodes="median"),
    "affine": GSDMethod(closure="affine"),
    "exp": GSDMethod(closure="exp"),
}
#: the default of every operator: EA, median collision nodes, Gauss rate nodes, no kernel averaging
DEFAULT_METHOD = METHODS["ea"]


def resolve(method=None) -> GSDMethod:
    """The ``GSDMethod`` for ``method``: None -> ``DEFAULT_METHOD`` (read at call time), a name of ``METHODS``,
    or a ``GSDMethod``; anything else raises."""
    if method is None:
        return DEFAULT_METHOD
    if isinstance(method, GSDMethod):
        return method
    if isinstance(method, str):
        if method not in METHODS:
            raise ValueError(f"unknown GSD method {method!r}; choose from {tuple(METHODS)} or pass a GSDMethod")
        return METHODS[method]
    raise TypeError(f"method must be None, a name or a GSDMethod, not {type(method).__name__}")


# ---------------------------------------------------------------------------------------------------------------------
# compiled kernels: power-law integrals, truncation solve, affine and exp closures, piece remap, cumulative number
# (verified in diagnostics/gsd/pl2m_variants.py)
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


@njit(cache=True)
def _amplitudes(n, lam, u1, u2, spike):
    A = np.zeros(n.shape)
    for i in range(n.shape[0]):
        for k in range(n.shape[1]):
            if n[i, k] != 0.0 and not spike[i, k]:
                A[i, k] = n[i, k] / _ip(lam[i, k], u1[i, k], u2[i, k])
    return A


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
                j = _bin_of(a_e, Xd)
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
                    j = _bin_of(a_e, Xp)
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
                    oM += mfac * _cubic_p(Ai, li, c0, d1, u_lo, uq_)
                j0 = max(_bin_of(a_e, Xp), 0)
                j1 = min(_bin_of(a_e, Xq), nb - 1)
                for j in range(j0, j1 + 1):
                    x1 = max(Xp, a_e[j])
                    x2 = min(Xq, a_e[j + 1])
                    if x2 <= x1:
                        continue
                    u1_ = (p + (x1 - Xp) / s) / a0
                    u2_ = (p + (x2 - Xp) / s) / a0
                    N_new[j] += Ai * _ip(li, u1_, u2_)
                    M_new[j] += mfac * _cubic_p(Ai, li, c0, d1, u1_, u2_)
    return N_new, M_new, lost, oN, oM


@njit(cache=True)
def _cubic_p(A, lam, d0, d1, u1, u2):
    return A * (d0 ** 3 * _ip(lam, u1, u2) + 3.0 * d0 ** 2 * d1 * _ip(lam + 1.0, u1, u2)
                + 3.0 * d0 * d1 ** 2 * _ip(lam + 2.0, u1, u2) + d1 ** 3 * _ip(lam + 3.0, u1, u2))


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


# ---------------------------------------------------------------------------------------------------------------------
# edge-aware closures on the PL2M base
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _ea_nb(N, r, uR, da, lam_b, delta_b, u_delta, eps, lam_front, fixed):
    """Edge-aware pieces [nb, 4]: base (PL2M: lam_b on [1, uR], or a spike at u_delta), right front, left front,
    isolated spike. fixed: eaf (front exponent lam_front); else ea (the neighbour's exponent, weighted by how far
    the neighbour is interior; lam_front where it is a front)."""
    nb = N.size
    eR_ = np.zeros(nb)
    eL_ = np.zeros(nb)
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        dens = N[i] / da[i]
        if not dens > 0.0:
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
        w0 = (1.0 - eL) * (1.0 - eR)
        w1 = eR * (1.0 - eL)
        w2 = eL * (1.0 - eR)
        w3 = eL * eR
        if delta_b[i]:
            sp[i, 0] = True
            u1[i, 0] = u2[i, 0] = u_delta[i]
        else:
            lam[i, 0], u1[i, 0], u2[i, 0] = lam_b[i], 1.0, uR[i]
        n[i, 0] = w0 * N[i]
        ri = r[i]
        if w1 > 0.0:
            lr = lam_front
            if not fixed and i > 0 and N[i - 1] > 0.0 and not delta_b[i - 1]:
                lr = (1.0 - eL_[i - 1]) * lam_b[i - 1] + eL_[i - 1] * lam_front
            if 1.0 + 1e-13 < ri <= _mom(lr, 1.0, uR[i], 3.0):
                lam[i, 1], u1[i, 1], u2[i, 1] = lr, 1.0, _solve_cut(ri, lr, uR[i], True)
                if u2[i, 1] < 1.0 + NARROW:
                    sp[i, 1] = True
                    u1[i, 1] = u2[i, 1] = ri ** (1.0 / 3.0)
                n[i, 1] = w1 * N[i]
            else:
                n[i, 0] += w1 * N[i]
        if w2 > 0.0:
            ll = lam_front
            if not fixed and i < nb - 1 and N[i + 1] > 0.0 and not delta_b[i + 1]:
                ll = (1.0 - eR_[i + 1]) * lam_b[i + 1] + eR_[i + 1] * lam_front
            if _mom(ll, 1.0, uR[i], 3.0) <= ri < uR[i] ** 3 * (1.0 - 1e-13):
                lam[i, 2], u1[i, 2], u2[i, 2] = ll, _solve_cut(ri, ll, uR[i], False), uR[i]
                if u2[i, 2] < u1[i, 2] * (1.0 + NARROW):
                    sp[i, 2] = True
                    u1[i, 2] = u2[i, 2] = ri ** (1.0 / 3.0)
                n[i, 2] = w2 * N[i]
            else:
                n[i, 0] += w2 * N[i]
        if w3 > 0.0:
            u1[i, 3] = u2[i, 3] = ri ** (1.0 / 3.0)
            sp[i, 3] = True
            n[i, 3] = w3 * N[i]
    return n, lam, u1, u2, sp


# ---------------------------------------------------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------------------------------------------------

class Pieces:
    """The pieces of every bin, arrays [nb, kp]: piece k of bin i is dN/du = A u^lam on [u1, u2] (u = a / a_L,i)
    with number n, or, if ``spike``, all of n at u1 (= u2). Unused pieces have n = 0."""

    def __init__(self, grid, n, A, lam, u1, u2, spike):
        self.grid = grid
        self.n, self.A, self.lam, self.u1, self.u2, self.spike = n, A, lam, u1, u2, spike
        self.aL = np.asarray(grid.a_edges[:-1], float)
        self.mL = np.asarray(grid.m_edges[:-1], float)

    @property
    def kp(self):
        return self.n.shape[1]


def _state(grid, N, M):
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    eta = grid.etas
    filled = N > 0
    r = np.where(filled, M / np.where(filled, N, 1.0) / grid.m_edges[:-1], 0.5 * (1 + eta))
    return N, M, np.clip(r, 1.0, eta), np.ascontiguousarray(eta ** (1.0 / 3.0), float), filled


def _from_profile(grid, prof):
    n = np.where(prof.N > 0, prof.N, 0.0)[:, None].copy()
    sp = np.asarray(prof.delta, bool)[:, None].copy()
    u1 = np.where(prof.delta, prof.u_delta, 1.0)[:, None].copy()
    u2 = np.where(prof.delta, prof.u_delta, prof.uR_bins)[:, None].copy()
    lam = np.asarray(prof.lam, float)[:, None].copy()
    A = np.where(prof.delta, 0.0, prof.A)[:, None].copy()
    return Pieces(grid, n, A, lam, u1, u2, sp)


def pieces(grid, N, M, method=None) -> Pieces:
    """The pieces of every bin of (N, M) under ``method``'s closure."""
    meth = resolve(method)
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    if meth.closure == "pl2m":
        return _from_profile(grid, BinProfile(grid, N, M))
    Nn, Mm, r, uR, filled = _state(grid, N, M)
    if meth.closure in ("ea", "eaf"):
        prof = BinProfile(grid, N, M)
        n, lam, u1, u2, sp = _ea_nb(Nn, r, uR, np.diff(grid.a_edges), np.asarray(prof.lam, float),
                                    np.asarray(prof.delta, np.bool_), np.asarray(prof.u_delta, float),
                                    float(meth.eps),
                                    float(meth.front_exponent if meth.closure == "eaf" else EA_DEFAULT_EXPONENT),
                                    meth.closure == "eaf")
    elif meth.closure == "affine":
        n, lam, u1, u2, sp = _aff_nb(Nn, r, uR, N_SUB)
    else:
        n, lam, u1, u2, sp = _exp_nb(Nn, r, uR, N_SUB)
    return Pieces(grid, n, _amplitudes(n, lam, u1, u2, sp), lam, u1, u2, sp)


def base_pieces(grid, N, M, method=None) -> Pieces:
    """The local base closure of ``method`` (PL2M for pl2m/ea/eaf, the closure itself for affine/exp): what
    per-moment drift velocities use."""
    meth = resolve(method)
    if meth.closure in ("pl2m", "ea", "eaf"):
        return pieces(grid, N, M, "pl2m")
    return pieces(grid, N, M, meth)


def component_fractions(grid, N, M, method=None) -> np.ndarray:
    """Number (= mass) fractions [nb, 4] of the components (base, right front, left front, spike) of every bin under
    ea/eaf; [1, 0, 0, 0] for the local closures. Every component carries the bin's mean mass, so a per-gram bin
    quantity is sum_c f_c kappa_c(s) (``optics.combine_components``)."""
    meth = resolve(method)
    N = np.asarray(N, float)
    out = np.zeros((N.size, 4))
    out[:, 0] = 1.0
    if meth.closure not in ("ea", "eaf"):
        return out
    P = pieces(grid, N, M, meth)
    filled = N > 0
    out[filled] = P.n[filled] / N[filled, None]
    return out


def bin_moments(P, k):
    """Integral of u^k dN per bin (u = a / a_L): k = 0 number, k = 3 mass / m_L."""
    return _bin_moments(P.n, P.A, P.lam, P.u1, P.u2, P.spike, float(k))


def radius_moments(P, k):
    """Integral of a^k dN per bin [cm^k]."""
    return bin_moments(P, k) * P.aL ** k


def remap_pieces(grid, P, image, n_pieces=4):
    """Remap the pieces along the flow map ``image`` (a vectorised, monotone a -> X(a)); each piece's support is
    split into ``n_pieces`` sub-intervals whose images are linear in a. Returns (N, M, lost_N, overflow_N,
    overflow_M) before the round-off clip of ``remap.realisable``."""
    t = np.linspace(0.0, 1.0, n_pieces + 1)
    lo = P.aL[:, None] * P.u1
    hi = P.aL[:, None] * P.u2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lo > 0, hi / np.where(lo > 0, lo, 1.0), 1.0)
    nodes = lo[:, :, None] * ratio[:, :, None] ** t[None, None, :]
    X = np.asarray(image(nodes.ravel()), float).reshape(nodes.shape)
    return _remap_pieces(grid.a_edges, float(grid.rho), P.aL, P.n, P.A, P.lam, P.spike, nodes, X)


def samples(grid, N, M, method=None, step=0.01):
    """Quadrature nodes (a, dN) of the reconstruction: trapezoid in ln a with steps of ``step`` per piece, spikes as
    single points."""
    P = pieces(grid, N, M, method)
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
            m = max(12, int(math.ceil(Lw / step)))
            u = np.geomspace(P.u1[i, j], P.u2[i, j], m + 1)
            w = np.full(m + 1, Lw / m)
            w[[0, -1]] *= 0.5
            a_out.append(P.aL[i] * u)
            w_out.append(P.A[i, j] * u ** (P.lam[i, j] + 1.0) * w)
    if not a_out:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(a_out), np.concatenate(w_out)


# ---------------------------------------------------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _quantile_one(lam, u1, u2, f):
    """u below which a fraction f of the number of u^lam on [u1, u2] lies (closed form)."""
    L = math.log(u2 / u1)
    p = lam + 1.0
    if abs(p * L) < 1e-12:
        return u1 * math.exp(f * L)
    if p * L > 0.0:
        # u^p = u1^p (1 + f expm1(pL)), in logs
        return u1 * math.exp(math.log1p(f * math.expm1(p * L)) / p)
    return u1 * math.exp(math.log1p(f * math.expm1(p * L)) / p)


@njit(cache=True)
def _lowest(n, A, lam, u1, u2, spike, i, uR, t, single):
    """(u*, mass / m_L of the t lowest grains of bin i): closed form for a single power-law piece, bisection on the
    cumulative number otherwise."""
    if single >= 0:
        ntot = n[i, single]
        f = min(max(t / ntot, 0.0), 1.0)
        if f <= 0.0:
            return u1[i, single], 0.0
        uq = _quantile_one(lam[i, single], u1[i, single], u2[i, single], f)
        uq = min(max(uq, u1[i, single]), u2[i, single])
        return uq, t * _mom(lam[i, single], u1[i, single], uq, 3.0)
    return _mass_of_lowest(n, A, lam, u1, u2, spike, i, uR, t)


@njit(cache=True)
def _nodes_nb(N, M, mL, mR, uR, n, A, lam, u1, u2, spike, gauss, n_sub):
    """Two nodes per bin (masses m, weights w, interval widths wid) and n_sub equal-number sub-slices per node
    (ms[nb, 2, n_sub]); gauss: the 2-point Gauss rule in radius, else the median split."""
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
            mb = min(max(mbar, mL[i]), mR[i])
            m[i, 0] = m[i, 1] = mb
            w[i, 0] = w[i, 1] = 0.5 * N[i]
            for h in range(2):
                for k in range(n_sub):
                    ms[i, h, k] = mb
            continue
        allspike = True
        active = 0
        single = -1
        for j in range(n.shape[1]):
            if n[i, j] != 0.0:
                active += 1
                single = j
                if not spike[i, j]:
                    allspike = False
        if active != 1 or spike[i, single]:
            single = -1
        if gauss:
            mu = np.zeros(4)
            for j in range(n.shape[1]):
                if n[i, j] == 0.0:
                    continue
                for k in range(4):
                    if spike[i, j]:
                        mu[k] += n[i, j] * u1[i, j] ** k
                    else:
                        mu[k] += n[i, j] * _mom(lam[i, j], u1[i, j], u2[i, j], float(k))
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
                x1 = (e1 + sg * z1) ** 3 * mL[i]
                x2 = (e1 + sg * z2) ** 3 * mL[i]
                if p1 < 1e-10 or p1 > 1.0 - 1e-10:
                    p1 = 0.5
                    x1 = x2 = mbar
            w[i, 0] = p1 * N[i]
            w[i, 1] = (1.0 - p1) * N[i]
            x1 = min(max(x1, mL[i]), mR[i])
            if w[i, 0] > 0.0 and w[i, 1] > 0.0:
                x2 = (M[i] - w[i, 0] * x1) / w[i, 1]
            else:
                x1 = x2 = mbar
            m[i, 0], m[i, 1] = x1, min(max(x2, mL[i]), mR[i])
            tsplit = w[i, 0]
        else:
            w[i, 0] = w[i, 1] = 0.5 * N[i]
            tsplit = 0.5 * N[i]
            if allspike:
                m[i, 0] = m[i, 1] = mbar
            else:
                us, mlow = _lowest(n, A, lam, u1, u2, spike, i, uR[i], tsplit, single)
                x1 = min(max(mL[i] * mlow / tsplit, mL[i]), mR[i])
                m[i, 0], m[i, 1] = x1, min(max(2.0 * mbar - x1, mL[i]), mR[i])
        if allspike or not (w[i, 0] > 0.0 and w[i, 1] > 0.0):
            for h in range(2):
                for k in range(n_sub):
                    ms[i, h, k] = mbar
            continue
        us, _ = _lowest(n, A, lam, u1, u2, spike, i, uR[i], tsplit, single)
        mq = mL[i] * us ** 3
        wid[i, 0] = mq - mL[i]
        wid[i, 1] = mR[i] - mq
        if n_sub <= 1:
            ms[i, 0, 0], ms[i, 1, 0] = m[i, 0], m[i, 1]
            continue
        for h in range(2):
            t0 = 0.0 if h == 0 else tsplit
            tw = w[i, h]
            prev = 0.0
            if h == 1:
                _u, prev = _lowest(n, A, lam, u1, u2, spike, i, uR[i], t0, single)
            for k in range(n_sub):
                t1 = t0 + tw * (k + 1) / n_sub
                if h == 1 and k == n_sub - 1:
                    cur = M[i] / mL[i]
                else:
                    _u, cur = _lowest(n, A, lam, u1, u2, spike, i, uR[i], t1, single)
                ms[i, h, k] = min(max(mL[i] * (cur - prev) / (tw / n_sub), mL[i]), mR[i])
                prev = cur
    return m, w, wid, ms


def nodes(grid, N, M, method=None, rule=None, n_sub=1):
    """(m[nb, 2], w[nb, 2], wid[nb, 2], m_sub[nb, 2, n_sub]): the two nodes of every bin under ``method``'s closure
    with ``rule`` ('median' or 'gauss'; default ``method.nodes``) and n_sub equal-number sub-slices of each node.
    The weights sum to N and the weighted masses to M exactly; 'gauss' also reproduces the integrals of a and a^2."""
    meth = resolve(method)
    rule = meth.nodes if rule is None else rule
    if rule not in NODE_RULES:
        raise ValueError(f"unknown node rule {rule!r}; choose from {NODE_RULES}")
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    P = pieces(grid, N, M, meth)
    uR = np.ascontiguousarray(grid.etas ** (1.0 / 3.0), float)
    return _nodes_nb(N, M, np.ascontiguousarray(grid.m_edges[:-1], float),
                     np.ascontiguousarray(grid.m_edges[1:], float), uR, P.n, P.A, P.lam, P.u1, P.u2, P.spike,
                     rule == "gauss", int(n_sub))


# ---------------------------------------------------------------------------------------------------------------------
# weights of bin averages (optics, rates, drift)
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _hat_into(Wrow, u, ud, scale):
    j = 0
    nu = u.size
    if ud <= u[0]:
        j = 0
    elif ud >= u[nu - 1]:
        j = nu - 2
    else:
        while j < nu - 2 and u[j + 1] <= ud:
            j += 1
    t = min(max((ud - u[j]) / (u[j + 1] - u[j]), 0.0), 1.0)
    Wrow[j] += scale * (1.0 - t)
    Wrow[j + 1] += scale * t


@njit(cache=True)
def _piece_weights(u, lam, u1, u2, spike, n, extra, out):
    """Weights over the nodes u of the average of a quantity linear in u between nodes, under the measure
    u^extra dN of each bin's pieces (extra = 0: number, 3: mass, 1: radius ...). out[nb, len(u)], rows sum to 1."""
    nb, kp = n.shape
    nu = u.shape[1]
    for i in range(nb):
        tot = 0.0
        for j in range(kp):
            if n[i, j] == 0.0:
                continue
            if spike[i, j]:
                wgt = n[i, j] * u1[i, j] ** extra
                _hat_into(out[i], u[i], u1[i, j], wgt)
                tot += wgt
                continue
            p = lam[i, j] + extra
            # weight of the piece in this measure, then the hat integrals over its support
            wgt = n[i, j] * _mom(lam[i, j], u1[i, j], u2[i, j], extra)
            norm = _ip(p, u1[i, j], u2[i, j])
            if not norm > 0.0:
                continue
            for k in range(nu - 1):
                a = max(u[i, k], u1[i, j])
                b = min(u[i, k + 1], u2[i, j])
                if b <= a:
                    continue
                h = u[i, k + 1] - u[i, k]
                i0 = _ip(p, a, b)
                i1 = _ip(p + 1.0, a, b)
                out[i, k] += wgt * (u[i, k + 1] * i0 - i1) / h / norm
                out[i, k + 1] += wgt * (i1 - u[i, k] * i0) / h / norm
            tot += wgt
        if tot > 0.0:
            for k in range(nu):
                out[i, k] /= tot


def bin_weights(grid, P, u_nodes, measure="mass"):
    """W[nb, n_u] such that the per-bin average of a quantity k(u), linear in u between the nodes u_nodes[i, :] of
    each bin, is W[i] @ k(u_nodes[i]): ``measure`` 'mass' (per-gram quantities: sum k m dN / M) or 'number'."""
    extra = {"mass": 3.0, "number": 0.0}[measure]
    u_nodes = np.ascontiguousarray(np.atleast_2d(u_nodes), float)
    if u_nodes.shape[0] == 1 and grid.n_bins > 1:
        u_nodes = np.ascontiguousarray(np.repeat(u_nodes, grid.n_bins, axis=0))
    out = np.zeros(u_nodes.shape)
    _piece_weights(u_nodes, P.lam, P.u1, P.u2, P.spike, P.n, extra, out)
    return out


def moment_velocities(grid, N, M, w_of_a, method=None, n_nodes=32):
    """Per-moment drift velocities (w_N, w_M) of every bin for a size-dependent velocity w(a) (vectorised): the
    number- and mass-weighted means of w over the bin's *local base* closure (``base_pieces``; never the isolated-bin
    spike of ea/eaf). w is taken linear in a between ``n_nodes`` + 1 nodes per bin (exact for w linear in a)."""
    P = base_pieces(grid, N, M, method)
    t = np.linspace(0.0, 1.0, n_nodes + 1)
    uR = grid.etas ** (1.0 / 3.0)
    u = 1.0 + (uR[:, None] - 1.0) * t[None, :]
    wv = np.asarray(w_of_a(P.aL[:, None] * u), float)
    wN = np.einsum("ij,ij->i", bin_weights(grid, P, u, "number"), wv)
    wM = np.einsum("ij,ij->i", bin_weights(grid, P, u, "mass"), wv)
    return wN, wM


# --- EAF components as functions of the in-bin state s (one-bin evaluations) ------------------------------------

def component_pieces(eta, s, component, method=None):
    """Pieces (arrays [len(s), 1]) of one component of the ea/eaf family in a bin of mass ratio eta at states s:
    'base' (PL2M), 'right' (u^front_exponent on [1, u_c]), 'left' ([u_c, u_R]) or 'spike' (at the mean mass). A
    front state that cannot reach s falls back to the base profile, as the closure does."""
    meth = resolve(method)
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; choose from {COMPONENTS}")
    if meth.closure != "eaf" and component != "base":
        raise ValueError("front components are tabulable in s only for eaf (ea's front exponent varies per cell)")
    s = np.atleast_1d(np.asarray(s, float))
    r = 1.0 + np.clip(s, 0.0, 1.0) * (eta - 1.0)
    uR = eta ** (1.0 / 3.0)
    from .grid import MassGrid
    g1 = MassGrid(np.array([1.0, float(eta)]), 1.0)
    out_n = np.zeros((s.size, 1))
    out_lam = np.zeros((s.size, 1))
    out_u1 = np.ones((s.size, 1))
    out_u2 = np.ones((s.size, 1))
    lam_f = float(meth.front_exponent)
    sp = np.zeros((s.size, 1), bool)
    for k, rk in enumerate(r):
        if component == "spike":
            out_n[k], out_u1[k], out_u2[k] = 1.0, rk ** (1.0 / 3.0), rk ** (1.0 / 3.0)
            sp[k] = True
            continue
        if component == "right" and 1.0 + 1e-13 < rk <= _mom(lam_f, 1.0, uR, 3.0):
            uc = _solve_cut(rk, lam_f, uR, True)
            if uc < 1.0 + NARROW:
                out_n[k], out_u1[k], out_u2[k], sp[k] = 1.0, rk ** (1 / 3), rk ** (1 / 3), True
            else:
                out_n[k], out_lam[k], out_u1[k], out_u2[k] = 1.0, lam_f, 1.0, uc
            continue
        if component == "left" and _mom(lam_f, 1.0, uR, 3.0) <= rk < uR ** 3 * (1.0 - 1e-13):
            uc = _solve_cut(rk, lam_f, uR, False)
            if uR < uc * (1.0 + NARROW):
                out_n[k], out_u1[k], out_u2[k], sp[k] = 1.0, rk ** (1 / 3), rk ** (1 / 3), True
            else:
                out_n[k], out_lam[k], out_u1[k], out_u2[k] = 1.0, lam_f, uc, uR
            continue
        prof = BinProfile(g1, np.array([1.0]), np.array([rk]))
        if prof.delta[0]:
            out_n[k], out_u1[k], out_u2[k], sp[k] = 1.0, prof.u_delta[0], prof.u_delta[0], True
        else:
            out_n[k], out_lam[k], out_u1[k], out_u2[k] = 1.0, prof.lam[0], 1.0, uR
    return out_n, out_lam, out_u1, out_u2, sp


def local_pieces(eta, s, method=None):
    """Pieces (arrays [len(s), kp]) of a local closure (pl2m, affine, exp) at states s in a bin of mass ratio eta."""
    meth = resolve(method)
    if not meth.local:
        raise ValueError(f"closure {meth.closure!r} is not local; use component_pieces")
    s = np.atleast_1d(np.asarray(s, float))
    r = 1.0 + np.clip(s, 0.0, 1.0) * (eta - 1.0)
    from .grid import MassGrid
    g1 = MassGrid(np.array([1.0, float(eta)]), 1.0)
    rows = [pieces(g1, np.array([1.0]), np.array([rk]), meth) for rk in r]
    kp = max(p.kp for p in rows)

    def stack(attr, fill):
        out = np.full((s.size, kp), fill, dtype=float if attr != "spike" else bool)
        for k, p in enumerate(rows):
            out[k, :p.kp] = getattr(p, attr)[0]
        return out

    return stack("n", 0.0), stack("lam", 0.0), stack("u1", 1.0), stack("u2", 1.0), stack("spike", False)


def state_weights(eta, s, u, n, lam, u1, u2, sp, measure="mass"):
    """Weights W[len(s), len(u)] of one-bin pieces at states s over the nodes u (shared by all states)."""
    s = np.atleast_1d(np.asarray(s, float))
    U = np.ascontiguousarray(np.repeat(np.asarray(u, float)[None, :], s.size, axis=0))
    out = np.zeros(U.shape)
    _piece_weights(U, np.asarray(lam, float), np.asarray(u1, float), np.asarray(u2, float), np.asarray(sp, np.bool_),
                   np.asarray(n, float), {"mass": 3.0, "number": 0.0}[measure], out)
    return out
