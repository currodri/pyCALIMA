"""The grain-size solver tests of McKinnon et al. (2018, MNRAS 478, 2851; arXiv:1805.04521), Sect. 3, run with PL2M.

McKinnon et al. store N and a slope per bin, linear in a ("piecewise linear", PL; slope limited so the profile stays
positive and the bin mass is kept), or N alone with a flat profile ("piecewise constant", PC). Their tests:

* Fig. 10, growth convergence: dn/da uniform on [a_min, a_cut], a_cut = a_min (a_max/a_min)^(1/4), a = 1 nm - 1 um,
  da/dt = 0.005 um/Gyr for 5 Gyr; fractional dust-mass error against the number of bins N. They find first order for
  PC, second order for PL, and at N = 64 errors of ~40 % (PC) and ~1 % (PL).
* Fig. 13, growth and shrinking: dn/da uniform over the middle quarter of N = 128 bins, da/dt sinusoidal with amplitude
  0.015 um/Gyr and period 1 Gyr; after one period PL needs 3 extra bins beyond the left analytic edge to fall to zero,
  PC 9.
* Fig. 15, shattering: log-normal dn/da (a0 = 0.1 um, sigma = 0.6), n_H = 0.4, D = 3.7e-3, v_rel = 3 km/s only between
  grains both above 0.1 um, every collision shatters both grains into dn/da ~ a^-3.3 over [a_min, a_max]; N = 8,
  100 Myr, against a many-bin integrator.
* Fig. 16, coagulation: dn/da ~ a^-3.5, n_H = 1e5, D = 0.01, v(a) = 1.1e3 (a/0.1 um)^1/2 cm/s, head-on (v_rel =
  v1 + v2), sticking below v_coag = 1.1e3 cm/s; N = 32, 20 Myr, against a many-bin integrator.

Our schemes: EA (the default gsd method since 2026-11-19: edge-aware power laws whose fronts continue the
neighbour's exponent, ``pycalima.gsd.closures``), EAF (fronts with a fixed exponent, -3.5) and PL2M (a power law in
every bin). Here PC and PL are re-implemented from their Sect. 3.2 (Eqs. 31-43) for the two growth
tests, which run in many
steps (their step size is not given; ``--steps``). PL2M uses ``gsd.remap`` (growth) and ``gsd.collisions`` (shattering,
coagulation). The collision references are an independent fine sectional integrator (640 bins, number at the bin
centre, products split between the two neighbouring grid masses so that number and mass are kept; scipy LSODA).
PL2M starts from the exact moments of the initial distribution; McKinnon et al. started both discretisations from
piecewise-constant initial conditions.

    python diagnostics/gsd/check_mckinnon18_tests.py --out-dir results/gsd_overview/figs
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from pycalima.gsd.collisions import Channel, CoagulationModel, collide
from pycalima.gsd.grid import MassGrid
from pycalima.gsd import closures as gc
from pycalima.gsd.remap import remap

UM = 1e-4
GYR = 3.15576e16
MYR = GYR * 1e-3
M_H = 1.6726e-24
RHO = 2.4                 # McKinnon et al.'s grain density [g cm^-3]
A_MIN, A_MAX = 1e-3 * UM, 1.0 * UM
SOLVERS = {"PL2M": "pl2m", "EAF": "eaf", "EA": "ea"}   # gsd methods run here (EA: the default since 2026-11-19)


def mass(a):
    return 4.0 / 3.0 * np.pi * RHO * np.asarray(a) ** 3


def grid_of(n):
    a = np.geomspace(A_MIN, A_MAX, n + 1)
    return a, MassGrid(mass(a), RHO)


def moments(a_edges, dnda, n_sub=400):
    """Exact (N_i, M_i) of a continuous dn/da over the bins (quadrature in ln a)."""
    N, M = np.zeros(a_edges.size - 1), np.zeros(a_edges.size - 1)
    for i in range(a_edges.size - 1):
        x = np.geomspace(a_edges[i], a_edges[i + 1], n_sub + 1)
        y = dnda(x) * x
        N[i] = np.trapezoid(y, np.log(x))
        M[i] = np.trapezoid(y * mass(x), np.log(x))
    return N, M


def boxcar_moments(a_edges, lo, hi, c=1.0):
    """Exact (N_i, M_i) of dn/da = c on [lo, hi]."""
    x0 = np.clip(a_edges[:-1], lo, hi)
    x1 = np.clip(a_edges[1:], lo, hi)
    return c * (x1 - x0), c * np.pi * RHO / 3.0 * (x1 ** 4 - x0 ** 4)


# ---------------------------------------------------------------------------------------------------------------------
# McKinnon et al.'s piecewise-linear (PL) and piecewise-constant (PC) discretisations, Sect. 3.2
# ---------------------------------------------------------------------------------------------------------------------

class LinearBins:
    """dn/da = N_i/(a_{i+1} - a_i) + s_i (a - a^c_i) in each bin (a^c the arithmetic centre); s = 0 for PC."""

    def __init__(self, a_edges, N, M=None, constant=False):
        self.a = np.asarray(a_edges, float)
        self.lo, self.hi = self.a[:-1], self.a[1:]
        self.w = self.hi - self.lo
        self.c = 0.5 * (self.lo + self.hi)
        self.constant = constant
        self.I3 = (self.hi ** 4 - self.lo ** 4) / 4.0
        self.I4c = (self.hi ** 5 - self.lo ** 5) / 5.0 - self.c * self.I3
        self.set(np.asarray(N, float), None if M is None else np.asarray(M, float))

    def set(self, N, M):
        """New (N, M): solve the slopes from the masses (Eq. 39) and slope-limit (Eqs. 40-43); PC drops M."""
        A = N / self.w
        if self.constant or M is None:
            self.N, self.s = N.copy(), np.zeros_like(N)
            return
        s = (3.0 * M / (4.0 * np.pi * RHO) - A * self.I3) / self.I4c
        m3 = 3.0 * M / (4.0 * np.pi * RHO)
        n_hi = A + s * (self.hi - self.c)
        n_lo = A + s * (self.lo - self.c)
        for edge, bad in ((self.hi, n_hi < 0), (self.lo, n_lo < 0)):
            if not bad.any():
                continue
            # Ntilde/w + stilde (edge - c) = 0 and (Ntilde/w) I3 + stilde I4c = m3: keep the mass, zero at the edge
            d = edge[bad] - self.c[bad]
            st = m3[bad] / (self.I4c[bad] - d * self.I3[bad])
            A[bad] = -st * d
            s[bad] = st
        self.N, self.s = A * self.w, s

    def mass_total(self):
        return 4.0 * np.pi * RHO / 3.0 * np.sum(self.N / self.w * self.I3 + self.s * self.I4c)

    def dnda(self, x):
        i = np.clip(np.searchsorted(self.a, x, side="right") - 1, 0, self.N.size - 1)
        return self.N[i] / self.w[i] + self.s[i] * (x - self.c[i])

    def _cum(self, x, da):
        """Number and mass (after a shift da) of the grains below radius x: piecewise antiderivatives."""
        x = np.clip(x, self.a[0], self.a[-1])
        i = np.clip(np.searchsorted(self.a, x, side="right") - 1, 0, self.N.size - 1)
        A = self.N / self.w
        alpha = A - self.s * (da + self.c)                       # n = alpha + s b, with b = a + da
        def part(j, x0, x1):
            n = A[j] * (x1 - x0) + 0.5 * self.s[j] * ((x1 - self.c[j]) ** 2 - (x0 - self.c[j]) ** 2)
            b0, b1 = x0 + da, x1 + da
            m = 4.0 * np.pi * RHO / 3.0 * (alpha[j] * (b1 ** 4 - b0 ** 4) / 4.0 + self.s[j] * (b1 ** 5 - b0 ** 5) / 5.0)
            return n, m
        full_n, full_m = part(np.arange(self.N.size), self.lo, self.hi)
        cn = np.concatenate([[0.0], np.cumsum(full_n)])
        cm = np.concatenate([[0.0], np.cumsum(full_m)])
        pn, pm = part(i, self.lo[i], x)
        return cn[i] + pn, cm[i] + pm

    def shift(self, da, rebin_top=False):
        """Move every grain by da (Eqs. 34, 37) and rebuild the bins. Grains grown past a_max are dropped, or with
        ``rebin_top`` put back into the top bin at a_max keeping their mass (Eqs. 44-46; PC: M/<m> grains)."""
        n1, m1 = self._cum(self.hi - da, da)
        n0, m0 = self._cum(self.lo - da, da)
        N, M = np.maximum(n1 - n0, 0.0), np.maximum(m1 - m0, 0.0)
        m_over = 0.0
        if rebin_top and da > 0:
            m_over = self._cum(self.a[-1], da)[1] - self._cum(self.a[-1] - da, da)[1]
        if m_over <= 0:
            self.set(N, M)
            return
        if self.constant:
            mbar = 4.0 * np.pi * RHO / 3.0 * self.I3[-1] / self.w[-1]
            N[-1] += m_over / mbar
            self.set(N, None)
            return
        self.set(N, M)                                   # slopes of the grains that stayed, then Eqs. 44-46
        j = -1
        A, sj, lo, hi, c, w = self.N[j] / self.w[j], self.s[j], self.lo[j], self.hi[j], self.c[j], self.w[j]
        J1 = (hi ** 2 - lo ** 2) / 2.0
        J2c = (hi ** 3 - lo ** 3) / 3.0 - c * J1
        Nj = self.N[j]
        a_mean = (A * J1 + sj * J2c) / Nj if Nj > 0 else hi
        n_reb = m_over / mass(self.a[-1])
        a_new = (Nj * a_mean + n_reb * self.a[-1]) / (Nj + n_reb)
        m3 = 3.0 * (M[j] + m_over) / (4.0 * np.pi * RHO)
        Nt, st = np.linalg.solve([[J1 / w - a_new, J2c], [self.I3[j] / w, self.I4c[j]]], [0.0, m3])
        M2 = M.copy()
        M2[j] = M[j] + m_over
        N2 = self.N.copy()
        N2[j] = Nt
        self.set(N2, M2)


# ---------------------------------------------------------------------------------------------------------------------
# growth tests
# ---------------------------------------------------------------------------------------------------------------------

def growth_convergence(ns, steps, adot=0.005 * UM / GYR, t_end=5.0 * GYR):
    """Fig. 10: fractional mass error at t_end against N, for PC, PL, PL2M and EAF (the last two also at other step
    counts)."""
    a_cut = A_MIN * (A_MAX / A_MIN) ** 0.25
    da_tot = adot * t_end
    exact = np.pi * RHO / 3.0 * ((a_cut + da_tot) ** 4 - (A_MIN + da_tot) ** 4)
    out = {"N": np.array(ns)}
    for name in ("PC", "PL"):
        err = []
        for n in ns:
            a, _ = grid_of(n)
            N0, M0 = boxcar_moments(a, A_MIN, a_cut)
            b = LinearBins(a, N0, M0, constant=(name == "PC"))
            for _ in range(steps):
                b.shift(da_tot / steps)
            err.append(abs(b.mass_total() / exact - 1.0))
        out[name] = np.array(err)
    for name, meth in SOLVERS.items():
        for k in sorted({1, steps, 10 * steps}):
            err = []
            for n in ns:
                a, g = grid_of(n)
                N, M = boxcar_moments(a, A_MIN, a_cut)
                for _ in range(k):
                    N, M, *_ = remap(g, N, M, lambda x, d=da_tot / k: x + d, method=meth)
                err.append(abs(M.sum() / exact - 1.0))
            out[f"{name}_{k}"] = np.array(err)
    return out


def sinusoid(n, steps_per_period=200, amp=0.015 * UM / GYR, period=1.0 * GYR, t_show=(0.25, 0.5, 0.75, 1.0)):
    """Fig. 13: dn/da at t_show [periods] for PC, PL, PL2M, EAF and the analytic boxcar."""
    a, g = grid_of(n)
    lo, hi = a[3 * n // 8], a[5 * n // 8]
    N0, M0 = boxcar_moments(a, lo, hi)
    pc, pl = LinearBins(a, N0, M0, constant=True), LinearBins(a, N0, M0)
    st = {name: (N0.copy(), M0.copy()) for name in SOLVERS}
    dt = period / steps_per_period
    shift = lambda t: amp * period / (2 * np.pi) * (1.0 - np.cos(2 * np.pi * t / period))   # noqa: E731
    snaps = {}
    hist = {"t": [0.0], "PC": [0.0], "PL": [0.0], **{name: [0.0] for name in SOLVERS}}
    for k in range(1, steps_per_period + 1):
        da = shift(k * dt) - shift((k - 1) * dt)
        pc.shift(da)
        pl.shift(da)
        for name, meth in SOLVERS.items():
            N, M, *_ = remap(g, *st[name], lambda x, d=da: x + d, method=meth)
            st[name] = (N, M)
        m_ex = np.pi * RHO / 3.0 * ((hi + shift(k * dt)) ** 4 - (lo + shift(k * dt)) ** 4)
        hist["t"].append(k * dt / period)
        hist["PC"].append(abs(pc.mass_total() / m_ex - 1.0))
        hist["PL"].append(abs(pl.mass_total() / m_ex - 1.0))
        for name in SOLVERS:
            hist[name].append(abs(st[name][1].sum() / m_ex - 1.0))
        tk = k / steps_per_period
        if any(abs(tk - ts) < 1e-9 for ts in t_show):
            snaps[round(tk, 6)] = dict(PC=(pc.N.copy(), pc.s.copy()), PL=(pl.N.copy(), pl.s.copy()),
                                       **{name: (st[name][0].copy(), st[name][1].copy()) for name in SOLVERS},
                                       analytic=(lo + shift(k * dt), hi + shift(k * dt)))
    snaps['hist'] = {k: np.array(v) for k, v in hist.items()}
    return a, g, (lo, hi), snaps


def left_spread(a_edges, dnda_bins, edge, plateau, tol=1e-3):
    """Bins left of the analytic left edge whose mean dn/da is above tol x plateau (McKinnon's 'extra bins')."""
    j = np.searchsorted(a_edges, edge, side="right") - 1
    left = dnda_bins[:j]
    above = np.nonzero(left > tol * plateau)[0]
    return 0 if above.size == 0 else j - above[0]


# ---------------------------------------------------------------------------------------------------------------------
# collision tests: PL2M models and the fine reference
# ---------------------------------------------------------------------------------------------------------------------

class FullShattering:
    """Fig. 15 model: v_rel = v only if both grains exceed a_thr; both shatter fully into a^-3.3 over the grid."""

    alpha_f = 3.3
    collector = "smallest"

    def __init__(self, v=3e5, a_thr=0.1 * UM):
        self.v, self.a_thr = v, a_thr

    def channels(self, a, m):
        A1, A2 = a[:, None], a[None, :]
        K = np.pi * (A1 + A2) ** 2 * self.v * ((A1 > self.a_thr) & (A2 > self.a_thr))
        one = np.ones_like(K)
        return [Channel(K, frag1=m[:, None] * one, a_hi1=A_MAX * one, a_lo1=A_MIN * one,
                        frag2=m[None, :] * one, a_hi2=A_MAX * one, a_lo2=A_MIN * one)]


def coag_kernel(m1, m2, v0=1.1e3, v_coag=1.1e3):
    a1 = (3.0 * m1 / (4.0 * np.pi * RHO)) ** (1.0 / 3.0)
    a2 = (3.0 * m2 / (4.0 * np.pi * RHO)) ** (1.0 / 3.0)
    vrel = v0 * (np.sqrt(a1 / (0.1 * UM)) + np.sqrt(a2 / (0.1 * UM)))
    return np.pi * (a1 + a2) ** 2 * vrel * (vrel < v_coag)


def fine_reference(kind, dnda0, rho_d, t_out, n_fine=640):
    """Independent fine sectional solution: bins of a geometric radius grid, all grains of a bin at its centre;
    coagulation products split between the two neighbouring centre masses (number and mass kept); shattering
    fragments laid as a^-3.3 over [a_min, a_max] with each bin's own mean fragment mass."""
    a_e = np.geomspace(A_MIN, A_MAX, n_fine + 1)
    a_c = np.sqrt(a_e[:-1] * a_e[1:])
    m_c = mass(a_c)
    N0, M0 = moments(a_e, dnda0, n_sub=40)
    N0 *= rho_d / M0.sum()
    n0 = N0
    A1, A2 = a_c[:, None], a_c[None, :]
    if kind == "coag":
        K = coag_kernel(m_c[:, None], m_c[None, :])
        mp = (m_c[:, None] + m_c[None, :]).ravel()
        L = np.clip(np.searchsorted(m_c, mp, side="right") - 1, 0, n_fine - 2)
        f = np.clip((m_c[L + 1] - mp) / (m_c[L + 1] - m_c[L]), 0.0, 1.0)

        def rhs(t, n):
            R = 0.5 * (K * np.outer(n, n)).ravel()
            gain = np.bincount(L, R * f, n_fine) + np.bincount(L + 1, R * (1.0 - f), n_fine)
            return gain - n * (K @ n)
    else:
        K = np.pi * (A1 + A2) ** 2 * 3e5 * ((A1 > 0.1 * UM) & (A2 > 0.1 * UM))
        frac = (a_e[1:] ** 0.7 - a_e[:-1] ** 0.7) / (A_MAX ** 0.7 - A_MIN ** 0.7)      # mass fraction per bin
        mbar = mass(1.0) * (-2.3 / 0.7) * (a_e[1:] ** 0.7 - a_e[:-1] ** 0.7) / (a_e[1:] ** -2.3 - a_e[:-1] ** -2.3)

        def rhs(t, n):
            loss = n * (K @ n)
            return np.sum(loss * m_c) * frac / mbar - loss
    sol = solve_ivp(rhs, (0.0, t_out[-1]), n0, method="LSODA", t_eval=t_out, rtol=1e-7, atol=1e-30 * n0.sum())
    return a_e, a_c, sol.y.T


def pl2m_collisions(kind, n, dnda0, rho_d, t_out, method="eaf"):
    a, g = grid_of(n)
    N, M = moments(a, dnda0)
    scale = rho_d / M.sum()
    N, M = N * scale, M * scale
    model = FullShattering() if kind == "shat" else CoagulationModel(coag_kernel)
    states, t_prev = [], 0.0
    for t in t_out:
        if t > t_prev:
            r = collide(g, N, M, model, t - t_prev, errmax=1e-4, method=method)
            N, M = r["N"], r["M"]
        states.append((N.copy(), M.copy()))
        t_prev = t
    return a, g, states


def profile_curve(g, N, M, n_pts=40, method="pl2m"):
    """(a, dn/da) of a method's reconstruction, piece by piece (NaN between pieces; spikes are not drawn)."""
    N = np.asarray(N, float)
    P = gc.pieces(g, N, np.asarray(M, float), method)
    xs, ys = [], []
    for i in range(g.n_bins):
        for j in range(P.kp):
            if P.n[i, j] <= 1e-6 * N[i] or P.spike[i, j]:     # skip negligible pieces (round-off remnants)
                continue
            u = np.geomspace(P.u1[i, j], P.u2[i, j], n_pts)
            xs += [g.a_edges[i] * u, [np.nan]]
            ys += [P.A[i, j] * u ** P.lam[i, j] / g.a_edges[i], [np.nan]]
    if not xs:
        return np.array([np.nan]), np.array([np.nan])
    return np.concatenate(xs), np.concatenate(ys)


def fig9_schematic(plt, out_dir):
    """Their Fig. 9: one growth step of a few bins, PC, PL (with the slope limiter), PL2M and EAF."""
    n = 5
    a = np.geomspace(0.02 * UM, 0.06 * UM, n + 1)
    g = MassGrid(mass(a), RHO)
    lo, hi = a[0], a[2] + 0.45 * (a[3] - a[2])
    N0, M0 = boxcar_moments(a, lo, hi)
    pc, pl = LinearBins(a, N0, M0, constant=True), LinearBins(a, N0, M0)
    da = 0.55 * (a[1] - a[0])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    x = np.linspace(a[0], a[-1], 3000)
    for ax, (title, shifted) in zip(axs, (("time $t$", 0.0), (r"time $t + \Delta t$ (all grains grown by $\Delta a$)", da))):
        if shifted:
            pc.shift(da)
            pl.shift(da)
            sol = {name: remap(g, N0, M0, lambda y: y + da, method=meth)[:2] for name, meth in SOLVERS.items()}
        else:
            sol = {name: (N0, M0) for name in SOLVERS}
        ax.fill_between(x / UM, ((x >= lo + shifted) & (x <= hi + shifted)).astype(float), color="0.88",
                        label="exact")
        ax.plot(x / UM, pc.dnda(x), c="C3", lw=1.0, ls="--", label="piecewise constant (PC)")
        ax.plot(x / UM, pl.dnda(x), c="C2", lw=1.2, label="piecewise linear (PL)")
        for name, (c, lab) in (("PL2M", ("C1", "PL2M (power law)")), ("EAF", ("C4", "EAF (fixed front slope)")),
                               ("EA", ("C0", "EA (default)"))):
            xp, yp = profile_curve(g, *sol[name], n_pts=200, method=SOLVERS[name])
            ax.plot(xp / UM, yp, c=c, lw=1.4, label=lab)
        for e in a:
            ax.axvline(e / UM, c="0.6", lw=0.5, ls=":")
        if shifted:
            # bins where PL needed the limiter: zero at an edge
            for j in range(n):
                edge_vals = (pl.dnda(np.array([a[j] * 1.000001]))[0], pl.dnda(np.array([a[j + 1] * 0.999999]))[0])
                if min(edge_vals) < 1e-6 * pl.N.max() / pl.w.min() and pl.N[j] > 0:
                    ax.annotate("PL slope limited", xy=((a[j] + a[j + 1]) / 2 / UM, 0.15), ha="center",
                                fontsize=7.5, color="C2")
        ax.set_xlabel(r"$a$ [µm]")
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel(r"$dn/da$ [arbitrary]")
    axs[0].legend(fontsize=7.5, loc="upper right")
    axs[0].set_ylim(-0.05, 2.4)
    fig.tight_layout()
    fig.savefig(out_dir / "mckinnon18_fig9.pdf")
    plt.close(fig)


def boundary_test(steps=160, adot=0.3 * UM / GYR, t_end=8.0 * GYR):
    """Their Fig. 11: grains growing past a_max. a_min = 0.015625 um, the bin ratio of 60 bins up to 1 um, and 60,
    70, 80 bins (a_max = 1, 2, 4 um); dn/da uniform on [a_min delta^45, 1 um]; da/dt = 0.3 um/Gyr for 8 Gyr. Returns
    {label: (t, m_d(t)/m_d(0))} for PL with rebinning, PL2M with the same rebinning, and EAF with the grains past
    a_max kept as one population at their mean mass that keeps growing (moving centre)."""
    a_min = 0.015625 * UM
    delta = (1.0 * UM / a_min) ** (1.0 / 60.0)
    a_l = a_min * delta ** 45
    t = np.linspace(0.0, t_end, steps + 1)
    exact = ((1.0 * UM + adot * t) ** 4 - (a_l + adot * t) ** 4) / ((1.0 * UM) ** 4 - a_l ** 4)
    out = {"analytic": (t, exact)}
    for n in (60, 70, 80):
        a = a_min * delta ** np.arange(n + 1)
        g = MassGrid(mass(a), RHO)
        N0, M0 = boxcar_moments(a, a_l, 1.0 * UM)
        pl = LinearBins(a, N0, M0)
        y = [1.0]
        for _ in range(steps):
            pl.shift(adot * t_end / steps, rebin_top=True)
            y.append(pl.mass_total() / M0.sum())
        out[f"PL, rebinning, a_max = {a[-1] / UM:.0f} um"] = (t, np.array(y))
        for mode, name in (("rebin", "PL2M"), ("carry", "EA")):
            N, M = N0.copy(), M0.copy()
            oN = oM = 0.0
            y = [1.0]
            d = adot * t_end / steps
            for _ in range(steps):
                if oN > 0:                              # the carried population grows like a spike
                    oM = oN * mass(((3.0 * oM / oN / (4.0 * np.pi * RHO)) ** (1.0 / 3.0)) + d)
                N, M, _, dN, dM = remap(g, N, M, lambda x, d=d: x + d, method=SOLVERS[name])
                if mode == "rebin" and dM > 0:
                    M[-1] += dM
                    N[-1] += dM / mass(a[-1])
                else:
                    oN += dN
                    oM += dM
                y.append((M.sum() + oM) / M0.sum())
            out[f"{name}, {mode}, a_max = {a[-1] / UM:.0f} um"] = (t, np.array(y))
    return out


def reservoir_test(D_list=(1e-2, 1e-3, 1e-4), Z=0.1, adot0=0.025 * UM / GYR, t_end=4.0 * GYR, steps=400, n=64):
    """Their Fig. 12, in one cell: dust (initial dust-to-gas D) accreting from gas-phase metals (mass fraction Z)
    with the implicit gas update of ``gsd.exchange`` (silicate, 20 K grains, no nucleation, no Kelvin term). The
    rate is the physical one, proportional to the gas-phase density, scaled to da/dt = adot0 at the start; the
    initial dn/da is that of the Fig. 10 test. Saturates at m_d/m_d(0) = 1 + Z/D."""
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.gsd.materials import get_material

    mat = get_material("silicate")
    model = mat.sublimation_model()
    a, _ = grid_of(n)
    g = MassGrid(4.0 / 3.0 * np.pi * model.density_gcm3 * a ** 3, model.density_gcm3)
    a_cut = A_MIN * (A_MAX / A_MIN) ** 0.25
    x0 = np.clip(a[:-1], A_MIN, a_cut)
    x1 = np.clip(a[1:], A_MIN, a_cut)
    N0 = x1 - x0
    M0 = np.pi * model.density_gcm3 / 3.0 * (x1 ** 4 - x0 ** 4)
    T_gas = 100.0
    probe = Population(model, g, N0, M0, np.full(n, 20.0), nucleation=False, kelvin=False)
    n_key0 = adot0 / float(probe.adot(1.0, T_gas)(np.array([1e-6]))[0])
    t = np.linspace(0.0, t_end, steps + 1)
    out = {}
    for D in D_list:
        scale = (D / Z) * n_key0 / probe.key_atoms()           # dust key atoms = (D/Z) x gas key atoms
        pop = Population(model, g, N0 * scale, M0 * scale, np.full(n, 20.0), nucleation=False, kelvin=False)
        gas = {mat.key: n_key0}
        m0 = pop.key_atoms()
        y = [1.0]
        for _ in range(steps):
            exchange_step([pop], gas, T_gas, t_end / steps)
            y.append(pop.key_atoms() / m0)
        out[D] = (t, np.array(y), (n_key0 + m0) / m0)
    exact = ((a_cut + adot0 * t) ** 4 - (A_MIN + adot0 * t) ** 4) / (a_cut ** 4 - A_MIN ** 4)
    return t, exact, out


# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycalima.plotting_style import use_calima_style
    use_calima_style()
    plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "xtick.bottom": True, "ytick.left": True, "legend.frameon": False})
    return plt


def run(out_dir, steps, lines):
    plt = _style()
    out_dir = Path(out_dir)
    cols = {"PC": "C3", "PL": "C2", "PL2M": "C1", "EAF": "C4", "EA": "C0"}
    fig9_schematic(plt, out_dir)

    # --- Figs. 11, 12: grains past a_max, and a finite metal reservoir ------------------------------------------------
    bt = boundary_test()
    t_r, exact_r, res = reservoir_test()
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axs[0]
    t, y = bt["analytic"]
    ax.plot(t / GYR, y, c="k", lw=1.6, label="analytic")
    styles = {"PL, rebinning": ("C2", "o"), "PL2M, rebin": ("C1", "s"), "EA, carry": ("C0", "^")}
    lines.append("Fig. 11 test (grains past a_max): m_d(8 Gyr)/analytic - 1")
    for lab, (t, y) in bt.items():
        if lab == "analytic":
            continue
        key = [k for k in styles if lab.startswith(k)][0]
        amax = lab.split("= ")[1]
        ms = {"1 um": 5, "2 um": 3.5, "4 um": 2}[amax]
        ax.plot(t[::8] / GYR, y[::8], styles[key][1], c=styles[key][0], ms=ms, mfc="none" if amax != "1 um" else None,
                label=lab.replace("um", "µm").replace("rebin,", "rebinning,").replace("carry,", "overflow carried,"))
        lines.append(f"  {lab:34s} {y[-1] / bt['analytic'][1][-1] - 1:+.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("$t$ [Gyr]")
    ax.set_ylabel(r"$m_d(t)/m_d(0)$")
    ax.set_title("(a) grains growing past $a_{\\max}$ (their Fig. 11)", fontsize=9)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], c="k", lw=1.6, label="analytic")]
    names = {"PL, rebinning": "PL, rebinning into the top bin", "PL2M, rebin": "PL2M, rebinning into the top bin",
             "EA, carry": "EA, overflow carried at its mean size"}
    hs += [Line2D([], [], c=c, marker=m, ls="none", label=names[k]) for k, (c, m) in styles.items()]
    hs += [Line2D([], [], c="0.4", marker="o", ls="none", ms=ms, mfc=mf, label=f"$a_{{\\max}}$ = {am}")
           for am, ms, mf in (("1 µm", 5, None), ("2 µm", 3.5, "none"), ("4 µm", 2, "none"))]
    ax.legend(handles=hs, fontsize=6.6, loc="upper left")
    ax = axs[1]
    ax.plot(t_r / GYR, exact_r, c="k", lw=1.2, ls="--", label="infinite reservoir (analytic)")
    lines.append("Fig. 12 test (finite reservoir, implicit gas): final m_d/m_d(0) against 1 + Z/D")
    for j, (D, (t, y, cap)) in enumerate(res.items()):
        ax.plot(t / GYR, y, c=f"C{j}", label=f"$D = 10^{{{int(round(np.log10(D)))}}}$")
        ax.axhline(cap, c=f"C{j}", lw=0.6, ls=":")
        lines.append(f"  D = {D:.0e}: {y[-1]:.6g} against {cap:.6g} (relative {y[-1] / cap - 1:+.1e})")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3e3)
    ax.set_xlabel("$t$ [Gyr]")
    ax.set_ylabel(r"$m_d(t)/m_d(0)$")
    ax.set_title("(b) growth from a finite metal reservoir, $Z = 0.1$ (their Fig. 12)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "mckinnon18_boundary.pdf")
    plt.close(fig)

    # --- growth convergence and sinusoid -----------------------------------------------------------------------------
    ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    conv = growth_convergence(ns, steps)
    lines.append(f"Fig. 10 test (growth, {steps} steps): fractional mass error at 5 Gyr")
    lines.append("   N " + " ".join(f"{k:>12s}" for k in conv if k != "N"))
    for i, n in enumerate(ns):
        lines.append(f"{n:4d} " + " ".join(f"{conv[k][i]:12.3e}" for k in conv if k != "N"))
    sin = {n: sinusoid(n) for n in (128, 16)}

    fig = plt.figure(figsize=(11.0, 8.4))
    gs = fig.add_gridspec(4, 3, width_ratios=(1.25, 1, 1))
    ax = fig.add_subplot(gs[:2, 0])
    for k in ("PC", "PL"):
        ax.plot(ns, conv[k], "o-", c=cols[k], ms=4, label=f"{k} (McKinnon+18, re-implemented)")
    for name in ("PL2M", "EAF", "EA"):
        ax.plot(ns, np.maximum(conv[f"{name}_{steps}"], 1e-15), "o-", c=cols[name], ms=4,
                label=f"{name}{' (default)' if name == 'EA' else ''}, {steps} steps")
        ax.plot(ns, np.maximum(conv[f"{name}_{10 * steps}"], 1e-15), "--", c=cols[name], lw=1,
                label=f"{name}, {10 * steps} steps")
    ax.plot([], [], " ", label=f"PL2M, EA: 1 step exact; EAF 1 step {conv['EAF_1'][0]:.0e} at 8 bins")
    nn = np.array(ns, float)
    ax.plot(nn, 3.0 / nn, c="0.6", lw=0.8, ls="-.", label=r"$\propto 1/N$")
    ax.plot(nn, 60.0 / nn ** 2, c="0.3", lw=0.8, ls="-.", label=r"$\propto 1/N^2$")
    ax.plot([64, 64], [0.40, 0.01], "*", c="k", ms=9, label="McKinnon+18 at $N = 64$ (PC, PL)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("bins $N$ (1 nm -- 1 µm)")
    ax.set_ylabel("fractional dust-mass error at 5 Gyr")
    ax.set_title("(a) growth convergence (their Fig. 10)", fontsize=9)
    ax.set_ylim(1e-7, 100)
    ax.legend(fontsize=6.6, loc="lower left")
    ax = fig.add_subplot(gs[2:, 0])
    for n, ls in ((128, "-"), (16, "--")):
        h = sin[n][3]["hist"]
        for k in ("PC", "PL", "PL2M", "EAF", "EA"):
            ax.plot(h["t"], np.maximum(h[k], 1e-15), ls, c=cols[k], lw=1, label=f"{k}, $N = {n}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-15, 3)
    ax.set_xlabel("$t$ [Gyr]")
    ax.set_ylabel("fractional dust-mass error")
    ax.set_title("(d) mass error through the growth--shrinking period", fontsize=9)
    ax.legend(fontsize=6.6, loc="lower right", ncol=2)
    # sinusoid snapshots
    for jc, n in enumerate((128, 16)):
        a, g, (lo0, hi0), snaps = sin[n]
        plateau = 1.0
        for jr, t in enumerate((0.25, 0.5, 0.75, 1.0)):
            ax = fig.add_subplot(gs[jr, 1 + jc])
            s = snaps[t]
            lo, hi = s["analytic"]
            ax.plot([A_MIN / UM, lo / UM, lo / UM, hi / UM, hi / UM, A_MAX / UM], [0, 0, 1, 1, 0, 0], c="k", lw=1.0)
            for k in ("PC", "PL"):
                b = LinearBins(a, *s[k][:1], constant=True)
                b.N, b.s = s[k]
                x = np.geomspace(a[0], a[-1], 4000)
                ax.plot(x / UM, b.dnda(x), c=cols[k], lw=0.9)
            for name in ("PL2M", "EAF", "EA"):
                xp, yp = profile_curve(g, *s[name], method=SOLVERS[name])
                ax.plot(xp / UM, yp, c=cols[name], lw=1.1)
            ax.set_xscale("log")
            ax.set_xlim(0.008, 0.12)
            ax.set_ylim(-0.05, 1.5)
            ax.text(0.03, 0.85, f"$t = {t:g}$ Gyr", transform=ax.transAxes, fontsize=8)
            if jr == 0:
                ax.set_title(f"({'b' if jc == 0 else 'c'}) growth and shrinking, $N = {n}$ (their Fig. 13)", fontsize=9)
            if jr == 3:
                ax.set_xlabel("$a$ [µm]")
            else:
                ax.set_xticklabels([])
            if jc == 0:
                ax.set_ylabel(r"$dn/da$")
            if jr == 3:
                spread = {}
                for k in ("PC", "PL"):
                    spread[k] = left_spread(a, s[k][0] / np.diff(a), lo, plateau)
                for name in SOLVERS:
                    spread[name] = left_spread(a, s[name][0] / np.diff(a), lo, plateau)
                lines.append(f"Fig. 13 test, N = {n}: bins beyond the left analytic edge after one period "
                             f"(mean dn/da > 1e-3 of the plateau): " + ", ".join(f"{k} {v}" for k, v in spread.items())
                             + f"; mass error PC {abs(LinearBins(a, s['PC'][0], constant=True).mass_total() / boxcar_moments(a, lo0, hi0)[1].sum() - 1):.3f}"
                             + "".join(f", {name} {abs(s[name][1].sum() / boxcar_moments(a, lo0, hi0)[1].sum() - 1):.2e}"
                                       for name in SOLVERS))
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], c="k", label="analytic")] +
               [Line2D([], [], c=c, label=k) for k, c in cols.items()],
               loc="lower right", ncol=4, fontsize=8, bbox_to_anchor=(0.98, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "mckinnon18_growth.pdf")
    plt.close(fig)

    # --- collisions --------------------------------------------------------------------------------------------------
    lognormal = lambda x: np.exp(-np.log(x / (0.1 * UM)) ** 2 / (2 * 0.6 ** 2)) / x   # noqa: E731
    mrn = lambda x: x ** -3.5                                                          # noqa: E731
    cases = {"shat": (lognormal, 3.7e-3 * 1.4 * M_H * 0.4, np.array([0.0, 100.0]) * MYR, (8, 16)),
             "coag": (mrn, 0.01 * 1.4 * M_H * 1e5, np.array([0.0, 2.0, 5.0, 10.0, 20.0]) * MYR, (32, 8))}
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for row, (kind, (f0, rho_d, t_out, n_list)) in enumerate(cases.items()):
        a_e, a_c, nref = fine_reference(kind, f0, rho_d, t_out)
        wref = np.diff(a_e)
        for col, n in enumerate(n_list):
            ax = axs[row, col]
            runs = {name: pl2m_collisions(kind, n, f0, rho_d, t_out, method=SOLVERS[name]) for name in ("PL2M", "EA")}
            a, g, states = runs["EA"]
            tcols = ["k"] + [f"C{j}" for j in range(len(t_out) - 1)]
            for k, t in enumerate(t_out):
                ax.plot(a_c / UM, nref[k] / wref * UM, c=tcols[k], lw=1.0 if k else 0.8, ls=":" if k == 0 else "-",
                        label=(f"reference, {t / MYR:g} Myr" if k else "initial"))
                if k == 0:
                    continue
                N, M = states[k]
                xp, yp = profile_curve(g, N, M, method="ea")
                ax.plot(xp / UM, yp * UM, c=tcols[k], lw=0.7, alpha=0.7)
                ac = np.sqrt(a[:-1] * a[1:])
                ax.plot(ac / UM, N / np.diff(a) * UM, "o", c=tcols[k], ms=4, mfc="none")
                Np = runs["PL2M"][2][k][0]
                ax.plot(ac / UM, Np / np.diff(a) * UM, "x", c=tcols[k], ms=4)
                # mass in bins against the reference's, on the coarse bins
                # 640 fine bins nest exactly in 8, 16 and 32 coarse ones: sum by index, not by matching edges
                if (a_e.size - 1) % n:
                    raise ValueError("the fine reference must nest in the coarse bins")
                edges_idx = np.arange(0, a_e.size, (a_e.size - 1) // n)
                M_ref = np.add.reduceat(nref[k] * mass(a_c), edges_idx[:-1])
                N_ref = np.add.reduceat(nref[k], edges_idx[:-1])
                good = M_ref > 1e-2 * M_ref.sum()
                for name in runs:
                    Nx, Mx = runs[name][2][k]
                    err = np.max(np.abs(Mx[good] / M_ref[good] - 1.0))
                    werr = np.sum(np.abs(Mx - M_ref)) / M_ref.sum()
                    errN = np.max(np.abs(Nx[good] / N_ref[good] - 1.0))
                    lines.append(f"Fig. {15 if kind == 'shat' else 16} test, {name}, N = {n}, t = {t / MYR:g} Myr: "
                                 f"bins with >= 1 % of the mass: worst error in mass {err:.3f}, in number {errN:.3f}; "
                                 f"mass misplaced (sum |dM| / M) {werr:.3f}; total mass {Mx.sum() / rho_d - 1:+.1e}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("$a$ [µm]")
            ax.set_ylabel(r"$dn/da$ [cm$^{-3}$ µm$^{-1}$]")
            if kind == "shat":
                ax.set_ylim(1e-14, 1e-3)
                ax.set_title(f"({'ab'[col]}) shattering, $N = {n}$, 100 Myr (their Fig. 15: $N = 8$)", fontsize=9)
            else:
                ax.set_ylim(1e-12, 1e8)
                ax.set_title(f"({'cd'[col]}) coagulation, $N = {n}$ (their Fig. 16: $N = 32$)", fontsize=9)
            ax.legend(fontsize=6.8, loc="lower left")
            ax.text(0.97, 0.95, "circles: EA (default)\ncrosses: PL2M", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "mckinnon18_collisions.pdf")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="results/gsd_overview/figs")
    ap.add_argument("--steps", type=int, default=100, help="steps of the growth convergence test")
    args = ap.parse_args(argv)
    lines = []
    run(args.out_dir, args.steps, lines)
    Path(args.out_dir, "mckinnon18_tests.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
