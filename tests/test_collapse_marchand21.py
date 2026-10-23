"""Benchmark M1: Marchand et al. (2021, A&A 649, A50), turbulent coagulation.

Their Fig. 1 (case nH4) is reproduced bin by bin from the curve read off the
paper's vector graphics (``data/marchand21_fig1_nH4.txt``, provenance in its
header), and their central claim, that at fixed initial distribution the
result depends only on chi = int g_local n_H dt, is checked with their three
constant-density cases (Table 1).

Their Eqs. 14, 17 and 21 carry T**-1/4, while their Eqs. 10-13 give T**+1/4
(see ``intermediate_turbulence_kernel``). Their figure and Table 1 follow the
printed -1/4, so the comparison uses ``T_power=-0.25``.
"""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from pycalima.collapse.coagulation import advance, intermediate_turbulence_kernel
from pycalima.collapse.grid import SizeGrid

YR = 3.15576e7
RHO_S, MU = 2.3, 2.3
FIG1 = Path(__file__).parent / "data" / "marchand21_fig1_nH4.txt"


def _marchand_grid(n_mrn=19, below=5, above=22):
    """Their Fig. 1 grid: ``n_mrn`` bins across 5-250 nm, extended both ways."""
    zeta = 50.0 ** (1.0 / n_mrn)
    g = SizeGrid.from_radii(5e-7 * zeta ** -below, 2.5e-5 * zeta ** above,
                            below + n_mrn + above, RHO_S)
    ae = g.a_edges
    w = np.zeros(g.n_bins)
    mrn = slice(below, below + n_mrn)
    # dn/da ∝ a^-3.5: the mass in [a-, a+] ∝ a+^0.5 - a-^0.5
    w[mrn] = ae[1:][mrn] ** 0.5 - ae[:-1][mrn] ** 0.5
    return g, w / w.sum()


def test_kernel_normalisation_is_their_eq_15():
    """dV = 174 cm/s at n_H = 1e4, T = 10 K, a = 10 nm (Marchand+21, Eq. 15)."""
    a = np.array([1e-6])
    K = intermediate_turbulence_kernel(1e4, 10.0, RHO_S, mu=MU, T_power=-0.25)(None, a)
    dV = K[0, 0] / (np.sqrt(8.0 / (3.0 * np.pi)) * np.pi * (2.0 * a[0]) ** 2)
    assert dV == pytest.approx(1.74e2, rel=0.01)


def test_fig1_nH4_is_reproduced_bin_by_bin():
    """Their initial MRN carries mX = 3.592e-26 g per H (the sum of their
    in-range bins), 7 % below d mu m_H; with that normalisation the result
    must match to the few percent of their own Fig. 1 error panel."""
    _, _, log_X, log_mX = np.loadtxt(FIG1, unpack=True)
    ref = 10.0 ** log_mX
    g, w = _marchand_grid()
    nH = 1e4
    rho, _, _ = advance(g, 3.592e-26 * nH * w, 4.58e7 * YR,
                        intermediate_turbulence_kernel(nH, 10.0, RHO_S, mu=MU, T_power=-0.25))
    mX = rho / nH
    phys = slice(5, 38)   # 5 nm to ~6 um: outside it their bins hold the code's floor value
    ratio = mX[phys] / ref[phys]
    assert np.abs(mX[phys] - ref[phys]).sum() / ref[phys].sum() < 0.04
    assert ratio.min() > 0.96 and ratio.max() < 1.05
    assert np.argmax(mX) == np.argmax(ref)
    # the mean sizes they plot at are the in-bin power-law means (bin 5, next
    # to their floor-valued bins, has a different slope there)
    beta = g.slopes(rho)
    log_a = np.loadtxt(FIG1, usecols=1)
    inner = slice(6, 38)
    np.testing.assert_allclose(np.log10(g.mean_radius(beta)[inner] * 1e4), log_a[inner], atol=0.01)


@pytest.mark.parametrize("T_power", [-0.25, 0.25])
def test_the_result_depends_only_on_chi(T_power):
    """Marchand+21 Table 1: nH0 (1e5 K), nH4 and nH8 (10 K) over durations of
    equal chi = int nH**3/4 T**T_power dt. Identical to round-off, for either
    sign of the T exponent (with chi defined to match)."""
    g, w = _marchand_grid(n_mrn=8, below=2, above=12)
    cases = [(1.0, 1e5), (1e4, 10.0), (1e8, 10.0)]
    chi = 8.12e17
    out = []
    for nH, T in cases:
        t = chi / (nH ** 0.75 * T ** T_power)
        rho, _, _ = advance(g, 0.01 * MU * 1.6726e-24 * nH * w, t,
                            intermediate_turbulence_kernel(nH, T, RHO_S, mu=MU, T_power=T_power))
        out.append(rho / nH)
    for o in out[1:]:
        np.testing.assert_allclose(o, out[0], rtol=1e-9, atol=1e-12 * out[0].max())
    # and something did happen: the mass peak moved up by more than a decade in size
    assert (np.argmax(out[0]) - np.argmax(w)) * np.log10(50.0) / 8 > 1.0


def test_free_fall_collapse_reaches_the_constant_density_result_at_equal_chi():
    """Guillet+20 pressure-free collapse from 1e4 cm^-3 at 10 K, integrated in
    steps of d ln n (dust mass fractions carried through the compression), to
    chi = 8.12e17 cgs: the distribution must match the constant-density run, as
    it does in their Fig. 1 (to < 0.1 % here, already at d ln n = 0.1: the
    remainder is the explicit integrator's own step-to-step noise)."""
    from scipy.integrate import quad

    from pycalima.collapse.tracks import freefall_steps, freefall_time_to_density

    g, w = _marchand_grid(n_mrn=8, below=2, above=12)
    T, mu, nH0, chi = 10.0, 2.3, 1e4, 8.12e17
    gloc = lambda n: n ** 0.75 * T ** -0.25

    def chi_to(n_end):   # chi = int g n dt = int g n (dt/dn) dn along the free fall
        f = lambda ln: gloc(math.exp(ln)) * (freefall_time_to_density(math.exp(ln) * 1.0001, nH0, mu)
                                             - freefall_time_to_density(math.exp(ln), nH0, mu)) / 1e-4
        return quad(f, math.log(nH0), math.log(n_end), limit=400)[0]

    n_end = math.exp(brentq(lambda ln: chi_to(math.exp(ln)) - chi, math.log(1e8), math.log(1e15)))
    ref, _, _ = advance(g, 0.01 * MU * 1.6726e-24 * nH0 * w, chi / gloc(nH0),
                        intermediate_turbulence_kernel(nH0, T, RHO_S, mu=MU, T_power=-0.25))
    ref /= nH0
    errs = []
    for dlnn in (0.1, 0.025):
        t, n = freefall_steps(nH0, n_end, mu, dlnn=dlnn)
        rho = 0.01 * MU * 1.6726e-24 * nH0 * w
        for k in range(n.size - 1):
            n_mid = math.sqrt(n[k] * n[k + 1])
            rho = rho * n_mid / (n[k] if k == 0 else math.sqrt(n[k - 1] * n[k]))
            rho, _, _ = advance(g, rho, t[k + 1] - t[k],
                                intermediate_turbulence_kernel(n_mid, T, RHO_S, mu=MU, T_power=-0.25))
        X = rho / math.sqrt(n[-2] * n[-1])
        errs.append(np.abs(X - ref).sum() / ref.sum())
    assert max(errs) < 1e-3
