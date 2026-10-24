"""Dark (UV-free) grain charging for the collapse model.

The exact P(Z) must reproduce the WDB06rt solver of rt_group_charging with no
photons, balance its currents, reach the Spitzer limit for large grains, and
interpolate from its table; the closure must satisfy ionisation balance and
neutrality. Benchmark M2 is Marchand et al. (2021, A&A 649, A50) Fig. 2, read
from the paper's vector graphics (``data/marchand21_fig2.txt``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from pycalima.collapse.charge import (
    ME_CGS,
    MH_CGS,
    DarkChargeTable,
    charge_balance,
    dark_charge_moments,
    marchand21_charge_balance,
    most_negative_charge,
    recombination_rate_coefficient,
)
from pycalima.collapse.grid import SizeGrid

M_P = 1.67262192e-24
KB, E2 = 1.380649e-16, 4.8032047e-10 ** 2
FIG2 = Path(__file__).parent / "data" / "marchand21_fig2.txt"


@pytest.mark.parametrize("material,a", [("graphite", 5e-8), ("graphite", 1e-5), ("silicate", 1e-6)])
def test_dark_distribution_reproduces_the_wdb06rt_solve_with_no_photons(material, a):
    """At T <= 1e3 K for any charge, and up to 3e3 K for negative grains: there
    the hot-gas terms omitted here (gas-electron secondaries, pass-through) are
    negligible. Above 1e3 K the secondaries shift positive grains by a few %
    (not tested: a known limitation). WDB06rt takes a Gaussian P(Z) once
    sigma > 3, so it is the approximate side of this comparison."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab = rg.build_bin_table(material, a, U=rg.u_grid(n=241), Ts=np.geomspace(3e3, 3e6, 5),
                             T_hot=np.geomspace(1e3, 1e9, 7))
    gt = rg.GroupChargingTable(tab)
    zmin = most_negative_charge(material, a)
    for T, rs in ((10.0, (1.0, 30.0, 1e3)), (300.0, (1.0, 30.0, 1e3)), (1e3, (1.0, 30.0, 1e3)), (3e3, (1.0,))):
        for r in rs:
            out = gt.solve(np.zeros(tab["L0"].size), 2.99792458e10, T, 1.0, [(r, 1.0, M_P)])
            zm, sg, _, _ = dark_charge_moments(a, T, r * math.sqrt(ME_CGS / M_P), zmin, 0.0)
            assert zm == pytest.approx(out["Zmean"], rel=1e-2, abs=1e-2)
            assert sg == pytest.approx(out["Zsigma"], rel=0.05, abs=0.02)


@pytest.mark.parametrize("a,T,xi", [(5e-8, 10.0, 0.02), (1e-6, 100.0, 1.0), (1e-4, 1e3, 5.0)])
def test_ion_and_electron_currents_balance(a, T, xi):
    zm, sg, J_i, J_e = dark_charge_moments(a, T, xi, most_negative_charge("silicate", a), 0.0)
    assert xi * J_i == pytest.approx(J_e, rel=1e-9)


def test_large_grains_reach_the_spitzer_potential():
    """tau, |Z| >> 1: <Z> = psi tau with (1 - psi) xi = s_e e^psi (Spitzer 1941;
    DS87). The DS87 corrections fall as tau^-1/2 and |Z|^-1/2, so tau ~ 6e3 here."""
    a, T, xi, s = 1e-2, 1e3, 0.05, 0.5
    tau = a * KB * T / E2
    psi = brentq(lambda p: (1.0 - p) * xi - s * math.exp(p), -30.0, 0.0)
    zm, sg, _, _ = dark_charge_moments(a, T, xi, -1e9, s)
    assert zm / tau == pytest.approx(psi, rel=0.02)
    # Gaussian width from the curvature of ln P: sigma^2 = -1/g' = tau (1 - psi) / (2 - psi)
    assert sg == pytest.approx(math.sqrt(tau * (1.0 - psi) / (2.0 - psi)), rel=0.02)


def test_table_interpolates_the_direct_solve_off_its_nodes():
    a = np.geomspace(5e-8, 1e-4, 6)
    tab = DarkChargeTable(a, ["graphite"] * 6, log_T=np.linspace(0.5, 3.5, 61))
    rng = np.random.default_rng(3)
    for _ in range(20):
        lT, lx = rng.uniform(0.6, 3.4), rng.uniform(-3.9, 3.9)
        _, Zt, St, Jt = tab.at_temperature(10.0 ** lT)
        h = tab.log_xi[1] - tab.log_xi[0]
        j = int((lx - tab.log_xi[0]) / h)
        w = (lx - tab.log_xi[j]) / h
        for k in range(a.size):
            zm, _, ji, _ = dark_charge_moments(a[k], 10.0 ** lT, 10.0 ** lx, tab.zmin[k], 0.0)
            Zi = (1 - w) * Zt[k, j] + w * Zt[k, j + 1]
            assert Zi == pytest.approx(zm, rel=0.02, abs=0.02)
            assert (1 - w) * Jt[k, j] + w * Jt[k, j + 1] == pytest.approx(math.log(ji), abs=0.02)


def test_charge_balance_satisfies_neutrality_and_ionisation_balance():
    g = SizeGrid.from_radii(5e-7, 2.5e-5, 20, 3.0)
    a = g.mean_radius(np.full(20, -11 / 6))
    tab = DarkChargeTable(a, ["silicate"] * 20, log_T=np.array([1.0, 2.0]), log_xi=np.linspace(-5, 5, 101))
    T, zeta, mu_i = 30.0, 1e-17, 25.0
    for nH in (1e3, 1e8, 1e12):
        n_k = 0.01 * 2.3 * MH_CGS * nH * g.power_law_rho(-11 / 6, 1.0) / g.mean_mass(np.full(20, -11 / 6))
        r = charge_balance(tab, n_k, T, nH, zeta, mu_i)
        assert r["n_i"] - r["n_e"] + np.dot(n_k, r["Zmean"]) == pytest.approx(0.0, abs=1e-9 * r["n_i"])
        v_i = math.sqrt(8 * KB * T / (math.pi * mu_i * MH_CGS))
        loss = (recombination_rate_coefficient(T) * r["n_i"] * r["n_e"]
                + r["n_i"] * v_i * np.dot(n_k * math.pi * a ** 2, r["Jion"]))
        assert loss == pytest.approx(zeta * nH, rel=1e-8)
    # grain-poor gas: the gas-phase limit n_i = n_e = (zeta nH / k)^1/2
    r = charge_balance(tab, n_k * 1e-12, T, 1e4, zeta, mu_i)
    n0 = math.sqrt(zeta * 1e4 / recombination_rate_coefficient(T))
    assert r["n_e"] == pytest.approx(n0, rel=1e-3) and r["n_i"] == pytest.approx(n0, rel=1e-3)


def _fig2_setup(f=1.0):
    """Marchand+21 Sect. 3.4: MRN 5-250 nm, d = 0.006, 50 bins, 8.7 nm mantles."""
    g = SizeGrid.from_radii(5e-7, 2.5e-5, 50, 2.3)
    beta = np.full(50, -11 / 6)
    a_core = g.mean_radius(beta)
    pick = [int(np.argmin(abs(np.log(a_core / s)))) for s in (1.5e-6, 3e-6, 1e-5, 2.5e-5)]
    per_nH = f * 0.006 * 2.3 * MH_CGS * g.power_law_rho(-11 / 6, 1.0) / g.mean_mass(beta)
    return a_core + 8.7e-7, per_nH, pick


def _fig2_run(solve, per_nH, pick):
    ref = np.loadtxt(FIG2)
    out = []
    for ln in ref[:, 0]:
        r = solve(per_nH * 10.0 ** ln, 10.0 ** ln)
        out.append(np.r_[np.log10(-r["Zmean"][pick]), np.log10(r["n_i"]), np.log10(r["n_e"])])
    return np.array(out) - ref[:, 1:7]


def test_fig2_their_analytic_method_is_reproduced():
    """With the grain abundance 0.85 x what d = 0.006, mu = 2.3, rho_s = 2.3
    imply (their unstated normalisation; rho_s = 2.7 would give it), all six
    curves match to 0.015 dex over n_H = 1e4-1e12. Without it, the mismatch
    grows to 0.15 dex where grains carry the charge: the factor is a fit."""
    a, per_nH, pick = _fig2_setup(f=0.85)
    d = _fig2_run(lambda n_k, nH: marchand21_charge_balance(a, n_k, 10.0, nH, 5e-17, 25.0), per_nH, pick)
    assert np.abs(d).max() < 0.015


def test_fig2_exact_distribution_is_close_to_their_method():
    """The exact P(Z) (WD01 sticking ~ 0.5 here) differs from their three-state
    plus Gaussian approximation by < 0.1 dex, most for large grains."""
    a, per_nH, pick = _fig2_setup(f=0.85)
    tab = DarkChargeTable(a, ["silicate"] * a.size, log_T=np.array([0.9, 1.1]),
                          log_xi=np.linspace(-5, 5, 201))
    d = _fig2_run(lambda n_k, nH: charge_balance(tab, n_k, 10.0, nH, 5e-17, 25.0), per_nH, pick)
    assert np.abs(d[:, :4]).max() < 0.1
    assert np.abs(d[:, 4:]).max() < 0.02
