"""Benchmark: turbulence-only coagulation in the collapse of Guillet et al. (2020).

Their standard model (A&A 643, A17, Sects. 2-3): MRN cores 5-250 nm (3 g/cm^3)
with 8.8 nm ice mantles (1 g/cm^3), dust-to-gas 0.9 %, n_H,0 = 1e4 cm^-3, 10 K,
pressure-free free fall; Ormel & Cuzzi (2007) turbulence driven at the Jeans
length with V_g^2 = 3/2 c_s^2 and Re = 6.2e7 (n / 1e5)^1/2; the coagulation
rate is the mean speed of the collisions below V = 0.4 km/s (a_hat / 0.1 um)^-5/6
(their Eq. 18). Their Fig. 8 (top left) is read from the paper's vector graphics
(``data/guillet20_fig8_turbulence.txt``).

Two of their prescriptions matter, and both are followed here:

* the ice is conserved and spread as a uniform mantle over the surviving
  grains (their Sect. 2.4): the 8.8 nm of the initial MRN thickens as small
  grains coagulate away (the 0.9 % is the cores; the ice, 0.43 %, comes on top);
* the Ormel & Cuzzi velocities switch between their limit formulas
  (``ormel_cuzzi_dv(form="limits")``), faster than the full form near t_eta.

With a fixed 8.8 nm mantle and the full form instead, we keep 0.36 dex more
small grains at 1e12 cm^-3 and the mass peaks at 3.5 um instead of 5.8 um.
Their ice accretion (to 1.2 %) is not followed; it moves the 1e12 peak by +0.1 dex.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pycalima.collapse import velocities as vel
from pycalima.collapse.coagulation import advance
from scipy.optimize import brentq

from pycalima.collapse.grid import SizeGrid, bin_means
from pycalima.collapse.tracks import freefall_steps

FIG8 = Path(__file__).parent / "data" / "guillet20_fig8_turbulence.txt"
MH, KB = 1.6735575e-24, 1.380649e-16
T, MU, RHO_S, RHO_ICE, MANTLE = 10.0, 2.33, 3.0, 1.0, 8.8e-7


def _mantle_thickness(g, rho, nH, V_ice):
    """Uniform mantle thickness h holding the ice volume V_ice per H on the cores."""
    beta = g.slopes(rho)
    N = g.number_density(rho, beta) / nH
    a1 = g.mean_radius(beta)
    a2 = g.a_edges[:-1] ** 2 * bin_means(np.ones(g.n_bins), g.eta, beta, 2.0 / 3.0)
    return brentq(lambda h: np.sum(N * (3 * a2 * h + 3 * a1 * h ** 2 + h ** 3)) - V_ice, 0.0, 1e-3)


def _run(bins_per_decade=8, form="limits", conserve_ice=True):
    nb = int(round(np.log10(1e-2 / 5e-7) * bins_per_decade))
    g = SizeGrid.from_radii(5e-7, 1e-2, nb, RHO_S)
    ae = g.a_edges
    top = int(np.searchsorted(ae, 2.5e-5 * (1 + 1e-9)))
    w = np.zeros(nb)
    w[:top] = np.clip(ae[1:top + 1], None, 2.5e-5) ** 0.5 - ae[:top] ** 0.5
    rho_per_n = 1.4 * MH
    t, n = freefall_steps(1e4, 1e12, mu=1.4, dlnn=0.05)
    rho = 0.009 * rho_per_n * 1e4 * w / w.sum()
    # the ice volume per H of an 8.8 nm mantle on the initial MRN cores
    beta = g.slopes(rho)
    N = g.number_density(rho, beta) / 1e4
    a1 = g.mean_radius(beta)
    a2 = g.a_edges[:-1] ** 2 * bin_means(np.ones(nb), g.eta, beta, 2.0 / 3.0)
    V_ice = np.sum(N * (3 * a2 * MANTLE + 3 * a1 * MANTLE ** 2 + MANTLE ** 3))
    cs = math.sqrt(KB * T / (MU * MH))
    vth = math.sqrt(8 * KB * T / (math.pi * MU * MH))
    snaps = {}
    for k in range(n.size):
        for mark in (1e4, 1e8, 1e10, 1e12):
            if abs(math.log10(n[k] / mark)) < 0.026 and mark not in snaps:
                snaps[mark] = rho / (rho_per_n * n[k])
        if k == n.size - 1:
            break
        nm = math.sqrt(n[k] * n[k + 1])
        rho = rho * nm / (n[k] if k == 0 else math.sqrt(n[k - 1] * n[k]))
        rho_g = rho_per_n * nm
        tau_L = 0.5 * math.sqrt(math.pi / (6.67430e-8 * rho_g))
        tau_eta = tau_L / math.sqrt(6.2e7 * math.sqrt(nm / 1e5))
        h = _mantle_thickness(g, rho, nm, V_ice) if conserve_ice else MANTLE

        def kernel(mean_m, mean_a):
            a = mean_a + h
            m = mean_m + 4 / 3 * math.pi * RHO_ICE * (a ** 3 - mean_a ** 3)
            ts = m / (4.0 / 3.0 * rho_g * vth * math.pi * a ** 2)     # 3 m / (4 rho v_th sigma)
            turb = vel.ormel_cuzzi_dv(ts, cs, tau_L, tau_eta, form=form)
            dV = np.sqrt(vel.brownian_rms(T, m) ** 2 + turb ** 2)
            ahat = a[:, None] * a[None, :] / (a[:, None] + a[None, :])
            V = 0.4e5 * (ahat / 1e-5) ** (-5.0 / 6.0)
            return math.pi * (a[:, None] + a[None, :]) ** 2 * vel.mean_collision_speed(dV, V)

        rho, _, _ = advance(g, rho, t[k + 1] - t[k], kernel, weights="turbulent")
    return g, snaps


@pytest.fixture(scope="module")
def run():
    return _run()


def _curves():
    d = np.loadtxt(FIG8)
    return {round(ln): (d[d[:, 0] == ln, 1], d[d[:, 0] == ln, 2]) for ln in np.unique(d[:, 0])}


def _ours(g, x):
    beta = g.slopes(x)
    v = x / np.log10(g.a_edges[1:] / g.a_edges[:-1])
    ok = v > 0
    return np.log10(g.mean_radius(beta) * 1e4)[ok], np.log10(v[ok])


def _normalised(g, snaps, ln):
    dlog = np.log10(g.a_edges[1:] / g.a_edges[:-1])
    norm = np.log10((snaps[1e4] / dlog).max())
    oa, oy = _ours(g, snaps[10.0 ** ln])
    return oa, oy - norm


def test_size_distribution_follows_guillet20_fig8(run):
    """Shapes, each normalised to its own initial peak (the MRN at 250 nm): within
    0.1 dex over 10 nm - 0.2 um at 1e8 and 1e12 cm^-3, 0.2 at 1e10 (the first bin, at
    the 5 nm grid edge, drops steeply in both), and the initial
    curves agree (same normalisation, same MRN)."""
    g, snaps = run
    ref = _curves()
    norm_ref = ref[4][1].max()
    for ln, tol in ((4, 0.05), (8, 0.1), (10, 0.2), (12, 0.1)):
        xa, ya = ref[ln]
        oa, oy = _normalised(g, snaps, ln)
        # below the 250 nm MRN edge, where their bin centres and our in-bin means differ
        for la in np.arange(-2.0, -0.75, 0.25):
            if xa.min() <= la <= xa.max():
                assert abs(np.interp(la, oa, oy) - (np.interp(la, xa, ya) - norm_ref)) < tol, (ln, la)


def test_mass_peak_moves_as_in_guillet20_fig8(run):
    g, snaps = run
    ref = _curves()
    for ln, tol in ((8, 0.1), (10, 0.1), (12, 0.25)):
        xa, ya = ref[ln]
        oa, oy = _ours(g, snaps[10.0 ** ln])
        assert oa[np.argmax(oy)] == pytest.approx(xa[np.argmax(ya)], abs=tol)


def test_fixed_mantle_and_full_velocities_coagulate_more_slowly():
    """Without their two prescriptions the small grains survive longer: the
    prescriptions, not the solver, set the remaining differences."""
    g0, s0 = _run(form="full", conserve_ice=False)
    g1, s1 = _run()
    a0, y0 = _normalised(g0, s0, 12)
    a1, y1 = _normalised(g1, s1, 12)
    assert np.interp(-1.5, a0, y0) - np.interp(-1.5, a1, y1) > 0.2
