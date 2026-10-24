"""Collapse tracks, growth by accretion, and the first literature checks (stage 4).

* Omukai et al. (2005) dynamics: compressional heating along their Z = 0 track
  against their Fig. 2a; free fall before the first temperature minimum.
* Accretion: the rigid shift in radius conserves mass exactly and matches the
  moment formula; gas-phase elements are conserved; growth sets in at
  n_grow ∝ Z^-2, at the densities of Chiaki, Nozawa & Yoshida (2013).
* Static dust: S D_cr is the constant of Schneider et al. (2012a, Eq. 16),
  whatever the material and size.
* Brownian coagulation: the grain cross-section halves just above the critical
  density of Hirashita & Omukai (2009, Eq. 15), which scales as Z^-2.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pycalima.collapse.accretion import formula_mass, mass_gain, shift_remap, solar_abundances
from pycalima.collapse.grid import SizeGrid, bin_means
from pycalima.collapse.integrator import (Population, brownian_kernel, critical_dust_to_gas,
                                          lognormal_population, run_collapse)
from pycalima.collapse.materials import get_material
from pycalima.collapse.tracks import Track, omukai_f

RHO_PER_H = (1 + 4 * 0.0833) * 1.6735575e-24
COMPR = Path(__file__).parent / "data" / "omukai05_fig2a_compr.txt"


@pytest.fixture(scope="module")
def z0():
    return Track.omukai2005("Z0")


# ---------------------------------------------------------------------------
# track and dynamics
# ---------------------------------------------------------------------------

def test_track_reproduces_the_primordial_collapse(z0):
    """T and f_H2 at landmark densities of the Z = 0 run (Omukai+05 Figs. 1, 3)."""
    assert 150 < z0.T(1e4) < 300           # the H2 cooling minimum
    assert 1000 < z0.T(1e12) < 1600
    assert 1e-3 < z0.f_H2(1e4) < 2e-3      # H- channel
    assert z0.f_H2(1e13) > 0.9             # three-body H2 formation done
    assert z0.log_n_frag == pytest.approx(4.0, abs=0.3)


def test_compression_heating_matches_omukai05_fig2a(z0):
    """Within 0.2 dex everywhere from 1 to 1e16 cm^-3 and 0.1 dex in the median
    (ours runs ~0.08 dex low throughout, in free fall too, so a normalisation
    convention, not the dynamics; the largest offsets sit at the f(gamma) switches)."""
    ln, lG = np.loadtxt(COMPR, unpack=True)
    n = 10.0 ** ln
    ours = np.log10(z0.compression_heating(n) / z0.rho(n))
    assert np.max(np.abs(ours - lG)) < 0.2
    assert abs(np.median(ours - lG)) < 0.1


def test_collapse_time_is_free_fall_before_fragmentation_and_capped(z0):
    tff = lambda n: math.sqrt(3 * math.pi / (32 * 6.67430e-8 * z0.rho(n)))
    assert z0.t_col(1e2) == pytest.approx(tff(1e2), rel=1e-12)
    assert omukai_f(4.0 / 3.0) == 0.95 and omukai_f(0.5) == 0.0
    assert z0.t_col(1e6) / tff(1e6) <= 1 / math.sqrt(0.05) + 1e-9


# ---------------------------------------------------------------------------
# accretion
# ---------------------------------------------------------------------------

def test_shift_remap_adds_exactly_the_moment_mass_even_with_steep_slopes():
    g = SizeGrid.from_radii(1e-7, 1e-3, 40, 3.2)
    rho = g.power_law_rho(-11 / 6, 1e-20)
    rho[25:] = 0.0
    rho[3] *= 1e-6
    beta = g.slopes(rho)
    N, a1 = g.number_density(rho, beta), g.mean_radius(beta)
    a2 = g.a_edges[:-1] ** 2 * bin_means(np.ones(40), g.eta, beta, 2 / 3)
    for da in (1e-8, 1e-6, 3e-5):
        new, over = shift_remap(g.a_edges, rho, 3 * beta + 2, g.rho_s, da)
        assert np.all(np.isfinite(new)) and np.all(new >= 0)
        assert new.sum() + over - rho.sum() == pytest.approx(mass_gain(g.rho_s, N, a1, a2, da), rel=1e-12)


@pytest.fixture(scope="module")
def mgsio3_growth(z0):
    """MgSiO3 at 0.01 um holding 10 % of Si, solar pattern scaled to Z (Chiaki+13 setup)."""
    out = {}
    for Z in (1e-4, 1e-5, 1e-6):
        y0 = solar_abundances(Z)
        nf = 0.1 * y0["Si"]
        pop = lognormal_population("MgSiO3", nf * formula_mass("MgSiO3"), a_peak=1e-6, sigma_ln=0.3)
        y = dict(y0)
        y["Si"] -= nf
        y["Mg"] -= nf
        h, _ = run_collapse(z0, [pop], y_gas=y, n0=1e2, n_end=1e17, dlnn=0.05, accretion=True)
        out[Z] = (h, y0, nf)
    return out


def test_accretion_conserves_each_element(mgsio3_growth):
    for h, y0, nf in mgsio3_growth.values():
        D = np.asarray(h.D)
        for k in (len(D) // 2, -1):
            si = D[k] / D[0] * nf + h.y_gas[k]["Si"]
            mg = D[k] / D[0] * nf + h.y_gas[k]["Mg"]
            assert si == pytest.approx(y0["Si"], rel=1e-9)
            assert mg == pytest.approx(y0["Mg"], rel=1e-9)
        assert D[-1] == pytest.approx(D[0] * y0["Si"] / nf, rel=1e-3)   # Si (the limiting element) used up


def test_growth_onset_scales_as_z_to_the_minus_two_at_chiaki13_densities(mgsio3_growth):
    """Chiaki, Nozawa & Yoshida (2013): MgSiO3 growth enhances cooling at
    1e10, 1e12, 1e14 cm^-3 for 1e-4, 1e-5, 1e-6 Z_sun (growth time ∝ 1/(Z n)
    against t_col ∝ n^-1/2). Here: the density where 90 % of Si is condensed."""
    n90 = {}
    for Z, (h, y0, nf) in mgsio3_growth.items():
        ySi = np.array([y["Si"] for y in h.y_gas])
        n90[Z] = np.log10(h.nH[int(np.argmax(ySi <= 0.1 * (y0["Si"] - nf)))])
    assert n90[1e-5] - n90[1e-4] == pytest.approx(2.0, abs=0.3)
    assert n90[1e-6] - n90[1e-5] == pytest.approx(2.0, abs=0.3)
    for Z, ref in ((1e-4, 10.0), (1e-5, 12.0), (1e-6, 14.0)):
        assert n90[Z] == pytest.approx(ref, abs=0.8)


# ---------------------------------------------------------------------------
# static dust: the critical dust-to-gas ratio
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("material,a_peak", [("MgSiO3", 3e-6), ("C", 1e-6)])
def test_critical_dust_to_gas_times_cross_section_is_schneider12_eq16(z0, material, a_peak, collapse_optics):
    """S D_cr = 1.4e-3 cm^2/g (T/1e3 K)^-1/2 at 1e12 cm^-3 (Schneider+12a Eq. 16,
    obeyed by their one-zone models): within 25 % for S differing 4x."""
    make = lambda D: ([lognormal_population(material, D * RHO_PER_H, a_peak=a_peak)], None)
    Dcr = critical_dust_to_gas(make, z0, rtol=0.05, dlnn=0.1)
    S = run_collapse(z0, make(1e-8)[0], n_end=1e3)[0].S[0]
    expect = 1.4e-3 * (z0.T(1e12) / 1e3) ** -0.5
    assert S * Dcr == pytest.approx(expect, rel=0.25)


# ---------------------------------------------------------------------------
# Brownian coagulation
# ---------------------------------------------------------------------------

def test_brownian_coagulation_follows_hirashita_omukai09(z0):
    """MRN (5-250 nm) at D = 0.006 Z/Z_sun: the cross-section halves 0.5-1.5 dex
    above their n_H,coag (Eq. 15, <a> = 10 nm, t_coag = t_ff), so ∝ Z^-2."""
    rho_s = get_material("Mg2SiO4").rho_s
    zeta = 50 ** (1 / 15)
    g = SizeGrid.from_radii(5e-7 * zeta ** -2, 2.5e-5 * zeta ** 25, 42, rho_s)
    ae = g.a_edges
    w = np.zeros(g.n_bins)
    w[2:17] = ae[3:18] ** 0.5 - ae[2:17] ** 0.5
    ns = np.logspace(2, 17, 3000)
    half = {}
    for Z in (1.0, 1e-2):
        h, _ = run_collapse(z0, [Population("Mg2SiO4", g, 0.006 * Z * RHO_PER_H * w / w.sum())],
                            n0=1e2, n_end=1e17, dlnn=0.1, coagulation=True, coag_kernel=brownian_kernel)
        S = np.asarray(h.S) / h.S[0]
        half[Z] = np.log10(h.nH[int(np.argmax(S <= 0.5))])
        crit = 1.6e7 * (rho_s / 3) ** 3 * Z ** -2 * (z0.T(ns) / 100) ** -1
        n_coag = np.log10(ns[int(np.argmax(ns >= crit))])
        assert 0.5 < half[Z] - n_coag < 1.5
    assert half[1e-2] - half[1.0] == pytest.approx(4.0, abs=0.6)


def test_accretion_focus_speeds_up_only_the_limiting_key_species():
    """``focus`` multiplies a key species' collision rate: doubling it for Fe doubles
    the growth of iron grains; for Mg it changes forsterite only where Mg, not
    SiO, limits the growth."""
    from pycalima.collapse.accretion import accrete
    from pycalima.collapse.grid import SizeGrid
    g = SizeGrid.from_radii(1e-7, 1e-4, 30, 7.87)
    rho = np.exp(-0.5 * (np.log(np.sqrt(g.a_edges[1:] * g.a_edges[:-1]) / 1e-5)) ** 2) * 1e-30
    y = {"C": 0.0, "O": 1e-8, "Mg": 1e-10, "Si": 1e-10, "Fe": 1e-10, "Al": 0.0, "S": 0.0}
    pops = lambda: [dict(material="Fe", grid=g, rho=rho.copy())]
    _, da1 = accrete(pops(), 500.0, 1e12, y, 1e3)
    assert da1[0] > 0.0
    _, da2 = accrete(pops(), 500.0, 1e12, y, 1e3, focus={"Fe": 2.0})
    assert da2[0] == pytest.approx(2.0 * da1[0], rel=1e-3)
    _, da3 = accrete(pops(), 500.0, 1e12, y, 1e3, focus={"Mg": 5.0})
    assert da3[0] == pytest.approx(da1[0], rel=1e-12)


def test_accretion_focus_may_depend_on_density(z0, collapse_optics):
    """A callable focus of (n_H, T) acts like the fixed one where it returns it, and
    not at all where it returns nothing."""
    from pycalima.collapse.ics import nozawa_ccsn
    pops, y, _ = nozawa_ccsn(20, 1e-4, n_amb=1, grains="grackle")
    run = lambda f: run_collapse(z0, pops, y_gas=y, n0=1e6, n_end=1e14, dlnn=0.1, accretion=True,
                                 accretion_focus=f)[0]
    fixed = {"Mg": 3.0, "Fe": 3.0}
    D = lambda h: np.asarray(h.D)
    assert np.array_equal(D(run(lambda n, T: fixed)), D(run(fixed)))
    assert np.array_equal(D(run(lambda n, T: None)), D(run(None)))
    assert D(run(fixed))[-1] > D(run(None))[-1]
