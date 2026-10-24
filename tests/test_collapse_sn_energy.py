"""Supernova-enriched initial conditions and the minimal energy equation (stage 5).

* ``ics.nozawa_ccsn`` reproduces the dust-to-metal ratios of Chiaki et al.
  (2015, Table 3) for the N07 models, keeps the elements consistent with the
  dust (no more of an element in grains than in the cloud, beyond small
  digitisation slack), and locks carbon in CO in these O-rich ejecta.
* The energy equation reproduces the dust-free track it is built from, finds no
  low-mass fragmentation in primordial gas, and with the Schneider & Omukai
  (2010) criterion gives the critical metallicity of Chiaki+15 Table 3 for the
  unprocessed 20 Msun dust.
* The digitised Nozawa+07 Fig. 6b carbon gives the S_C of N07M20n1 (Chiaki+15).
* The solver's shattering fragments follow Jones et al. (1996), n(a) ∝ a^-3.3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycalima.collapse.grid import SizeGrid
from pycalima.collapse.ics import (chiaki2015_models, distribution_on_grid, nozawa2007_m20_n1_sizes,
                                   nozawa_ccsn)
from pycalima.collapse.integrator import (critical_dust_to_gas, fragmentation, lognormal_population,
                                          run_collapse)
from pycalima.collapse.materials import get_material
from pycalima.collapse.tracks import Track

RHO_PER_H = (1 + 4 * 0.0833) * 1.6735575e-24


@pytest.fixture(scope="module")
def z0():
    return Track.omukai2005("Z0")


@pytest.mark.parametrize("M", [13, 20, 25, 30])
def test_ics_reproduce_chiaki15_dust_to_metal_ratios(M, collapse_optics):
    pops, y, info = nozawa_ccsn(M, 1e-5)
    model = chiaki2015_models()[float(M)]
    assert info["f_dep"] == pytest.approx(model["f_dep"], rel=1e-9)
    # metal mass per H is Z Z_sun m_H / X_H
    assert info["metal_per_H"] == pytest.approx(1e-5 * 0.02 * RHO_PER_H, rel=1e-12)
    # the elements hold what the grains contain, to the digitisation slack (Fe up
    # to 20 %; Al, from another yield set, is looser; others < 2 %)
    for e, short in info["shortfall"].items():
        assert short < {"Al": 1.0, "Fe": 0.2}.get(e, 0.02), (e, short)
    assert y["C"] == 0.0      # O-rich: carbon in CO


def test_ics_carbon_and_silicate_shares_match_chiaki15(collapse_optics):
    """N07M20n0: f_dep,C = 0.023, f_dep,Sil = 0.049 (Table 3)."""
    pops, y, info = nozawa_ccsn(20, 1e-5)
    x = {p.material: p.x.sum() for p in pops}
    m = info["metal_per_H"]
    assert x["C"] / m == pytest.approx(0.023, rel=0.05)
    assert (x["Mg2SiO4"] + x["MgSiO3"]) / m == pytest.approx(0.049, rel=0.05)


def test_nozawa07_m20n1_carbon_sizes_give_chiaki15_cross_section():
    """S_C = 3 <a^2> / (4 rho_s <a^3>) = 2.0e4 cm^2/g for N07M20n1 (Chiaki+15 Table 3)."""
    g = SizeGrid.from_radii(1e-8, 1e-3, 50, get_material("C").rho_s)
    w = distribution_on_grid(g, "C", nozawa2007_m20_n1_sizes)
    assert w.sum() == pytest.approx(1.0)
    beta = g.slopes(w)
    S = float(np.sum(g.number_density(w, beta) * g.mean_area(beta)))
    assert S == pytest.approx(2.0e4, rel=0.15)
    # a table without the species gives nothing rather than failing
    assert not distribution_on_grid(g, "Al2O3", nozawa2007_m20_n1_sizes).any()


def test_energy_equation_reproduces_the_dust_free_track(z0):
    pop = lognormal_population("MgSiO3", 1e-14 * RHO_PER_H, a_peak=3e-6)
    h, _ = run_collapse(z0, [pop], n0=1e-1, n_end=1e16, dlnn=0.05, energy=True)
    a = h.arrays()
    assert np.max(np.abs(np.log10(a["T"] / z0.T(a["nH"])))) < 0.05
    assert fragmentation(h) == []          # primordial gas: no sub-solar fragments


def test_slowed_collapse_scales_the_time_and_keeps_the_track(z0):
    """t_col_factor stretches the collapse time; the energy equation, built from the same
    t_col, still reproduces the track."""
    slow = Track(z0.log_n, z0.log_T, z0.log_fH2, t_col_factor=2.0)
    n = np.array([1e2, 1e8, 1e12])
    assert np.allclose(slow.t_col(n), 2.0 * z0.t_col(n))
    pop = lognormal_population("MgSiO3", 1e-14 * RHO_PER_H, a_peak=3e-6)
    h, _ = run_collapse(slow, [pop], n0=1e-1, n_end=1e16, dlnn=0.05, energy=True)
    a = h.arrays()
    assert np.max(np.abs(np.log10(a["T"] / slow.T(a["nH"])))) < 0.05
    h1, _ = run_collapse(z0, [pop], n0=1e-1, n_end=1e16, dlnn=0.05, energy=True)
    assert a["t"][-1] == pytest.approx(2.0 * h1.arrays()["t"][-1], rel=0.05)


def test_dust_cooling_lowers_the_temperature_and_fragments(z0):
    pop = lognormal_population("MgSiO3", 1e-5 * RHO_PER_H, a_peak=3e-6)
    h, _ = run_collapse(z0, [pop], n0=1e-1, n_end=1e16, dlnn=0.05, energy=True)
    a = h.arrays()
    i = np.argmin(np.abs(np.log10(a["nH"]) - 13))
    assert a["T"][i] < 0.5 * z0.T(a["nH"][i])
    frag = fragmentation(h)
    assert frag and all(m < 0.8 for _, _, m in frag)


def test_static_dust_critical_metallicity_matches_chiaki15_table3(z0, collapse_optics):
    """N07M20n0 without growth: [Z_cr] = -5.2 (Chiaki+15), within 0.35 dex. The
    gamma criterion converges for d ln n <= 0.1 (-4.91, -4.95, -4.97 at 0.1, 0.05,
    0.025); 0.2 is too coarse for the energy step (-4.47)."""
    make = lambda Z: nozawa_ccsn(20, Z)[:2]
    Zcr = critical_dust_to_gas(make, z0, criterion="frag", lo=1e-7, hi=1e-4, rtol=0.15, dlnn=0.1, n0=1e-1)
    assert math.log10(Zcr) == pytest.approx(-5.2, abs=0.35)


def test_solver_fragment_slope_is_jones96():
    """slope_frag_func s is the exponent of the cumulative fragment mass, M(<m) ∝ m^s,
    so n(a) ∝ a^(3 s - 4); the default must give Jones et al. (1996), a^-3.3."""
    from pycalima.solvers.chemistry_state import DustChemistryState
    import dataclasses
    s = {f.name: f.default for f in dataclasses.fields(DustChemistryState)}["slope_frag_func"]
    assert 3 * s - 4 == pytest.approx(-3.3, abs=1e-12)
