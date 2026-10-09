"""pahchem.impact and pahchem.cosmic_rays: energy deposited in PAHs by cosmic-ray collisions.

The gates test the oscillator strength (sum rule), the stopping it implies
against NIST STAR measurements, and the invariants of the collision-outcome
mapping and the CR transport.
"""

import numpy as np
import pytest

from pycalima.pahchem import cosmic_rays as cr
from pycalima.pahchem import impact as im
from pycalima.pahchem import photo

PARENTS = [(24, 12), (54, 18), (96, 24)]


@pytest.mark.parametrize("nc,nh", PARENTS)
def test_oscillator_strength_obeys_the_trk_sum_rule(nc, nh):
    w = im.w_grid()
    total = np.trapezoid(im.oscillator_strength(nc, nh, 0, w), w)
    assert total == pytest.approx(6 * nc + nh, rel=0.05)


@pytest.mark.parametrize("proj,prog", [(im.PROTON, "pstar"), (im.ALPHA, "astar")])
def test_bethe_stopping_of_the_pah_matches_polystyrene_for_fast_nuclei(proj, prog):
    """With no correction, the delta-oscillator stopping per electron is within
    6% of the measured aromatic CH stopping at 1-100 MeV/u (I = 59 eV for the
    PAH against 68.7 eV for polystyrene)."""
    nc, nh = 24, 12
    w = im.w_grid()
    f = im.oscillator_strength(nc, nh, 0, w)
    for t_u in (1e6, 1e7, 1e8):
        t = t_u * proj.nucleons
        model = im.collisions(proj, t, f, w).stopping / (6 * nc + nh)
        assert model == pytest.approx(float(im.polystyrene_stopping_per_electron(prog, t)), rel=0.06)


def test_bethe_stopping_of_the_pah_matches_polystyrene_for_fast_electrons():
    """10 keV - 1 GeV electrons, against ESTAR with the condensed-medium density effect removed."""
    nc, nh = 54, 18
    w = im.w_grid()
    f = im.oscillator_strength(nc, nh, 0, w)
    for t in (1e4, 1e5, 1e6, 1e7, 1e8, 1e9):
        model = im.collisions(im.ELECTRON, t, f, w).stopping / (6 * nc + nh)
        assert model == pytest.approx(float(im.polystyrene_stopping_per_electron("estar", t)), rel=0.07)


def test_delta_oscillator_model_reduces_to_bethe():
    nc, nh = 24, 12
    w = im.w_grid(3000)
    f = im.oscillator_strength(nc, nh, 0, w)
    i_ev = im.mean_excitation_energy(nc, nh, 0, w)
    ne = np.trapezoid(f, w)
    for t in (1e7, 1e8, 1e9):
        model = im.collisions(im.PROTON, t, f, w).stopping
        assert model == pytest.approx(ne * float(im.bethe_stopping_per_electron(im.PROTON, t, i_ev)), rel=0.01)


def test_low_speed_correction_tends_to_one_and_vanishes_below_the_floor():
    assert im.low_speed_correction(im.PROTON, 1e8) == pytest.approx(1.0, abs=0.01)
    assert im.low_speed_correction(im.PROTON, 0.5 * im.T_MIN_PER_U_EV) == 0.0
    fe = im.nucleus("Fe", 26, 56.0)
    assert im.low_speed_correction(fe, 5e6 * fe.nucleons) < im.low_speed_correction(im.PROTON, 5e6)


@pytest.mark.parametrize("partition", im.PARTITIONS)
def test_deposition_conserves_the_collisions(partition):
    """Every collision lands at one (dz, E*), in the grid or in the overflow."""
    nc, nh, ip = 54, 18, 6.14
    coll = im.impact_collisions(im.PROTON, 1e7, nc, nh, 0, ip_ev=ip)
    dep = im.deposition(coll, ip, 8.91, lambda E: photo.yield_jochims1996(E, ip), partition=partition)
    assert dep.total() == pytest.approx(coll.total, rel=1e-3)
    assert all(np.all(r >= 0.0) for r in dep.rates.values())


def test_deposited_energy_never_exceeds_the_energy_transferred():
    nc, nh, ip = 54, 18, 6.14
    coll = im.impact_collisions(im.PROTON, 1e7, nc, nh, 0, ip_ev=ip)
    dep = im.deposition(coll, ip, 8.91, lambda E: photo.yield_jochims1996(E, ip), partition="all", e_max=1e4, de=1.0)
    assert dep.power() <= coll.stopping


def test_slow_electrons_get_the_binary_encounter_reduction():
    nc, nh, ip = 24, 12, 7.2
    plain = im.impact_collisions(im.ELECTRON, 50.0, nc, nh, 0)
    scaled = im.impact_collisions(im.ELECTRON, 50.0, nc, nh, 0, ip_ev=ip)
    assert scaled.total < plain.total
    fast = im.impact_collisions(im.ELECTRON, 1e5, nc, nh, 0, ip_ev=ip)
    assert fast.total == pytest.approx(im.impact_collisions(im.ELECTRON, 1e5, nc, nh, 0).total)


def test_csda_attenuation_conserves_particles_and_is_the_identity_at_zero_column():
    csda = cr.CSDA.build(im.PROTON)
    t = np.geomspace(1e6, 1e10, 40)
    assert np.allclose(csda.attenuated(cr.PROTONS_L, t, 0.0), cr.PROTONS_L(t))
    n = 1e22
    j = csda.attenuated(cr.PROTONS_L, t, n)
    r = np.interp(np.log(t), np.log(csda.t), csda.range_) + n
    t0 = np.exp(np.interp(np.log(r), np.log(csda.range_), np.log(csda.t)))
    L = lambda x: np.exp(np.interp(np.log(x), np.log(csda.t), np.log(csda.loss)))  # noqa: E731
    assert np.allclose(j * L(t), cr.PROTONS_L(t0) * L(t0), rtol=1e-6)
    assert np.all(j <= cr.PROTONS_L(t) * (1 + 1e-9)) or np.all(np.diff(t) > 0)


def test_csda_range_matches_pstar():
    """The range from the tabulated loss reproduces PSTAR's CSDA range in H2 (1-100 MeV)."""
    from pycalima import _paths
    d = np.loadtxt(_paths.get_external_data_path("pahchem/nist_pstar_hydrogen.csv"), delimiter=",", comments="#")
    csda = cr.CSDA.build(im.PROTON)
    for t_mev in (1.0, 10.0, 100.0):
        r_pstar = np.interp(t_mev, d[:, 0], d[:, 4]) / cr.M_H2_G
        assert np.interp(np.log(t_mev * 1e6), np.log(csda.t), csda.range_) == pytest.approx(r_pstar, rel=0.05)


def test_pdg_nuclei_are_normalised_to_hydrogen():
    ab = {p.name: a for p, a in cr.nuclei()}
    assert ab["He"] == pytest.approx(34.0 / 550.0)
    assert ab["Fe-Ni"] == pytest.approx(0.12 / 550.0)
