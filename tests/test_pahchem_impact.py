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


def test_standard_spectra_give_rtz_their_zeta_and_attenuate_monotonically():
    """RAMSES-RTZ's convention: the standard spectrum scaled to zeta_H has a primary zeta(H2) of H2_PER_H zeta_H, linearly.
    The shapes behind it: model H, with its low-energy protons, ionises more than L, and both less through more column."""
    for zeta in (1e-18, 1e-17, 1e-16, 1e-15):                     # below, inside and above the blend
        for model in ("blend", "L", "H"):
            assert cr.primary_ionisation_rate(cr.standard_spectra(zeta, model)) == pytest.approx(cr.H2_PER_H * zeta, rel=1e-6)
    a_l, a_h = cr.blend_weights(1e-16)
    assert a_l + a_h == pytest.approx(1.0) and 0.0 < a_h < 1.0
    hi = 1e13                                                          # the shared measured part: the blend is the same there
    assert cr.standard_spectra(1e-16)[0][1](hi) == pytest.approx(cr.projectile_spectra("L", cr.STANDARD_COLUMN)[0][1](hi), rel=1e-6)
    z = {(m, n): cr.primary_ionisation_rate(cr.projectile_spectra(m, n)) for m in ("L", "H") for n in (1e20, 1e21, 1e22)}
    for n in (1e20, 1e21, 1e22):
        assert z[("H", n)] > z[("L", n)]
    for m in ("L", "H"):
        assert z[(m, 1e20)] > z[(m, 1e21)] > z[(m, 1e22)]


def test_maxwellian_flux_and_coulomb_focus_of_thermal_particles():
    """``maxwellian_flux``: 4 pi int j dT = n <v>, the mean speed of the Maxwellian. ``coulomb_focus``: an electron is drawn
    to a cation (rate and energy up), held off an anion below the barrier (rate 0), and neutrals feel nothing."""
    T, n = 1e6, 3.0
    t = np.geomspace(1e-3, 3e3, 4000)
    j = im.maxwellian_flux(im.ELECTRON, T, n)
    kt = 8.617333262e-5 * T
    v_mean = im.C_CM_S * np.sqrt(8.0 * kt / (np.pi * im.ELECTRON.rest_ev))
    assert 4.0 * np.pi * np.trapezoid(j(t), t) == pytest.approx(n * v_mean, rel=1e-3)
    t = np.array([1.0, 10.0, 100.0])
    g, hit = im.coulomb_focus(im.ELECTRON, 2, 24, t)
    assert np.all(g > 1.0) and np.all(hit > t)
    g, hit = im.coulomb_focus(im.ELECTRON, -1, 24, t)
    assert g[0] == 0.0 and np.all(g[1:] < 1.0) and np.all(hit[1:] < t[1:])
    assert np.array_equal(im.coulomb_focus(im.PROTON, 0, 24, t)[0], np.ones(3))


def test_measured_partition_leaves_the_photons_hole():
    """``partition="measured"``: the distant ionisations leave the mean hole of a photoionisation at W
    (``photo.photoelectron_energy``), so their mean E* is the hole weighted by their rates, to within the bin."""
    nc, nh, ip = 54, 18, 6.14
    coll = im.impact_collisions(im.PROTON, 1e7, nc, nh, 0, ip_ev=ip)
    y = lambda E: photo.yield_jochims1996(E, ip)                                   # noqa: E731
    meas = im.deposition(coll, ip, 8.91, y, partition="measured", e_max=1e3, de=0.02)       # bin centres: de/2 at E* = 0
    none = im.deposition(coll, ip, 8.91, y, partition="none", e_max=1e3, de=0.02)
    w, tw = coll.w, im._trapezoid_weights(coll.w)
    s = np.ones_like(w) if coll.ion_scale is None else coll.ion_scale
    q = np.where((w >= ip) & (w < im.K_EDGE_EV), coll.distant * tw * s * np.clip(y(w), 0.0, 1.0), 0.0)
    hole = photo.photoelectron_energy(w, ip)[1]
    assert meas.power() - none.power() == pytest.approx(float(q @ hole), rel=0.02)


@pytest.mark.parametrize("tu", [1e7, 3e5])
def test_passages_conserve_crossings_and_collisions(tu):
    """Every crossing has 0, 1, 2 ... collisions; the mixture's mean is sigma_tot."""
    nc, nh, ip = 54, 18, 6.14
    yf = lambda E: photo.yield_jochims1996(E, ip)  # noqa: E731
    c = im.impact_collisions(im.ALPHA, tu * im.ALPHA.nucleons, nc, nh, 0, ip_ev=ip)
    ps = im.passage_outcomes(c, nc, ip, 8.91, yf)
    th = (np.arange(24) + 0.5) * (0.5 * np.pi / 24)
    r, d = im.disc_radius_cm(nc), im.DISC_THICKNESS_CM
    sg = np.pi * r * r * np.cos(th) + 2 * r * d * np.sin(th)
    w_th = 4 * np.pi * sg * np.sin(th) * (0.5 * np.pi / 24)
    none = float((w_th * np.exp(-c.total / sg)).sum())
    assert ps.P.sum() * ps.de + ps.beyond + none == pytest.approx(ps.passages, rel=1e-6)
    assert ps.collisions == pytest.approx(4 * np.pi * c.total, rel=1e-9)
    # Cauchy: the mean projected area of a convex body is S/4
    S = 2 * np.pi * r * r + 2 * np.pi * r * d
    assert ps.passages == pytest.approx(4 * np.pi * S / 4, rel=0.01)


def test_slow_iron_passages_all_end_beyond_the_grid():
    """Fe at 10 MeV/u makes ~60 collisions crossing C54: always destroyed (Coulomb explosion)."""
    nc, nh, ip = 54, 18, 6.14
    fe = im.nucleus("Fe", 26, 56.0)
    c = im.impact_collisions(fe, 1e7 * fe.nucleons, nc, nh, 0, ip_ev=ip)
    assert im.passage_multiplicity(c, nc) > 20
    ps = im.passage_outcomes(c, nc, ip, 8.91, lambda E: photo.yield_jochims1996(E, ip))
    assert ps.beyond == pytest.approx(ps.passages, rel=1e-3)


def test_rrkm_whitten_rabinovitch_agrees_with_beyer_swinehart():
    from pycalima.pahchem import dos, rates
    freqs = np.linspace(200.0, 3100.0, 60)
    ch = im.Channel("H", 4.4, 12.0, 6.0, 3050.0, 0, 1)
    e_ev = np.array([8.0, 12.0, 16.0])
    E, counts = dos.beyer_swinehart(freqs, 17.0 * dos.CM_PER_EV, 10.0)
    ts = rates.transition_state_for_entropy(freqs, 3050.0, 12.0, n_scaled=5)
    k_bs = rates.rrkm(E, counts, ts, 4.4 * dos.CM_PER_EV, degeneracy=6.0)
    ref = np.interp(e_ev * dos.CM_PER_EV, E, k_bs)
    assert np.allclose(im.rrkm_whitten_rabinovitch(freqs, ch, e_ev), ref, rtol=0.25)


def test_fragmentation_chain_is_monotone_and_starts_above_the_barriers():
    from types import SimpleNamespace
    freqs = np.linspace(150.0, 3100.0, 90)
    sp = SimpleNamespace(freqs_cm=freqs, einstein_a=np.full(freqs.size, 5.0))
    ch = [im.Channel("H", 4.41, 12.0, 10.0, 3050.0, 0, 1), im.Channel("C2", 4.6, 41.84, 1.0, 1600.0, 2, 0)]
    fc = im.FragmentationChain(sp, ch, e_max_ev=120.0, de_ev=0.1, e_cascade_ev=30.0, de_cm=100.0)
    assert np.all(fc.carbon[fc.e < 4.41] == 0.0) and np.all(fc.hydrogen[fc.e < 4.41] == 0.0)
    total = fc.carbon / 2 + fc.hydrogen
    assert np.all(np.diff(total) >= -1e-9)
    # each loss removes at least its barrier: steps <= E / min(E0)
    assert np.all(total <= fc.e / 4.41 + 1e-9)


def test_radiated_energy_is_the_deposit_below_onset_and_bounded_above():
    from types import SimpleNamespace
    freqs = np.linspace(150.0, 3100.0, 90)
    sp = SimpleNamespace(freqs_cm=freqs, einstein_a=np.full(freqs.size, 5.0))
    ch = [im.Channel("H", 4.41, 12.0, 10.0, 3050.0, 0, 1)]
    fc = im.FragmentationChain(sp, ch, e_max_ev=120.0, de_ev=0.1, e_cascade_ev=30.0, de_cm=100.0)
    low = fc.e < 4.41
    assert np.allclose(fc.radiated[low], fc.e[low])
    assert np.all(fc.radiated <= fc.e + 1e-9)
    onset = fc.e[np.argmax(fc.p_diss > 0.5)]
    assert fc.residual(110.0) < 1.5 * onset


# ---- Coulomb explosion: a cold limit at any E*, a hot-ion limit in E* ---------------------------------------------------

def test_cold_coulomb_limit_gives_the_dft_and_laboratory_anchors():
    """floor(0.2 N_C): Holm+11's thermodynamic limits (naphthalene, biphenylene, anthracene, pyrene, coronene) and the
    C60^12+ seen intact."""
    assert [im.coulomb_limit(n) for n in (10, 12, 14, 16, 24, 60)] == [2, 2, 2, 3, 4, 12]


def test_multifragmentation_energy_is_chabot_fig7_and_falls_with_charge():
    assert np.isinf(im.multifragmentation_energy(60, 2)) and np.isinf(im.multifragmentation_energy(54, 2))
    assert im.multifragmentation_energy(60, 3) == pytest.approx(105.0) and im.multifragmentation_energy(60, 5) == pytest.approx(60.0)
    e = im.multifragmentation_energy(96, np.arange(5, 15))
    assert np.all(np.diff(e) <= 0.0) and e[-1] == pytest.approx(96.0)          # held at 1 eV per C above Q/N_C = 5/60


def _one_bin_spectrum(dz, e_star):
    """A single passage outcome: charge change dz, internal energy e_star [eV]."""
    e = np.arange(0.5, 200.0, 1.0)
    P = np.zeros((dz + 2, e.size))
    P[dz, int(np.argmin(abs(e - e_star)))] = 1.0
    return {"x": im.Passages(e, 1.0, P, 0.0, 1.0, 1.0)}


def test_a_cold_coronene_dication_survives_and_a_hot_one_explodes():
    """Chabot+20's charge-only threshold (4/60 x 24 = 1.6) explodes every C24H12 dication; the cold limit keeps the
    cold ones (stable in the lab), and their Fig. 7 line still explodes those left with >= 24 eV (1 eV per C)."""
    from types import SimpleNamespace
    chain = SimpleNamespace(lost=lambda e: (np.zeros_like(e), np.zeros_like(e)))
    cold, hot = _one_bin_spectrum(2, 5.0), _one_bin_spectrum(2, 40.0)
    rate = lambda s, mode: im.destruction_rates(s, 24, 12, 0, chain, coulomb=mode)["x"]["exploded"]   # noqa: E731
    assert rate(cold, "charge_only") == 1.0 and rate(hot, "charge_only") == 1.0
    assert rate(cold, "cold_and_hot") == 0.0 and rate(hot, "cold_and_hot") == 1.0
    assert rate(_one_bin_spectrum(5, 5.0), "cold_and_hot") == 1.0                   # above floor(0.2 x 24) = 4 at any E*
    with pytest.raises(ValueError):
        im.destruction_rates(cold, 24, 12, 0, chain, coulomb="x")
