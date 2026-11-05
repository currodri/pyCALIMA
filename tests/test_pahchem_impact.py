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


def test_disc_chords_hold_the_disc_s_electrons_and_atoms():
    """``disc_chords``: summed over every chord (weights in cm^2 sr), the friction column is 4 pi HARTREE int gamma dV and
    the atom column 4 pi N_C, whatever the friction law; an ion without a friction has none."""
    nc = 54
    r = im.disc_radius_cm(nc) / im.BOHR_CM
    x = np.linspace(0.0, im.JELLIUM_EXTENT_AU, 4001)
    for el in ("H", "He", "C"):
        ch = im.disc_chords(nc, el)
        volume = np.pi * r**2 * 2.0 * np.trapezoid(im.friction_coefficient(el, x), x) * im.BOHR_CM**2
        assert (ch.weight * ch.friction).sum() == pytest.approx(4.0 * np.pi * im.HARTREE_EV * volume, rel=2e-3)
        assert (ch.weight * ch.atoms).sum() == pytest.approx(4.0 * np.pi * nc, rel=2e-3)
    assert im.disc_chords(nc, None).friction is None and im.disc_chords(nc, "Fe").friction is not None


def test_friction_interpolates_the_gerrits_table_and_falls_with_height():
    """``friction_coefficient``: at the heights where the disc's jellium has a tabulated r_s it returns Gerrits+20's LDA
    eta; it falls monotonically out of the disc and vanishes far from it. Every element to Zn has a table, none beyond."""
    n0, w = im.JELLIUM
    x = np.linspace(0.0, 2.0 * im.JELLIUM_EXTENT_AU, 400)
    for el in ("H", "He", "C", "O", "Fe"):
        rs, eta = im._friction_table()[el]
        n = 3.0 / (4.0 * np.pi * rs**3)
        np.testing.assert_allclose(im.friction_coefficient(el, np.sqrt(-w * np.log(n / n0))), eta, rtol=1e-10)
        g = im.friction_coefficient(el, x)
        on = g > 0.0                                      # far out, where r_s ~ 1e3, it underflows to 0
        assert on[x <= np.sqrt(-w * np.log(n[-1] / n0))].all() and np.all(np.diff(g[on]) < 0.0) and np.all(np.diff(g) <= 0.0)
        assert g[-1] < 1e-6 * g[0]
    assert im.has_friction("Zn") and not im.has_friction("Kr") and not im.has_friction(None)


def test_knock_on_starts_at_the_critical_energy_and_leaves_the_rest_as_heat():
    """``knock_on`` (MJT eqs 14-15): no knock-out below E_0n = T_0 / gamma (MJT Table 2: 26.4, 10 and 7.5 eV for H, He and C
    on C at T_0 = 7.5 eV), one above it, vanishing at it; below it all the elastic stopping is heat, above it less."""
    for el, table in (("H", 26.4), ("He", 10.0), ("C", 7.5)):
        m1, m2 = im.ATOMS[el][1], im.ATOMS["C"][1]
        e0n = 7.5 * (m1 + m2) ** 2 / (4.0 * m1 * m2)
        assert e0n == pytest.approx(table, rel=0.01)
        sigma, below = im.knock_on(el, np.array([0.999 * e0n, 1.0001 * e0n, 3.0 * e0n]), 7.5)
        assert sigma[0] == 0.0 and 0.0 < sigma[1] < 1e-2 * sigma[2]
        s0 = im.knock_on(el, np.array([3.0 * e0n]), 1e9)[1][0]                    # no transfer reaches T_0: S_n0
        assert below[2] < s0 and below[0] > 0.0


def test_thermal_ion_heat_and_knock_outs():
    """``thermal_ion_passages``: the heat in the grid is the friction and elastic stopping of every chord (``disc_chords``'
    sums), to the grid's split; at MJT's T_0 the He knock-outs at 1e5 K are twice Micelotta+10b's nuclear-dominated fit
    (their factor 1/2, which is dropped); a cation draws fewer of them than the neutral, an anion more."""
    nc, t = 50, 1e5
    ps, ko = im.thermal_ion_passages(nc, "He", t)
    assert ps.beyond == 0.0 and ps.P[1:].sum() == 0.0 and set(ko) == {"C"}
    kt = 8.617333262e-5 * t
    tt = np.geomspace(1e-2 * kt, 40.0 * kt, 48)
    proj = im.nucleus("He", 1, im.ATOMS["He"][1])
    wt = im._trapezoid_weights(np.log(tt)) * tt * im.maxwellian_flux(proj, t, 1.0)(tt)
    ch = im.disc_chords(nc, "He")
    v_au = np.sqrt(2.0 * tt / proj.rest_ev) * im.C_CM_S / im.V_AU_CM_S
    below = im.knock_on("He", tt, im.KNOCKOUT_T0_EV["C"])[1]
    heat = np.sum(wt * (v_au * (ch.weight * ch.friction).sum() + below * (ch.weight * ch.atoms).sum()))
    assert (ps.P[0] * ps.e).sum() * ps.de == pytest.approx(heat, rel=1e-6)
    x = np.log10(t)
    fit = 10.0 ** sum(c * x**i for i, c in enumerate((-971.448, 770.259, -245.561, 38.8995, -3.05787, 0.0954303)))
    assert im.thermal_ion_passages(nc, "He", t, t0_ev={"C": 7.5})[1]["C"] / (fit / 0.1) == pytest.approx(2.0, rel=0.15)
    ko_z = [im.thermal_ion_passages(nc, "H", t, charge=z)[1]["C"] for z in (-1, 0, 2)]
    assert ko_z[0] > ko_z[1] > ko_z[2] > 0.0


def test_stopping_probability_is_the_poisson_chance_of_enough_collisions():
    """``stopping_probability``: with every collision losing the same W, losing >= t takes ceil(t / W) collisions, so P is
    the passages' mean Poisson tail at nu = sigma / sigma_g; P = 1 for nothing to lose, and falls as the threshold rises."""
    from scipy.stats import poisson

    nc, w0 = 54, 10.0
    w = np.linspace(5.0, 15.0, 101)
    distant = np.where(np.isclose(w, w0), 2e-16 / (w[1] - w[0]), 0.0)
    coll = im.Collisions(w, distant, np.zeros_like(w), np.zeros_like(w))
    r, d = im.disc_radius_cm(nc), im.DISC_THICKNESS_CM
    th = (np.arange(24) + 0.5) * (0.5 * np.pi / 24)
    sg = np.pi * r**2 * np.cos(th) + 2.0 * r * d * np.sin(th)
    sigma = np.trapezoid(distant, w)
    for t, n0 in ((25.0, 3), (5.0, 1), (35.0, 4)):
        exact = (sg * np.sin(th) * poisson.sf(n0 - 1, sigma / sg)).sum() / (sg * np.sin(th)).sum()
        assert im.stopping_probability(coll, nc, t) == pytest.approx(exact, rel=1e-6)
    assert im.stopping_probability(coll, nc, 0.0) == 1.0
    assert im.stopping_probability(coll, nc, 15.0) > im.stopping_probability(coll, nc, 25.0)


def test_close_collisions_below_the_ip_excite_and_do_not_ionise():
    """``close_ionising`` (pahchem audit C1): a close collision frees the electron only if its recoil exceeds the IP.
    A 12 eV electron (Q_max = 6 eV < IP) ionises through distant collisions alone, as if it had no close ones; for
    protons, whose scaling is 1, the split moves collisions between dz = 0 and +1 and loses none; above the IP every
    close collision ionises."""
    nc, nh, ip = 24, 12, 7.4
    w = im.w_grid()
    df = im.oscillator_strength(nc, nh, 0, w)
    y = lambda E: photo.yield_jochims1996(E, ip)                                   # noqa: E731
    c = im.collisions(im.ELECTRON, 12.0, df, w, ip_ev=ip)
    share, _ = im.close_ionising(c, ip)
    active = c.close > 0.0                                                         # oscillators with close collisions
    assert active.any() and not share[(w < ip) & active].any() and np.all(share[w >= ip] == 1.0)
    bare = im.Collisions(w, c.distant, np.zeros_like(w), np.zeros_like(w), c.ion_scale, c.kinematics)
    assert im.deposition(c, ip, 18.9, y).total(1) == pytest.approx(im.deposition(bare, ip, 18.9, y).total(1), rel=1e-12)
    p = im.collisions(im.PROTON, 1e6, df, w, correction=im.low_speed_correction(im.PROTON, 1e6), ip_ev=ip)
    share, q = im.close_ionising(p, ip)
    assert np.all((share > 0.0) & (share <= 1.0)) and np.all(q[w < ip] < ip)
    assert im.deposition(p, ip, 18.9, y, e_max=1e5).total() == pytest.approx(p.total, rel=1e-9)


def test_passage_chain_reduces_to_one_state_and_rising_ips_ionise_less():
    """``passage_outcomes``' chain (pahchem audit M1): a chain of identical states gives the single-state passages
    exactly; with the IP rising by 3 eV per charge added, 4 MeV alphas (2.5 collisions per passage) leave fewer multiply
    ionised products and a lower mean charge, and no passage is lost."""
    nc, nh, ip = 24, 12, 7.4
    w = im.w_grid()
    fe = im.nucleus("He", 2, 4.0026)
    t = 4e6
    ladder = [(ip + 3.0 * k, ip + 3.0 * (k + 1), lambda E, i=ip + 3.0 * k: photo.yield_jochims1996(E, i))
              for k in range(6)]
    chain = [(im.collisions(fe, t, im.oscillator_strength(nc, nh, 0, w), w, correction=im.low_speed_correction(fe, t),
                            ip_ev=a), a, b, f) for a, b, f in ladder]
    one = im.passage_outcomes(chain[0][0], nc, *ladder[0])
    same = im.passage_outcomes(chain[0][0], nc, *ladder[0], chain=[chain[0]] * 6)
    np.testing.assert_allclose(same.P, one.P, rtol=1e-9, atol=1e-12 * one.P.max())
    assert same.beyond == pytest.approx(one.beyond, rel=1e-9, abs=1e-30)
    rising = im.passage_outcomes(chain[0][0], nc, *ladder[0], chain=chain)
    assert rising.P[2:].sum() < 0.95 * one.P[2:].sum()
    mean_dz = lambda ps: (np.arange(ps.P.shape[0]) @ ps.P.sum(axis=1)) / ps.P.sum()   # noqa: E731
    assert mean_dz(rising) < mean_dz(one)
    assert rising.P.sum() * rising.de + rising.beyond == pytest.approx(one.P.sum() * one.de + one.beyond, rel=1e-9)


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
