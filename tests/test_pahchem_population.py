"""Population fits of observed PAH bands (M7d): the machinery's invariants.

The PAHdb-based tests need PAHdb v3.20 ($CALIMA_PAHDB_DIR, manual dataset
``pahdb-theoretical-v3-20``) and skip without it; the others run anywhere.
"""

import re

import numpy as np
import pytest

from pycalima.pahchem import dos, emission, population
from pycalima.pahchem.baseline import E_PH, Environment

_CACHE = {}
WINDOWS = {"3.3": (3.2, 3.36), "6.2": (6.0, 6.6), "11.2": (11.1, 11.8), "12.7": (12.3, 13.1)}


def _ascii():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_suite

    try:
        return ames_suite.ascii_path()
    except DatasetUnavailable:
        pytest.skip("PAHdb v3.20 not installed (calima-fetch-data import pahdb-theoretical-v3-20 ...)")


def _library():
    if "lib" not in _CACHE:
        _CACHE["lib"] = population.library_m20(_ascii())
    return _CACHE["lib"]


def _numeric_window(freqs, strength, lo_um, hi_um, local):
    grid = np.arange(400.0, 4000.0, 0.05)
    s = emission.spectrum(freqs, strength, grid)
    m = (grid >= 1e4 / hi_um) & (grid <= 1e4 / lo_um)
    flux = s[m].sum() * 0.05
    if local:
        lam = 1e4 / grid[m]
        f1, f2 = np.interp([1e4 / lo_um, 1e4 / hi_um], grid, s)
        flux -= ((f1 + (f2 - f1) * (lam - lo_um) / (hi_um - lo_um)).sum() * 0.05)
    return flux


@pytest.mark.parametrize("local", [False, True])
def test_band_matrix_equals_numeric_integral_of_the_spectrum(local):
    freqs = np.array([3050.0, 1610.0, 890.0, 845.0, 790.0, 1300.0])
    strength = np.array([1.0, 0.5, 2.0, 1.0, 0.7, 0.3])
    M = population._band_matrix(freqs, WINDOWS, local=tuple(WINDOWS) if local else ())
    for b, (lo, hi) in WINDOWS.items():
        got = strength @ M[:, list(WINDOWS).index(b)]
        assert got == pytest.approx(_numeric_window(freqs, strength, lo, hi, local), rel=2e-3, abs=1e-6)


def test_local_baseline_removes_a_line_far_outside_the_window():
    M = population._band_matrix([2000.0], {"11.2": (11.1, 11.8)}, local=("11.2",))
    M0 = population._band_matrix([2000.0], {"11.2": (11.1, 11.8)})
    assert abs(M[0, 0]) < 1e-2 * M0[0, 0]        # left: the line's curvature across the window


def test_fit_recovers_a_synthetic_population():
    rng = np.random.default_rng(1)
    F = rng.uniform(0.05, 1.0, size=(20, 8))
    bands = [f"b{i}" for i in range(8)]
    true = np.zeros(20)
    true[[2, 7, 15]] = [3.0, 1.0, 0.5]
    model = true @ F
    obs = dict(zip(bands, model / model[0]))
    res = population.fit_abundances(F, obs, "b0")
    assert res["chi2"] < 1e-8
    for b in bands:
        assert res["ratios"][b] == pytest.approx(obs[b], rel=1e-6)
    # with the true species only, the weights themselves come back (normalised to b0)
    sub = population.fit_abundances(F[[2, 7, 15]], obs, "b0")
    assert sub["weights"] == pytest.approx(true[[2, 7, 15]] / model[0], rel=1e-6)


def test_bootstrap_is_reproducible_and_non_negative():
    rng = np.random.default_rng(2)
    F = rng.uniform(0.05, 1.0, size=(6, 5))
    obs = dict(zip("abcde", (np.ones(6) @ F) / (np.ones(6) @ F)[0]))
    b1 = population.bootstrap(F, obs, "a", n=20, seed=3)
    b2 = population.bootstrap(F, obs, "a", n=20, seed=3)
    assert np.array_equal(b1, b2) and (b1 >= 0).all()


def test_population_statistics_counts_edge_hydrogens():
    lib = [population.Molecule("a", 24, 12, {"solo": 0, "duo": 12, "trio": 0, "quartet": 0}, {}, {}, 7.0),
           population.Molecule("b", 54, 18, {"solo": 6, "duo": 12, "trio": 0, "quartet": 0}, {}, {}, 6.0)]
    s = population.population_statistics(lib, [1.0, 1.0])
    assert s["mean_nc"] == pytest.approx(39.0)
    assert s["solo/duo"] == pytest.approx(6.0 / 24.0)
    assert s["solo/trio"] == np.inf
    assert population.population_statistics(lib, [1.0, 1.0], by="carbon")["mean_nc"] == pytest.approx(
        (24**2 + 54**2) / 78.0)


# -- PAHdb ----------------------------------------------------------------------------------------
def test_m20_sample_uids_match_their_v3_20_formulas():
    from pycalima.pahchem import ames_suite, maragkoudakis2020

    s = maragkoudakis2020.sample()
    meta = ames_suite.read_metadata(_ascii(), s.uid)
    for r in s.itertuples():
        m = re.match(r"C(\d+)H(\d+)", meta[r.uid]["FORMULA"])
        assert (int(m[1]), int(m[2]), meta[r.uid]["CHARGE"]) == (r.nc, r.nh, r.charge), r.uid


def test_library_m20_pairs_and_dft_energetics():
    lib = _library()
    assert len(lib) == 133
    assert sum(m.ea_ev is not None for m in lib) == 38
    for m in lib:
        assert sum(m.edges.values()) <= m.nh
        assert 3.5 < m.ip_ev < 8.0          # lowest: C82H24 (561), 3.76 eV
        assert m.ea_ev is None or 0.0 < m.ea_ev < m.ip_ev
    c54 = next(m for m in lib if m.name == "C54H18 (37)")
    assert c54.ip_ev == pytest.approx(6.14, abs=0.3)       # Andrews+16 Table A.1 (Malloci+07)
    nc, ip = np.array([m.nc for m in lib]), np.array([m.ip_ev for m in lib])
    assert np.corrcoef(np.log(nc), ip)[0, 1] < -0.5



def test_all_anions_extends_but_keeps_the_table_h3_anions():
    lib, ext = _library(), population.library_m20(_ascii(), all_anions=True)
    assert [m.name for m in lib] == [m.name for m in ext]
    assert sum(m.ea_ev is not None for m in ext) == 94          # anions matched by bond graph
    for a, b in zip(lib, ext):
        if -1 in a.uids:
            assert b.uids[-1] == a.uids[-1]
        if b.ea_ev is not None:
            assert 0.0 < b.ea_ev < b.ip_ev and b.species[-1].charge == -1 and b.species[-1].nc == b.nc

def test_charge_balance_follows_the_conditions():
    lib = _library()
    mol = next(m for m in lib if m.ea_ev is not None and m.nc >= 40)
    base = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    f = population.charge_fractions(mol, base)
    assert sum(f.values()) == pytest.approx(1.0) and set(f) == {-1, 0, 1}
    brighter = population.charge_fractions(mol, Environment("g", 1e5, 5e4, 10.0, 500.0, 40000))
    denser = population.charge_fractions(mol, Environment("e", 1e4, 5e4, 100.0, 500.0, 40000))
    assert brighter[1] > f[1] > denser[1]
    assert brighter[-1] < f[-1] < denser[-1]


def test_emitted_band_energy_equals_absorbed_heating():
    mol = next(m for m in _library() if m.name == "C54H18 (37)")
    pop = population.Population([mol], {"all": (1.0, 1e4)})
    env = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    for z in mol.charges:
        b = pop.basis(mol, z, population.field_shape(env))
        absorbed = env.g0 * (b.absorbed + b.recombination)                  # photons plus recombination heating
        assert pop.state_band_fluxes(mol, z, env)[0] == pytest.approx(absorbed, rel=0.01)


def test_recombination_heating_deposits_the_threshold_at_the_ionisation_rate():
    from pycalima.pahchem import dos
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in population.library_m20(_ascii()) if m.name == "C54H18 (37)")
    shape = ("kurucz", 40000)
    on = population.Population([mol], {"all": (1.0, 1e4)})
    off = population.Population([mol], {"all": (1.0, 1e4)}, params=Params(recombination_heating=False))
    b0, b1 = on.basis(mol, 0, shape), on.basis(mol, 1, shape)
    assert b0.recombination == pytest.approx(b0.k_up * mol.ionisation_ev[0] * dos.CM_PER_EV)
    assert b1.recombination == pytest.approx(b1.k_up * on.library[0].ionisation_ev[1] * dos.CM_PER_EV)
    library = population.Population([mol], {"all": (1.0, 1e4)}, params=Params(charge_ladder="library"))
    assert library.basis(mol, 1, shape).recombination == 0.0                # the top of the chain
    assert off.basis(mol, 0, shape).recombination == 0.0
    assert b0.emitted.sum() == pytest.approx(b0.absorbed + b0.recombination, rel=0.01)
    assert b0.emitted.sum() > 1.2 * off.basis(mol, 0, shape).emitted.sum()  # neutrals: tens of per cent more power


def test_photoelectron_power_is_the_partitioned_excess_energy():
    """int sigma Y F p (E - IP) dE: zero above the field's edge, linear in p, and the analytic value for flat inputs."""
    from pycalima.pahchem import fields, photo

    field = fields.draine1978(g0=1.0)
    sigma, y = np.full(E_PH.size, 1e-16), np.ones(E_PH.size)
    ip = 7.0
    on = E_PH >= ip
    exact = np.trapezoid((1e-16 * (E_PH - ip) * field.flux_at(E_PH))[on], E_PH[on])
    assert photo.photoelectron_power(field, sigma, y, E_PH, ip, 1.0) == pytest.approx(exact, rel=2e-3)
    assert photo.photoelectron_power(field, sigma, y, E_PH, ip, 0.46) == pytest.approx(
        0.46 * photo.photoelectron_power(field, sigma, y, E_PH, ip, 1.0))
    assert photo.photoelectron_power(field, sigma, y, E_PH, 13.6, 1.0) == 0.0


def test_photoelectric_heating_shares_the_charge_balance_rates():
    """Per state, Gamma / (p k_up) is the mean photoelectron excess, between 0 and 13.6 - IP, with k_up the basis
    rate that sets the charge balance; the anion detaches with p = 1; eps falls as gamma = G0 T^0.5 / n_e rises."""
    from pycalima.pahchem import photo

    mol = next(m for m in _library() if m.name == "C54H18 (37)")
    pop = population.Population([mol], WINDOWS)
    env = Environment("test", 1e3, 1e4, 3.0, 300.0, 40000)
    shape = population.field_shape(env)
    G, _ = pop._pe_stack(shape, photo.PHOTOELECTRON_PARTITION)
    own = pop.library[0]                                                    # its ladder, to the cold limit
    live = pop._live_charges(own, shape)
    for z in live[:-1]:
        p = 1.0 if z < 0 else photo.PHOTOELECTRON_PARTITION
        mean_excess = G[0, z - pop.zmin] / (p * pop.basis(own, z, shape).k_up)
        assert 0.0 < mean_excess < 13.6 - own.ionisation_ev[z]
    # neither the last state the field reaches nor the top of the chain ionises
    assert G[0, live[-1] - pop.zmin] == 0.0 and G[0, max(own.charges) - pop.zmin] == 0.0
    gamma, gz, eps = pop.photoelectric_heating(env)
    assert gamma[0] == pytest.approx(gz[0].sum()) and 0.0 < eps[0] < 0.2
    _, gz5, _ = pop.photoelectric_heating(env, partition=0.5)
    i0, i1 = 0 - pop.zmin, 1 - pop.zmin
    assert gz5[0, i0] == pytest.approx(gz[0, i0] * 0.5 / photo.PHOTOELECTRON_PARTITION)
    if -1 in mol.charges:
        assert gz5[0, -1 - pop.zmin] == pytest.approx(gz[0, -1 - pop.zmin])     # detachment: p = 1 either way
    assert gz5[0, i1] == 0.0 or gz5[0, i1] == pytest.approx(gz[0, i1] * 0.5 / photo.PHOTOELECTRON_PARTITION)
    _, _, eps_hi = pop.photoelectric_heating(Environment("bright", 1e5, 1e4, 3.0, 300.0, 40000))
    assert eps_hi[0] < eps[0]                                              # more charge, higher barrier


def test_statistical_detachment_gives_slow_electrons_and_bounded_efficiency():
    """detachment = "statistical": the yield is a probability at least the prompt WD01 share; detached electrons
    carry less than E - EA (mostly thermionic, < 0.5 eV above ~6 eV); with anions present the PE efficiency
    stays in the BT94/WD01 range instead of the ~0.3-0.6 that full-excess electrons give."""
    from pycalima.models.dust_charge.shared_physics import photodetachment_cross_section_vec
    from pycalima.pahchem import photo
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in _library() if m.ea_ev is not None and m.nc >= 40)
    y, ek = population.statistical_detachment(mol)
    fuv = slice(population.E_PH_VIS.size, population.E_PH_VIS.size + E_PH.size)   # the 5-13.6 eV part
    y, ek = y[fuv], ek[fuv]
    assert np.array_equal(population.E_PH[fuv], E_PH)
    ea = mol.ionisation_ev[-1]
    prompt = photodetachment_cross_section_vec(E_PH, ea, -1) / photo.sigma_li_draine(mol.nc, -1, E_PH)
    assert np.all((y >= np.clip(prompt, 0, 1) - 1e-12) & (y <= 1 + 1e-12))
    assert np.all(ek[E_PH > ea] < E_PH[E_PH > ea] - ea) and np.all(ek[E_PH > 7.0] < 0.5)
    env = Environment("cnm", 1.7, 0.0, 0.04, 100.0, None)
    att = {"attachment_model": "constant"}                                     # f = 1: anion-rich, so detachment decides
    stat = population.Population([mol], WINDOWS, params=Params(detachment="statistical", **att))
    assert stat.charge_matrix(env)[0, -1 - stat.zmin] > 0.1                 # anions matter here
    assert 0.0 < stat.photoelectric_heating(env)[2][0] < 0.08
    jochims = population.Population([mol], WINDOWS, params=Params(detachment="jochims", **att))
    assert jochims.photoelectric_heating(env)[2][0] > 0.15                     # full-excess electrons


@pytest.mark.parametrize("detachment", ["jochims", "statistical"])
def test_ionising_photons_heat_conserves_the_absorbed_energy(detachment):
    """With ``ionising_photons_heat``, internal energy + electron energy + binding work = absorbed energy, for
    every charge (the anion's detached electron takes all, or the statistical share, of the excess)."""
    from pycalima.pahchem import photo
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in _library() if m.ea_ev is not None and m.nc >= 40)
    p = Params(ionising_photons_heat=True, detachment=detachment)
    field = population._unit_field(("kurucz", 40000, 500.0))
    for z in mol.charges:
        inp = population._basis_inputs(mol, z, ("kurucz", 40000, 500.0), 200.0, 1e5, p)
        sigma = photo.sigma_li_draine(mol.nc, z, population.E_PH)
        h = population.deposit_spectrum(field, sigma, inp["yield_"], inp["threshold_ev"], True, inp["electron_ev"])
        kept = np.trapezoid(h * population.E_DEP, population.E_DEP)
        out = field.photon_rate(sigma * inp["yield_"] * (inp["electron_ev"] + inp["threshold_ev"]), population.E_PH)
        assert (kept + out) / field.photon_rate(sigma * population.E_PH, population.E_PH) == pytest.approx(1.0, abs=1e-3)


def test_photoelectron_energy_conserves_the_excess_and_follows_the_measured_share():
    """<E_k> + E_int = E - IP; share 0.46 up to 13.6 eV, the Hrodmarsson+25 means to 21.9 eV, then a constant E_int."""
    from pycalima.pahchem import photo

    E = np.array([5.0, 8.0, 13.6, 18.0, 21.9, 50.0, 300.0])
    ke, ei = photo.photoelectron_energy(E, 6.5)
    assert np.allclose(ke + ei, np.clip(E - 6.5, 0, None)) and ke[0] == 0.0 == ei[0]
    assert ke[1] == pytest.approx(0.46 * 1.5) and ke[2] == pytest.approx(0.46 * 7.1)
    assert ke[3] == pytest.approx(0.56 * 11.5) and ei[5] == pytest.approx(ei[4]) == pytest.approx(ei[6])


def test_euv_grid_leaves_lyman_cut_fields_unchanged_and_closes_the_energy():
    """For a field cut at 13.6 eV the EUV grid adds nothing (rates as on the 5-13.6 eV grid alone). With EUV and
    ionising deposits, every absorbed eV ends up as vibrational deposit, photoelectron energy or ionisation
    work; the EUV raises the charge and the heating, and the top state's ionisations are reported."""
    from dataclasses import replace

    from pycalima.pahchem import dos, photo
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in _library() if m.name == "C54H18 (37)")
    fuv = population.E_PH <= 13.6
    pop = population.Population([mol], WINDOWS)
    own = pop.library[0]                                                    # its ladder, to the cold limit
    f1 = population._unit_field(("kurucz", 40000))
    for z in pop._live_charges(own, ("kurucz", 40000)):
        sig = photo.sigma_uv(mol.nc, z, population.E_PH)
        y = population._yield(own, z, pop.p)
        assert pop.basis(own, z, ("kurucz", 40000)).k_up == pytest.approx(
            photo.ionisation_rate(f1, sig[fuv], y[fuv], population.E_PH[fuv]), rel=1e-6)
    heat = population.Population([mol], WINDOWS, params=replace(Params(), ionising_photons_heat=True))
    shape = ("kurucz", 40000, 99.0)
    fe = population._unit_field(shape)
    own = heat.library[0]
    for z in heat._live_charges(own, shape):                                # every state, highly charged ones too
        b = heat.basis(own, z, shape)
        sig = photo.sigma_uv(mol.nc, z, population.E_PH)
        thr = population.state_threshold(own, z)
        y = population._yield(own, z, heat.p)
        ke, _ = photo.photoelectron_energy(population.E_PH, thr)
        parts = (b.absorbed / dos.CM_PER_EV + thr * b.k_up
                 + photo.photoelectron_power(fe, sig, y, population.E_PH, thr, electron_energy_ev=ke))
        assert parts == pytest.approx(fe.photon_rate(sig * population.E_PH, population.E_PH), rel=1e-3)
    cut = Environment("cut", 1e4, 1e3, 30.0, 500.0, 40000)
    euv = Environment("euv", 1e4, 1e3, 30.0, 500.0, 40000, 99.0)
    z_mean = lambda env: float(heat.charge_matrix(env)[0] @ np.array(heat.zs))      # noqa: E731
    assert z_mean(euv) > z_mean(cut)
    assert heat.photoelectric_heating(euv)[0][0] > 2.0 * heat.photoelectric_heating(cut)[0][0]
    assert heat.chain_overflow(cut)[0] == 0.0 < heat.chain_overflow(euv)[0]


def test_photons_below_5ev_detach_anions_and_leave_the_higher_thresholds_alone():
    """The 1-5 eV grid: a field cut at 5 eV gives the rates of the old 5-13.6 eV grid; the photons below 5 eV only
    reach the anion's detachment (EA ~1.5 eV), not the neutral's or the cation's thresholds, and every state absorbs
    more (the cations in the visible). The Draine shape carries the old stars per unit G0."""
    from pycalima.pahchem import fields, photo

    mol = next(m for m in _library() if m.name == "C54H18 (37)")
    pop = population.Population([mol], WINDOWS)
    full, cut = ("kurucz", 40000), ("kurucz", 40000, 13.6, 5.0)
    uncut = fields.kurucz(40000, 1.0)
    for z in mol.charges[:-1]:
        b_full, b_cut = pop.basis(mol, z, full), pop.basis(mol, z, cut)
        y_fuv = population._yield(mol, z, pop.p, energy_ev=E_PH)
        assert b_cut.k_up == pytest.approx(
            photo.ionisation_rate(uncut, photo.sigma_uv(mol.nc, z, E_PH), y_fuv, E_PH), rel=1e-6)
        if mol.ionisation_ev[z] < 5.0:
            assert b_full.k_up > 1.05 * b_cut.k_up
        else:
            assert b_full.k_up == pytest.approx(b_cut.k_up, rel=1e-9)
        assert b_full.absorbed > b_cut.absorbed
    lo = Environment("lo", 1.7, 50.0, 8e-3, 80.0, None)
    hi = Environment("hi", 170.0, 50.0, 8e-3, 80.0, None)
    assert population.field_shape(Environment("old", 1.7, 50.0, 8e-3, 80.0, None, field_e_min_ev=5.0)) == ("draine",)
    E = np.linspace(1.0, 4.9, 20)
    for env in (lo, hi):
        f = population._unit_field(population.field_shape(env))
        assert env.g0 * f.flux_at(E) == pytest.approx(fields.old_stars_flux(E), rel=1e-4)


def test_recombination_cooling_takes_kt_per_capture_and_grows_with_temperature():
    """Each captured electron takes kT Lambda~/J~ (DS87): ~1 kT on cations (focused), 1.5-2 kT on neutrals. The
    cooling is the charge balance's capture flux times that energy, and at fixed gamma it rises against the
    photoelectric heating with T (BT94: it matters above ~3e3 K)."""
    from pycalima.pahchem import chemistry

    mol = next(m for m in _library() if m.ea_ev is not None and m.nc >= 40)        # neutral capture needs the anion
    pop = population.Population([mol], WINDOWS)
    for T in (100.0, 8000.0):
        e = pop._capture_energy(T)[0] / (chemistry.KB_EV * T)
        assert 1.0 <= e[1 - pop.zmin] < 1.3 and 1.45 < e[0 - pop.zmin] < 2.0
    ratio = []
    for T, ne in ((300.0, 3.0), (8000.0, 3.0 * np.sqrt(8000.0 / 300.0))):          # same gamma = G0 T^0.5 / n_e
        env = Environment("x", 1e4, 0.0, ne, T, None)
        lam, lz = pop.recombination_cooling(env)
        f = pop.charge_matrix(env)[0]
        flux = ne * f * pop._capture_rates(T)[0] * pop._capture_energy(T)[0] * chemistry.EV_ERG
        assert lam[0] == pytest.approx(flux.sum()) and np.allclose(lz[0], flux)
        ratio.append(lam[0] / pop.photoelectric_heating(env)[0][0])
    assert ratio[1] > 3.0 * ratio[0]


def test_ion_collision_rates_match_their_lab_anchors():
    """Mutual neutralisation is 0.58 x DS87 (the Shuman+14 law for C50H18- + C+ at 300 K, 7.9e-8); the ion-neutral
    rate is 0.9 x Langevin (anthracene + C+ measured 3 +- 1.5e-9 at 293 K, Canosa+95)."""
    from pycalima.pahchem import chemistry

    mu = 12.011 * (50 * 12.011 + 18 * 1.008) / (12.011 + 50 * 12.011 + 18 * 1.008)
    assert chemistry.mutual_neutralisation(50, 18, -1, "C+", 300.0) == pytest.approx(2.7e-7 * mu**-0.5, rel=0.02)
    assert 1.5e-9 < chemistry.ion_neutral_rate(14, 10, "C+") < 4.5e-9
    assert chemistry.mutual_neutralisation(50, 18, -1, "H+", 100.0) > chemistry.mutual_neutralisation(50, 18, -1, "H+", 300.0)


def test_ions_neutralise_anions_and_charge_neutrals_above_their_threshold():
    """C+ in cold neutral gas removes PAH anions (mutual neutralisation competes with photodetachment) and adds
    cations by charge transfer; an ion whose atom's IP is below the molecule's does nothing unless adducts are on;
    the scalar and vectorised chains agree with ions."""
    from dataclasses import replace

    from pycalima.pahchem.baseline import Params

    mol = next(m for m in _library() if m.ea_ev is not None and m.nc >= 40)
    pop = population.Population([mol], WINDOWS)
    bare = Environment("cnm", 1.7, 30.0, 0.04, 100.0, None)
    cplus = replace(bare, ions=(("C+", 0.04),))
    f0, f1 = pop.charge_matrix(bare)[0], pop.charge_matrix(cplus)[0]
    i_m, i_p = -1 - pop.zmin, 1 - pop.zmin
    assert f1[i_m] < f0[i_m] and f1[i_p] > f0[i_p]
    own = pop.library[0]                                                    # its ladder, to the cold limit
    scalar = population.charge_fractions(own, cplus)
    assert np.allclose([scalar[z] for z in own.charges], f1[[z - pop.zmin for z in own.charges]], rtol=1e-6, atol=1e-300)
    low = ("Na+", 0.04)                                                     # IP(Na) 5.14 eV < any PAH IP
    assert population.ion_up_rate(mol, 0, replace(bare, ions=(low,))) == 0.0
    assert population.ion_up_rate(mol, 0, replace(bare, ions=(low,)), Params(metal_adducts=True)) > 0.0


def test_gas_rates_conserve_charge_in_steady_state():
    """Without ions the PAHs make as many electrons as they capture; with ions the net electron source equals minus
    the ion-driven up-flux (the PAHs pass the ions' charge to electrons), and alpha(X) is the abundance-weighted
    flux of each ion."""
    from dataclasses import replace

    mol = next(m for m in _library() if m.ea_ev is not None and m.nc >= 40)
    pop = population.Population([mol], WINDOWS)
    bare = Environment("cnm", 1.7, 30.0, 0.04, 100.0, None)
    x = np.array([3e-7])
    r0 = pop.gas_rates(bare, x)
    made = float(x @ (r0["f"] * bare.g0 * pop._stack(population.field_shape(bare))[0]).sum(axis=1))
    assert abs(r0["electron_source"]) < 1e-9 * made and r0["alpha_mn"] == {}
    env = replace(bare, ions=(("C+", 0.04), ("H+", 1e-3)))
    r = pop.gas_rates(env, x)
    ion_flux = sum(r["alpha_mn"][i] * n + r["alpha_ct"][i] * n for i, n in env.ions)
    assert r["electron_source"] == pytest.approx(-ion_flux, rel=1e-6)
    assert r["alpha_mn"]["C+"] > 0 and r["carbon_accretion"] == pytest.approx(0.4 * r["alpha_ct"]["C+"])
    assert r["gamma_pe"] > 0 and r["lambda_rec"] > 0


def test_vectorised_charges_equal_the_scalar_chain():
    lib = population.library_m20(_ascii(), all_anions=True)
    pop = population.Population(lib, WINDOWS)
    for env in (Environment("a", 2e4, 1e5, 100.0, 700.0, 40000), Environment("b", 7e4, 5e4, 10.0, 700.0, 40000),
                Environment("c", 1e3, 3e4, 5.0, 700.0, None)):
        _, fz = pop.band_fluxes(env)
        for m in lib[::7]:
            own = pop._own(m)                                               # its ladder, to the cold limit
            ref = population.charge_fractions(own, env)
            for z in own.charges:
                assert fz[m.name][z] == pytest.approx(ref[z], rel=1e-9, abs=1e-14)


def test_k_h_table_interpolates_the_multiphoton_steady_state():
    from pycalima.pahchem import energy_me, photo, rates

    mol = next(m for m in _library() if m.nc <= 24)
    pop = population.Population([mol], WINDOWS)
    shape = ("kurucz", 40000)
    b = pop.basis(mol, 1, shape)
    assert np.all(np.diff(b.k_h / population.K_H_G0_GRID) >= -1e-12 * b.k_h[1:] / population.K_H_G0_GRID[1:])
    g0 = np.sqrt(population.K_H_G0_GRID[8] * population.K_H_G0_GRID[9])
    sp = mol.species[1]
    sigma = photo.sigma_uv(mol.nc, 1, population.E_PH)
    own = pop.library[0]                         # the population's copy: its ladder, so +1 is not the top state
    h = population.deposit_spectrum(population._unit_field(shape), sigma, population._yield(own, 1),
                                    population.state_threshold(own, 1), False)
    cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, pop.e_max_cm, pop.de_cm)
    E, counts = dos.beyer_swinehart(sp.freqs_cm, cm.energy_cm[-1] + 200.0)
    ts = rates.transition_state_for_entropy(sp.freqs_cm, 3050.0, pop.p.h_even[1], n_scaled=5)
    k = rates.rrkm(E, counts, ts, pop.p.h_even[0] * dos.CM_PER_EV, degeneracy=mol.nh)
    k = k[np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)]
    direct = cm.steady_state(population.E_DEP, g0 * h, {"H": k}).rates["H"]
    assert b.k_h_at(g0) == pytest.approx(direct, rel=0.05)


def test_basis_disk_cache_round_trips(tmp_path):
    mol = next(m for m in population.library_m20(_ascii()) if m.name == "C54H18 (37)")   # fresh: no memory cache
    env = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    first = population.Population([mol], WINDOWS, cache_dir=tmp_path)
    F1, _ = first.band_fluxes(env)
    assert len(list(tmp_path.glob("*.npz"))) == len(first._live_charges(first.library[0], population.field_shape(env)))
    mol2 = next(m for m in population.library_m20(_ascii()) if m.name == "C54H18 (37)")
    second = population.Population([mol2], WINDOWS, cache_dir=tmp_path)
    F2, _ = second.band_fluxes(env)
    assert np.array_equal(F1, F2)
    assert second.intact_fraction(mol2, env) == first.intact_fraction(mol, env)


def test_basis_stored_under_its_legacy_key_is_found_not_rebuilt(tmp_path, monkeypatch):
    mol = next(m for m in population.library_m20(_ascii()) if m.name == "C54H18 (37)")
    env = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    first = population.Population([mol], WINDOWS, cache_dir=tmp_path)
    F1, _ = first.band_fluxes(env)
    own = first.library[0]                      # its ladder, to the cold limit
    live = first._live_charges(own, population.field_shape(env))
    for z in live:                              # move each file to the key the previous version gave it
        inp = population._basis_inputs(own, z, population.field_shape(env), first.de_cm, first.e_max_cm, first.p)
        (tmp_path / f"{population._basis_key(inp)}.npz").rename(tmp_path / f"{population._basis_key(inp, legacy=True)}.npz")
    monkeypatch.setattr(population, "_build", lambda inp: pytest.fail("rebuilt a stored basis"))
    mol2 = next(m for m in population.library_m20(_ascii()) if m.name == "C54H18 (37)")
    F2, _ = population.Population([mol2], WINDOWS, cache_dir=tmp_path).band_fluxes(env)
    assert np.array_equal(F1, F2)
    assert len(list(tmp_path.glob("*.npz"))) == 2 * len(live)           # new keys linked to the old files


def test_large_compact_pahs_stay_intact_in_the_orion_bar():
    """C96H24 stays intact at the Atomic most-ionising corner; C54H18, near threshold, does not (the network
    gives 0.000 for C54H18 there); the chain never exceeds the single first-step ratio."""
    from pycalima.pahchem import conditions

    lib = _library()
    c54 = next(m for m in lib if m.name == "C54H18 (37)")
    c96 = next(m for m in lib if m.name == "C96H24 (108)")
    env = conditions.bracketing_environments("orion_bar", "Atomic")["most ionising"]
    pop = population.Population([c54, c96], WINDOWS)
    chain, single = pop.intact_fractions(env), pop.intact_fractions(env, method="single")
    assert chain[1] > 0.9 and chain[0] < chain[1]
    assert np.all(chain <= single + 1e-9)


# -- PAHdb XML and the v4.00 library ----------------------------------------------------------------
def _benzene(scale=1.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.arange(6) * np.pi / 3
    c = np.c_[1.39 * np.cos(a), 1.39 * np.sin(a), np.zeros(6)] * scale
    h = np.c_[2.47 * np.cos(a), 2.47 * np.sin(a), np.zeros(6)] * scale
    xyz = np.vstack([c, h]) + jitter * rng.standard_normal((12, 3))
    return [6] * 6 + [1] * 6, xyz


def test_bond_graph_hash_ignores_geometry_but_not_connectivity():
    from pycalima.pahchem import ames_suite

    t, x = _benzene()
    h0 = ames_suite.bond_graph_hash(t, x)
    assert ames_suite.bond_graph_hash(*_benzene(scale=1.02, jitter=0.02, seed=3)) == h0   # a charge state's geometry
    t2, x2 = list(t), x.copy()
    x2[6] = x2[0] + (x2[1] - x2[0]) * 0.5 + np.array([0.0, 0.0, 1.09])  # move one H onto a C-C bond: other graph
    assert ames_suite.bond_graph_hash(t2, x2) != h0


def _v400():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_suite

    if "v400" not in _CACHE:
        try:
            path = ames_suite.xml_path("pahdb-theoretical-v4-00")
        except DatasetUnavailable:
            pytest.skip("PAHdb v4.00 not installed")
        recs = ames_suite.xml_index(path)                       # no cache_dir: tests do not write to model data
        _CACHE["v400"] = (recs, population.library_v400(recs))
    return _CACHE["v400"]


def test_v400_xml_frequencies_are_used_as_written():
    from pycalima._paths import get_model_data_dir
    from pycalima.pahchem.molecule import PAHSpecies

    recs, _ = _v400()
    rec = next(r for r in recs if r["uid"] == 37)
    f = get_model_data_dir() / "PAH_states" / "C54H18_0.dat"
    if not f.exists():
        pytest.skip("PAH_states not generated")
    ref = PAHSpecies.from_states_file(f)
    assert rec["formula"] == "C54H18" and rec["basis"] == "6-31G*"
    assert np.allclose(np.sort(rec["freqs_cm"]), np.sort(ref.freqs_cm))


def test_v400_library_charge_ladders_and_thresholds():
    _, lib = _v400()
    assert len(lib) > 1300
    assert sum(2 in m.charges for m in lib) > 700
    for m in lib:
        zs = m.charges
        assert zs == tuple(range(zs[0], zs[-1] + 1)) and 0 in zs and 1 in zs
        th = [m.ionisation_ev[z] for z in zs[:-1]]
        assert all(b > a for a, b in zip(th, th[1:])), m.name
    c54 = next(m for m in lib if m.name == "C54H18 (37)")
    assert c54.charges == (-1, 0, 1, 2, 3)
    assert c54.ionisation_ev[0] == pytest.approx(6.14, abs=0.4)          # Andrews+16 Table A.1 (Malloci+07)
    assert c54.ionisation_ev[1] == pytest.approx(8.91, abs=0.4)


def test_v400_aromatic_library_holds_only_edge_h():
    """Every H of the default library sits on an aromatic edge; super-H isomers whose PAHdb tags miss
    the extra H (C96H25, UIDs 684-690) are aliphatic and come in only with include_superh."""
    recs, lib = _v400()
    assert all(sum(m.edges[e] for e in population.EDGES) == m.nh for m in lib)
    assert not {684, 685, 686, 687, 688, 689, 690} & {m.uids[0] for m in lib}
    full = population.library_v400(recs, include_superh=True)
    extra = [m for m in full if m.edges["aliphatic"] > 0]
    assert len(full) == len(lib) + len(extra)
    assert {684, 682, 190} <= {m.uids[0] for m in extra}
    assert all(sum(m.edges[e] for e in population.EDGES) + m.edges["aliphatic"] == m.nh for m in full)
    assert all(m.edges["ch3"] <= m.edges["aliphatic"] for m in full)


# -- charge-input options -------------------------------------------------------------------------
def test_charge_inputs_move_the_charge_balance_the_right_way():
    from pycalima.pahchem.baseline import Params

    lib = population.library_m20(_ascii(), all_anions=True)
    mol = [m for m in lib if m.ea_ev is not None and m.nc >= 40][:3]
    env = Environment("t", 2e4, 5e4, 30.0, 700.0, 40000)
    base = population.Population(mol, WINDOWS).charge_matrix(env)
    hi_ip = population.Population(population.shift_thresholds(mol, {0: 0.6}), WINDOWS).charge_matrix(env)
    slow = population.Population(mol, WINDOWS, params=Params(capture_scale=0.5)).charge_matrix(env)
    i0, i1 = 1, 2                                     # slots of Z = 0, +1 (zs = -1, 0, 1)
    assert (hi_ip[:, i1] < base[:, i1]).all() and (slow[:, i1] > base[:, i1]).all()
    assert (hi_ip[:, i0] > base[:, i0]).all()
    shifted = population.shift_thresholds(mol, {0: 0.6, -1: 0.3})
    for a, b in zip(mol, shifted):
        assert b.ip_ev == pytest.approx(a.ip_ev + 0.6) and b.ea_ev == pytest.approx(a.ea_ev + 0.3)
        assert b._cache == {} and a.ionisation_ev[0] == a.ip_ev       # the input is not modified


def test_associative_detachment_leaves_the_parent_anion_unless_asked():
    """Demarais+12 measured PAH- + H on dehydrogenated anions: the chain (N_H0) has none by default."""
    from pycalima.pahchem.baseline import Params

    _, lib = _v400()
    mol = next(m for m in lib if m.name == "C54H18 (37)")
    env = Environment("t", 2.0, 50.0, 0.01, 100.0, None)
    no_h = Environment("t", 2.0, 0.0, 0.01, 100.0, None)
    f = population.charge_fractions(mol, env)
    assert f == pytest.approx(population.charge_fractions(mol, no_h), rel=1e-12)
    f_all = population.charge_fractions(mol, env, Params(associative_detachment="all"))
    assert f_all[-1] < f[-1]
    assert population.anion_associative_detachment(Params(associative_detachment="all")) == 7.8e-10
    with pytest.raises(ValueError):
        population.anion_associative_detachment(Params(associative_detachment="some"))


def test_wenzel_yield_is_the_cation_one_only():
    """Wenzel+20 measured cation -> dication: the dication takes Jochims', as in the baseline."""
    from pycalima.pahchem import photo

    _, lib = _v400()
    mol = next(m for m in lib if m.name == "C54H18 (37)")
    E = np.linspace(8.0, 13.6, 50)
    assert np.allclose(population._yield(mol, 1, energy_ev=E), photo.yield_wenzel2020(E, mol.ionisation_ev[1], mol.nc))
    assert np.allclose(population._yield(mol, 2, energy_ev=E), photo.yield_jochims1996(E, mol.ionisation_ev[2]))


def test_unit_detachment_leaves_an_anion_nothing_to_emit():
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in population.library_m20(_ascii(), all_anions=True) if m.ea_ev is not None)
    pop = population.Population([mol], WINDOWS, params=Params(detachment="unity", recombination_heating=False))
    b = pop.basis(mol, -1, ("kurucz", 40000, 13.6, 5.0))                 # every photon above the EA detaches
    assert b.absorbed == 0.0 and not b.emitted.any() and b.k_up > 0.0
    assert pop.basis(mol, -1, ("kurucz", 40000)).absorbed > 0.0          # 1 eV - EA: absorbed, not detached
    # with recombination heating the anion still emits what electron attachment deposits (EA per event)
    hot = population.Population([mol], WINDOWS, params=Params(detachment="unity"))
    bh = hot.basis(mol, -1, ("kurucz", 40000, 13.6, 5.0))
    assert bh.absorbed == 0.0 and bh.emitted.sum() == pytest.approx(bh.recombination, rel=0.01) and bh.recombination > 0


def test_lab_recombination_factor_is_the_falp_power_law():
    from pycalima.pahchem import chemistry

    a, b, n = chemistry.lab_recombination_fit()
    assert n == 9 and b == pytest.approx(1.01, abs=0.1)          # 9 points: one fluoranthene duplicate removed
    f = chemistry.lab_recombination_factor([6, 16, 54, 100])
    assert f[0] < 0.2 and 0.1 < f[1] < 0.4 and np.all(np.diff(f) > 0)
    assert chemistry.lab_recombination_factor(1000, cap=3.0) == 3.0


def test_lab_capture_slows_small_and_speeds_large_cation_recombination():
    from pycalima.pahchem.baseline import Params

    lib = population.library_m20(_ascii(), all_anions=True)
    small = next(m for m in lib if m.nc <= 30)
    large = next(m for m in lib if m.nc >= 130)
    env = Environment("t", 2e4, 5e4, 30.0, 700.0, 40000)
    for mol, more_ionised in ((small, True), (large, False)):
        f0 = population.charge_fractions(mol, env)[1]
        f1 = population.charge_fractions(mol, env, Params(capture_model="lab_trend"))[1]
        assert (f1 > f0) == more_ionised


@pytest.mark.parametrize("subset", ("cold", "pahs"))
def test_lab_sticking_rises_with_size_saturates_at_one_and_follows_its_data(subset):
    """The saturating fit to the lab recombination data: monotone, in (0, 1], 1 for large PAHs, and within x3 of every point it
    is fitted to (the systematic scatter of the data: the two naphthalene measurements differ by 3x)."""
    import pandas as pd

    from pycalima._paths import get_external_data_path
    from pycalima.pahchem import chemistry

    n = np.arange(6, 400)
    s = chemistry.lab_sticking(n, subset)
    assert np.all((s > 0) & (s <= 1)) and np.all(np.diff(s) >= 0) and s[-1] == pytest.approx(1.0)
    t = pd.read_csv(get_external_data_path("pahchem/pah_electron_recombination.csv"), comment="#")
    t = t[(t.n_c >= 10) & ((t.method != "FALP-MS") if subset == "cold" else True)]
    ratio = t.k_cm3_s.to_numpy() / np.array([chemistry.electron_capture(x, 1, 300.0) for x in t.n_c])
    model = chemistry.lab_sticking(t.n_c.to_numpy(float), subset)
    assert np.all(np.abs(np.log10(model / ratio)) < np.log10(3.0))
    with pytest.raises(ValueError):
        chemistry.lab_sticking(54, "hot")


def test_cation_capture_models_and_t_power_touch_cations_only():
    from pycalima.pahchem import chemistry
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in population.library_m20(_ascii()) if m.nc <= 30)
    for model in ("lab_cold", "lab_pahs"):
        p = Params(capture_model=model)
        assert population._capture_factor(mol, 0, p) == 1.0
        assert population._capture_factor(mol, 1, p) == pytest.approx(float(chemistry.lab_sticking(mol.nc, model.split("_")[1])))
    p = Params(capture_model="bt94", capture_t_power=0.25)
    assert population._capture_factor(mol, 1, p, 300.0) == pytest.approx(1.0)
    assert population._capture_factor(mol, 1, p, 3000.0) == pytest.approx(10.0 ** -0.25)
    assert population._capture_factor(mol, 0, p, 3000.0) == 1.0
    w = Params(capture_model="wd01")
    assert population._capture_factor(mol, 0, w, 1000.0) == 1.0
    assert all(0.05 < population._capture_factor(m, 1, w, 1000.0) < 0.5 for m in (mol,))       # WD01's sticking ~0.1-0.3
    with pytest.raises(ValueError, match="capture_model"):
        population._capture_factor(mol, 1, Params(capture_model="hot"))


def test_attachment_fraction_models_meet_their_anchors_and_feed_attachment_and_detachment():
    """f per molecule: "constant" is Params.attachment_fraction; the EA and size logistics rise monotonically from attachment_f_low
    to 1, through ~0.003 at the small-PAH anchors and ~1 at C60's; the same f enters the neutral's sticking."""
    from dataclasses import replace

    from pycalima.pahchem.baseline import Params

    mol = next(m for m in population.library_m20(_ascii(), all_anions=True) if m.ea_ev is not None)
    assert population.attachment_fraction(mol, Params(attachment_model="constant", attachment_fraction=0.2)) == 0.2
    size = Params(attachment_model="size")
    f = [population.attachment_fraction(replace(mol, nc=n, _cache={}), size) for n in (14, 24, 60, 216)]
    assert f[0] < f[1] < 0.006 and f[2] > 0.6 and f[3] > 0.99
    ea = Params(attachment_model="ea", attachment_ea_half=2.5)
    g = [population.attachment_fraction(replace(mol, ionisation_ev={**mol.ionisation_ev, -1: e}, _cache={}), ea) for e in (0.5, 0.97, 2.68)]
    assert g[0] < g[1] < 0.006 and g[2] > 0.6
    assert population.attachment_sticking(mol, 300.0, size) == pytest.approx(
        population.attachment_fraction(mol, size) * population.neutral_sticking(mol, 300.0, population.attachment_fraction(mol, size)))
    with pytest.raises(ValueError, match="attachment_model"):
        population.attachment_fraction(mol, Params(attachment_model="x"))


def test_capture_cannot_exceed_the_classical_limit():
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in population.library_m20(_ascii()) if m.nc >= 130)
    assert population._capture_factor(mol, 1, Params(capture_model="lab_trend")) <= population.CAPTURE_MAX
    with pytest.raises(ValueError, match="classical limit"):
        population._capture_factor(mol, 1, Params(capture_scale=3.0))


def test_stretch_scale_moves_power_out_of_the_ch_stretch_only():
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in _library() if m.name == "C54H18 (37)")
    env = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    wins = {"3.3": (3.2, 3.36), "all": (1.0, 1e4)}
    base = population.Population([mol], wins)
    lab = population.Population([mol], wins, params=Params(stretch_scale=population.LAB_STRETCH_SCALE))
    for z in mol.charges:
        b0, b1 = base.state_band_fluxes(mol, z, env), lab.state_band_fluxes(mol, z, env)
        assert b1[1] == pytest.approx(b0[1], rel=0.01)                      # same total power
        assert 0.4 < b1[0] / b0[0] < 0.75                                    # the 3.3 um band roughly halves
    a = population._einstein_a(mol.species[0], Params(stretch_scale=0.5))
    f = mol.species[0].freqs_cm
    s = (f >= 2900) & (f <= 3200)
    assert np.allclose(a[s], 0.5 * mol.species[0].einstein_a[s]) and np.allclose(a[~s], mol.species[0].einstein_a[~s])


def test_chain_reproduces_the_baseline_network_given_its_rates():
    """The dehydrogenation chain solver, fed the baseline's own (Z, N_H) rates, gives the network's f(N_H0)."""
    from pycalima._paths import get_model_data_dir
    from pycalima.pahchem import chemistry
    from pycalima.pahchem.andrews16 import CORONENE_GEOMETRIC_A2, PARENTS
    from pycalima.pahchem.baseline import BaselineModel, Params
    from pycalima.pahchem.molecule import PAHSpecies

    d = get_model_data_dir() / "PAH_states"
    if not (d / "C54H18_0.dat").exists():
        pytest.skip("PAH_states not generated")
    sp = {z: PAHSpecies.from_states_file(d / f"C54H18_{z}.dat") for z in (-1, 0)}
    base = BaselineModel("C54", sp[0], Params(uv_cross_section="malloci_average"), anion=sp[-1])
    nc, nh0, n_solo = PARENTS["C54"]
    # recorded with the Malloci average (0.000 and 0.650 with Li & Draine; only the cross section changed), which this test pins: the
    # library default gives other values (the check is that the chain follows the network for whatever the rates are)
    for env, expected in ((Environment("Atomic most", 7.1e4, 5e4, 10.0, 722.0, 40000), 0.025),
                          (Environment("DF3 most", 5e3, 3.5e3, 0.49, 661.0, 40000), 0.880)):
        pr = base.photo_rates(env)
        net = base.network(pr, env)
        fss = net.steady_state()
        fz = net.charge_fractions(fss)
        lh = sum(fz[z] * np.array([pr[(z, nh0 - k)].get("H", 0.0) for k in range(4)]) for z in base.charges)
        lh2 = sum(fz[z] * np.array([pr[(z, nh0 - k)].get("H2", 0.0) for k in range(4)]) for z in base.charges)
        neutral = 0.07 * chemistry.cross_section_rate(env.temperature_k, CORONENE_GEOMETRIC_A2 * nc / 24.0)
        k_add = sum(fz[z] * (1.4e-10 if z > 0 else neutral) * env.n_h for z in base.charges)
        f_net = net.hydrogenation_fractions(fss)[nh0]
        assert f_net == pytest.approx(expected, abs=0.01)
        assert population._chain_intact(nh0, n_solo, lh, lh2, k_add) == pytest.approx(f_net, abs=0.02)


def _hydrogenation_pair():
    mols = [next(m for m in _library() if m.nc == n) for n in (24, 54)]
    return (population.Population(mols, WINDOWS), population.Population(mols, WINDOWS, dehydrogenated_emission=False))


def test_dehydrogenated_states_emit_what_they_absorb():
    """N_H0-1 ... N_H0-3 are heated by the same deposits as N_H0, so each emits the same total power."""
    pop, _ = _hydrogenation_pair()
    E = pop.state_powers(("kurucz", 40000))
    ok = E[:, :, :1] > 0
    assert np.allclose(np.where(ok, E[:, :, 1:], 1.0), np.where(ok, E[:, :, :1], 1.0), rtol=1e-6)


def test_hydrogenation_shares_and_the_unmodelled_rest_add_up():
    pop, _ = _hydrogenation_pair()
    for env in (Environment("dim", 1e2, 1e4, 10.0, 500.0, 40000), Environment("hot", 1e4, 1e3, 10.0, 500.0, 40000)):
        h, deep = pop.hydrogenation_fractions(env)
        assert np.all(h >= 0) and np.allclose(h.sum(axis=1) + deep, 1.0)
        assert np.allclose(h[:, 0], pop.intact_fractions(env))
        assert np.allclose(pop.emitting_fractions(env), h.sum(axis=1))


def test_dehydrogenated_emission_lowers_the_ch_bands_and_leaves_intact_molecules_alone():
    """C54H18 stays intact here: both treatments give its bands. C24H14 is 0.34 intact and 0.71 in the
    emitting states: its total power falls to that share (the rest is not modelled) and its C-H bands
    fall against the C-C 6.2 um band (3.3/6.2 and 11.2/6.2 by 5-8 %)."""
    pop, old = _hydrogenation_pair()
    env = Environment("dim", 1e2, 1e4, 10.0, 500.0, 40000)
    F, _ = pop.band_fluxes(env)
    F0, _ = old.band_fluxes(env)
    assert F[1] == pytest.approx(F0[1], rel=1e-6)
    assert F[0].sum() / F0[0].sum() == pytest.approx(pop.emitting_fractions(env)[0], rel=0.01)
    assert F[0, 0] / F[0, 1] < 0.97 * F0[0, 0] / F0[0, 1] and F[0, 2] / F[0, 1] < 0.95 * F0[0, 2] / F0[0, 1]
    nu = np.linspace(600.0, 3300.0, 2000)
    assert np.allclose(pop.spectra(env, nu)[1], old.spectra(env, nu)[1], rtol=1e-6, atol=0.0)


def test_chain_kernel_matches_the_numpy_solve():
    rng = np.random.default_rng(5)
    for nh0 in (2, 5, 12, 30):
        for _ in range(5):
            lh, lh2 = 10 ** rng.uniform(-8, -2, 4), 10 ** rng.uniform(-9, -3, 4)
            k_add = 10 ** rng.uniform(-7, -2)
            fast = population._chain_distribution(nh0, nh0 // 3, lh, lh2, k_add)
            slow = population._chain_distribution_numpy(nh0, nh0 // 3, lh, lh2, k_add)
            assert fast.sum() == pytest.approx(1.0) and np.all(fast >= 0)
            assert fast == pytest.approx(slow, abs=1e-12)


def _ladder_molecules():
    """A cation-only molecule and one with its anion (N_C >= 50): both take LADDER_STEP_FIT's +1 -> +2 step."""
    return [_library()[0], next(m for m in _library() if -1 in m.charges and m.nc >= 50)]


def test_extend_ladder_reaches_the_cold_limit_with_stand_ins_and_constant_steps():
    """The ladder runs to floor(0.2 N_C) (``impact.coulomb_limit``). The states above the highest computed charge take
    its modes, the computed thresholds stay, and the new ones rise by the last computed step (LADDER_STEP_FIT's for a
    molecule with no computed dication)."""
    from dataclasses import replace

    from pycalima.pahchem import impact

    a, b = population.LADDER_STEP_FIT
    for mol in _ladder_molecules():
        ext = population.extend_ladder(mol)
        z_top = impact.coulomb_limit(mol.nc)
        assert ext.charges == tuple(range(min(mol.charges), z_top + 1))
        assert ext.stand_ins == tuple(range(2, z_top + 1))
        assert all(ext.species[z] is mol.species[1] for z in ext.stand_ins)
        assert all(ext.ionisation_ev[z] == v for z, v in mol.ionisation_ev.items())
        steps = np.diff([ext.ionisation_ev[z] for z in range(0, z_top)])
        assert np.allclose(steps, a + b / np.sqrt(mol.nc))
        assert population.top_threshold(ext) == pytest.approx(ext.ionisation_ev[z_top - 1] + steps[-1])
        assert population.extend_ladder(ext) is ext
    mol = _library()[0]
    dication = replace(mol, species=mol.species | {2: mol.species[1]}, ionisation_ev={0: 6.5, 1: 9.3}, _cache={})
    ext = population.extend_ladder(dication)
    assert ext.stand_ins[0] == 3
    assert np.allclose(np.diff([ext.ionisation_ev[z] for z in range(0, max(ext.charges))]), 2.8)


def test_cold_limit_ladder_gives_no_basis_to_states_the_field_cannot_reach():
    """In a field cut at 13.6 eV a state above +1 is live only if every threshold below it is under 13.6 eV (WD01 eq. 22);
    the others hold f = 0 exactly and get no basis. Among the computed charges the balance is the library ladder's (the
    same ratio of neighbours); only the states above them are new. The population maps the caller's molecules to its
    copies."""
    from pycalima.pahchem import fields
    from pycalima.pahchem.baseline import Params

    mols = _ladder_molecules()
    new = population.Population(mols, WINDOWS)
    old = population.Population(mols, WINDOWS, params=Params(charge_ladder="library"))
    env = Environment("cloud", 10.0, 50.0, 10.0 * np.sqrt(100.0) / 1e3, 100.0, None)          # psi = 1e3
    shape = population.field_shape(env)
    f = new.charge_matrix(env)
    for s, m in enumerate(new.library):
        reach = tuple(z for z in m.charges if z < 2 or all(m.ionisation_ev[k] < fields.LYMAN_EV for k in range(1, z)))
        assert new._live_charges(m, shape) == reach and max(reach) < max(m.charges)
        dead = [z for z in m.charges if z not in reach]
        assert np.all(f[s, [z - new.zmin for z in dead]] == 0.0)
        assert all(new._ckey(shape, z, m) not in m._cache for z in dead)
    fo = old.charge_matrix(env)
    assert new.zmin == old.zmin
    for s, m in enumerate(mols):
        i = [z - new.zmin for z in m.charges]
        assert f[s, i[1:]] / f[s, i[:-1]] == pytest.approx(fo[s, i[1:]] / fo[s, i[:-1]], rel=1e-12)
    assert new.basis(mols[1], 1, shape) is new.basis(new.library[1], 1, shape)
    with pytest.raises(ValueError, match="charge_ladder"):
        population.Population(mols, WINDOWS, params=Params(charge_ladder="x"))


def test_compact_state_spectra_hold_every_emitting_state_and_no_unreachable_basis():
    """``spectral_fit.state_table`` and ``molecule_table`` skip the states with f = 0 (``Population._live``) as the
    population's own tables do (each PDRs4All worker otherwise built all of them, serially). The compact state spectra
    (one row per live state with modes) give the spectra ``molecule_spectra`` sums from the modes, and so do
    ``Population.spectra`` and the split by charge."""
    from pycalima.pahchem import spectral_fit

    pop = population.Population(_ladder_molecules(), WINDOWS)
    env = Environment("cloud", 10.0, 50.0, 10.0 * np.sqrt(100.0) / 1e3, 100.0, None)          # psi = 1e3
    shape = population.field_shape(env)
    nu = np.linspace(500.0, 3200.0, 50)
    S = spectral_fit.shaped_state_spectra(pop, shape, nu)
    P = spectral_fit.molecule_spectra(pop, env, nu)
    for m in pop.library:
        dead = [z for z in m.charges if z not in pop._live_charges(m, shape)]
        assert dead and all(pop._ckey(shape, z, m) not in m._cache for z in dead)
    assert S.rows.shape[0] < len(pop.library) * len(pop.zs) * population.N_EMITTING / 2
    W = pop.emission_weights(env)
    assert env.g0 * S.combine(W) == pytest.approx(P, rel=1e-12)
    w = np.arange(1.0, len(pop.library) + 1.0)
    assert S.by_slot(W, w, len(pop.zs)).sum(axis=0) == pytest.approx(w @ S.combine(W), rel=1e-12)
    # the population's own, with BandShape()'s fixed lines (15 cm^-1 Lorentzians shifted by -15 cm^-1)
    assert pop.spectra(env, nu, fwhm_cm=15.0, shift_cm=-15.0) == pytest.approx(P, rel=1e-12)


def test_highly_charged_states_ionise_only_above_their_threshold_and_conserve_energy():
    """A state whose threshold lies above Y_UNITY_EV ionises from it on, not below. In an EUV field its non-ionising
    deposits run above E_DEP on the photon nodes (``deposit_grid``), and its IR power is its non-ionising absorption
    plus the energy each recombination into it leaves."""
    from pycalima.pahchem import photo

    pop = population.Population(_ladder_molecules()[1:], WINDOWS)
    mol = pop.library[0]
    z = next(z for z in mol.charges[:-1] if mol.ionisation_ev[z] > 25.0)
    th = mol.ionisation_ev[z]
    y = population._yield(mol, z)
    assert np.all(y[population.E_PH < th] == 0.0) and np.all(y[population.E_PH >= th] == 1.0)
    shape = population.field_shape(Environment("hii", 1e5, 1.0, 1e2, 1e4, 40000, field_e_max_ev=100.0))
    assert z in pop._live_charges(mol, shape)
    sigma = photo.sigma_uv(mol.nc, z, population.E_PH, pop.p.uv_cross_section)
    grid = population.deposit_grid(population._unit_field(shape), sigma, y)
    assert grid[-1] >= th > population.E_DEP[-1]
    b = pop.basis(mol, z, shape)
    assert b.emitted.sum() == pytest.approx(b.absorbed + b.recombination, rel=1e-3)


def test_multiphoton_loss_ceiling_rises_until_converged():
    """For the largest PAHs, multiphoton heating at G0 = 1e6 runs past 45 eV (LOSS_E_MAX_EV): there the clamped jumps
    exceed LOSS_OVERFLOW_TOL of the loss rate, so the ceiling rises, and the converged k_H equals that at the cap."""
    from pycalima.pahchem import photo
    from pycalima.pahchem.baseline import Params

    p = Params()
    mol = max(_library(), key=lambda m: m.nc)
    shape = population.field_shape(Environment("pdr", 1e4, 1e4, 1.0, 1000.0, None))
    inp = population._basis_inputs(mol, 0, shape, 100.0, 21.5 * dos.CM_PER_EV, p)
    freqs, a_ein, nh, h2_ok = inp["ladder"][0]
    field1 = population._unit_field(shape)
    sigma = photo.sigma_uv(mol.nc, 0, population.E_PH, p.uv_cross_section)
    e_dep = population.deposit_grid(field1, sigma, inp["yield_"])
    h = population.deposit_spectrum(field1, sigma, inp["yield_"], inp["threshold_ev"], False, None, e_dep)
    cm, E, counts, beta = population._loss_grid(freqs, a_ein, population.LOSS_E_MAX_EV)
    k = population._h_loss_rate(freqs, E, counts, np.rint(cm.energy_cm / (E[1] - E[0])).astype(int), nh, p.h_even,
                                p.h_odd, beta)
    assert not population._loss_converged(cm.steady_state(e_dep, population.K_H_G0_GRID[-1] * h, {"H": k}))
    kh, _ = population._loss_tables(freqs, a_ein, nh, h2_ok, h, p.h_even, p.h_odd, p.h2, e_dep=e_dep)
    kh_cap, _ = population._loss_tables(freqs, a_ein, nh, h2_ok, h, p.h_even, p.h_odd, p.h2, e_dep=e_dep,
                                        e_max_ev=population.LOSS_E_CAP_EV)
    assert kh == pytest.approx(kh_cap, rel=0.02)


def test_anion_valence_threshold_is_one_coulomb_step_down_the_ladder():
    """2 IP(0) - IP(+1) with the molecule's own IP(+1); without one, LADDER_STEP_FIT's step (the one extend_ladder
    gives it), so the library and cold-limit ladders agree; never below the EA."""
    from dataclasses import replace

    mol = next(m for m in _library() if -1 in m.charges and m.nc >= 50 and 1 not in m.ionisation_ev)
    a, b = population.LADDER_STEP_FIT
    th = mol.ionisation_ev
    assert population.anion_valence_threshold(mol) == pytest.approx(max(th[0] - (a + b / np.sqrt(mol.nc)), th[-1]))
    assert population.anion_valence_threshold(population.extend_ladder(mol)) == pytest.approx(
        population.anion_valence_threshold(mol))
    computed = replace(mol, species=mol.species | {2: mol.species[1]}, ionisation_ev=th | {1: th[0] + 2.0}, _cache={})
    assert population.anion_valence_threshold(computed) == pytest.approx(max(th[0] - 2.0, th[-1]))
