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
    assert b1.recombination == 0.0                                          # the top of the chain
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
    for z in mol.charges[:-1]:
        p = 1.0 if z < 0 else photo.PHOTOELECTRON_PARTITION
        mean_excess = G[0, z - pop.zmin] / (p * pop.basis(mol, z, shape).k_up)
        assert 0.0 < mean_excess < 13.6 - mol.ionisation_ev[z]
    assert G[0, max(mol.charges) - pop.zmin] == 0.0                       # the top of the chain does not ionise
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
    y, ek = y[: E_PH.size], ek[: E_PH.size]                                 # the 5-13.6 eV part (population.E_PH starts with it)
    assert np.array_equal(population.E_PH[: E_PH.size], E_PH)
    ea = mol.ionisation_ev[-1]
    prompt = photodetachment_cross_section_vec(E_PH, ea, -1) / photo.sigma_li_draine(mol.nc, -1, E_PH)
    assert np.all((y >= np.clip(prompt, 0, 1) - 1e-12) & (y <= 1 + 1e-12))
    assert np.all(ek[E_PH > ea] < E_PH[E_PH > ea] - ea) and np.all(ek[E_PH > 7.0] < 0.5)
    env = Environment("cnm", 1.7, 0.0, 0.04, 100.0, None)
    stat = population.Population([mol], WINDOWS, params=Params(detachment="statistical"))
    assert stat.charge_matrix(env)[0, -1 - stat.zmin] > 0.1                 # anions matter here
    assert 0.0 < stat.photoelectric_heating(env)[2][0] < 0.08
    assert population.Population([mol], WINDOWS).photoelectric_heating(env)[2][0] > 0.15   # full-excess electrons


def test_photoelectron_energy_conserves_the_excess_and_follows_the_measured_share():
    """<E_k> + E_int = E - IP; share 0.46 up to 13.6 eV, the Verstraete+25 means to 21.9 eV, then a constant E_int."""
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
    f1 = population._unit_field(("kurucz", 40000))
    for z in mol.charges[:-1]:
        sig = photo.sigma_li_draine(mol.nc, z, population.E_PH)
        y = population._yield(mol, z, pop.p)
        assert pop.basis(mol, z, ("kurucz", 40000)).k_up == pytest.approx(
            photo.ionisation_rate(f1, sig[fuv], y[fuv], population.E_PH[fuv]), rel=1e-6)
    heat = population.Population([mol], WINDOWS, params=replace(Params(), ionising_photons_heat=True))
    shape = ("kurucz", 40000, 99.0)
    fe = population._unit_field(shape)
    for z in mol.charges:
        b = heat.basis(mol, z, shape)
        sig = photo.sigma_li_draine(mol.nc, z, population.E_PH)
        thr = population.state_threshold(mol, z)
        y = population._yield(mol, z, heat.p)
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


def test_vectorised_charges_equal_the_scalar_chain():
    lib = population.library_m20(_ascii(), all_anions=True)
    pop = population.Population(lib, WINDOWS)
    for env in (Environment("a", 2e4, 1e5, 100.0, 700.0, 40000), Environment("b", 7e4, 5e4, 10.0, 700.0, 40000),
                Environment("c", 1e3, 3e4, 5.0, 700.0, None)):
        _, fz = pop.band_fluxes(env)
        for m in lib[::7]:
            ref = population.charge_fractions(m, env)
            for z in m.charges:
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
    sigma = photo.sigma_li_draine(mol.nc, 1, population.E_PH)
    h = population.deposit_spectrum(population._unit_field(shape), sigma, population._yield(mol, 1),
                                    population.state_threshold(mol, 1), False)
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
    assert len(list(tmp_path.glob("*.npz"))) == len(mol.charges)
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
    for z in mol.charges:                       # move each file to the key the previous version gave it
        inp = population._basis_inputs(mol, z, population.field_shape(env), first.de_cm, first.e_max_cm, first.p)
        (tmp_path / f"{population._basis_key(inp)}.npz").rename(tmp_path / f"{population._basis_key(inp, legacy=True)}.npz")
    monkeypatch.setattr(population, "_build", lambda inp: pytest.fail("rebuilt a stored basis"))
    mol2 = next(m for m in population.library_m20(_ascii()) if m.name == "C54H18 (37)")
    F2, _ = population.Population([mol2], WINDOWS, cache_dir=tmp_path).band_fluxes(env)
    assert np.array_equal(F1, F2)
    assert len(list(tmp_path.glob("*.npz"))) == 2 * len(mol.charges)    # new keys linked to the old files


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


def test_unit_detachment_leaves_an_anion_nothing_to_emit():
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in population.library_m20(_ascii(), all_anions=True) if m.ea_ev is not None)
    pop = population.Population([mol], WINDOWS, params=Params(detachment="unity", recombination_heating=False))
    b = pop.basis(mol, -1, ("kurucz", 40000))
    assert b.absorbed == 0.0 and not b.emitted.any() and b.k_up > 0.0
    # with recombination heating the anion still emits what electron attachment deposits (EA per event)
    hot = population.Population([mol], WINDOWS, params=Params(detachment="unity"))
    bh = hot.basis(mol, -1, ("kurucz", 40000))
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
    base = BaselineModel("C54", sp[0], Params(), anion=sp[-1])
    nc, nh0, n_solo = PARENTS["C54"]
    for env, expected in ((Environment("Atomic most", 7.1e4, 5e4, 10.0, 722.0, 40000), 0.000),
                          (Environment("DF3 most", 5e3, 3.5e3, 0.49, 661.0, 40000), 0.650)):
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


def test_chain_kernel_matches_the_numpy_solve():
    rng = np.random.default_rng(5)
    for nh0 in (2, 5, 12, 30):
        for _ in range(5):
            lh, lh2 = 10 ** rng.uniform(-8, -2, 4), 10 ** rng.uniform(-9, -3, 4)
            k_add = 10 ** rng.uniform(-7, -2)
            fast = population._chain_intact(nh0, nh0 // 3, lh, lh2, k_add)
            assert fast == pytest.approx(population._chain_intact_numpy(nh0, nh0 // 3, lh, lh2, k_add), abs=1e-12)
