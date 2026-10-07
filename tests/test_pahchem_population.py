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
    from pycalima.pahchem import ames_pahdb

    try:
        return ames_pahdb.ascii_path()
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
    from pycalima.pahchem import ames_pahdb, maragkoudakis2020

    s = maragkoudakis2020.sample()
    meta = ames_pahdb.read_metadata(_ascii(), s.uid)
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
    assert sum(m.ea_ev is not None for m in ext) == 81
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
        absorbed = env.g0 * pop.basis(mol, z, population.field_shape(env)).absorbed
        assert pop.state_band_fluxes(mol, z, env)[0] == pytest.approx(absorbed, rel=0.01)


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
    sigma = photo.sigma_li_draine(mol.nc, 1, E_PH)
    h = photo.heating_spectrum(population._unit_field(shape), sigma, population._yield(mol, 1), E_PH)
    cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, pop.e_max_cm, pop.de_cm)
    E, counts = dos.beyer_swinehart(sp.freqs_cm, cm.energy_cm[-1] + 200.0)
    ts = rates.transition_state_for_entropy(sp.freqs_cm, 3050.0, pop.p.h_even[1], n_scaled=5)
    k = rates.rrkm(E, counts, ts, pop.p.h_even[0] * dos.CM_PER_EV, degeneracy=mol.nh)
    k = k[np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)]
    direct = cm.steady_state(E_PH, g0 * h, {"H": k}).rates["H"]
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


def test_large_compact_pahs_stay_intact_in_the_orion_bar():
    from pycalima.pahchem import conditions

    mol = next(m for m in _library() if m.name == "C54H18 (37)")
    env = conditions.bracketing_environments("orion_bar", "Atomic")["most ionising"]
    assert population.Population([mol], WINDOWS).intact_fraction(mol, env) > 0.9


# -- PAHdb XML and the v4.00 library ----------------------------------------------------------------
def _benzene(scale=1.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.arange(6) * np.pi / 3
    c = np.c_[1.39 * np.cos(a), 1.39 * np.sin(a), np.zeros(6)] * scale
    h = np.c_[2.47 * np.cos(a), 2.47 * np.sin(a), np.zeros(6)] * scale
    xyz = np.vstack([c, h]) + jitter * rng.standard_normal((12, 3))
    return [6] * 6 + [1] * 6, xyz


def test_bond_graph_hash_ignores_geometry_but_not_connectivity():
    from pycalima.pahchem import ames_pahdb

    t, x = _benzene()
    h0 = ames_pahdb.bond_graph_hash(t, x)
    assert ames_pahdb.bond_graph_hash(*_benzene(scale=1.02, jitter=0.02, seed=3)) == h0   # a charge state's geometry
    t2, x2 = list(t), x.copy()
    x2[6] = x2[0] + (x2[1] - x2[0]) * 0.5 + np.array([0.0, 0.0, 1.09])  # move one H onto a C-C bond: other graph
    assert ames_pahdb.bond_graph_hash(t2, x2) != h0


def _v400():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_pahdb

    if "v400" not in _CACHE:
        try:
            path = ames_pahdb.xml_path("pahdb-theoretical-v4-00")
        except DatasetUnavailable:
            pytest.skip("PAHdb v4.00 not installed")
        recs = ames_pahdb.xml_index(path)                       # no cache_dir: tests do not write to model data
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


# -- charge-input options -------------------------------------------------------------------------
def test_charge_inputs_move_the_charge_balance_the_right_way():
    from pycalima.pahchem.baseline import Params

    lib = population.library_m20(_ascii(), all_anions=True)
    mol = [m for m in lib if m.ea_ev is not None and m.nc >= 40][:3]
    env = Environment("t", 2e4, 5e4, 30.0, 700.0, 40000)
    base = population.Population(mol, WINDOWS).charge_matrix(env)
    hi_ip = population.Population(population.shift_thresholds(mol, {0: 0.6}), WINDOWS).charge_matrix(env)
    fast = population.Population(mol, WINDOWS, params=Params(capture_scale=3.0)).charge_matrix(env)
    i0, i1 = 1, 2                                     # slots of Z = 0, +1 (zs = -1, 0, 1)
    assert (hi_ip[:, i1] < base[:, i1]).all() and (fast[:, i1] < base[:, i1]).all()
    assert (hi_ip[:, i0] > base[:, i0]).all()
    shifted = population.shift_thresholds(mol, {0: 0.6, -1: 0.3})
    for a, b in zip(mol, shifted):
        assert b.ip_ev == pytest.approx(a.ip_ev + 0.6) and b.ea_ev == pytest.approx(a.ea_ev + 0.3)
        assert b._cache == {} and a.ionisation_ev[0] == a.ip_ev       # the input is not modified


def test_unit_detachment_leaves_an_anion_nothing_to_emit():
    from pycalima.pahchem.baseline import Params

    mol = next(m for m in population.library_m20(_ascii(), all_anions=True) if m.ea_ev is not None)
    pop = population.Population([mol], WINDOWS, params=Params(detachment="unity"))
    b = pop.basis(mol, -1, ("kurucz", 40000))
    assert b.absorbed == 0.0 and not b.emitted.any() and b.k_up > 0.0
