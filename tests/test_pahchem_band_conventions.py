"""Band conventions (``pahchem.band_conventions``): profiles, window fractions, emission-temperature shares.

The library tests need PAHdb v3.20 ($CALIMA_PAHDB_DIR) and skip without it; the others run anywhere.
"""

import numpy as np
import pytest

from pycalima.pahchem import band_conventions as bc
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Environment

WINDOWS = {"3.3": (3.2, 3.36), "6.2": (6.0, 6.6), "11.0": (10.9, 11.1), "11.2": (11.1, 11.8), "12.7": (12.3, 13.1)}
FREQS = np.array([3050.0, 1610.0, 1300.0, 1150.0, 908.0, 890.0, 845.0, 790.0, 500.0])


def _numeric_window(freqs, strength, shift, fwhm, eta, lo_um, hi_um, local):
    grid = np.arange(100.0, 6000.0, 0.02)
    s = bc.line_spectrum(freqs, strength, grid, shift, fwhm, eta)
    m = (grid >= 1e4 / hi_um) & (grid <= 1e4 / lo_um)
    flux = s[m].sum() * 0.02
    if local:
        lam = 1e4 / grid[m]
        f1, f2 = np.interp([1e4 / lo_um, 1e4 / hi_um], grid, s)
        flux -= ((f1 + (f2 - f1) * (lam - lo_um) / (hi_um - lo_um)).sum() * 0.02)
    return flux


@pytest.mark.parametrize("local", [False, True])
def test_lorentzian_window_matrix_equals_the_legacy_band_matrix(local):
    loc = tuple(WINDOWS) if local else ()
    M = bc.window_matrix(FREQS, WINDOWS, loc, np.full(FREQS.size, -15.0), 15.0, 1.0)
    assert np.allclose(M, population._band_matrix(FREQS, WINDOWS, loc), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("eta", [1.0, 0.562, 0.0])
def test_window_matrix_equals_the_numeric_integral_of_the_lines(local, eta):
    rng = np.random.default_rng(1)
    strength = rng.uniform(0.2, 2.0, FREQS.size)
    shift = rng.uniform(-20.0, 5.0, FREQS.size)
    fwhm = rng.uniform(3.0, 30.0, FREQS.size)
    loc = tuple(WINDOWS) if local else ()
    M = bc.window_matrix(FREQS, WINDOWS, loc, shift, fwhm, eta)
    for b, (lo, hi) in WINDOWS.items():
        got = strength @ M[:, list(WINDOWS).index(b)]
        assert got == pytest.approx(_numeric_window(FREQS, strength, shift, fwhm, eta, lo, hi, local), rel=3e-3, abs=1e-5)


@pytest.mark.parametrize("eta", [1.0, 0.5, 0.0])
def test_lines_integrate_to_their_strength(eta):
    grid = np.arange(-2e5, 2e5, 0.05)
    s = bc.line_spectrum([0.0], [2.5], grid, 0.0, 8.0, eta)
    assert s.sum() * 0.05 == pytest.approx(2.5, rel=2e-4)


def test_named_conventions_and_validation():
    assert bc.get("legacy") is bc.LEGACY and bc.get(bc.SOURCED) is bc.SOURCED
    with pytest.raises(ValueError):
        bc.get("nope")
    with pytest.raises(ValueError):
        bc.BandConvention("x", kind="other")
    with pytest.raises(ValueError):
        bc.BandConvention("x", t_map="other")
    assert bc.BandConvention("q", t_map=("factor", 0.5)).t_classical(FREQS, 800.0) == pytest.approx(400.0)


def test_energy_matching_has_the_low_temperature_plateau_and_the_classical_limit():
    T = np.array([25.0, 100.0, 300.0, 800.0, 3000.0, 3e4])
    tc = bc.classical_temperature(FREQS, T)
    assert np.all(np.diff(tc) > 0.0) and np.all(tc < T)
    assert tc[0] < 1e-3 * T[0]                       # the modes freeze out: no shift at low T
    zero_point = FREQS.sum() / 2.0 / (FREQS.size * 0.6950348)
    assert tc[-1] == pytest.approx(T[-1] - zero_point, rel=1e-3)   # E_vib -> s k T - sum(h nu)/2


def test_sourced_lines_follow_the_class_laws():
    """Cold: the v4.00 positions (solos +3 cm^-1); hot: red shifts and broader lines; CH a pseudo-Voigt."""
    cls = bc.mode_classes(FREQS)
    sh0, fw0, eta = bc.SOURCED_TDEP.lines_at(FREQS, 96, 0, 0.0, 0.0)
    assert np.all(sh0[cls == bc.SOLO] == 3.0) and np.all(sh0[cls != bc.SOLO] == 0.0)
    t = 800.0
    sh, fw, _ = bc.SOURCED_TDEP.lines_at(FREQS, 96, 0, t, bc.classical_temperature(FREQS, t))
    laws = cls != bc.OTHER
    assert np.all(sh[laws] - sh0[laws] <= 0.0) and np.all(fw[laws] > fw0[laws])
    moving = np.isin(cls, (0, 1, 2, 3, bc.SOLO, 5))     # Mai's trio/quartet slope is 0 at N_C 80-130
    assert np.all(sh[moving] - sh0[moving] < 0.0)
    assert eta[cls == 0][0] == pytest.approx(0.562) and np.all(eta[cls != 0] == 1.0)
    cation = bc.BandConvention("c", cation_solo_offset_cm=7.0).lines_at(FREQS, 96, 1, 0.0, 0.0)[0]
    assert np.all(cation[cls == bc.SOLO] == 10.0) and np.all(cation[cls != bc.SOLO] == 0.0)
    # the constant convention is the T-dependent one at the class's T_REP
    shc, fwc, _ = bc.SOURCED.constant_lines(FREQS, 96, 0)
    T = np.array([bc.SOURCED.t_rep(c, 96) for c in cls])
    shr, fwr, _ = bc.SOURCED_TDEP.lines_at(FREQS, 96, 0, T, bc.classical_temperature(FREQS, T))
    assert np.allclose(shc, shr) and np.allclose(fwc, fwr)


def test_class_temperature_shares_are_normalised_per_class():
    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, (FREQS.size, 40))
    T = np.linspace(50.0, 1500.0, 40)
    W = bc.class_temperature_shares(FREQS, X, T)
    present = np.unique(bc.mode_classes(FREQS))
    assert np.allclose(W[present].sum(axis=1), 1.0)
    assert np.all(W[[c for c in range(bc.N_CLASSES) if c not in present]] == 0.0)


def test_tdep_window_matrix_at_one_temperature_is_the_window_matrix_there():
    W = np.zeros((bc.N_CLASSES, bc.T_MID.size))
    t = 20
    W[:, t] = 1.0
    M = bc.tdep_window_matrix(bc.SOURCED_TDEP, FREQS, 54, 0, WINDOWS, (), W)
    sh, fw, et = bc.SOURCED_TDEP.lines_at(FREQS, 54, 0, bc.T_MID[t], bc.classical_temperature(FREQS, bc.T_MID[t]))
    assert np.allclose(M, bc.window_matrix(FREQS, WINDOWS, (), sh, fw, et))


# -- with the library -------------------------------------------------------------------------------------
def _molecule():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_suite

    try:
        path = ames_suite.ascii_path()
    except DatasetUnavailable:
        pytest.skip("PAHdb v3.20 not installed (calima-fetch-data import pahdb-theoretical-v3-20 ...)")
    return next(m for m in population.library_m20(path) if m.name == "C54H18 (37)")


def test_default_population_is_the_legacy_convention():
    mol = _molecule()
    env = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    shape = population.field_shape(env)
    a = population.Population([mol], WINDOWS)
    b = population.Population([mol], WINDOWS, band_convention="legacy")
    assert np.array_equal(a._hydrogenation_bands(shape), b._hydrogenation_bands(shape))
    nu = np.arange(600.0, 3300.0, 1.0)
    bas = a.basis(mol, 0, shape)
    assert np.array_equal(a.state_spectra(shape, nu)[0, -a.zmin],
                          population.emission.spectrum(bas.freqs_cm, bas.emitted, nu, 15.0, -15.0))


@pytest.mark.parametrize("name", ["sourced", "sourced_tdep"])
def test_band_conventions_conserve_the_emitted_energy(name):
    """A window over the whole spectrum holds every state's IR power, as for the legacy lines."""
    mol = _molecule()
    env = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    pop = population.Population([mol], {"all": (1.0, 1e4)}, band_convention=name)
    for z in mol.charges:
        b = pop.basis(mol, z, population.field_shape(env))
        absorbed = env.g0 * (b.absorbed + b.recombination)
        assert pop.state_band_fluxes(mol, z, env)[0] == pytest.approx(absorbed, rel=0.01)


def test_emission_temperature_shares_are_normalised_and_order_the_classes():
    mol = _molecule()
    env = Environment("test", 1e4, 5e4, 10.0, 500.0, 40000)
    pop = population.Population([mol], WINDOWS, band_convention="sourced_tdep")
    shape = population.field_shape(env)
    for z in mol.charges:
        W = pop.temperature_shares(mol, z, shape)
        rows = W.sum(axis=1)
        assert np.all(np.isclose(rows, 1.0) | (rows == 0.0))
        t_mean = W[rows > 0] @ bc.T_MID
        assert np.all((t_mean > 100.0) & (t_mean < 3000.0))
        # the CH stretches leave the hottest bins (they emit early in the cascade), the oop modes the coolest
        assert (W[0] @ bc.T_MID) > (W[4:7].sum(axis=0) / max(W[4:7].sum(), 1e-300) @ bc.T_MID)
