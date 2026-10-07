"""The joint PAH + continuum spectral fit (spectral_fit): it must return what was put in.

The PAHdb-based tests need PAHdb v3.20 ($CALIMA_PAHDB_DIR) and skip without it.
"""

import numpy as np
import pytest

from pycalima.pahchem import population, spectral_fit
from pycalima.pahchem.baseline import Environment

WINDOWS = {"3.3": (3.2, 3.36), "6.2": (6.0, 6.6), "7.7": (7.05, 8.15), "8.6": (8.15, 9.1), "11.2": (11.1, 11.8),
           "12.7": (12.3, 13.1)}
_CACHE = {}


def _setup():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_pahdb

    if "pop" not in _CACHE:
        try:
            path = ames_pahdb.ascii_path()
        except DatasetUnavailable:
            pytest.skip("PAHdb v3.20 not installed")
        lib = [m for m in population.library_m20(path) if m.nc >= 40][:12]
        _CACHE["pop"] = population.Population(lib, WINDOWS)
    return _CACHE["pop"]


def test_grid_bins_and_masks():
    g = spectral_fit.Grid.build()
    assert np.all(np.diff(g.nu[g.segment == 1]) > 0)
    lam = g.lam_um
    assert not np.any((lam > 13.71) & (lam < 13.94))
    x = np.linspace(2.9, 16.0, 200000)
    y = 3.0 + 0.0 * x
    b = spectral_fit.bin_spectrum(x, y, g)
    assert np.allclose(b, 3.0)


def test_continuum_basis_is_a_partition_of_unity_per_segment():
    g = spectral_fit.Grid.build()
    C = spectral_fit.continuum_basis(g, knot_um=1.5)
    assert np.allclose(C.sum(axis=1), 1.0, atol=1e-9)
    assert C.min() >= 0.0


def test_fit_returns_the_input_population_and_continuum():
    pop = _setup()
    g = spectral_fit.Grid.build()
    env = Environment("t", 2e4, 5e4, 50.0, 700.0, 40000)
    P = pop.spectra(env, g.nu).T                                 # (n_grid, n_mol)
    w_true = np.zeros(P.shape[1])
    w_true[[1, 4, 9]] = [2.0, 1.0, 0.5]
    pah = P @ w_true
    cont = 0.3 * pah.max() * (g.lam_um / 10.0) ** 2 * (g.segment == 1) + 0.05 * pah.max() * (g.segment == 0)
    C = spectral_fit.continuum_basis(g, knot_um=1.5)
    fit = spectral_fit.fit_spectrum(pah + cont, P, C)
    F, _ = pop.band_fluxes(env)
    true_b, fit_b = w_true @ F, fit.weights @ F
    i = list(WINDOWS).index("11.2")
    assert fit_b / fit_b[i] == pytest.approx(true_b / true_b[i], rel=0.03)
    assert fit.cont == pytest.approx(cont, rel=0.05, abs=0.01 * pah.max())
    assert fit.rms < 1e-3
    # no continuum in: (almost) none out
    fit0 = spectral_fit.fit_spectrum(pah, P, C)
    assert fit0.cont.max() < 0.02 * pah.max()


def test_band_fluxes_of_the_fitted_spectrum_are_the_model_bands():
    """The derived band fluxes integrate the fitted PAH spectrum itself (no continuum operator)."""
    pop = _setup()
    env = Environment("t", 2e4, 5e4, 50.0, 700.0, 40000)
    nu = np.arange(500.0, 3500.0, 0.25)
    P = pop.spectra(env, nu)
    F, _ = pop.band_fluxes(env)
    for b, (lo, hi) in WINDOWS.items():
        m = (nu >= 1e4 / hi) & (nu <= 1e4 / lo)
        num = P[:, m].sum(axis=1) * 0.25
        assert num == pytest.approx(F[:, list(WINDOWS).index(b)], rel=1.5e-2)    # Riemann sum vs analytic window
