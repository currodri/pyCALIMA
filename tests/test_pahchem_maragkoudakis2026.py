"""Maragkoudakis+26 (PDRs4All XVIII): pyPAHdb's decomposition and its matrix, from their stated inputs.

The matrix tests need the public pyPAHdb v4.00-alpha matrix (dataset
``pypahdb-matrix-v4-00-alpha``) and, for the column test, PAHdb v3.20; they
skip without them.
"""

import numpy as np
import pytest

from pycalima.pahchem import maragkoudakis2026 as M

_CACHE = {}


def _matrix():
    from pycalima._datasets import DatasetUnavailable

    if "pre" not in _CACHE:
        try:
            _CACHE["pre"] = M.load_precomputed()
        except DatasetUnavailable:
            pytest.skip("pyPAHdb v4.00-alpha matrix not installed")
    return _CACHE["pre"]


def test_table1_gamma_is_their_eq1_except_hii_if():
    t = M.table1().set_index("zone")
    g = 2.66 * t.f_cat / t.f_neut
    off = (g - t.gamma_1e4).abs()
    assert off.drop("HII-IF").max() < 0.1
    assert off["HII-IF"] == pytest.approx(0.38, abs=0.02)       # as printed: 2.7 against 3.08
    assert np.allclose(t[["f_neut", "f_cat", "f_an"]].sum(axis=1), 1.0, atol=0.011)


def test_decompose_recovers_synthetic_weights():
    rng = np.random.default_rng(0)
    x = np.linspace(600.0, 2100.0, 3001)
    centres = rng.uniform(650.0, 2050.0, size=(8, 3))
    cols = np.stack([sum(np.exp(-0.5 * ((x - c) / 6.0) ** 2) for c in cs) for cs in centres], axis=1)
    pre = {"abscissa": x, "matrix": cols, "charge": np.array([0, 1, 1, 2, -1, 0, 1, 0]),
           "size": np.array([30, 40, 60, 80, 50, 100, 70, 24])}
    w_true = np.array([0.0, 2.0, 0.5, 0.0, 1.0, 0.0, 0.0, 3.0])
    lam = np.sort(1e4 / x)
    y = np.interp(1e4 / lam, x, cols @ w_true)
    w, fit = M.decompose(lam, y, pre, fit_range_um=(lam.min(), lam.max()))
    assert w == pytest.approx(w_true, abs=1e-8)
    assert M.fit_error(lam, y, fit, fit_range_um=(lam.min(), lam.max())) < 1e-8


def test_emission_shares_equal_pypahdb_fractions_for_equal_energy_columns():
    rng = np.random.default_rng(1)
    pre = {"charge": rng.integers(-1, 3, 50), "size": rng.integers(21, 200, 50)}
    w = rng.uniform(0, 1, 50)
    a, b = M.fractions(w, pre), M.emission_shares(w, np.full(50, 7.0), pre["charge"], pre["size"])
    for k in a:
        assert a[k] == pytest.approx(b[k])


def test_every_matrix_column_holds_one_7_ev_photon():
    pre = _matrix()
    integ = np.trapezoid(pre["matrix"], pre["abscissa"], axis=0)
    assert integ == pytest.approx(np.full(integ.size, 7.0 * 1.602176634e-12), rel=1e-3)
    assert set(np.unique(pre["charge"])) == {-1, 0, 1, 2, 3}
    assert pre["size"].min() == 21 and pre["size"].max() == 384


def test_a_v320_column_is_rebuilt_by_pahchem():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_pahdb

    pre = _matrix()
    try:
        path = ames_pahdb.ascii_path()
    except DatasetUnavailable:
        pytest.skip("PAHdb v3.20 not installed")
    sp = ames_pahdb.read_ascii(path, [564])[564]                  # C96 neutral, B3LYP/4-31G
    ours = M.matrix_from_species([sp], pre["abscissa"])["matrix"][:, 0]
    col = pre["matrix"][:, 6603]
    assert np.abs(ours - col).max() < 2e-3 * col.max()
