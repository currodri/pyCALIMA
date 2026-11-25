"""Planck-mean optics of grains from tabulated optical constants (``material_optics``).

Limits and invariants: the Planck mean of a constant Q is that constant, Rayleigh
grains have <Q>/a independent of size, the far-infrared extrapolation keeps
absorption physical, the fetched constants parse, and Q_abs is continuous across
the switch from Mie to the small-particle expansion.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycalima.models.dust_radiation.material_optics import (LAMBDA_UM, PlanckTable, load_nk, materials,
                                                            optics_source, planck_mean, qabs_grid, qabs_small,
                                                            refractive_index)

BUNDLED = [n for n in materials() if optics_source(n).startswith("bundled:")]


def test_planck_mean_of_a_constant_is_that_constant():
    Q = np.full((2, LAMBDA_UM.size), 0.37)
    np.testing.assert_allclose(planck_mean(Q, LAMBDA_UM, np.array([5.0, 300.0, 2500.0])), 0.37, rtol=1e-3)


@pytest.mark.parametrize("name", ["Mg2SiO4", "SiO2", "MgO"])
def test_small_dielectric_grains_are_in_the_rayleigh_regime(name):
    """<Q> ∝ a for a << lambda_peak / 2 pi: here 1-10 nm at 100 K."""
    tab = PlanckTable.compute(name, radii_cm=np.geomspace(1e-7, 1e-6, 5), T=np.array([50.0, 100.0, 200.0]))
    q = tab(np.array([1e-7, 1e-6]), 100.0) / np.array([1e-7, 1e-6])
    assert q[1] == pytest.approx(q[0], rel=0.02)


@pytest.mark.parametrize("name", BUNDLED)
def test_far_infrared_extrapolation_keeps_absorption_physical(name):
    lam, n, k = load_nk(name)
    m = refractive_index(np.geomspace(lam[-1], 2e4, 20), lam, n, k)
    assert np.all(m.real > 0.0) and np.all(-m.imag > 0.0)
    if k[-1] < n[-1]:      # dielectric: k does not grow beyond the data
        assert np.all(np.diff(-m.imag) <= 1e-15)


def _load_nk_or_skip(name):
    """``load_nk``, skipping the test for a registered download (data/registry.toml: C, Si, Fe3O4) that has
    not been fetched with calima-fetch-data."""
    from pycalima._datasets import DatasetUnavailable
    try:
        return load_nk(name)
    except DatasetUnavailable:
        pytest.skip(f"optical constants of {name} not fetched")


@pytest.mark.parametrize("name,n_points", [("C", 1245), ("Si", 4001), ("Fe3O4", 401)])
def test_fetched_optical_constants_parse_completely(name, n_points):
    """The three materials whose optical constants are registered downloads
    (data/registry.toml); skipped until fetched with calima-fetch-data."""
    lam, n, k = _load_nk_or_skip(name)
    assert lam.size == n_points
    assert np.all(np.diff(lam) > 0) and np.all(n > 0) and np.all(k >= 0)


@pytest.mark.parametrize("name", ["Fe", "Mg2SiO4", "C", "Si"])
def test_absorption_is_continuous_across_the_mie_switch(name):
    """Q_abs from the small-particle expansion and from Mie agree where both
    are accurate, so switching at |m| x = 0.1 leaves no step; Fe checks the
    magnetic-dipole term (Draine & Hensley 2013), which Mie and the expansion
    share and the electric dipole alone misses by orders of magnitude."""
    import miepython
    lam = np.geomspace(1.0, 1e4, 200)
    m = refractive_index(lam, *_load_nk_or_skip(name))
    a_um = 0.01
    x = 2 * np.pi * a_um / lam
    qe, qs, _, _ = miepython.efficiencies(m, 2 * a_um, lam)
    near = (np.abs(m) * x > 0.03) & (np.abs(m) * x < 0.3) & ((qe - qs) > 1e-4 * qs)   # Mie free of cancellation
    if near.any():
        np.testing.assert_allclose(qabs_small(m[near], x[near]), (qe - qs)[near], rtol=6e-3)


def test_weak_absorption_has_no_cancellation_floor():
    """Crystalline Si at 1-5 um (k ~ 1e-10): Q_abs follows k, not Mie's round-off."""
    _load_nk_or_skip("Si")
    lam = np.geomspace(1.5, 4.5, 40)
    Q = qabs_grid("Si", radii_cm=np.array([1e-6]), lam_um=lam)[0]
    k = -refractive_index(lam, *load_nk("Si")).imag
    ratio = Q / (k / lam)
    assert ratio.max() / ratio.min() < 1.5      # Rayleigh: Q ∝ k / lambda at fixed n


def test_unknown_material_raises_with_the_known_ones():
    with pytest.raises(KeyError, match="Mg2SiO4"):
        optics_source("unobtainium")
