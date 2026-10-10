"""Collapse grain materials, Planck-mean optics, dust temperature and gas-grain cooling.

Limits and invariants: bulk densities from both key species agree, the Planck
mean of a constant Q is that constant, Rayleigh grains have <Q>/a independent of
size, the dust temperature tends to T_rad with no gas and to T with strong
coupling or no escape, the energy balance holds at the root, and Lambda_gd is
Chiaki et al. (2015) Eq. 5 and scales with the grain cross-section.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycalima.collapse import dust_temperature as dt
from pycalima.collapse import gas_grain as gg
from pycalima.collapse.materials import materials
from pycalima.collapse.optics import LAMBDA_UM, PlanckTable, planck_mean, refractive_index, load_nk

BUNDLED = [n for n, m in materials().items() if m.optics.startswith("bundled:")]


def test_bulk_density_is_the_same_from_every_key_species():
    for m in materials().values():
        rhos = [k.mu * 1.66053906660e-24 / (4 / 3 * math.pi * k.a0 ** 3) for k in m.keys]
        assert max(rhos) == pytest.approx(min(rhos), rel=0.01), m.name
        assert 2.0 < m.rho_s < 8.0


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


@pytest.fixture(scope="module")
def silicate():
    return PlanckTable.compute("Mg2SiO4", radii_cm=np.geomspace(1e-7, 1e-3, 21), T=np.geomspace(2.0, 3000.0, 61))


def test_dust_temperature_limits(silicate):
    a = np.geomspace(1e-7, 1e-4, 7)
    T, Tr = 500.0, 40.0
    np.testing.assert_allclose(dt.dust_temperature(silicate, a, T, 1e-30, Tr), Tr, rtol=1e-6)    # no gas
    np.testing.assert_allclose(dt.dust_temperature(silicate, a, T, 1e30, Tr), T, rtol=1e-6)      # strong coupling
    np.testing.assert_allclose(dt.dust_temperature(silicate, a, T, 1e15, Tr, beta=1e-12), T, rtol=1e-4)
    np.testing.assert_allclose(dt.dust_temperature(silicate, a, T, 1e15, Tr, beta=0.0), T)


def test_dust_temperature_balances_heating_and_emission(silicate):
    a = np.geomspace(1e-7, 1e-4, 7)
    T, Tr, nH2 = 500.0, 40.0, 1e10
    F = gg.collision_flux_factor(T, 0.0, nH2, 0.08 * 2 * nH2)
    Td = dt.dust_temperature(silicate, a, T, F, Tr, beta=0.5)
    heat = gg.heating_per_grain(a, T, Td, F)
    emit = 4 * math.pi * a ** 2 * dt.SIGMA_SB * 0.5 * (silicate(a, Td) * Td ** 4 - silicate(a, Tr) * Tr ** 4)
    np.testing.assert_allclose(heat, emit, rtol=1e-6)
    assert np.all((Td > Tr) & (Td < T))
    assert np.all(np.diff(Td) < 0)       # small (Rayleigh) grains radiate worse and run hotter


def test_cooling_rate_is_chiaki15_eq5_and_scales_with_cross_section():
    T, nH, nH2, nHe = 300.0, 1e9, 3e8, 1.2e8
    F = gg.collision_flux_factor(T, nH, nH2, nHe)
    v = math.sqrt(8 * 1.380649e-16 * T / (math.pi * 1.6735575e-24))
    assert F == pytest.approx((nH + nH2 / math.sqrt(2) + nHe / 2) * v, rel=1e-12)
    a, n, Td = np.array([1e-6, 1e-5]), np.array([1e-3, 1e-6]), np.array([80.0, 60.0])
    L = gg.cooling_rate(n, a, T, Td, F)
    assert L == pytest.approx(np.sum(n * math.pi * a ** 2 * F * 2 * 1.380649e-16 * (T - Td)), rel=1e-12)
    assert gg.cooling_rate(2 * n, a, T, Td, F) == pytest.approx(2 * L, rel=1e-12)


@pytest.mark.parametrize("name,n_points", [("C", 1245), ("Si", 4001), ("Fe3O4", 401)])
def test_fetched_optical_constants_parse_completely(name, n_points):
    """The three materials whose optical constants are registered downloads
    (data/registry.toml); skipped until fetched with calima-fetch-data."""
    from pycalima._datasets import DatasetUnavailable
    try:
        lam, n, k = load_nk(name)
    except DatasetUnavailable:
        pytest.skip(f"optical constants of {name} not fetched")
    assert lam.size == n_points
    assert np.all(np.diff(lam) > 0) and np.all(n > 0) and np.all(k >= 0)


@pytest.mark.parametrize("name", ["Fe", "Mg2SiO4", "C", "Si"])
def test_absorption_is_continuous_across_the_mie_switch(name):
    """Q_abs from the small-particle expansion and from Mie agree where both
    are accurate, so switching at |m| x = 0.1 leaves no step; Fe checks the
    magnetic-dipole term (Draine & Hensley 2013), which Mie and the expansion
    share and the electric dipole alone misses by orders of magnitude."""
    import miepython
    from pycalima.collapse.optics import qabs_small
    lam = np.geomspace(1.0, 1e4, 200)
    m = refractive_index(lam, *load_nk(name))
    a_um = 0.01
    x = 2 * np.pi * a_um / lam
    qe, qs, _, _ = miepython.efficiencies(m, 2 * a_um, lam)
    near = (np.abs(m) * x > 0.03) & (np.abs(m) * x < 0.3) & ((qe - qs) > 1e-4 * qs)   # Mie free of cancellation
    if near.any():
        np.testing.assert_allclose(qabs_small(m[near], x[near]), (qe - qs)[near], rtol=6e-3)


def test_weak_absorption_has_no_cancellation_floor():
    """Crystalline Si at 1-5 um (k ~ 1e-10): Q_abs follows k, not Mie's round-off."""
    from pycalima.collapse.optics import qabs_grid
    lam = np.geomspace(1.5, 4.5, 40)
    Q = qabs_grid("Si", radii_cm=np.array([1e-6]), lam_um=lam)[0]
    k = -refractive_index(lam, *load_nk("Si")).imag
    ratio = Q / (k / lam)
    assert ratio.max() / ratio.min() < 1.5      # Rayleigh: Q ∝ k / lambda at fixed n
