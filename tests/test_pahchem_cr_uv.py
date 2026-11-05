"""The cosmic-ray-induced UV field (pycalima.pahchem.cr_uv) and its "cr_uv" field shape in the population."""

import numpy as np
import pytest

from pycalima.pahchem import cr_uv, population


def test_heays17_spectrum_is_a_cumulative_with_its_lyman_alpha_step():
    """The digitised Heays+17 Fig. 32: non-decreasing, 0.36 photons per ionisation in all (their text), and Lyman alpha
    a step of about 15 % of it at 121.567 nm (their text; 13.5 % as digitised)."""
    lam, cum = cr_uv.heays17()
    assert np.all(np.diff(lam) > 0) and np.all(np.diff(cum) >= 0)
    assert cum[-1] == pytest.approx(0.36, abs=0.01)
    step = np.interp(121.568, lam, cum) - np.interp(121.567, lam, cum)
    assert 0.12 * cum[-1] < step < 0.16 * cum[-1]


def test_field_counts_every_photon_once_on_the_population_grid():
    """``cr_uv.field`` on E_PH: int F sigma_abs dE = zeta(H2) x(H2) times the photons generated at 1-13.6 eV, exactly;
    linear in zeta(H2) and x(H2); nothing above 13.6 eV; Lyman alpha in one node."""
    E = population.E_PH
    lam, cum = cr_uv.heays17()
    for zeta, x in ((1e-16, 0.5), (3e-17, 0.2)):
        f = cr_uv.field(zeta, x, energy_ev=E)
        generated = zeta * x * (np.interp(cr_uv.HC_EV_NM / E[0], lam, cum) - np.interp(cr_uv.HC_EV_NM / 13.6, lam, cum))
        assert np.trapezoid(f.photon_flux * cr_uv.dust_absorption_per_h(E), E) == pytest.approx(generated, rel=1e-10)
        assert not f.photon_flux[E > 13.6].any()
    a, b = cr_uv.field(1e-16, 0.5, energy_ev=E), cr_uv.field(2e-16, 0.25, energy_ev=E)
    np.testing.assert_allclose(a.photon_flux, b.photon_flux, rtol=1e-12)
    i = int(np.argmin(np.abs(E - cr_uv.HC_EV_NM / 121.567)))
    assert a.photon_flux[i] > 5.0 * max(a.photon_flux[i - 2], a.photon_flux[i + 2])


def test_primary_scale_and_the_cr_uv_environment():
    """The field scales with the primary zeta(H2) = 2 zeta_H, as Heays+17 define P(lambda) (pahchem audit A1): g0 of
    ``cr_uv.environment`` is 2 zeta_H / 1e-16, its field shape "cr_uv", and its field g0 times the unit one."""
    env = cr_uv.environment("dark", 1e-16, 1.0, 1e-3, 10.0)
    assert env.g0 == pytest.approx(cr_uv.scale(1e-16)) and env.g0 == pytest.approx(2.0)
    shape = population.field_shape(env)
    assert shape[:3] == ("cr_uv", "heays17", 0.5) and shape[3] == cr_uv.signature("heays17")
    np.testing.assert_allclose(population.env_field(env).photon_flux,
                               env.g0 * population._unit_field(shape).photon_flux, rtol=1e-12)
    f = env.field()
    assert f.photon_rate(np.ones_like(f.energy_ev)) == pytest.approx(
        population.env_field(env).photon_rate(np.ones_like(population.E_PH), population.E_PH), rel=0.01)
