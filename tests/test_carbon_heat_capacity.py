"""Harmonic heat capacity of a-C(:H) (pycalima.models.dust_radiation.carbon_heat_capacity)."""
import numpy as np
import pytest

from pycalima.models.dust_radiation import carbon_heat_capacity as ch

T = np.geomspace(1.0, 1e5, 60)


@pytest.mark.parametrize("term", [ch.per_sp2_carbon, ch.per_sp3_carbon, ch.per_aliphatic_h, ch.per_aromatic_h])
def test_every_term_has_three_modes_and_never_exceeds_dulong_petit(term):
    c = term(T)
    assert np.all(c >= 0) and np.all(c <= 3.0 + 1e-9)
    assert np.all(np.diff(c) >= -1e-12)
    assert c[-1] == pytest.approx(3.0, rel=2e-3)


def test_composition_follows_the_jones_2012_networks():
    x_h, x_sp2, x_sp3 = ch.composition(0.1)              # defected graphite: X_sp3 = X_H / 2
    assert x_sp3 == pytest.approx(x_h / 2) and x_h + x_sp2 + x_sp3 == pytest.approx(1.0)
    x_h, x_sp2, x_sp3 = ch.composition(2.5)              # random network, Angus & Jansen eq. 7
    assert x_sp3 / x_sp2 == pytest.approx((8 * x_h - 3) / (8 - 13 * x_h))
    with pytest.raises(ValueError, match="outside"):
        ch.composition(1.55)                             # X_H = 0.36: between the networks


def test_grain_heat_capacity_counts_3n_minus_6_modes():
    n = np.array([30.0, 1e6])
    C = ch.grain_heat_capacity(n, [1e5], 0.1)[0]
    np.testing.assert_allclose(C / ch.K_B, 3.0 * (n - 2.0), rtol=3e-3)


def test_cm20_structure_is_continuous_at_the_mantle(themis_cm20):
    a = np.array([0.0199999, 0.0200001])
    C = ch.heat_capacity_cm20(a, [30.0, 300.0, 3000.0])
    np.testing.assert_allclose(C[:, 0], C[:, 1], rtol=1e-4)


def test_graphite_term_matches_dustem_graphite_and_dustem_cm20_exceeds_dulong_petit(themis_cm20):
    from pycalima.models.dust_radiation import themis_optics as th

    rows = [r.split() for r in th._numbers(themis_cm20 / "hcap" / "C_Gra.DAT")]
    d = np.array(rows[3:3 + int(rows[2][0])], float)
    TG = 10 ** d[:, 0]
    per_c = 10 ** d[:, -1] / (2.24 / (12.011 * ch.AMU)) / ch.K_B
    ok = TG >= 3.0
    np.testing.assert_allclose(ch.per_sp2_carbon(TG[ok]), per_c[ok], rtol=0.05)
    a = np.array([6e-3])
    n_atoms = 4 / 3 * np.pi * (a * 1e-4) ** 3 * 1.6 / (ch.mean_atomic_mass(0.1) * ch.AMU)
    assert th.heat_capacity(a, [5000.0])[0, 0] / (3 * ch.K_B * n_atoms[0]) > 1.3
    assert ch.heat_capacity_cm20(a, [5000.0])[0, 0] / (3 * ch.K_B * n_atoms[0]) < 1.0
