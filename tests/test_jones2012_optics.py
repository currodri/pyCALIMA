"""a-C(:H) optics of Jones (2012) (pycalima.models.dust_radiation.jones2012_optics).

The tests need the dataset optical-jones-2012-size and skip without it; the
comparison with THEMIS also needs the DustEM dataset.
"""
import numpy as np
import pytest

from pycalima.models.dust_radiation import jones2012_optics as jo


@pytest.fixture
def jones2012():
    from pycalima._datasets import get_dataset

    if get_dataset(jo.DATASET).locate() is None:
        pytest.skip("Jones (2012) tables not fetched; run `calima-fetch-data fetch optical-jones-2012-size`")


def test_band_gap_columns_are_in_increasing_order(jones2012):
    """k falls with E_g in the visible for a >= 3 nm, so the ReadMe's 1.5/1.15 eV
    labels of columns 7 and 8 are 1.25/1.5 eV. (Below 3 nm the size-limited aromatic
    clusters lower k of the low-E_g columns, and k is not monotonic in E_g.)"""
    lam, _, k = jo.load()
    i = np.argmin(abs(lam - 0.55))
    assert np.all(np.diff(k[3:, i, :], axis=1) < 0)
    assert jo.E_GAPS[7:9] == (1.25, 1.5)


def test_tables_are_reproduced_at_their_nodes(jones2012):
    lam, n, k = jo.load()
    il = np.array([0, 300, 600, 999])
    for r, a_nm in enumerate(jo.RADII_NM):
        m = jo.refractive_index(a_nm * 1e-3, lam[il], 2.5)[:, 0]
        np.testing.assert_allclose(m.real, n[r, il, 12], rtol=1e-10)
        np.testing.assert_allclose(-m.imag, k[r, il, 12], rtol=1e-10)


def test_composition_and_density_follow_jones_2012(jones2012):
    np.testing.assert_allclose(jo.density([0.1, 2.5]), [1.5964, 1.3269], atol=1e-4)
    np.testing.assert_allclose(jo.hydrogen_fraction([0.1, 2.5]), [0.1 / 4.3, 2.5 / 4.3])
    with pytest.raises(ValueError, match="outside the tables"):
        jo.refractive_index(1e-3, [0.5], 3.0)


def test_efficiencies_are_physical_on_both_sides_of_the_splice(jones2012):
    a = np.array([3.3e-4, 2e-3, 2e-2, 0.2])
    lam = np.geomspace(2.5e-3, 1e3, 30)
    for e_gap in (0.1, 2.5):
        qabs, qsca, g = jo.efficiencies(a, lam, e_gap)
        assert np.all(np.isfinite(qabs)) and np.all(qabs > 0)
        assert np.all(qsca >= 0) and np.all(np.abs(g) <= 1)


def test_a_c_matches_themis_where_themis_did_not_modify_it(jones2012, themis_cm20):
    """At 10-20 nm the CM20 grains are a-C (E_g = 0.1 eV) computed from these
    tables; THEMIS's changes (Jones et al. 2013, App. C) act at smaller radii."""
    from pycalima.models.dust_radiation import themis_optics as th

    a = np.array([1e-2, 2e-2])
    lam = np.geomspace(0.1, 1000.0, 60)
    r = jo.efficiencies(a, lam, 0.1)[0] / th.efficiencies(a, lam)[0]
    assert np.median(np.abs(r - 1)) < 0.005 and np.all(np.abs(r - 1) < 0.1)


# -- hydrogen of a particle: bulk plus surface passivation (paper III, App. A); no dataset needed --

def test_surface_hydrogen_vanishes_for_large_particles():
    for e in (0.1, 1.0, 2.5):
        x_h = e / 4.3
        np.testing.assert_allclose(jo.h_per_carbon(1e-2, e), x_h / (1 - x_h), rtol=1e-3)


def test_h_per_carbon_falls_with_radius_and_rises_with_band_gap():
    a = np.geomspace(3e-8, 3e-5, 40)
    for e in (0.1, 1.0, 2.5):
        assert np.all(np.diff(jo.h_per_carbon(a, e)) < 0)
    e_dg, e_rcn = np.linspace(0.1, 1.5, 29), np.linspace(1.64, 2.5, 18)   # each network on its own
    for r in (5e-8, 1e-7, 1e-6):
        for es in (e_dg, e_rcn):
            assert np.all(np.diff([jo.h_per_carbon(r, e) for e in es]) > 0)


def test_surface_hydrogen_matches_paper_iii_statements():
    """Paper III, Sect. 2.4: at 0.5 nm there can be as many H atoms on the surface as
    inside; for H-rich grains of 30 nm the surface share is a few per cent."""
    for e in (1.75, 2.0, 2.5):
        x_h = e / 4.3
        assert jo.surface_hydrogen_fraction(5e-8, e) >= x_h
        assert jo.surface_hydrogen_fraction(3e-6, e) < 0.07 * x_h        # 3.2-5.8 %


def test_aromatic_nanoparticle_h_per_carbon_matches_jones_ysard_2025():
    """Jones & Ysard (2025): [H]/[C] ~ 1.4 -> 0.8 from 0.4 to 1.2 nm, even at E_g = 0.1 eV."""
    np.testing.assert_allclose(jo.h_per_carbon(np.array([4e-8, 1.2e-7]), 0.1), [1.4, 0.8], rtol=0.15)


def test_band_gap_of_h_per_carbon_inverts_h_per_carbon():
    for r in (1.2e-7, 3e-7, 1e-6, 5e-6):
        for e in (0.1, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 2.5):
            np.testing.assert_allclose(jo.e_gap_of_h_per_carbon(r, jo.h_per_carbon(r, e)), e, atol=2e-3)


def test_band_gap_of_h_per_carbon_clips_to_the_carrier_range():
    r = 1e-6
    lo, hi = jo.h_per_carbon(r, 0.1), jo.h_per_carbon(r, 2.5)
    np.testing.assert_allclose(jo.e_gap_of_h_per_carbon(r, [0.5 * lo, 2.0 * hi]), [0.1, 2.5])
    e = jo.e_gap_of_h_per_carbon(np.array([1e-7, 1e-6, 1e-5]), 0.8)
    assert e.shape == (3,) and np.all(np.diff(e) > 0)


# -- band gaps between columns and radii between tables --

def test_both_size_schemes_reproduce_the_tables_at_their_radii(jones2012):
    lam, n, k = jo.load()
    il = np.array([0, 300, 600, 999])
    for r, a_nm in enumerate(jo.RADII_NM):
        for j in (2, 12):                                   # E_g = 0.1, 2.5
            for scheme in jo.SIZE_INTERP:
                m = jo.refractive_index(a_nm * 1e-3, lam[il], jo.E_GAPS[j], scheme)[:, 0]
                np.testing.assert_allclose(m.real, n[r, il, j], rtol=1e-10)
                np.testing.assert_allclose(-m.imag, k[r, il, j], rtol=1e-10)


def test_effective_gap_scheme_is_plain_interpolation_where_size_does_not_limit_the_gap(jones2012):
    """For a-C:H (E_g 2.5 > E_lim at every tabulated radius) the two schemes agree."""
    a = np.array([4e-4, 2e-3, 6e-3, 2e-2, 6e-2])
    lam = np.geomspace(0.03, 900.0, 40)
    np.testing.assert_allclose(jo.refractive_index(a, lam, 2.5), jo.refractive_index(a, lam, 2.5, "lna"),
                               rtol=1e-10)


def test_band_gaps_between_columns_are_continuous(jones2012):
    a, lam = np.array([2e-3, 3e-2]), np.geomspace(0.1, 500.0, 25)
    for e in (0.25, 1.0, 2.0):
        lo, hi = jo.refractive_index(a, lam, e - 1e-6), jo.refractive_index(a, lam, e + 1e-6)
        np.testing.assert_allclose(lo, hi, rtol=1e-4)
    a = a[1:]                       # 30 nm: E_lim < 0.5 eV, so m is linear in E_g between columns
    m = jo.refractive_index(a, lam, 0.625)                  # midway between 0.5 and 0.75
    m0, m1 = jo.refractive_index(a, lam, 0.5), jo.refractive_index(a, lam, 0.75)
    np.testing.assert_allclose(m.real, 0.5 * (m0.real + m1.real), rtol=1e-10)
    np.testing.assert_allclose(np.log(-m.imag), 0.5 * (np.log(-m0.imag) + np.log(-m1.imag)), rtol=1e-8)


def test_effective_gap_scheme_follows_themis_between_table_radii(jones2012, themis_cm20):
    """At 1.5, 2 and 5 nm (between the 1, 3 and 10 nm tables) a-C follows Q_CM20, built
    from Jones's continuous-in-radius model: the far-IR, which E_lim was fitted to, within
    10% (THEMIS and the tables differ by 9% at the 3 nm table itself), and every band
    within 25% (THEMIS changed bands and the bump below 10 nm)."""
    from pycalima.models.dust_radiation import themis_optics as th

    a = np.array([1.5e-3, 2e-3, 5e-3])
    lam = np.geomspace(0.0912, 1000.0, 300)
    r = jo.efficiencies(a, lam, 0.1)[0] / th.efficiencies(a, lam)[0]
    fir = lam > 30.0
    assert np.all(np.abs(np.median(r[fir], axis=0) - 1.0) < 0.10)
    for l0, l1 in ((0.0912, 0.2), (0.2, 0.25), (0.3, 1.0), (1.0, 5.0), (5.0, 30.0)):
        b = (lam >= l0) & (lam < l1)
        assert np.all(np.abs(np.median(r[b], axis=0) - 1.0) < 0.25)
    r_lna = jo.efficiencies(a[1:2], lam, 0.1, "lna")[0] / th.efficiencies(a[1:2], lam)[0]
    assert np.median(r_lna[fir]) < 0.6                        # what plain interpolation gives at 2 nm
