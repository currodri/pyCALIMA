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
    with pytest.raises(ValueError, match="not tabulated"):
        jo.refractive_index(1e-3, [0.5], 2.4)


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
