"""CosmicPAH-IRDB reader (pahchem.cosmicpah): the law rules on a synthetic payload of the database's schema,
and Demyk+26 Table 4 on the fetched data (skipped until ``calima-fetch-data fetch cosmicpah-irdb``)."""

import numpy as np
import pytest

from pycalima.pahchem import cosmicpah as cp


def _payload(rows):
    return ["", "{...}", rows]


ROW = [3000.0, 0.4, 1.5, "P", [[3010.0, -0.02, 0.0, 400.0, 500.0], [3012.0, -0.024, 0.0, 500.0, 700.0]],
       "F", [[5.0, 0.02, 0.0, 400.0, 500.0], [3.0, 0.024, 0.0, 500.0, 700.0]]]


def test_laws_hold_below_and_extrapolate_positions_above():
    b = cp.parse_laws(_payload([ROW]))[0]
    t = np.array([100.0, 400.0, 450.0, 500.0, 600.0, 700.0, 1000.0])
    assert b.position(t) == pytest.approx([3002.0, 3002.0, 3001.0, 3000.0, 2997.6, 2995.2, 2988.0])
    assert b.fwhm(t) == pytest.approx([13.0, 13.0, 14.0, 15.0, 17.4, 19.8, 27.0])
    assert b.t_range == (400.0, 700.0) and b.lorentz == 0.4


def test_a_falling_width_is_held_above_the_data():
    row = ROW[:6] + [[[30.0, -0.01, 0.0, 400.0, 700.0]]]
    b = cp.parse_laws(_payload([row]))[0]
    assert b.fwhm([700.0, 2000.0]) == pytest.approx([23.0, 23.0])


@pytest.mark.parametrize("bad", [
    [ROW[:3] + ["X"] + ROW[4:]],                                                    # not a P/F row
    [ROW[:4] + [[[3010.0, -0.02, 0.0, 400.0]]] + ROW[5:]],                          # short piece
    [ROW[:4] + [[[3010.0, -0.02, 0.0, 400.0, 500.0], [3012.0, -0.024, 0.0, 550.0, 700.0]]] + ROW[5:]],   # gap
    [ROW[:4] + [[["__import__('os')", 0, 0, 400.0, 500.0]]] + ROW[5:]],             # strings are not numbers
])
def test_malformed_payloads_are_rejected(bad):
    with pytest.raises(ValueError):
        cp.parse_laws(_payload(bad))


def test_payload_must_be_the_three_element_list():
    with pytest.raises(ValueError):
        cp.parse_laws({"bands": []})


def test_pseudo_voigt_has_unit_area_and_the_stated_fwhm():
    nu = np.linspace(-4000.0, 4000.0, 800001)
    for lor in (0.0, 0.3, 1.0):
        p = cp.pseudo_voigt(nu, 0.0, 12.0, lor)
        assert np.trapezoid(p, nu) == pytest.approx(1.0, abs=1e-3 if lor else 1e-9)
        half = nu[p >= 0.5 * p.max()]
        assert half[-1] - half[0] == pytest.approx(12.0, abs=0.02)


def test_registry_lists_the_files_the_fetcher_writes():
    from pycalima._datasets import get_dataset

    ds = get_dataset(cp.DATASET)
    assert tuple(ds.files) == cp.FILES and ds.kind == "fetch" and ds.fetcher == "pycalima.pahchem.cosmicpah:fetch"


@pytest.fixture(scope="module")
def root():
    from pycalima._datasets import get_dataset

    found = get_dataset(cp.DATASET).locate()
    if found is None:
        pytest.skip("CosmicPAH-IRDB not fetched (calima-fetch-data fetch cosmicpah-irdb)")
    return found


@pytest.mark.parametrize("key,label,pieces", [
    ("6h-pyrene", 2942.7, [(2947.10, -0.00846, 423, 523), (2948.71, -0.01152, 523, 673)]),
    ("6h-pyrene", 3028.0, [(3041.08, -0.02493, 423, 523), (3040.21, -0.02353, 523, 623)]),
    ("6h-pyrene", 1597.0, [(1612.73, -0.0304, 423, 673)]),
    ("me-pyrene", 3048.8, [(3061.01, -0.02137, 423, 548), (3061.56, -0.02232, 548, 623)]),
    ("pyrene", 3051.3, [(3063.42, -0.02306, 523, 548), (3062.52, -0.02159, 548, 673)]),
])
def test_position_laws_are_demyk26_table4(root, key, label, pieces):
    b = cp.band(key, label, root)
    got = [(p.c0, p.c1, p.t_min, p.t_max) for p in b.position_pieces]
    assert len(got) == len(pieces)
    for g, e in zip(got, pieces):
        assert g[0] == pytest.approx(e[0], abs=0.01) and g[1] == pytest.approx(e[1], abs=1e-5) and g[2:] == e[2:]


def test_every_law_gives_its_listed_position_at_523k(root):
    for key in cp.DATASETS:
        for b in cp.laws(key, root):
            if b.intensity > 0:
                assert float(b.position(523.0)) == pytest.approx(b.label_cm, abs=1.5), (key, b.label_cm)


def test_spectra_cover_the_bands_and_the_3um_band_reddens_with_temperature(root):
    temps = cp.temperatures("6h-pyrene", root)
    assert temps[0] == 423 and temps[-1] == 673
    peaks = []
    for t in (temps[0], temps[-1]):
        nu, a = cp.spectrum("6h-pyrene", t, root)
        assert np.all(np.diff(nu) > 0) and nu[0] < 500 and nu[-1] > 7000
        w = (nu > 2900) & (nu < 2990)
        peaks.append(nu[w][np.argmax(a[w])])
    assert peaks[1] < peaks[0]
