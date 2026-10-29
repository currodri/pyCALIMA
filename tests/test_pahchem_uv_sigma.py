"""pahchem's UV absorption cross sections (``photo.sigma_uv``): Berne+22's Malloci average, the default since 2026-10-26.

These need the fetched ``malloci2007-spectra`` (``calima-fetch-data fetch malloci2007-spectra``) and skip without it.
"""

import numpy as np
import pytest

from pycalima.pahchem import photo, population
from pycalima.pahchem.baseline import Params

E = np.linspace(1.0, 40.0, 781)


@pytest.fixture(autouse=True)
def _malloci():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import malloci2007

    try:
        malloci2007.spectra()
    except DatasetUnavailable:
        pytest.skip("malloci2007-spectra not fetched (calima-fetch-data fetch malloci2007-spectra)")


def test_default_is_the_malloci_average():
    assert Params().uv_cross_section == "malloci_average"
    assert np.array_equal(photo.sigma_uv(54, 0, E), photo.sigma_malloci_average(54, 0, E))
    assert np.array_equal(photo.sigma_uv(54, 0, E, "li_draine"), photo.sigma_li_draine(54, 0, E))
    with pytest.raises(ValueError):
        photo.sigma_uv(54, 0, E, "nope")


@pytest.mark.parametrize("z", [-1, 0, 1, 2])
def test_malloci_average_scales_with_n_c_inside_and_is_li_draine_outside(z):
    lo, hi = photo.MALLOCI_AVERAGE_RANGE_EV
    inside = (E >= lo) & (E <= hi)
    a, b = photo.sigma_malloci_average(54, z, E), photo.sigma_malloci_average(108, z, E)
    assert np.allclose(b[inside], 2.0 * a[inside])
    assert np.array_equal(a[~inside], photo.sigma_li_draine(54, z, E)[~inside])


@pytest.mark.parametrize("z", [-1, 0, 1, 2])
def test_malloci_average_lies_within_its_molecules(z):
    """The per-C mean sits between the per-C extremes of the nine molecules, at every energy inside the range."""
    from pycalima.pahchem import malloci2007

    spec = malloci2007.spectra()
    lo, hi = photo.MALLOCI_AVERAGE_RANGE_EV
    e = E[(E >= lo) & (E <= hi)]
    per_c = np.array([np.interp(e, *spec[m]["sigma"][z]) / spec[m]["nc"] for m in photo.MALLOCI_AVERAGE_MOLECULES])
    avg = photo.sigma_malloci_average(1, z, e)
    assert np.all(avg >= per_c.min(axis=0) * (1 - 1e-12)) and np.all(avg <= per_c.max(axis=0) * (1 + 1e-12))


def test_trications_take_the_dication_average():
    assert np.array_equal(photo.sigma_malloci_average(96, 3, E), photo.sigma_malloci_average(96, 2, E))


def test_malloci_molecule_takes_the_nearest_computed_charge_and_scales():
    from pycalima.pahchem import malloci2007

    # coronene's served anion file is C54H18-'s (left out): the anion takes the neutral
    assert np.array_equal(photo.sigma_malloci_molecule("coronene", -1, E), photo.sigma_malloci_molecule("coronene", 0, E))
    e, s = malloci2007.sigma_of("circumcoronene", 1)
    assert np.allclose(photo.sigma_malloci_molecule("circumcoronene", 1, E, nc=96),
                       96 / 54 * np.interp(E, e, s, left=0.0, right=0.0))


def test_li_draine_bases_keep_their_cache_keys():
    """The cross section enters the basis inputs only when it is not Li & Draine's, so the earlier bases stay valid."""
    mol = population.Molecule("x", 24, 12, {"solo": 0, "duo": 12, "trio": 0, "quartet": 0},
                              {0: _species()}, {0: 0}, 7.2)
    shape = ("kurucz", 40000)
    ld = population._basis_inputs(mol, 0, shape, 100.0, 21.5 * 8065.54, Params(uv_cross_section="li_draine"))
    new = population._basis_inputs(mol, 0, shape, 100.0, 21.5 * 8065.54, Params())
    assert "uv_cross_section" not in ld and new["uv_cross_section"] == "malloci_average"
    assert population._basis_key(ld) != population._basis_key(new)


def _species():
    from pycalima.pahchem.molecule import PAHSpecies

    # a synthetic C24H12 mode set (3N - 6 = 102): 12 CH stretches, 12 CH in-plane, 12 out-of-plane, 66 skeletal
    f = np.concatenate([np.full(12, 3050.0), np.full(12, 1150.0), np.full(12, 850.0), np.geomspace(60.0, 1600.0, 66)])
    return PAHSpecies(24, 12, 0, f, np.full(f.size, 1.0), "test")
