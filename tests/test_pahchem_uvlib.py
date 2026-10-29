"""The Octopus cross-section library as sigma(E, N_C, Z): pycalima.pahchem.uvlib. Invariants of the size and charge rule, on the
bundled anchor table (skipped if it is absent)."""
import numpy as np
import pytest

from pycalima.pahchem import uvlib

MB_EV_PER_F = 109.761
E = np.geomspace(0.05, 5000.0, 6000)


@pytest.fixture(scope="module")
def lib():
    if not uvlib.default_path().exists():
        pytest.skip("external_data/pahchem/pah_uv_library.npz is not bundled")
    return uvlib.library()


def f_per_c(s, e_max=30.0):
    k = E <= e_max
    return np.trapezoid(s[k], E[k]) / 1e-18 / MB_EV_PER_F


def test_anchors_are_reproduced_exactly(lib):
    for i, n in enumerate(lib["nc"]):
        for q, z in enumerate(lib["charges"]):
            for ip, pos in enumerate(uvlib.POSITIONS):
                for ist, st in enumerate(uvlib.STRUCTURES):
                    got = uvlib.per_carbon(n, z, lib["energy_ev"], pos, st)
                    assert np.allclose(got, lib["sigma"][ip, ist, i, q].astype(float), rtol=1e-6, atol=0)


def test_sigma_scales_with_the_number_of_carbons(lib):
    s = uvlib.sigma_library(64, 0, E)
    assert np.allclose(s, 64 * uvlib.per_carbon(64, 0, E))


def test_interpolated_strength_lies_between_the_neighbouring_anchors(lib):
    """Linear interpolation of spectra is a convex combination, so the oscillator strength per carbon is bracketed and varies monotonically
    between two anchors."""
    for z in (-1, 0, 1, 2):
        for lo, hi in zip(lib["nc"][:-1], lib["nc"][1:]):
            f = [f_per_c(uvlib.per_carbon(n, z, E)) for n in np.linspace(lo, hi, 7)]
            assert min(f[0], f[-1]) - 1e-9 <= min(f) and max(f) <= max(f[0], f[-1]) + 1e-9
            assert np.all(np.diff(f) * np.sign(f[-1] - f[0]) >= -1e-9)


def test_sizes_outside_the_anchors_take_the_nearest_one(lib):
    assert np.array_equal(uvlib.per_carbon(8, 0, E), uvlib.per_carbon(lib["nc"][0], 0, E))
    assert np.array_equal(uvlib.per_carbon(500, 1, E), uvlib.per_carbon(lib["nc"][-1], 1, E))


def test_charge_plus_three_takes_the_dication_and_charge_below_minus_one_is_refused(lib):
    assert np.array_equal(uvlib.per_carbon(96, 3, E), uvlib.per_carbon(96, 2, E))
    with pytest.raises(ValueError):
        uvlib.per_carbon(96, -2, E)


def test_the_dication_of_the_small_molecules_is_the_cations(lib):
    small = np.nonzero(lib["nc"] < 54)[0]
    assert lib["proxy"][small, -1].all() and not lib["proxy"][lib["nc"] >= 54].any()
    for n in lib["nc"][small]:
        assert np.array_equal(uvlib.per_carbon(n, 2, E), uvlib.per_carbon(n, 1, E))


def test_the_position_shift_conserves_strength_and_moves_the_neutral_up(lib):
    """The shifted and TD-LDA variants have the same f per carbon, and the shifted neutral absorbs at higher energy; the ions are
    identical."""
    for n in lib["nc"]:
        a, b = uvlib.per_carbon(n, 0, E, "shifted"), uvlib.per_carbon(n, 0, E, "tdlda")
        assert f_per_c(a, 30.0) == pytest.approx(f_per_c(b, 30.0), rel=2e-3)
        k = (E > 1.0) & (E < 7.0)
        assert np.trapezoid(a[k] * E[k], E[k]) / np.trapezoid(a[k], E[k]) > np.trapezoid(b[k] * E[k], E[k]) / np.trapezoid(b[k], E[k])
        for z in (-1, 1):
            assert np.array_equal(uvlib.per_carbon(n, z, E, "shifted"), uvlib.per_carbon(n, z, E, "tdlda"))


def test_mixed_and_compact_agree_below_c54_and_differ_above(lib):
    for n in (16, 24, 32):
        assert np.array_equal(uvlib.per_carbon(n, 0, E, structure="mixed"), uvlib.per_carbon(n, 0, E, structure="compact"))
    assert not np.array_equal(uvlib.per_carbon(54, 0, E, structure="mixed"), uvlib.per_carbon(54, 0, E, structure="compact"))


def test_thomas_reiche_kuhn_count_is_kept(lib):
    """f to 150 eV per carbon is ~98.8 % of the valence electrons per carbon (4 + H/C - Z/N_C) in every state."""
    for n in (24, 54, 96, 216):
        f = f_per_c(uvlib.per_carbon(n, 0, E, structure="compact"), 150.0)
        assert 3.8 < f < 4.5


def test_spectra_are_non_negative_to_the_ringing_level_and_vanish_outside_the_range(lib):
    for n in (16, 70, 216):
        for z in (-1, 0, 1, 2):
            s = uvlib.per_carbon(n, z, E)
            assert s.min() >= -2e-3 * s.max()
    assert uvlib.per_carbon(54, 0, np.array([0.001, 2e4])).tolist() == [0.0, 0.0]


def test_bad_options_are_refused(lib):
    with pytest.raises(ValueError):
        uvlib.per_carbon(54, 0, E, positions="x")
    with pytest.raises(ValueError):
        uvlib.per_carbon(54, 0, E, structure="x")


def test_compact_is_the_circumcoronene_family_and_mixed_the_isomer_mean(lib):
    """C54H18's strong band (a sharp, high peak) dominates the compact anchor; the mixed one averages it with four non-compact isomers
    whose bands are broader, so its peak per carbon is lower. Guards the order of the table's structure axis."""
    c = uvlib.per_carbon(54, 0, E, structure="compact")
    m = uvlib.per_carbon(54, 0, E, structure="mixed")
    k = (E > 3.0) & (E < 8.0)
    assert c[k].max() > 1.15 * m[k].max()
