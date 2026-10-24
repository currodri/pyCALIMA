"""The fetched Malloci+07 cross sections obey the Thomas-Reiche-Kuhn sum rule, except the two files flagged.

Skip without ``calima-fetch-data fetch malloci2007-spectra``.
"""
import pytest

from pycalima.pahchem import malloci2007 as M
from pycalima.pahchem import photo


def _spectra(**kw):
    from pycalima._datasets import DatasetUnavailable

    try:
        return M.spectra(**kw)
    except DatasetUnavailable:
        pytest.skip("Malloci+07 spectra not fetched (calima-fetch-data fetch malloci2007-spectra)")


def _f(m, z):
    return M.oscillator_strength(*m["sigma"][z], 0, 30)


def test_every_kept_state_holds_the_same_share_of_its_electrons_and_one_per_charge_step():
    for name, m in _spectra().items():
        for z in m["sigma"]:
            assert 0.58 < _f(m, z) / M.valence_electrons(m["nc"], m["nh"], z) < 0.66, (name, z)
            if z + 1 in m["sigma"]:
                assert -1.0 < _f(m, z) - _f(m, z + 1) < 3.0, (name, z)


def test_the_flagged_files_do_break_the_sum_rule():
    s = _spectra(include_inconsistent=True)
    cor = s["coronene"]
    assert _f(cor, -1) / M.valence_electrons(24, 12, -1) > 1.2
    assert _f(cor, -1) / M.valence_electrons(54, 18, -1) == pytest.approx(0.61, abs=0.02)
    bc = s["benzocoronene"]
    assert _f(bc, 1) - _f(bc, 0) > 10.0
    assert ("coronene", -1) in M.INCONSISTENT and -1 not in _spectra()["coronene"]["sigma"]


def test_bundled_coronene_anion_is_not_served():
    with pytest.raises(ValueError):
        photo.sigma_malloci_coronene(-1, [10.0])
