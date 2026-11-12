"""The Delta-SCF threshold shifts of dehydrogenated and super-hydrogenated states (``pahchem.threshold_shifts``)."""

import pytest

from pycalima.pahchem import threshold_shifts as ts
from pycalima.pahchem.baseline import BaselineModel, Params

PARENTS = ("C24", "C54", "C96")
EA, IP0, IP1 = -1, 0, 1


def test_ladder_covers_minus4_to_plus2_and_is_zero_at_the_parent():
    for parent in PARENTS:
        assert sorted(ts.LADDER[parent]) == list(range(-4, 3))
        assert ts.LADDER[parent][0] == (0.0, 0.0, 0.0)


def test_lower_configuration_signs():
    """A missing H raises the EA (the sigma anion) and the IPs; an odd extra H (a CH2 pi radical) lowers IP0 and
    raises the EA."""
    for parent in PARENTS:
        for dnh in range(-4, 0):
            assert ts.ladder_shift(parent, dnh, EA) > 0.0
            assert ts.ladder_shift(parent, dnh, IP0) > 0.0
        assert ts.ladder_shift(parent, 1, IP0) < 0.0 < ts.ladder_shift(parent, 1, EA)


def test_shifts_shrink_with_size():
    """The sigma-anion and CH2-radical steps fall towards the large-PAH limit, where the parent's own EA and IP catch
    up with them."""
    for dnh, z in ((-1, EA), (-3, EA), (1, IP0), (1, EA), (-4, IP0)):
        a, b, c = (abs(ts.ladder_shift(p, dnh, z)) for p in PARENTS)
        assert a > b > c


def test_extension_beyond_the_ladder():
    for parent in PARENTS:
        for z in (EA, IP0, IP1, 2, 3):
            assert ts.ladder_shift(parent, 3, z) == ts.ladder_shift(parent, 1, z)
            assert ts.ladder_shift(parent, 4, z) == ts.ladder_shift(parent, 2, z)
        assert ts.ladder_shift(parent, -5, EA) == ts.ladder_shift(parent, -3, EA)
        assert ts.ladder_shift(parent, -6, EA) == ts.ladder_shift(parent, -4, EA)
        for z in (IP0, IP1):
            assert ts.ladder_shift(parent, -8, z) == pytest.approx(2.0 * ts.ladder_shift(parent, -4, z))
        assert ts.ladder_shift(parent, -2, 2) == ts.ladder_shift(parent, -2, IP1)


def test_baseline_thresholds_follow_params():
    for parent in PARENTS:
        nh0 = BaselineModel(parent, None).nh0
        dft = BaselineModel(parent, None)
        rule = BaselineModel(parent, None, Params(threshold_shifts="andrews16"))
        for z in dft.charges:
            base = dft.ionisation_energy(z, nh0)
            assert rule.ionisation_energy(z, nh0) == base
            for nh in dft.nh_values:
                assert dft.ionisation_energy(z, nh) == pytest.approx(base + ts.ladder_shift(parent, nh - nh0, z))
                assert rule.ionisation_energy(z, nh) == pytest.approx(base + 0.1 * (nh0 - nh))
    with pytest.raises(ValueError, match="threshold_shifts"):
        BaselineModel("C24", None, Params(threshold_shifts="nope")).ionisation_energy(0, 11)


def test_size_laws_reproduce_the_ladder():
    """``law_shift`` at the ladder parents' own DFT thresholds gives ``LADDER`` to 0.07 eV (0.06 where C96 is estimated
    or interpolated; the rest to 0.03)."""
    for parent, (nc, ip0, _, ea) in ts.LADDER_PARENTS.items():
        for dnh in (-4, -3, -2, -1, 1, 2):
            for z in (EA, IP0, IP1):
                law = ts.law_shift(nc, dnh, z, ip0, ea - ts.DIFFUSE_EA_EV)
                assert law == pytest.approx(ts.ladder_shift(parent, dnh, z), abs=0.07)


def test_lower_configuration_forms_depend_on_the_parent():
    """The transfer test's forms: a parent whose EA already exceeds the sigma anion's gains nothing from a missing H, and
    a parent whose IP is below the CH2 radical's loses nothing from an extra one (C28H14, UID 2226: EA 1.54, IP 5.79 at
    N_C 28; the law gives +0.19 and -0.36, DFT +0.14 to +0.41 and -0.02 to -0.40 by site)."""
    nc = 28
    assert ts.law_shift(nc, -1, EA, 6.0, 2.2) == 0.0
    assert ts.law_shift(nc, 1, IP0, 5.0, 1.0) == 0.0
    assert ts.law_shift(nc, -1, EA, 5.79, 1.54 - ts.DIFFUSE_EA_EV) == pytest.approx(0.19, abs=0.01)
    assert ts.law_shift(nc, 1, IP0, 5.79, 1.54 - ts.DIFFUSE_EA_EV) == pytest.approx(-0.36, abs=0.01)
    assert ts.law_shift(nc, 0, EA, 5.79, 1.0) == 0.0


def test_size_laws_outside_the_ladder():
    """Below N_C 24, C24's shifts; above 96 the shift forms hold their C96 value; the dN_H extensions as the ladder's."""
    for z in (EA, IP0, IP1):
        for dnh in (-6, -1, 1, 3):
            assert ts.law_shift(10, dnh, z, 8.1, -0.2) == ts.ladder_shift("C24", dnh, z)
    assert ts.law_shift(400, 2, IP0, 5.0, 2.5) == ts.law_shift(96, 2, IP0, 5.0, 2.5)
    assert ts.law_shift(400, -8, IP0, 5.0, 2.5) == pytest.approx(2.0 * ts.law_shift(96, -4, IP0, 5.0, 2.5))
    assert ts.law_shift(60, -7, EA, 6.0, 1.0) == ts.law_shift(60, -3, EA, 6.0, 1.0)
    assert ts.law_shift(60, 4, IP1, 6.0, 1.0) == ts.law_shift(60, 2, IP1, 6.0, 1.0)
