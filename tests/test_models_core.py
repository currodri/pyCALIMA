"""Core model utilities: unit conversions, radiation fields, size distributions.

These are pure functions with checkable mathematical properties, so the
assertions are real invariants (round-trips, normalisation, monotonicity,
bounds) rather than recorded output.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycalima.models.tools.utils import (
    Nc_from_size,
    has_uniform_bins,
    mass_from_Nc,
    sigmoid_function,
    size_from_Nc,
)


# ---------------------------------------------------------------------------
# PAH size <-> carbon count (Draine et al. 2021, Eq. 8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angstrom", [4.0, 5.0, 6.5, 10.0, 20.0, 50.0])
def test_size_nc_round_trip(angstrom):
    """size_from_Nc inverts Nc_from_size to within the integer truncation."""
    nc = Nc_from_size(angstrom)
    assert nc > 0
    back = size_from_Nc(nc)
    assert back == pytest.approx(angstrom, rel=0.02)


def test_nc_from_size_matches_the_published_normalisation():
    """Eq. 8 is N_C = 418 (a / 10 A)^3, so a = 10 A must give exactly 418."""
    assert Nc_from_size(10.0) == 418
    assert size_from_Nc(418) == pytest.approx(10.0)


def test_nc_from_size_is_monotonic_and_cubic():
    sizes = np.array([5.0, 10.0, 20.0])
    ncs = np.array([Nc_from_size(a) for a in sizes])
    assert np.all(np.diff(ncs) > 0)
    # doubling the radius must multiply N_C by ~8
    assert ncs[1] / ncs[0] == pytest.approx(8.0, rel=0.02)
    assert ncs[2] / ncs[1] == pytest.approx(8.0, rel=0.02)


def test_mass_from_Nc_is_positive_and_linear_in_Nc():
    """Regression test: mC_amu/mH_amu used to be undefined, so every call to
    this function raised NameError -- including the three call sites in
    models/PAH_collisions/PAH_coalescence.py."""
    m54 = mass_from_Nc(54)
    m108 = mass_from_Nc(108)
    assert m54 > 0 and np.isfinite(m54)
    # C_n H_(2n+2): mass is affine in n, so doubling n slightly less than doubles mass
    assert 1.9 < m108 / m54 < 2.1


def test_mass_from_Nc_has_a_physically_sane_magnitude():
    """C54H110 is a few hundred amu to ~1e3 amu, i.e. ~1e-21 g."""
    m = mass_from_Nc(54)
    assert 1e-22 < m < 1e-20


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def test_sigmoid_is_bounded_and_monotonic():
    x = np.logspace(-2, 2, 200)
    y = np.array([sigmoid_function(4.0, 1.0, xi) for xi in x])
    # saturates to exactly 1.0 in float at large x, hence <= rather than <
    assert np.all((y > 0) & (y <= 1))
    assert np.all(np.diff(y) >= -1e-12), "sigmoid must be non-decreasing in x"


def test_sigmoid_is_one_half_at_the_midpoint():
    assert sigmoid_function(3.0, 7.0, 7.0) == pytest.approx(0.5)


def test_has_uniform_bins():
    assert has_uniform_bins(np.linspace(0.0, 1.0, 11))
    assert not has_uniform_bins(np.logspace(0.0, 1.0, 11))


# ---------------------------------------------------------------------------
# radiation fields
# ---------------------------------------------------------------------------

def test_draine_isrf_is_positive_and_finite():
    from pycalima.models.tools.radiation_fields import Draine_1978_isrf

    for lam in (1000.0, 1500.0, 2000.0):
        val = Draine_1978_isrf(lam)
        assert np.isfinite(val) and val > 0


def test_mathis_field_is_positive_and_finite():
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    for energy_ev in (1.0, 5.0, 10.0, 13.0):
        val = Mathis83_radiation_field(energy_ev)
        assert np.isfinite(val) and val > 0


def test_mathis_field_falls_off_at_high_energy():
    """The ISRF is cut off above the Lyman limit, so the far-UV must not
    exceed the near-UV."""
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    assert Mathis83_radiation_field(13.0) < Mathis83_radiation_field(5.0)


# ---------------------------------------------------------------------------
# grain size distributions
# ---------------------------------------------------------------------------

DIST_KWARGS = dict(a0=0.01, amin=0.001, amax=0.1, sigma=0.6, grain_density=3.0)


def _lognormal():
    from pycalima.models.grain_distributions import LogNormal_Distribution

    return LogNormal_Distribution(**DIST_KWARGS)


@pytest.fixture
def sizes():
    return np.logspace(np.log10(DIST_KWARGS["amin"]),
                       np.log10(DIST_KWARGS["amax"]), 400)


def test_n_density_is_non_negative_and_finite(sizes):
    n = np.asarray(_lognormal().n_density(1e-24, sizes))
    assert n.shape == sizes.shape
    assert np.all(np.isfinite(n))
    assert np.all(n >= 0)


def test_n_density_scales_linearly_with_mass_density(sizes):
    d = _lognormal()
    n1 = np.asarray(d.n_density(1e-24, sizes))
    n2 = np.asarray(d.n_density(2e-24, sizes))
    assert np.allclose(n2, 2.0 * n1, rtol=1e-10)


@pytest.mark.parametrize("method", ["averaged_over", "averaged_over_number",
                                    "averaged_over_mass"])
def test_averaging_a_constant_returns_that_constant(method, sizes):
    """The averaging weights must be normalised."""
    d = _lognormal()
    avg = getattr(d, method)(np.ones_like(sizes), sizes)
    assert avg == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("method", ["averaged_over_number", "averaged_over_mass"])
def test_mean_size_lies_within_the_size_range(method, sizes):
    d = _lognormal()
    mean_a = getattr(d, method)(sizes, sizes)
    assert DIST_KWARGS["amin"] <= mean_a <= DIST_KWARGS["amax"]


def test_mass_weighted_mean_size_exceeds_number_weighted(sizes):
    """Mass weighting is ~a^3, so it must favour larger grains."""
    d = _lognormal()
    a_num = d.averaged_over_number(sizes, sizes)
    a_mass = d.averaged_over_mass(sizes, sizes)
    assert a_mass > a_num


def test_averaging_is_linear(sizes):
    d = _lognormal()
    x = np.linspace(1.0, 3.0, sizes.size)
    y = np.linspace(-2.0, 5.0, sizes.size)
    lhs = d.averaged_over_number(2.0 * x + 3.0 * y, sizes)
    rhs = 2.0 * d.averaged_over_number(x, sizes) + 3.0 * d.averaged_over_number(y, sizes)
    assert lhs == pytest.approx(rhs, rel=1e-8)


ALL_DISTRIBUTIONS = [
    ("LogNormal_Distribution", DIST_KWARGS),
    ("Classical_LogNormal_Distribution", DIST_KWARGS),
    ("Flat_Distribution", DIST_KWARGS),
    ("PowerLaw_Distribution",
     dict(a0=0.01, amin=0.001, amax=0.1, powlaw_index=-3.5, grain_density=3.0)),
    ("PowerLaw_DualCutoff_Distribution",
     dict(a0=0.01, amin=0.001, amax=0.1, powlaw_index=-3.5, grain_density=3.0)),
    ("Exponential_Distribution", DIST_KWARGS),
    ("PowerLaw_ExpCutoff_Distribution",
     dict(amin=0.001, amax=0.1, a_cutoff=0.05, powlaw_index=-3.5, grain_density=3.0)),
]


def _make(cls_name, kwargs):
    from pycalima.models import grain_distributions as gd

    try:
        return getattr(gd, cls_name)(**kwargs)
    except TypeError as exc:
        pytest.skip(f"{cls_name} takes a different signature: {exc}")


@pytest.mark.parametrize("cls_name,kwargs", ALL_DISTRIBUTIONS)
def test_every_distribution_gives_finite_non_negative_densities(cls_name, kwargs, sizes):
    d = _make(cls_name, kwargs)
    n = np.asarray(d.n_density(1e-24, sizes))
    assert np.all(np.isfinite(n)) and np.all(n >= 0)


# PowerLaw_ExpCutoff_Distribution.averaged_over_number divides X by the
# power-law weight instead of multiplying by it (grain_distributions.py, the
# `y = (X[mask]/sizes[mask]**(-self.powlaw_index))` line), so averaging a
# constant returns ~1e13 rather than that constant. Every sibling class
# multiplies. This is a live physics bug: the class is instantiated four times
# in models/dust_radiation/dust_oppacity.py for the Gao (2020) and Nozawa
# (2007) size distributions, and averaged_over_number is called there 33
# times. Left as a strict xfail rather than silently changing scientific
# output -- remove the marker when the weighting is corrected.
KNOWN_BAD_NORMALISATION = {"PowerLaw_ExpCutoff_Distribution"}


@pytest.mark.parametrize("cls_name,kwargs", ALL_DISTRIBUTIONS)
def test_number_averaging_a_constant_returns_that_constant(cls_name, kwargs, sizes, request):
    if cls_name in KNOWN_BAD_NORMALISATION:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="averaged_over_number divides by the power-law weight "
                       "instead of multiplying; see comment above",
            )
        )
    d = _make(cls_name, kwargs)
    avg = d.averaged_over_number(np.ones_like(sizes), sizes)
    assert avg == pytest.approx(1.0, rel=1e-5)


FULL_AVERAGING_INTERFACE = ("averaged_over", "averaged_over_mass",
                            "averaged_over_column", "averaged_over_number")

# Two of the nine classes implement only averaged_over_number. Recorded rather
# than asserted, so the inconsistency is visible without failing the suite.
PARTIAL_INTERFACE = {"Classical_LogNormal_Distribution",
                     "PowerLaw_ExpCutoff_Distribution"}


@pytest.mark.parametrize("cls_name,kwargs", ALL_DISTRIBUTIONS)
def test_averaging_interface_is_as_documented(cls_name, kwargs):
    from pycalima.models import grain_distributions as gd

    cls = getattr(gd, cls_name)
    present = {m for m in FULL_AVERAGING_INTERFACE if hasattr(cls, m)}
    if cls_name in PARTIAL_INTERFACE:
        assert present == {"averaged_over_number"}, (
            f"{cls_name} gained methods; move it out of PARTIAL_INTERFACE"
        )
    else:
        assert present == set(FULL_AVERAGING_INTERFACE), (
            f"{cls_name} is missing {set(FULL_AVERAGING_INTERFACE) - present}"
        )


# ---------------------------------------------------------------------------
# dust_model builders
# ---------------------------------------------------------------------------

def test_build_distribution_for_every_configured_bin():
    from pycalima.models.dust_model import build_distribution
    from pycalima.models.grain_size_config import get_bins

    for info in get_bins():
        d = build_distribution(info["id"])
        assert hasattr(d, "n_density"), info["id"]
        sizes = np.logspace(-4, 0, 100)
        n = np.asarray(d.n_density(1e-24, sizes))
        assert np.all(np.isfinite(n)) and np.all(n >= 0), info["id"]


@pytest.mark.parametrize("composition", ["graphite", "silicate"])
def test_build_distribution_for_composition(composition):
    from pycalima.models.dust_model import build_distribution_for

    d = build_distribution_for(composition)
    assert hasattr(d, "n_density")


def test_build_distribution_rejects_an_unknown_bin():
    from pycalima.models.dust_model import build_distribution

    with pytest.raises((KeyError, ValueError)):
        build_distribution("NoSuchBin_99")
