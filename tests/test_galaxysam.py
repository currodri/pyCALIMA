"""The galaxysam subpackage: IMFs, abundances and stellar-yield tables.

galaxysam already resolved its own data correctly before packaging (it used
``Path(__file__).parent`` rather than walking up to a repository root), so
these tests mostly pin the numerical contract of the IMFs and confirm the
bundled yield tables are reachable from the installed package.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycalima import galaxysam


# ---------------------------------------------------------------------------
# package surface
# ---------------------------------------------------------------------------

def test_version_is_exposed():
    assert isinstance(galaxysam.__version__, str) and galaxysam.__version__


def test_documented_submodules_are_re_exported():
    for name in ("constants", "imf", "sn1a", "yield_models", "galaxy_sam", "plotting"):
        assert hasattr(galaxysam, name), f"galaxysam.{name} is not re-exported"


def test_routines_mapping_parses_and_imports():
    """This module held the only syntax error in the repository: an unmatched
    paren at `print_mapping())`, which meant it did not parse on any Python
    version and would break compileall during a build."""
    import pycalima.galaxysam.ROUTINES_MAPPING as rm

    assert hasattr(rm, "print_mapping")


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

def test_solar_metallicities_are_plausible():
    from pycalima.galaxysam import constants as c

    assert 0.005 < c.ZSUN_ASPLUND < 0.03
    assert 0.005 < c.ZSUN_ANDERS < 0.03


def test_abundance_tables_are_positive_and_non_empty():
    from pycalima.galaxysam import constants as c

    for name in ("ASPLUND_ABUNDANCES", "ANDERS_GREVESSE_ABUNDANCES"):
        table = getattr(c, name)
        assert table, f"{name} is empty"
        for element, value in table.items():
            assert isinstance(element, str)
            assert value > 0, f"{name}[{element}] = {value}"


def test_hydrogen_dominates_the_abundance_table():
    from pycalima.galaxysam import constants as c

    table = c.ASPLUND_ABUNDANCES
    if "H" not in table:
        pytest.skip("no hydrogen entry to compare against")
    assert table["H"] == max(table.values())


def test_element_lists_are_unique():
    from pycalima.galaxysam import constants as c

    for name in ("ELEMENTS_KOBAYASHI", "ELEMENTS_LC18"):
        elements = list(getattr(c, name))
        assert elements
        assert len(elements) == len(set(elements)), f"{name} has duplicates"


def test_hubble_time_and_mass_separatrix_are_positive():
    from pycalima.galaxysam import constants as c

    assert c.HUBBLE_TIME > 0
    assert c.MASS_SEPARATRIX > 0


# ---------------------------------------------------------------------------
# initial mass functions
# ---------------------------------------------------------------------------

IMF_NAMES = ["salpeter", "chabrier"]


@pytest.mark.parametrize("name", IMF_NAMES)
def test_create_imf_returns_a_usable_imf(name):
    from pycalima.galaxysam.imf import create_imf

    imf = create_imf(name)
    for attr in ("phi", "mmin", "mmax", "normalize"):
        assert hasattr(imf, attr), f"{name} IMF is missing {attr}"
    assert imf.mmin > 0
    assert imf.mmax > imf.mmin


@pytest.mark.parametrize("name", IMF_NAMES)
def test_imf_is_positive_across_its_mass_range(name):
    from pycalima.galaxysam.imf import create_imf

    imf = create_imf(name)
    masses = np.logspace(np.log10(imf.mmin), np.log10(imf.mmax), 200)
    phi = np.asarray([imf.phi(m) for m in masses], dtype=float)
    assert np.all(np.isfinite(phi)), f"{name} IMF is not finite"
    assert np.all(phi > 0), f"{name} IMF is not strictly positive"


@pytest.mark.parametrize("name", IMF_NAMES)
def test_imf_decreases_towards_high_mass(name):
    """Every standard IMF is bottom-heavy, so phi must fall from low to high
    mass overall."""
    from pycalima.galaxysam.imf import create_imf

    imf = create_imf(name)
    lo = float(imf.phi(max(imf.mmin * 1.5, 0.2)))
    hi = float(imf.phi(min(imf.mmax * 0.5, 50.0)))
    assert hi < lo, f"{name}: phi(high mass) {hi:.3e} >= phi(low mass) {lo:.3e}"


def test_salpeter_slope_matches_the_published_value():
    from pycalima.galaxysam.imf import SalpeterIMF

    imf = SalpeterIMF()
    # stored signed: alpha = -2.35 for dN/dm ~ m^-2.35
    assert float(imf.alpha) == pytest.approx(-2.35, rel=1e-6)


def test_salpeter_is_a_power_law_of_the_expected_slope():
    """phi(2m)/phi(m) must equal 2^alpha for a pure power law."""
    from pycalima.galaxysam.imf import SalpeterIMF

    imf = SalpeterIMF()
    ratio = float(imf.phi(2.0)) / float(imf.phi(1.0))
    assert ratio == pytest.approx(2.0 ** float(imf.alpha), rel=1e-6)


@pytest.mark.parametrize("alpha", [-1.3, -2.35, -2.7])
def test_salpeter_slope_is_configurable(alpha):
    from pycalima.galaxysam.imf import SalpeterIMF

    imf = SalpeterIMF(alpha=alpha)
    ratio = float(imf.phi(4.0)) / float(imf.phi(1.0))
    assert ratio == pytest.approx(4.0 ** alpha, rel=1e-6)


def test_create_imf_rejects_an_unknown_name():
    from pycalima.galaxysam.imf import create_imf

    with pytest.raises((ValueError, KeyError)):
        create_imf("not_an_imf")


def test_imf_weighted_average_of_a_constant_is_that_constant():
    from pycalima.galaxysam.imf import create_imf, imf_weighted_quantity

    imf = create_imf("salpeter")
    masses = np.logspace(np.log10(imf.mmin), np.log10(imf.mmax), 300)
    got = imf_weighted_quantity(masses, np.ones_like(masses), imf)
    assert float(got) == pytest.approx(1.0, rel=1e-3)


def test_broken_power_law_imf_is_constructible():
    from pycalima.galaxysam.imf import BrokenPowerLawIMF

    imf = BrokenPowerLawIMF(alpha_slopes=[-1.3, -2.35], mass_bounds=[0.1, 0.5, 100.0])
    assert float(imf.phi(0.2)) > 0
    assert float(imf.phi(5.0)) > 0


def test_broken_power_law_is_steeper_above_the_break():
    """A Kroupa-like break must make the high-mass side fall off faster."""
    from pycalima.galaxysam.imf import BrokenPowerLawIMF

    imf = BrokenPowerLawIMF(alpha_slopes=[-1.3, -2.35], mass_bounds=[0.1, 1.0, 100.0])
    low_ratio = float(imf.phi(0.4)) / float(imf.phi(0.2))
    high_ratio = float(imf.phi(20.0)) / float(imf.phi(10.0))
    assert high_ratio < low_ratio


# ---------------------------------------------------------------------------
# type Ia supernovae
# ---------------------------------------------------------------------------

def test_sn1a_model_is_constructible_and_exposes_callables():
    import inspect

    from pycalima.galaxysam import sn1a

    fns = [n for n, o in vars(sn1a).items()
           if (inspect.isfunction(o) or inspect.isclass(o)) and not n.startswith("_")]
    assert fns, "sn1a exposes nothing public"


# ---------------------------------------------------------------------------
# bundled yield tables
# ---------------------------------------------------------------------------

def test_nozawa_dust_yields_load_from_bundled_data():
    """These two .dat files are the only coupling models/yields ever had to
    galaxysam, and they ship inside the package."""
    from pycalima.galaxysam.yield_models import load_nozawa2003_dust_yields

    data = load_nozawa2003_dust_yields()
    assert data is not None
    arr = np.asarray(getattr(data, "values", data))
    assert arr.size > 0


def test_nozawa_dust_size_distribution_loads_from_bundled_data():
    from pycalima.galaxysam.yield_models import load_nozawa2003_dust_dist

    data = load_nozawa2003_dust_dist()
    assert data is not None
    arr = np.asarray(getattr(data, "values", data))
    assert arr.size > 0


def test_default_yield_directories_point_inside_the_package():
    """galaxysam anchors its data on Path(__file__).parent, which is the
    pattern the rest of the package was migrated to."""
    from pathlib import Path

    import pycalima
    from pycalima.galaxysam import yield_models as ym

    pkg = Path(pycalima.__path__[0])
    checked = 0
    for attr in ("DEFAULT_YIELD_DIR", "DEFAULT_KOBAYASHI_RAW_DAT",
                 "DEFAULT_NOZAWA2003_DUST_YIELDS", "DEFAULT_NOZAWA2003_DUST_DIST"):
        if not hasattr(ym, attr):
            continue
        path = Path(str(getattr(ym, attr)))
        assert pkg in path.parents, f"{attr} escapes the package: {path}"
        checked += 1
    assert checked, "no DEFAULT_* data constants found to check"


def test_create_yield_model_rejects_an_unknown_name():
    from pycalima.galaxysam.yield_models import create_yield_model

    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        create_yield_model("not_a_yield_model")
