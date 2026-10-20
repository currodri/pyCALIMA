"""Gate G4: Micelotta, Jones & Tielens (2011) reproduced from method_cards/micelotta2011.toml."""

import pytest

from pycalima.pahchem import micelotta2011 as m

T = m.CARD["targets"]


@pytest.mark.parametrize("t0,key", [(4.5, "electron_lifetime_t0_4p5"), (7.5, "electron_lifetime_t0_7p5"),
                                    (15.0, "electron_lifetime_t0_15")])
def test_electron_lifetimes_reproduce_table2_without_the_eq31_factor(t0, key):
    assert m.electron_lifetime_yr(t0, orientation=1.0) == pytest.approx(T[key]["value"], rel=0.06)


def test_the_printed_eq31_factor_doubles_the_electron_lifetimes():
    """The recorded discrepancy (card: electrons.orientation_discrepancy)."""
    ratio = m.electron_lifetime_yr(7.5) / m.electron_lifetime_yr(7.5, orientation=1.0)
    assert ratio == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize("f,key,tol", [(0.5, "crossover_nc_f0p5", 0.10), (1.0, "crossover_nc_f1", 0.15)])
def test_cr_lifetime_equals_the_shock_lifetime_at_the_stated_size(f, key, tol):
    """Sect. 7.1: CRs dominate PAH destruction below N_C ~ 150 (f = 0.5) and ~280 (f = 1)."""
    nc = m.crossover_nc(T["shock_lifetime"]["value"], e0=4.6, t0=7.5, f=f)
    assert nc == pytest.approx(T[key]["value"], rel=tol)


def test_he_and_cno_dominate_destruction_of_small_pahs():
    """Abstract: 'He and the CNO group being the more effective projectiles'."""
    tau = {k: m.ion_lifetime_yr(50, 4.58, 1.0, ions=(k,)) for k in m.IONS}
    assert min(tau, key=tau.get) in ("He", "CNO")
    assert tau["H"] > 10 * min(tau.values())
