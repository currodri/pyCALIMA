"""Invariants of the SSP chemical-evolution tables (pycalima.gce)."""

from __future__ import annotations

import numpy as np
import pytest

from pycalima.gce import imf, lifetimes, snia, ssp, stellar_dust, stellar_yields
from pycalima.gce.stellar_yields import ELEMENTS

YIELD_SETS = ("lc18_karakas", "kobayashi11")
T_HUBBLE = 1.37e10


@pytest.fixture(scope="module", params=YIELD_SETS)
def table(request):
    return ssp.build_ssp_table(request.param)


# -- IMF ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["chabrier", "kroupa", "salpeter"])
def test_imf_is_normalised_to_unit_mass(name):
    phi = imf.create_imf(name)
    m = np.logspace(-1, 2, 200001)
    assert np.trapezoid(m * phi(m), m) == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("name, breaks", [("chabrier", [1.0]), ("kroupa", [0.5])])
def test_imf_is_continuous_at_breaks(name, breaks):
    phi = imf.create_imf(name)
    for b in breaks:
        assert phi(b * (1 - 1e-9)) == pytest.approx(phi(b * (1 + 1e-9)), rel=1e-6)


def test_imf_vanishes_outside_its_mass_range():
    phi = imf.create_imf("chabrier", m_low=0.1, m_up=100.0)
    assert phi(0.05) == 0.0 and phi(150.0) == 0.0


# -- lifetimes ------------------------------------------------------------

@pytest.mark.parametrize("Z", [0.0, 1e-4, 0.004, 0.02, 0.05])
def test_lifetime_decreases_with_mass_and_inverts(Z):
    m = np.logspace(np.log10(0.6), 2, 300)
    t = lifetimes.lifetime(m, Z)
    assert np.all(np.diff(t) < 0)
    np.testing.assert_allclose(lifetimes.dying_mass(t, Z), m, rtol=1e-8)


def test_no_star_dies_before_the_shortest_lifetime():
    assert np.isinf(lifetimes.dying_mass(1e5, 0.02))


# -- per-star yield tables --------------------------------------------------

@pytest.mark.parametrize("name", YIELD_SETS)
def test_every_yield_table_is_complete_and_physical(name):
    ys = stellar_yields.load_yield_set(name)
    assert len(ys.agb) == len(ys.snii) == len(ys.Z)
    for table_ in ys.agb + ys.snii:
        assert table_.masses.size >= 5, table_.source
        assert np.all(np.isfinite(table_.ejecta)) and np.all(table_.ejecta >= 0)
        assert np.all(table_.total <= table_.masses), table_.source
        # tracked elements are a subset of everything ejected
        assert np.all(table_.ejecta.sum(axis=1) <= table_.total * 1.01 + 1e-8), table_.source


def test_out_of_range_masses_keep_ejected_fractions():
    t = stellar_yields.load_yield_set("lc18_karakas").snii[3]
    ej_hi, tot_hi = t.per_star([2 * t.masses[-1]])
    np.testing.assert_allclose(ej_hi[0], 2 * t.ejecta[-1])
    assert tot_hi[0] == pytest.approx(2 * t.total[-1])


# -- SN Ia and dust -----------------------------------------------------------

def test_w7_ejecta_are_a_subset_of_a_chandrasekhar_mass():
    assert 1.3 < snia.W7_TOTAL < 1.45
    assert snia.W7_EJECTA.sum() <= snia.W7_TOTAL
    assert snia.W7_EJECTA[ELEMENTS.index("Fe")] == pytest.approx(0.743, rel=0.01)


@pytest.mark.parametrize("slope", [1.0, 1.1])
def test_dtd_normalisation_and_onset(slope):
    dtd = snia.PowerLawDTD(slope=slope)
    assert dtd.cumulative(dtd.t_min * 0.5) == 0.0
    assert dtd.cumulative(dtd.t_norm) == pytest.approx(dtd.n_per_msun)


@pytest.mark.parametrize("preset", sorted(stellar_dust.PRESETS))
def test_condensed_dust_never_exceeds_available_elements(preset):
    rng = np.random.default_rng(1)
    ej = rng.uniform(0, 1e-2, size=(500, len(ELEMENTS)))
    for channel in ("agb", "snii"):
        carbon, sil = stellar_dust.condense(ej, channel, preset)
        assert np.all(carbon >= 0) and np.all(sil >= 0)
        assert np.all(carbon <= ej[:, ELEMENTS.index("C")])
        for el, frac in stellar_dust.OLIVINE.items():
            assert np.all(sil * frac <= ej[:, ELEMENTS.index(el)] * (1 + 1e-12))


def test_carbon_rich_agb_winds_make_no_silicate():
    ej = np.zeros(len(ELEMENTS))
    ej[ELEMENTS.index("C")], ej[ELEMENTS.index("O")] = 2e-2, 1e-2
    ej[[ELEMENTS.index(e) for e in ("Mg", "Si", "Fe")]] = 1e-3
    carbon, sil = stellar_dust.condense(ej, "agb")
    assert carbon > 0 and sil == 0


# -- SSP tables ---------------------------------------------------------------

def test_ssp_cumulative_quantities_never_decrease(table):
    for arr in (table.summed("total"), table.summed("ejecta"), table.summed("dust"),
                table.n_sn["snii"], table.n_sn["snia"]):
        assert np.all(np.diff(arr, axis=1) >= -1e-12)
        assert np.all(arr[:, 0] == 0.0)


def test_ssp_mass_budget(table):
    i = np.searchsorted(table.ages, T_HUBBLE)
    returned = table.summed("total")[:, i]
    # returned fraction of a Chabrier population after a Hubble time
    assert np.all((returned > 0.35) & (returned < 0.5))
    ejecta = table.summed("ejecta")[:, i]
    assert np.all(ejecta.sum(axis=1) <= returned * 1.01)
    metals = sum(table.metals[c] for c in ssp.CHANNELS)[:, i]
    assert np.all((metals > 0) & (metals < returned))


def test_ssp_supernova_numbers(table):
    i = np.searchsorted(table.ages, T_HUBBLE)
    # stars above 8 Msun per Msun formed, Chabrier 0.1-100 Msun
    assert np.all((table.n_sn["snii"][:, i] > 0.009) & (table.n_sn["snii"][:, i] < 0.013))
    assert table.n_sn["snia"][0, i] == pytest.approx(1.3e-3, rel=0.02)
    # core-collapse SNe are all over by 50 Myr
    j = np.searchsorted(table.ages, 5e7)
    np.testing.assert_allclose(table.n_sn["snii"][:, j], table.n_sn["snii"][:, -1])


def test_ssp_released_is_a_difference_of_cumulatives(table):
    Z = float(table.Z[2])
    a = table.released(Z, 0.0, 1e8)
    b = table.released(Z, 1e8, 1e9)
    ab = table.released(Z, 0.0, 1e9)
    assert a["total"] + b["total"] == pytest.approx(ab["total"])
    np.testing.assert_allclose(a["ejecta"] + b["ejecta"], ab["ejecta"], rtol=1e-12)
    assert ab["metals"] == pytest.approx(ab["total"] - ab["ejecta"][0] - ab["ejecta"][1])


def test_ssp_round_trips_through_disk(table, isolated_env):
    path = table.save(ssp.default_table_path(table.meta["yield_set"]))
    assert path.is_relative_to(isolated_env)
    back = ssp.SSPTable.load(path)
    np.testing.assert_array_equal(back.summed("ejecta"), table.summed("ejecta"))
    assert back.meta["yield_set"] == table.meta["yield_set"]
