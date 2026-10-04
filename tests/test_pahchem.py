"""pahchem: method cards and the gates of the benchmark ladder.

A method card records the inputs of a published PAH model. The invariants here
keep the cards honest: every number names where in the paper it comes from,
and every card cites a bibliography entry that exists.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

CARDS = sorted(
    p for p in resources.files("pycalima.pahchem").joinpath("method_cards").iterdir()
    if p.name.endswith(".toml")
)
REFS_BIB = Path(__file__).resolve().parents[1] / "docs" / "refs.bib"
REQUIRED_SECTIONS = {"card", "species", "dissociation", "targets"}
NUMERIC_KEYS = {"value", "E_act", "efficiency"}


def _load(card):
    with card.open("rb") as fh:
        return tomllib.load(fh)


def _inline_tables(node, path=()):
    """Yield (path, table) for every table holding a number to be sourced."""
    if isinstance(node, dict):
        if NUMERIC_KEYS & node.keys():
            yield path, node
        for key, value in node.items():
            yield from _inline_tables(value, path + (key,))


def test_there_are_method_cards():
    assert {c.name for c in CARDS} >= {"andrews16.toml", "montillaud13.toml"}


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_card_has_required_sections(card):
    data = _load(card)
    assert REQUIRED_SECTIONS <= data.keys(), REQUIRED_SECTIONS - data.keys()
    assert data["card"]["citation"]


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_card_bib_key_is_in_the_bibliography(card):
    key = _load(card)["card"]["bib_key"]
    assert re.search(r"@\w+\{" + re.escape(key) + ",", REFS_BIB.read_text())


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_every_number_has_a_source_and_confidence(card):
    tables = list(_inline_tables(_load(card)))
    assert tables
    for path, table in tables:
        where = ".".join(path)
        assert table.get("source"), f"{card.name}: {where} has no source"
        assert table.get("confidence") in {"A", "B", "C", "D", "E"}, (
            f"{card.name}: {where} has no A-E confidence")


# ===========================================================================
# M1: statistical mechanics (level 0 of the benchmark ladder)
# ===========================================================================

import numpy as np

from pycalima.pahchem import dos
from pycalima.pahchem.molecule import PAHSpecies


def _random_modes(n, seed=0):
    """Incommensurate frequencies spread like a PAH's, 50-3100 cm^-1."""
    rng = np.random.default_rng(seed)
    return np.sort(np.exp(rng.uniform(np.log(50.0), np.log(3100.0), n)))


@pytest.fixture
def pah_states(model_data):
    states = model_data / "PAH_states"
    if not (states / "C54H18_0.dat").exists():
        pytest.skip("model_data/PAH_states/ has no PAHdb mode files")
    return states


@pytest.mark.parametrize("freqs", [
    [1000],
    [300, 410, 520, 777, 900, 1013, 1250, 1600],
    [150, 150, 150, 433, 433, 1200, 1201, 2999, 3001, 3050],
])
def test_beyer_swinehart_equals_direct_count(freqs):
    """With frequencies on the grid, the two counts must agree exactly."""
    e_bs, c_bs = dos.beyer_swinehart(freqs, 6000.0)
    e_dc, c_dc = dos.direct_count(freqs, 6000.0)
    assert np.array_equal(e_bs, e_dc)
    assert np.array_equal(c_bs, c_dc)


def test_whitten_rabinovitch_density_is_derivative_of_sum():
    """Holds to 3 %: the published piecewise w(E') has a derivative jump at
    E' = 1 (measured 2.6 % there), and is smooth elsewhere."""
    f = _random_modes(30)
    E = np.linspace(0.5, 4.0, 3001) * 0.5 * f.sum()
    log_sum, log_rho = dos.whitten_rabinovitch(f, E)
    numeric = np.log(np.gradient(np.exp(log_sum - log_sum[0]), E)) + log_sum[0]
    away = np.abs(E / (0.5 * f.sum()) - 1.0) > 0.05
    assert np.max(np.abs(log_rho - numeric)[2:-2]) < 0.03
    assert np.max(np.abs(log_rho - numeric)[2:-2][away[2:-2]]) < 0.005


def test_whitten_rabinovitch_within_3_percent_above_zpe_for_30_modes():
    """Measured 2.2 % at E = E_z for PAH-like (50-3100 cm^-1) frequencies."""
    f = _random_modes(30)
    ez = 0.5 * f.sum()
    E, counts = dos.beyer_swinehart(f, 4.0 * ez)
    N = np.cumsum(counts)
    for frac in (1.0, 2.0, 3.9):
        i = int(frac * ez)
        log_sum, _ = dos.whitten_rabinovitch(f, E[i])
        assert abs(np.exp(log_sum - np.log(N[i])) - 1.0) < 0.03, frac


@pytest.mark.parametrize("T", [50.0, 300.0, 1000.0, 5000.0])
def test_canonical_temperature_inverts_thermal_energy(T):
    f = _random_modes(60)
    assert dos.canonical_temperature(f, dos.thermal_energy(f, T)) == pytest.approx(T, rel=1e-8)


def test_species_rejects_a_mode_count_other_than_3n_minus_6():
    with pytest.raises(ValueError, match="3N-6"):
        PAHSpecies(nc=24, nh=12, charge=0, freqs_cm=np.ones(100))


def test_species_loads_pahdb_states_file(pah_states):
    sp = PAHSpecies.from_states_file(pah_states / "C54H18_0.dat")
    assert (sp.nc, sp.nh, sp.charge, sp.n_modes) == (54, 18, 0, 210)
    assert sp.formula == "C54H18"
    assert 1.0 < sp.zpe_cm / dos.CM_PER_EV < 20.0


@pytest.mark.parametrize("name, tolerance", [("C24H12_0.dat", 0.04), ("C54H18_0.dat", 0.10)])
def test_whitten_rabinovitch_error_grows_with_mode_count(pah_states, name, tolerance):
    """Measured: -3 % (C24H12) and -9 % (C54H18) at E = E_z, so WR is a check,
    not the production state count. A larger error means a regression."""
    sp = PAHSpecies.from_states_file(pah_states / name)
    E, counts = dos.beyer_swinehart(sp.freqs_cm, 2.0 * sp.zpe_cm)
    N = np.cumsum(counts)
    for frac in (1.0, 2.0):
        i = int(frac * sp.zpe_cm)
        log_sum, _ = dos.whitten_rabinovitch(sp.freqs_cm, E[i])
        assert abs(np.exp(log_sum - np.log(N[i])) - 1.0) < tolerance, frac


@pytest.mark.parametrize("name", ["C24H12_0.dat", "C54H18_0.dat", "C96H24_0.dat"])
def test_microcanonical_temperature_tends_to_canonical(pah_states, name):
    """T_micro / T_can - 1 = O(1/s); classically exactly 1/(s-1).

    Measured: within 3 % from 1 eV up, with s (T_micro/T_can - 1) falling
    monotonically to 1.1-1.3 at 20 eV.
    """
    sp = PAHSpecies.from_states_file(pah_states / name)
    s = sp.n_modes
    E, counts = dos.beyer_swinehart(sp.freqs_cm, 21.0 * dos.CM_PER_EV)
    T_micro = dos.microcanonical_temperature(E, dos.smoothed_log_density(E, counts, 200.0))
    rel = []
    for e_ev in (1.0, 2.0, 5.0, 10.0, 20.0):
        i = int(e_ev * dos.CM_PER_EV)
        rel.append(T_micro[i] / dos.canonical_temperature(sp.freqs_cm, E[i]) - 1.0)
    rel = np.array(rel)
    assert np.all((rel > 0.0) & (rel < 0.03)), rel
    assert np.all(np.diff(rel) < 0), rel                # approaches the limit from above
    assert 0.8 < s * rel[-1] < 1.5, s * rel


def test_thermal_energy_matches_tielens_table_per_mode(pah_states):
    """Tielens (2005) E-T relation for circumcoronene is energy per mode.

    The legacy pah_temperature.py scaled it by N_C (54) instead of the number
    of modes (210). Measured agreement with U(T)/(3N-6): 2-7 %.
    """
    from pycalima._paths import get_external_data_path

    sp = PAHSpecies.from_states_file(pah_states / "C54H18_0.dat")
    table = np.loadtxt(get_external_data_path(
        "E-Trelation_neutral_circumcoronene_Tielens2005.csv"), delimiter=",")
    T, e_per_mode_ev = table[table[:, 0] >= 150.0].T
    ours = dos.thermal_energy(sp.freqs_cm, T) / sp.n_modes / dos.CM_PER_EV
    assert np.all(np.abs(ours / e_per_mode_ev - 1.0) < 0.10)
