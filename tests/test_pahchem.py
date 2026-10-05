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


# ===========================================================================
# M2: unimolecular rates (levels 0-1)
# ===========================================================================

from pycalima.pahchem import rates

EV = dos.CM_PER_EV


@pytest.fixture(scope="module")
def coronene_grid():
    """C24H12-like modes; the identities below hold for any mode set.

    The grid reaches 40 eV so that the Boltzmann weight at 2000 K (mean
    energy ~18 eV for 102 modes) is not truncated.
    """
    f = _random_modes(102, seed=3)
    E, counts = dos.beyer_swinehart(f, 40.0 * EV)
    return f, E, counts


def _ln_q(freqs, T):
    return -np.sum(np.log(-np.expm1(-np.asarray(freqs) / (dos.KB_CM * T))))


@pytest.mark.parametrize("T", [500.0, 1000.0, 2000.0])
def test_inverse_laplace_arrhenius_averages_to_arrhenius(coronene_grid, T):
    """<k(E)>_T = A exp(-E0/kT) exactly: the form is the inverse Laplace
    transform of the Arrhenius law."""
    _, E, counts = coronene_grid
    k = rates.inverse_laplace_arrhenius(E, counts, 6.8e17, 4.8 * EV)
    expected = 6.8e17 * np.exp(-4.8 * EV / (dos.KB_CM * T))
    assert rates.thermal_rate(E, counts, k, T) == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("T", [500.0, 1000.0, 2000.0])
def test_rrkm_averages_to_transition_state_theory(coronene_grid, T):
    """<k(E)>_T = (kT/h) Q_TS/Q exp(-E0/kT).

    Exact for the frequencies the grid actually counts, which Beyer-Swinehart
    rounds to whole grains: that rounding alone moves a ~30 cm^-1 mode's Q by
    ~1 %, so Q is evaluated on the rounded frequencies.
    """
    f, E, counts = coronene_grid
    ts = rates.transition_state_for_entropy(f, 3050.0, 44.8)
    k = rates.rrkm(E, counts, ts, 4.6 * EV)
    tst = (rates.KB_J_K * T / rates.H_J_S
           * np.exp(_ln_q(np.rint(ts), T) - _ln_q(np.rint(f), T))
           * np.exp(-4.6 * EV / (dos.KB_CM * T)))
    assert rates.thermal_rate(E, counts, k, T) == pytest.approx(tst, rel=0.005)


def test_transition_state_has_the_requested_entropy(coronene_grid):
    f, _, _ = coronene_grid
    for n_scaled in (None, 5):
        for dS in (-53.1, 0.0, 44.8):
            ts = rates.transition_state_for_entropy(f, 3050.0, dS, n_scaled=n_scaled)
            assert ts.size == f.size - 1
            got = (rates.vibrational_entropy(ts, 1000.0) - rates.vibrational_entropy(f, 1000.0))
            assert got * rates.R_J_MOL_K == pytest.approx(dS, abs=1e-6)


def test_rates_vanish_below_threshold_and_grow_above(coronene_grid):
    f, E, counts = coronene_grid
    i0 = int(4.6 * EV)
    for k in (rates.inverse_laplace_arrhenius(E, counts, 1e16, 4.6 * EV),
              rates.rrkm(E, counts, rates.transition_state_for_entropy(f, 3050.0, 10.0), 4.6 * EV),
              rates.rrk(E, f.size, 1e16, 4.6 * EV)):
        assert np.all(k[:i0 - 1] == 0.0)
        assert np.all(np.diff(k[i0 + 200:]) >= 0.0)


def test_rrk_tends_to_the_prefactor():
    assert rates.rrk(1e12, 100, 3e16, 4.6 * EV) == pytest.approx(3e16, rel=1e-4)


def test_effective_temperature_form_matches_legacy(pah_states):
    """Same physics as models.PAH_photophysics.pah_mol_data."""
    from pycalima.models.PAH_photophysics.pah_mol_data import compute_rrkm_dissociation_rate

    path = pah_states / "C54H18_0.dat"
    sp = PAHSpecies.from_states_file(path)
    for e_ev in (6.0, 10.0, 20.0):
        ours = rates.arrhenius_effective_temperature(sp.freqs_cm, e_ev * EV, 4.6 * EV, 44.8)
        assert ours == pytest.approx(compute_rrkm_dissociation_rate(str(path), e_ev, 4.6, 44.8), rel=1e-5)


def test_montillaud13_fig3_coronene(pah_states):
    """k = A rho(E-E0)/rho(E), A = 6.8e17 s^-1, E0 = 4.8 eV (Montillaud+13 eq. 3).

    Measured against the digitised Fig. 3: -0.11 to -0.26 dex at 7-15 eV, a
    near-constant offset from using PAHdb neutral modes where Montillaud used
    Malloci+07 modes (method card: modes).
    """
    import pandas as pd
    from pycalima._paths import get_external_data_path

    sp = PAHSpecies.from_states_file(pah_states / "C24H12_0.dat")
    E, counts = dos.beyer_swinehart(sp.freqs_cm, 16.0 * EV)
    k = rates.inverse_laplace_arrhenius(E, counts, 6.8e17, 4.8 * EV)
    fig = pd.read_csv(get_external_data_path("dissoc_rate_coronene_Montillaud.csv"), skiprows=1)
    x, y = fig.iloc[:, 0].dropna().to_numpy(), fig.iloc[:, 1].dropna().to_numpy()
    dex = np.log10(k[(x * EV).astype(int)] / y)
    assert np.all(np.abs(dex) < 0.3), dex
    assert np.ptp(dex) < 0.2, dex                  # shape: offset, not slope


@pytest.mark.parametrize("name", ["C24H12", "C54H18", "C96H24"])
@pytest.mark.parametrize("channel, E_act, dS", [("H-loss", 4.6, 44.8), ("H2-loss", 3.52, -53.1)])
def test_andrews16_fig5_rates(pah_states, name, channel, E_act, dS):
    """Andrews+16 Table 2 parameters with the T_e form, against Fig. 5.

    Points within 1.5 eV of E_act are dropped: the curves are steep there and
    a small digitising error in E is a large one in k. Measured elsewhere:
    median |dex| <= 0.22 (C96 sits uniformly ~0.2 dex high), max 0.51.
    """
    from pycalima._paths import get_external_data_path

    sp = PAHSpecies.from_states_file(pah_states / f"{name}_0.dat")
    d = np.loadtxt(get_external_data_path(f"{channel}_{name}_Andrews16.csv"), delimiter=",")
    d = d[d[:, 0] > E_act + 1.5]
    k = rates.arrhenius_effective_temperature(sp.freqs_cm, d[:, 0] * EV, E_act * EV, dS)
    dex = np.abs(np.log10(k / d[:, 1]))
    assert np.median(dex) < 0.25 and dex.max() < 0.55, np.round(dex, 2)


# ===========================================================================
# M2: laboratory gates (level 2) -- West+18 iPEPICO, Jochims+94 PIMS
# ===========================================================================

from pycalima.pahchem import labdata

# Reactions whose ESI k(E) was computed with the Table 1 E0 (E0 fitted to the
# ESI k(E) within 0.01 eV of Table 1), and the integer reaction-path degeneracy
# that k(E) implies. Others differ: R16 implies E0 = 2.61 eV (Table 1: 2.63);
# R5, R8-R10, R14, R15, R18, R20 by 0.03-0.28 eV; R6, R12, R13, R17, R19 and
# R21-R23 by ~1 eV with a non-integer degeneracy, so their ESI columns do not
# match Table 1.
WEST_CLEAN = {1: 1, 2: 6, 3: 2, 4: 1, 7: 1, 24: 2, 25: 1}


@pytest.fixture(scope="module")
def west_esi():
    from pycalima._datasets import DatasetUnavailable

    try:
        return labdata.west2018_esi()
    except DatasetUnavailable:
        pytest.skip("west2018-esi not installed (calima-fetch-data import west2018-esi ...)")


def test_jochims94_table1_obeys_their_eq1():
    """E_int = AP - IP + <E_th> (eq. 1): a check of the transcription.

    Rows flagged c (isomerisation) or d (isotopic shift) add a term; skipped.
    """
    t = labdata.jochims94_table(1)
    t = t[t["flag"].isna() & t["ip_eV"].notna()]
    checked = 0
    for ch in ("H", "H2", "C2H2"):
        rows = t[t[f"ap_{ch}_eV"].notna() & t[f"eint_{ch}_eV"].notna()]
        resid = rows[f"ap_{ch}_eV"] - rows["ip_eV"] + rows["eth_eV"] - rows[f"eint_{ch}_eV"]
        assert np.all(np.abs(resid) <= 0.05), resid
        checked += len(rows)
    assert checked >= 15


def test_rrk_reproduces_jochims94_table2():
    """k = nu ((E - E0)/E)^(s-1), nu = 1e16 s^-1, E0 = 2.8 eV (their eq. 2).

    Measured: every printed E_int(calc) and E_crit(calc) to 0.01 eV.
    """
    from scipy.optimize import brentq

    for name, row in labdata.jochims94_table(2).iterrows():
        s = int(row["s_minus_1"]) + 1
        for k, col in ((1e4, "eint_calc_eV"), (1e2, "ecrit_calc_eV")):
            E = brentq(lambda e: np.log(rates.rrk(e * EV, s, 1e16, 2.8 * EV)) - np.log(k), 2.81, 60.0)
            assert E == pytest.approx(row[col], abs=0.006), (name, col, E)


def test_west2018_table1_is_complete():
    t = labdata.west2018_table1()
    assert list(t.index) == list(range(1, 26))
    assert (t["E0_err_eV"] > 0).all() and (t["dS_err_J_K_mol"] > 0).all()
    assert t["ie_eV"].between(6.5, 8.5).all()


@pytest.mark.parametrize("n, sigma", sorted(WEST_CLEAN.items()))
def test_rrkm_reproduces_west2018_esi_rates(west_esi, n, sigma):
    """Their B3LYP frequencies and Table 1 E0 in our rrkm give their k(E).

    Measured: median 0.0002-0.01 dex over 10 eV, the larger where Table 1
    rounds E0 to 0.01 eV (R7). A code check of rrkm against minimal-PEPICO.
    """
    r = west_esi[n]
    E0 = labdata.west2018_table1().loc[n, "E0_eV"] * EV
    grid, counts = dos.beyer_swinehart(r.reactant_freqs_cm, r.energy_cm.max() + 10.0)
    k = rates.rrkm(grid, counts, r.ts_freqs_cm, E0, degeneracy=sigma, window_cm=1.0)
    ours = k[np.rint(r.energy_cm).astype(int)]
    ok = (r.k_s > 0) & (ours > 0)
    dex = np.abs(np.log10(ours[ok] / r.k_s[ok]))
    assert ok.sum() > 100
    assert np.median(dex) < 0.02 and dex.max() < 0.05, (np.median(dex), dex.max())


@pytest.mark.parametrize("n", [1, 4, 7, 24, 25])
def test_west2018_transition_states_follow_their_convention(west_esi, n):
    """Their TS: the reactant less one C-H stretch, with the 5 lowest modes
    scaled to give dS(1000 K). Our builder reproduces it to 2 %."""
    r = west_esi[n]
    rx, ts = np.sort(r.reactant_freqs_cm), np.sort(r.ts_freqs_cm)
    dS = labdata.west2018_table1().loc[n, "dS_J_K_mol"]
    theirs = (rates.vibrational_entropy(ts, 1000.0) - rates.vibrational_entropy(rx, 1000.0)) * rates.R_J_MOL_K
    assert theirs == pytest.approx(dS, abs=1.0)
    ours = rates.transition_state_for_entropy(rx, 3100.0, dS, n_scaled=5)
    assert np.max(np.abs(ours / ts - 1.0)) < 0.02
