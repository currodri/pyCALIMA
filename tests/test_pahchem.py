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


# ===========================================================================
# M3: radiation fields, absorption, IR cascade, internal-energy master equation
# ===========================================================================

from pycalima.pahchem import cooling, energy_me, fields, photo

E_PH = np.linspace(5.0, 13.6, 345)


def test_isrf_g0_values():
    """Habing G0: Draine (1978) 1.7 and Mathis et al. (1983) 1.14 in the literature."""
    assert fields.draine1978().g0() == pytest.approx(1.67, abs=0.02)
    assert fields.mathis1983().g0() == pytest.approx(1.13, abs=0.02)


@pytest.mark.parametrize("make", [lambda: fields.draine1978(g0=37.0),
                                  lambda: fields.blackbody(30000.0, 37.0),
                                  lambda: fields.kurucz(15000, 37.0)])
def test_fields_scale_to_the_requested_g0(make):
    f = make()
    assert f.g0() == pytest.approx(37.0, rel=1e-10)
    assert f.scaled(370.0).photon_rate(np.ones_like(E_PH), E_PH) == pytest.approx(
        10 * f.photon_rate(np.ones_like(E_PH), E_PH), rel=1e-10)


def test_li_draine_radius_and_cross_section():
    assert photo.li_draine_radius_um(468) == pytest.approx(1e-3)
    for nc, Z in ((24, 0), (54, 1), (96, 0)):
        s = photo.sigma_li_draine(nc, Z, E_PH)
        assert np.all(np.isfinite(s)) and np.all(s > 0)
        per_c = np.trapezoid(s, E_PH) / nc                 # cm^2 eV per C atom
        assert 2e-17 < per_c < 2e-16, (nc, Z, per_c)


def test_malloci_and_li_draine_agree_for_coronene():
    """Measured: int sigma dE over 5-13.6 eV differs by 18 %."""
    ld = np.trapezoid(photo.sigma_li_draine(24, 0, E_PH), E_PH)
    ma = np.trapezoid(photo.sigma_malloci_coronene(0, E_PH), E_PH)
    assert ld / ma == pytest.approx(1.0, abs=0.3)


@pytest.mark.parametrize("yield_", [lambda E: photo.yield_jochims1996(E, 7.0),
                                    lambda E: photo.yield_lepage2001(E, 7.0)])
def test_ionisation_yields_are_probabilities_and_vanish_below_ip(yield_):
    E = np.linspace(4.0, 20.0, 400)
    y = yield_(E)
    assert np.all((y >= 0) & (y <= 1))
    assert np.all(y[E < 7.0] == 0)


def test_heating_and_ionisation_add_up_to_absorption():
    f = fields.draine1978(g0=10.0)
    sigma = photo.sigma_malloci_coronene(1, E_PH)
    Y = photo.yield_jochims1996(E_PH, 11.5)
    total = f.photon_rate(sigma, E_PH)
    assert (np.trapezoid(photo.heating_spectrum(f, sigma, Y, E_PH), E_PH)
            + photo.ionisation_rate(f, sigma, Y, E_PH)) == pytest.approx(total, rel=1e-10)


def test_single_oscillator_emits_at_n_times_A():
    """One mode with n quanta emits at n A (the v -> v-1 rate is v A)."""
    nu, A = 1000.0, 3.0
    E, counts = dos.beyer_swinehart([nu], 10 * nu)
    k = cooling.mode_emission_rates(E, counts, [nu], [A], window_cm=1.0,
                                    at_cm=np.arange(1, 10) * nu)
    assert np.allclose(k[0], A * np.arange(1, 10))


@pytest.fixture(scope="module")
def toy_cascade():
    """60 PAH-like modes, A_i = 1-10 s^-1, and one channel with E0 = 3 eV."""
    f = _random_modes(60, seed=7)
    A = np.random.default_rng(7).uniform(1.0, 10.0, f.size)
    cm = energy_me.CascadeModel(f, A, e_max_cm=25.0 * EV, de_cm=50.0)
    E, counts = dos.beyer_swinehart(f, 25.0 * EV + 200.0)
    k = rates.inverse_laplace_arrhenius(E, counts, 1e16, 3.0 * EV)
    return cm, {"X": k[np.rint(cm.energy_cm).astype(int)]}


def test_photon_yields_are_probabilities(toy_cascade):
    cm, ch = toy_cascade
    Y = cm.photon_yields(ch)["X"]
    assert np.all((Y >= 0) & (Y <= 1 + 1e-12))
    assert np.all(Y[cm.energy_cm < 3.0 * EV] == 0)
    assert Y[-1] > 0.99                                  # k >> k_IR at 25 eV


def _toy_heating(g0):
    f = fields.draine1978(g0=g0)
    return photo.heating_spectrum(f, photo.sigma_li_draine(20, 0, E_PH), np.zeros_like(E_PH), E_PH)


def test_steady_state_conserves_energy(toy_cascade):
    """Absorbed power = radiated power + energy carried off by dissociation."""
    cm, ch = toy_cascade
    h = _toy_heating(1e4)
    no_loss = cm.steady_state(E_PH, h, {})
    assert no_loss.radiated_power == pytest.approx(no_loss.absorbed_power, rel=1e-6)
    ss = cm.steady_state(E_PH, h, ch)
    carried = ss.probability @ (ch["X"] * cm.energy_cm)
    assert ss.radiated_power + carried == pytest.approx(ss.absorbed_power, rel=1e-6)


def test_steady_state_reduces_to_single_photon_yields_at_low_g0(toy_cascade):
    cm, ch = toy_cascade
    Y = cm.photon_yields(ch)["X"]
    for g0 in (1.0, 10.0):
        h = _toy_heating(g0)
        single = np.trapezoid(h * np.interp(E_PH * EV, cm.energy_cm, Y), E_PH)
        ss = cm.steady_state(E_PH, h, ch)
        assert ss.rates["X"] == pytest.approx(single, rel=0.01)
        assert ss.rates["X"] <= np.trapezoid(h, E_PH)


def test_cooling_tends_to_the_thermal_approximation(pah_states):
    """Measured for C24H12: P_micro/P_thermal = 0.95 at 1 eV, 0.999 at 40 eV."""
    from pycalima.models.PAH_photophysics.pah_mol_data import compute_thermal_ir_rate

    path = pah_states / "C24H12_0.dat"
    sp = PAHSpecies.from_states_file(path)
    E, counts = dos.beyer_swinehart(sp.freqs_cm, 41.0 * EV, 2.0)
    at = np.array([1.0, 3.0, 8.0, 20.0, 40.0]) * EV
    micro = cooling.cooling_rate(sp.freqs_cm, cooling.mode_emission_rates(
        E, counts, sp.freqs_cm, sp.einstein_a, at_cm=at))
    thermal = np.array([cooling.cooling_rate(sp.freqs_cm, cooling.thermal_mode_rates(
        sp.freqs_cm, sp.einstein_a, dos.canonical_temperature(sp.freqs_cm, e))) for e in at])
    legacy = np.array([compute_thermal_ir_rate(str(path), e / EV)[2] * e for e in at])
    assert np.allclose(thermal, legacy, rtol=1e-5)
    ratio = micro / thermal
    assert np.all((ratio > 0.9) & (ratio <= 1.0)) and np.all(np.diff(ratio) > 0), ratio
    assert ratio[-1] > 0.995


@pytest.mark.parametrize("name", ["C24H12", "C96H24"])
def test_andrews16_fig5_ir_rate(pah_states, name):
    """k_IR(E) = (dE/dt)/E (Andrews+16 eq. 3) against Fig. 5.

    Measured: within 0.22 dex (C24H12) and 0.26 dex (C96H24).
    """
    from pycalima._paths import get_external_data_path

    sp = PAHSpecies.from_states_file(pah_states / f"{name}_0.dat")
    d = np.loadtxt(get_external_data_path(f"kIR_{name}_Andrews16.csv"), delimiter=",")
    d = d[d[:, 0] > 0.5]
    E, counts = dos.beyer_swinehart(sp.freqs_cm, d[:, 0].max() * EV + 200.0, 2.0)
    k = cooling.cooling_rate(sp.freqs_cm, cooling.mode_emission_rates(
        E, counts, sp.freqs_cm, sp.einstein_a, at_cm=d[:, 0] * EV)) / (d[:, 0] * EV)
    assert np.all(np.abs(np.log10(k / d[:, 1])) < 0.3)


def test_circumcoronene_becomes_multiphoton(pah_states):
    """C54H18 H loss: single-photon below G0 ~ 10, ~G0^2 above (measured
    enhancement 1.01, 1.5, 6, 56 at G0 = 1, 1e2, 1e3, 1e4)."""
    sp = PAHSpecies.from_states_file(pah_states / "C54H18_0.dat")
    cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=45.0 * EV, de_cm=100.0)
    ch = {"H": rates.arrhenius_effective_temperature(sp.freqs_cm, cm.energy_cm, 4.6 * EV, 44.8)}
    Y = cm.photon_yields(ch)["H"]
    sigma, ion = photo.sigma_li_draine(54, 0, E_PH), photo.yield_jochims1996(E_PH, 6.14)
    enh = []
    for g0 in (1.0, 1e3, 1e4):
        h = photo.heating_spectrum(fields.draine1978(g0=g0), sigma, ion, E_PH)
        single = np.trapezoid(h * np.interp(E_PH * EV, cm.energy_cm, Y), E_PH)
        enh.append(cm.steady_state(E_PH, h, ch).rates["H"] / single)
    assert enh[0] == pytest.approx(1.0, abs=0.03)
    assert 3.0 < enh[1] < 15.0 and 30.0 < enh[2] < 120.0, enh
    assert 0.8 < np.log10(enh[2] / enh[1]) < 1.1, enh       # slope ~ 2 in G0


# --- Monte Carlo check of the master equation (continuous E, no binning) ---

from pycalima.pahchem import montecarlo


def test_monte_carlo_matches_photon_yields(toy_cascade):
    cm, ch = toy_cascade
    Y = np.interp(5.0 * EV, cm.energy_cm, cm.photon_yields(ch)["X"])
    n = 2000
    y = montecarlo.MonteCarlo(cm, ch, seed=1).photon_yields(5.0 * EV, n)["X"]
    assert abs(y - Y) < 3.0 * np.sqrt(Y * (1 - Y) / n), (y, Y)


def test_monte_carlo_matches_multiphoton_steady_state():
    """A barrier (8 eV) that one 5-13.6 eV photon rarely clears: the rate is
    ~4e5 times the single-photon value at G0 = 1e8. Measured MC/ME = 0.95,
    z = -0.8, with 3000 photons (~200 events)."""
    f = _random_modes(60, seed=7)
    A = np.random.default_rng(7).uniform(1.0, 10.0, f.size)
    cm = energy_me.CascadeModel(f, A, e_max_cm=45.0 * EV, de_cm=50.0)
    E, counts = dos.beyer_swinehart(f, 45.0 * EV + 200.0)
    k = rates.inverse_laplace_arrhenius(E, counts, 1e16, 8.0 * EV)
    ch = {"X": k[np.rint(cm.energy_cm).astype(int)]}
    h = _toy_heating(1e8)
    me = cm.steady_state(E_PH, h, ch).rates["X"]
    n_ph = 1500
    mc = montecarlo.MonteCarlo(cm, ch, E_PH, h, seed=4).steady_rates(n_ph)["X"]
    events = me / np.trapezoid(h, E_PH) * n_ph
    assert events > 50
    assert abs(mc - me) < 3.0 * me / np.sqrt(events), (mc, me, events)


# ===========================================================================
# M4: (Z, N_H) network and collisional rate coefficients
# ===========================================================================

from pycalima.pahchem import chemistry
from pycalima.pahchem.network import PAHNetwork


@pytest.mark.parametrize("T", [50.0, 100.0, 500.0])
def test_eley_rideal_reproduces_andrews16_eq10(T):
    """sigma_er = 0.06 A^2 (Mennella+12) gives 8.7e-13 sqrt(T/100) cm^3 s^-1."""
    k = chemistry.cross_section_rate(T, 0.06)
    assert k == pytest.approx(8.7e-13 * np.sqrt(T / 100.0), rel=0.01)


def test_electron_capture_orders_and_low_t_scaling():
    """Coulomb focusing: attractive > neutral > repulsive; for a cation at
    low T the DS87 J~ ~ 1/tau, so the coefficient goes as T^-1/2."""
    T = 100.0
    k = {z: chemistry.electron_capture(24, z, T) for z in (-1, 0, 1, 2)}
    assert k[2] > k[1] > k[0] > k[-1] > 0
    slope = np.log(chemistry.electron_capture(24, 1, 40.0) / chemistry.electron_capture(24, 1, 20.0)) / np.log(2)
    assert slope == pytest.approx(-0.5, abs=0.05)


def _toy_network(g0=1.0, ne=0.1, nH=100.0, T=100.0):
    """Coronene-like: Z = -1..2, N_H = 0..14 (N_H0 = 12). Rates are illustrative."""
    net = PAHNetwork(charges=(-1, 0, 1, 2), nh_values=range(0, 15))
    nz, nn = net.shape
    nh = np.array(net.nh_values)[None, :] * np.ones((nz, 1))
    z = np.array(net.charges)[:, None] * np.ones((1, nn))
    ion = np.where(z < 2, 1e-8 * g0 * (1.0 + 0.1 * (z + 1)), 0.0)
    rec = np.where(z > -1, [[chemistry.electron_capture(24, int(q), T) * ne] * nn for q in net.charges], 0.0)
    h_loss = np.where(nh > 0, 3e-9 * g0 * np.where(nh > 12, 1e2, 1.0), 0.0)
    h2_loss = np.where((nh >= 2) & (nh % 2 == 0) & (nh <= 12), 2e-10 * g0, 0.0)
    add = np.where(nh < 14, chemistry.cross_section_rate(T, 1.1) * nH, 0.0)
    er = np.where(nh > 12, chemistry.cross_section_rate(T, 0.06) * nH, 0.0)
    net.add("photoionisation", ion, dz=+1)
    net.add("recombination", rec, dz=-1)
    net.add("H loss", h_loss, dnh=-1)
    net.add("H2 loss", h2_loss, dnh=-2, h2_released=1)
    # neutrals and cations add H; anions by associative detachment, to (0, N_H + 1)
    net.add("H addition", np.where(z >= 0, add, 0.0), dnh=+1)
    net.add("associative detachment", np.where(z == -1, add, 0.0), dz=+1, dnh=+1)
    net.add("Eley-Rideal", er, dnh=-1, h2_released=1)
    return net


def test_network_generator_conserves_probability():
    Q = _toy_network().generator()
    assert np.allclose(np.asarray(Q.sum(axis=0)).ravel(), 0.0, atol=1e-20)


def test_network_rejects_rates_that_leave_the_state_space():
    net = PAHNetwork(charges=(0, 1), nh_values=range(0, 3))
    with pytest.raises(ValueError, match="outside"):
        net.add("bad", 1.0, dnh=+1)                 # N_H = 2 has nowhere to go


def test_network_steady_state_is_normalised_and_reached_in_time():
    net = _toy_network(g0=1e3, ne=0.1, nH=1e3)
    f = net.steady_state()
    assert f.sum() == pytest.approx(1.0) and np.all(f >= 0)
    f0 = net.zeros()
    f0[1, 12] = 1.0                                    # neutral, normally hydrogenated
    late = net.evolve(f0, [1e14])[-1]
    assert np.allclose(late, f, atol=1e-6)


def test_network_balances_hydrogen_at_steady_state():
    """Net H-atom flux into the PAH population is zero at steady state."""
    net = _toy_network(g0=1e2, nH=1e3)
    f = net.steady_state()
    flux = net.fluxes(f)
    dnh = {p.name: p.dnh for p in net.processes}
    gained = sum(flux[n] * dnh[n] for n in flux)
    scale = sum(abs(flux[n] * dnh[n]) for n in flux)
    assert abs(gained) < 1e-8 * scale


def test_charge_chain_matches_the_four_state_closed_form():
    """A pure charge network is the four_state_charge_fractions of PAH_charge."""
    from pycalima.models.PAH_charge.PAH_photoelectric_heating import four_state_charge_fractions

    k_att, k_det, k_pe0, k_pe1, k_rec1, k_rec2, ne = 3e-9, 2e-8, 5e-8, 1e-9, 4e-6, 9e-6, 0.05
    net = PAHNetwork(charges=(-1, 0, 1, 2), nh_values=(12,))
    net.add("up", np.array([[k_det], [k_pe0], [k_pe1], [0.0]]), dz=+1)
    net.add("down", np.array([[0.0], [k_att * ne], [k_rec1 * ne], [k_rec2 * ne]]), dz=-1)
    f = net.steady_state()[:, 0]
    expected = four_state_charge_fractions(k_att, k_det, k_pe0, k_pe1, k_rec1, k_rec2, ne)
    assert np.allclose(f, expected, rtol=1e-10)


def test_charge_balance_depends_on_g0_over_ne_only():
    """With photo-rates proportional to G0 and capture to n_e, scaling both by
    the same factor leaves every fraction unchanged (the gamma parameter)."""
    a = _toy_network(g0=10.0, ne=0.01, nH=0.0).steady_state()
    b = _toy_network(g0=1e3, ne=1.0, nH=0.0).steady_state()
    assert np.allclose(a.sum(axis=1), b.sum(axis=1), atol=1e-12)


def test_measured_cation_recombination_implies_sticking_below_one_rising_with_size():
    """Measured k_rec (C6-C16 cations, 300 K) over capture with s_e = 1 at the
    Li & Draine radius: 0.03-0.35, rising with N_C, reaching 1 near N_C ~ 20.
    So s_e(cation) ~ 1 is an extrapolation for C24-C96, and WD01's
    s_e = 0.5(1 - exp(-a/l_e)) ~ 0.1 already underestimates pyrene+."""
    from pycalima.pahchem.photo import li_draine_radius_um

    t = labdata.electron_recombination()
    assert t["n_c"].between(6, 16).all() and t["k_cm3_s"].between(1e-7, 1e-5).all()
    ratio = np.array([k / chemistry.electron_capture(n, 1, 300.0, disc_factor=1.0,
                                                     radius_cm=li_draine_radius_um(n) * 1e-4)
                      for n, k in zip(t["n_c"], t["k_cm3_s"])])
    assert np.all((ratio > 0.01) & (ratio < 1.0)), ratio
    slope, intercept = np.polyfit(t["n_c"], np.log10(ratio), 1)
    assert slope > 0
    assert 15.0 < -intercept / slope < 35.0                 # N_C where s_e -> 1


def test_stripped_species_keeps_3n_minus_6_and_removes_ch_modes(pah_states):
    sp = PAHSpecies.from_states_file(pah_states / "C54H18_0.dat")
    bare = sp.stripped(0)
    assert bare.n_modes == 3 * 54 - 6
    assert not np.any(bare.freqs_cm > 2900.0)                 # no C-H stretch left
    assert sp.stripped(10).n_modes == 3 * 64 - 6


# ===========================================================================
# M5: reproductions of published models through their method cards
# ===========================================================================


_MONTILLAUD_CACHE = {}


@pytest.fixture
def montillaud_c54(pah_states):
    """Built once per session: each photo_rates call solves ~36 cascades."""
    from pycalima.pahchem.montillaud13 import Montillaud13

    if "C54" not in _MONTILLAUD_CACHE:
        sp = PAHSpecies.from_states_file(pah_states / "C54H18_0.dat")
        _MONTILLAUD_CACHE["C54"] = Montillaud13(sp, "C54", modes="stripped", e_max_ev=40.0, de_cm=300.0)
    return _MONTILLAUD_CACHE["C54"]


# A point on the digitised 50 % contour of Montillaud+13 Fig. 10 (C54H18).
M13_G0, M13_NH = 44.4561, 1275.63


def _m13_dehydrogenated(model, g0):
    net = model.network(model.photo_rates(g0), 0.5 * M13_NH, 1.4e-4 * M13_NH, 100.0)
    return net.steady_state()[:, 0].sum()


def test_montillaud13_fig10_contour_offset(montillaud_c54):
    """Our 50 % contour lies +0.25 to +0.5 dex in G0 above theirs (measured +0.38
    at nH = 1276 with per-state stripped modes; +1.2 with the parent's modes).
    A regression bound on the documented gap, not agreement."""
    assert _m13_dehydrogenated(montillaud_c54, M13_G0 * 10**0.25) < 0.5
    assert _m13_dehydrogenated(montillaud_c54, M13_G0 * 10**0.5) > 0.5


@pytest.mark.xfail(strict=True, reason="pahchem's C54H18 is ~0.4 dex in G0 more photostable than "
                   "Montillaud+13 Fig. 10. The gap closes from 1.2 dex with per-state modes; the rest needs "
                   "DFT modes of each state, Malloci's C54 sigma and cation IR intensities. Remove this "
                   "marker when it passes.")
def test_montillaud13_fig10_agreement(montillaud_c54):
    assert 0.25 < _m13_dehydrogenated(montillaud_c54, M13_G0) < 0.75


def test_h2_formation_counts_eley_rideal_and_h2_loss():
    net = _toy_network(g0=1e2, nH=1e3)
    f = net.steady_state()
    flux = net.fluxes(f)
    assert net.h2_formation_rate(f) == pytest.approx(flux["H2 loss"] + flux["Eley-Rideal"])
