"""Recurrent fluorescence (RF) in the cascade (cooling.recurrent_fluorescence_rates, M3c).

Invariant gates on a synthetic electronic level (E_n = 8000 cm^-1, f = 0.05)
of coronene's modes; the published RF inputs and storage-ring benchmarks are
separate (method cards).
"""

import numpy as np
import pytest

from pycalima.pahchem import cooling, dos, energy_me, fields, photo, rates
from pycalima.pahchem.molecule import PAHSpecies

LEVEL = (8000.0, 0.05)
EV = dos.CM_PER_EV


@pytest.fixture
def coronene(model_data):
    path = model_data / "PAH_states" / "C24H12_1.dat"
    if not path.exists():
        pytest.skip("model_data/PAH_states/ has no C24H12_1.dat")
    return PAHSpecies.from_states_file(path)


def test_rf_rate_is_in_detailed_balance(coronene):
    """<k_RF>_T over the canonical ensemble = A (g_n/g_0) exp(-E_n/kT) = 0.667 nu^2 f exp(-E_n/kT):
    the thermal population of the level times its Einstein A, at 600-2000 K."""
    E, counts = dos.beyer_swinehart(coronene.freqs_cm, 300000.0, 2.0)
    k = cooling.recurrent_fluorescence_rates(E, counts, [LEVEL[0]], [LEVEL[1]], window_cm=20.0)[0]
    for T in (600.0, 1000.0, 2000.0):
        w = counts * np.exp(-E / (dos.KB_CM * T))
        expected = cooling.RF_A_PER_F * LEVEL[0] ** 2 * LEVEL[1] * np.exp(-LEVEL[0] / (dos.KB_CM * T))
        assert (k @ w) / w.sum() == pytest.approx(expected, rel=2e-3)


def test_rf_level_is_a_radiative_channel_of_the_cascade(coronene):
    """With the level, CascadeModel holds it after the IR modes; the steady state
    radiates (IR + optical) what it absorbs, and RF carries a share of it."""
    cm = energy_me.CascadeModel(coronene.freqs_cm, coronene.einstein_a, 20 * EV, de_cm=100.0, electronic=[LEVEL])
    assert cm.n_ir == coronene.n_modes and cm.freqs_cm[-1] == LEVEL[0]
    e = np.linspace(5.0, 13.6, 345)
    h = photo.heating_spectrum(fields.draine1978(g0=1e4), photo.sigma_li_draine(24, 1, e), np.zeros_like(e), e)
    ss = cm.steady_state(e, h, {})
    assert ss.radiated_power == pytest.approx(ss.absorbed_power, rel=1e-3)
    rf_power = LEVEL[0] * (cm.k_modes[cm.n_ir:] @ ss.probability).sum()
    assert 0.0 < rf_power < ss.radiated_power


def test_rf_suppresses_dissociation_and_matches_the_monte_carlo(coronene):
    """H loss (E0 = 4.41 eV, West+18 coronene+) from 12 eV: RF lowers the yield, and the
    master-equation yield equals 4000 Gillespie cascades on the same rates (3 sigma)."""
    from pycalima.pahchem.montecarlo import MonteCarlo

    yields = {}
    for lev in ((), (LEVEL,)):
        cm = energy_me.CascadeModel(coronene.freqs_cm, coronene.einstein_a, 14 * EV, de_cm=100.0, electronic=lev)
        E, counts = dos.beyer_swinehart(coronene.freqs_cm, cm.energy_cm[-1] + 200.0)
        ts = rates.transition_state_for_entropy(coronene.freqs_cm, 3050.0, 12.0, n_scaled=5)
        k = rates.rrkm(E, counts, ts, 4.41 * EV, degeneracy=12)[np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)]
        j = int(round(12.0 * EV / cm.de_cm))
        yields[len(lev)] = cm.photon_yields({"H": k})["H"][j]
        if lev:
            n = 4000
            mc = MonteCarlo(cm, {"H": k}, seed=5).photon_yields(cm.energy_cm[j], n)["H"]
            sigma = np.sqrt(yields[1] * (1 - yields[1]) / n)
            assert abs(mc - yields[1]) < 3 * sigma + 1e-3
    assert yields[1] < yields[0]


def test_rf_in_the_baseline_cools_coronene_cations(model_data):
    """Baseline RF (Malloci+07 coronene+ bands, baseline.RF_SPECTRA) at the NGC 7023 edge:
    - the cation's H-loss rate falls 2.3x (N_H = 12);
    - fully hydrogenated C24H12+ carries 32 % of its absorbed power off as RF photons
      (10 % without the weak 0.6 eV band: the lowest state dominates at low energy);
    - its 3.29/6.2 falls from 0.063 to 0.046;
    - IR + RF power closes on the absorbed power (1e-15)."""
    from pycalima.pahchem.baseline import NGC7023_EDGE, BaselineModel, Params
    from pycalima.pahchem.family_emission import FamilyEmission, band_fluxes

    files = {z: model_data / "PAH_states" / f"C24H12_{z}.dat" for z in (-1, 0, 1)}
    if not all(p.exists() for p in files.values()):
        pytest.skip("model_data/PAH_states/ lacks C24H12 -1/0/+1")
    sp = {z: PAHSpecies.from_states_file(p) for z, p in files.items()}
    grid = np.arange(500.0, 3500.0, 0.5)
    out = {}
    for rf in (False, True):
        model = BaselineModel("C24", sp[0], Params(recurrent_fluorescence=rf), anion=sp[-1])
        fam = FamilyEmission(model, sp)
        f, ir, absorbed, rf_power = fam.state_emission(NGC7023_EDGE)[(1, 12)]
        assert f @ ir + rf_power == pytest.approx(absorbed, rel=1e-9)
        b = band_fluxes(grid, fam.spectrum(NGC7023_EDGE, grid, {(z, n): float((z, n) == (1, 12))
                                                                 for z in model.charges for n in model.nh_values})[0])
        out[rf] = (model.photo_rates(NGC7023_EDGE)[(1, 12)]["H"], rf_power / absorbed, b["3.29"] / b["6.2"])
    assert out[False][1] == 0.0 and 0.25 < out[True][1] < 0.4
    assert out[True][0] < out[False][0] / 2
    assert out[True][2] < 0.8 * out[False][2]


def test_rf_suppresses_circumcoronene_cation_h_loss(model_data):
    """C54H18+ (Malloci+07 bands 1.05 eV, f = 0.076; 2.40 eV, f = 1.08) at the NGC 7023 edge: its
    H loss needs multiphoton energies, where RF through the strong 2.4 eV band is fast.
    Measured (de = 200 cm^-1): k_H falls 68x at N_H = 18 (50x at the Orion Bar, where the
    family's dehydrogenated fraction halves, 0.115 -> 0.052)."""
    from pycalima.pahchem.baseline import NGC7023_EDGE, BaselineModel, Params

    files = {z: model_data / "PAH_states" / f"C54H18_{z}.dat" for z in (-1, 0)}
    if not all(p.exists() for p in files.values()):
        pytest.skip("model_data/PAH_states/ lacks C54H18 -1/0")
    sp = {z: PAHSpecies.from_states_file(p) for z, p in files.items()}
    k = {rf: BaselineModel("C54", sp[0], Params(recurrent_fluorescence=rf, de_cm=400.0), anion=sp[-1])
         .photo_rates(NGC7023_EDGE)[(1, 18)]["H"] for rf in (False, True)}
    assert k[True] < k[False] / 10, k


_BORJA = {"Np+": "C10H8_1", "An+": "C14H10_1", "Py+": "C16H10_1"}


@pytest.mark.parametrize("molecule", list(_BORJA))
def test_rf_rates_against_borja2026(model_data, molecule):
    """k_RF(E) of each excited state against Borja+26 Fig. 8, with their Table I inputs
    (experimental D0-Dn gaps, f = f_FC + f_HT(T), f_HT linear in T between their 0 and
    2000 K values) and PAHdb cation modes for rho; their model adds the excited states'
    own modes (Duschinsky) and Herzberg-Teller dipoles. Measured at 4-13 eV:
    - allowed states (Np+ D2, An+ D2, Py+ D2, D3): -0.18 to +0.16 dex, within 0.07 at 6-10 eV;
    - forbidden D1 (pure Herzberg-Teller): -0.14 to +0.49 dex, ours high at low E."""
    import pandas as pd
    from pycalima._paths import get_external_data_path

    path = model_data / "PAH_states" / f"{_BORJA[molecule]}.dat"
    if not path.exists():
        pytest.skip(f"model_data/PAH_states/ has no {path.name}")
    sp = PAHSpecies.from_states_file(path)
    table = pd.read_csv(get_external_data_path("pahchem/borja2026_table1.csv"), comment="#")
    curves = pd.read_csv(get_external_data_path("pahchem/borja2026_fig8_krf.csv"), comment="#")
    rows = table[table.molecule == molecule].set_index("state")
    d0 = rows.loc["D0", "transition_energy_exp_eV"]
    E, counts = dos.beyer_swinehart(sp.freqs_cm, 15 * EV, 2.0)
    for state in [s for s in rows.index if s != "D0"]:
        r = rows.loc[state]
        c = curves[(curves.molecule == molecule) & (curves.state == state)]
        allowed = not np.isnan(r.fFC_x1e3)
        for e_ev in (4.0, 6.0, 8.0, 10.0, 13.0):
            T = float(dos.canonical_temperature(sp.freqs_cm, np.array([e_ev * EV]))[0])
            f = (r.fFC_x1e3 if allowed else 0.0) * 1e-3 + 1e-3 * (
                r.fHT_0K_x1e3 + (r.fHT_2000K_x1e3 - r.fHT_0K_x1e3) * min(T, 2000.0) / 2000.0)
            k = cooling.recurrent_fluorescence_rates(E, counts, [(r.transition_energy_exp_eV - d0) * EV], [f],
                                                     window_cm=50.0, at_cm=[e_ev * EV])[0, 0]
            dex = np.log10(k) - np.interp(e_ev, c.e_ev, c.log10_k_rf)
            if allowed:
                assert abs(dex) < 0.2, (state, e_ev, dex)
            else:
                assert -0.2 < dex < 0.55, (state, e_ev, dex)
