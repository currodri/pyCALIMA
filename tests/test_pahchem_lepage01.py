"""Gates for pahchem.lepage01: Le Page, Snow & Bierbaum 2001/2003 in their configuration.

Each test isolates one input of method_cards/lepage01.toml. The numbers in
the docstrings were measured on 2026-10-05.
"""
import numpy as np
import pytest

from pycalima.pahchem import lepage01 as L
from pycalima.pahchem import rates

R_CAL = 8.314462618 / 4.184


# --- 2nu-RRKM (Paper I Sect. 3.1.3) ---

@pytest.mark.parametrize("nc, nh", [(6, 6), (24, 12), (100, 28), (200, 50)])
def test_two_nu_transition_state_has_the_stated_entropy_at_every_size(nc, nh):
    """250 cm^-1 bends give dS(1000 K) = 5.03 cal/K, 950 cm^-1 give 0.00, for any size."""
    r = L.two_nu_modes(nc, nh)
    for dS, bend in ((5.0, 5.03), (0.0, 0.0)):
        ts = L.two_nu_ts(nc, nh, dS)
        assert ts.size == r.size - 1
        got = (rates.vibrational_entropy(ts, 1000.0) - rates.vibrational_entropy(r, 1000.0)) * R_CAL
        assert got == pytest.approx(bend, abs=0.02)


def test_coronene_rate_reproduces_lepage01_fig9():
    """C24H12(+) H loss (E0 = 4.8 eV, dS = 5 cal/K, sigma = 12) against the
    'LePage' curve of Montillaud+13 Fig. 3 (= Paper I Fig. 9). Measured
    log10(ours/theirs) = +0.23 at 7.6 eV (threshold, where steepest descent
    and a lattice count differ most), -0.07 to +0.15 over 8.3-15 eV.
    sigma = 1 is ~1 dex low, which is how sigma = 12 was inferred."""
    import pandas as pd
    from pycalima._paths import get_external_data_path

    d = pd.read_csv(get_external_data_path("dissoc_rate_coronene_Montillaud.csv"), skiprows=1)
    x, y = d["X.1"].dropna().to_numpy(), d["Y.1"].dropna().to_numpy()
    dex = np.log10(L.k_of_E(24, 12, 4.8, 5.0, 12.0, x) / y)
    assert np.all(np.abs(dex) < 0.25) and np.all(np.abs(dex[1:]) < 0.16), np.round(dex, 2)


@pytest.mark.parametrize("nc, nh, E0, energies, quoted", [
    (6, 6, 3.8, [4.5, 5.5], [3.19, 6.1]),      # benzene+, Table 2 E0; measured 3.05, 6.07
    (14, 10, 4.4, [7.9, 13.6], [2.2, 7.6]),    # anthracene+; measured 2.09, 7.54
])
def test_two_nu_reproduces_the_quoted_cation_rates(nc, nh, E0, energies, quoted):
    """Paper I Sect. 3.1.3, log10 k at two energies, with sigma = N_H."""
    got = np.log10(L.k_of_E(nc, nh, E0, 5.0, float(nh), np.array(energies)))
    assert np.allclose(got, quoted, atol=0.15), got


@pytest.mark.xfail(strict=True, reason=(
    "Naphthalene+ with the Table 2 E0 = 4.5 eV gives log k ~ 2.8-2.9, 6.8-6.9 at 7 and 10 eV against the "
    "quoted 3.9, 7.5; E0 ~ 4.2 eV reproduces it. The E0 Le Page used for this check is not stated."))
def test_two_nu_reproduces_naphthalene_with_table2_e0():
    got = np.log10(L.k_of_E(10, 8, 4.5, 5.0, 8.0, np.array([7.0, 10.0])))
    assert np.allclose(got, [3.9, 7.5], atol=0.12), got


def test_lp01_table1_two_nu_column_is_the_neutral_pahh_h_loss():
    """Paper I Table 1: log10 k(C50H21) - log10 k(PAHH), 2nu-RRKM, at 5 and 10 eV.

    The channel is not stated. The neutral PAHH H loss (E0 = 1.2 eV, dS = 0)
    reproduces it: measured 1.31/1.44, 2.05/2.22, 2.66/2.82, 3.57/3.70,
    4.70/4.73 against 1.5/1.3, 2.3/2.0, 2.9/2.6, 3.8/3.5, 4.9/4.5 (|dex| <= 0.25).
    The cation PAHH channel (2.9 eV) gives 3.4-12 dex.
    """
    table = {(80, 25): (1.5, 1.3), (100, 29): (2.3, 2.0), (120, 31): (2.9, 2.6),
             (150, 41): (3.8, 3.5), (200, 51): (4.9, 4.5)}
    E = np.array([5.0, 10.0])
    ref = np.log10(L.k_of_E(50, 21, 1.2, 0.0, 1.0, E))
    for (nc, nh), theirs in table.items():
        got = ref - np.log10(L.k_of_E(nc, nh, 1.2, 0.0, 1.0, E))
        assert np.allclose(got, theirs, atol=0.35), ((nc, nh), got)


def test_lp01_table1_rrk_column_implies_e0_near_0p7_ev():
    """The RRK column of Table 1 (nu = 1e16, s = 3N - 6) is reproduced only
    with E0 ~ 0.7 eV, not the eq. 3 value 2.8 eV nor the Table 3 1.2 eV
    (which give 2-5x the printed differences). A finding about the table,
    recorded so the RRK row is not used as a target with E0 = 2.8 eV."""
    table = {(80, 25): (6.8, 3.2), (100, 29): (12.0, 5.4), (120, 31): (16.0, 7.5),
             (150, 41): (24.0, 11.0), (200, 51): (36.0, 17.0)}
    E = np.array([5.0, 10.0]) * L.EV
    s = lambda nc, nh: 3 * (nc + nh) - 6

    def misfit(E0):
        ref = np.log10(rates.rrk(E, s(50, 21), 1e16, E0 * L.EV))
        return sum(np.sum((ref - np.log10(rates.rrk(E, s(*k), 1e16, E0 * L.EV)) - np.array(v)) ** 2)
                   for k, v in table.items())

    grid = np.arange(0.3, 3.0, 0.05)
    best = grid[np.argmin([misfit(e) for e in grid])]
    assert 0.6 <= best <= 0.8, best


def test_branching_is_a_probability_and_the_rate_is_bounded_by_absorption():
    """k_ph <= the absorption rate over the same photon range (the whole field)."""
    E = L.energy_grid(L.field().energy_ev.min())
    for nh, z in ((12, 0), (13, 0), (13, 1), (5, 1)):
        P = L.branching(24, nh, 12, z, E)
        total = sum(P.values())
        assert np.all((total >= 0) & (total <= 1 + 1e-9))
        k = L.k_photodissociation(24, nh, 12, z)
        assert sum(k.values()) <= L.field().photon_rate(L.sigma_uv(24, E), E)


# --- Charge (Paper I Sects. 3.2-3.5; LP03 Table 2, Fig. 4) ---

def test_paper1_appendix_three_state_c50():
    """C50H20+ (A), C50H20 (B), C50H21+ (C) with the four printed rates
    [month^-1]: k_ion B->A 0.052, k_rec n_e A->B and C->B 0.019 (PAHH+
    recombination is dissociative, Sect. 3.3), H addition A->C 0.011, H loss
    C->A 0.022. Gives 0.578/0.268/0.155 against the printed 57/27/16 %."""
    k_ion, k_rec, k_add, k_loss = 0.052, 0.019, 0.011, 0.022
    Q = np.array([[-(k_rec + k_add), k_ion, k_loss],
                  [k_rec, -k_ion, k_rec],
                  [k_add, 0.0, -(k_loss + k_rec)]])
    Q[0] = 1.0
    f = np.linalg.solve(Q, [1.0, 0.0, 0.0])
    assert np.allclose(f, [0.57, 0.27, 0.16], atol=0.01), f


LP03_TABLE2 = {16: (4.0e-9, 2.8e-9, 3.2e-8), 24: (6.9e-9, 3.8e-9, 4.8e-8),
               50: (2.0e-8, 6.6e-9, 1.0e-7), 100: (5.0e-8, 1.1e-8, 2.0e-7)}   # k_ion, k_e, k_p at nt = 25
LP03_FIG4 = {10: {16: (0.61, 0.37, 0.01), 24: (0.74, 0.26, 0.01), 50: (0.89, 0.11, 0.0), 100: (0.95, 0.05, 0.0)},
             25: {16: (0.38, 0.58, 0.05), 24: (0.52, 0.45, 0.04), 50: (0.76, 0.23, 0.02), 100: (0.89, 0.11, 0.01)},
             100: {16: (0.11, 0.66, 0.23), 24: (0.18, 0.62, 0.20), 50: (0.39, 0.48, 0.13), 100: (0.63, 0.30, 0.07)}}


def _fig4(nt, nc, k_ion):
    n_e = L.NE_OVER_NT * nt
    return L.three_state_charge(k_ion, L.k_recombination(100.0, n_e, "benzene"),
                                L.k_attachment(nc, n_e), L.k_photodetachment(nc))


def test_closed_form_charge_rates_match_lp03_table2():
    """k_rec n_e, k_e n_e and k_p at nt = 25, T = 100 K: within 1.5 %."""
    n_e = L.NE_OVER_NT * 25.0
    assert L.k_recombination(100.0, n_e, "benzene") == pytest.approx(6.0e-9, rel=0.015)
    for nc, (_, ke, kp) in LP03_TABLE2.items():
        assert L.k_attachment(nc, n_e) == pytest.approx(ke, rel=0.015)
        assert L.k_photodetachment(nc) == pytest.approx(kp, rel=0.015)


@pytest.mark.parametrize("nt", sorted(LP03_FIG4))
def test_lp03_fig4_follows_from_lp03_table2_rates(nt):
    """With their k_ion and our closed-form k_rec, k_e, k_p, the three-state
    balance gives Fig. 4 within 0.015 (max measured 0.013): a check of the
    transcription and of the charge solver, isolated from sigma."""
    for nc, theirs in LP03_FIG4[nt].items():
        got = _fig4(nt, nc, LP03_TABLE2[nc][0])
        assert np.allclose(got, theirs, atol=0.015), (nc, np.round(got, 3), theirs)


def test_pahchem_k_ion_against_lp03_table2():
    """eq. 15 with the um^-1 yield, the Allain IP law and their sigma (Paper I
    Fig. 11): k_ion is 1.06-1.34x LP03 Table 2 (C100 to C16). With the eV form
    of the yield (Montillaud's transcription) it is 0.96-0.99x for C16-C32 but
    0.82x and 0.72x for C50 and C100; the IPs are not stated, so the two cannot
    be told apart. Substitutes: Li & Draine 1.17-1.71x, Malloci per C
    0.92-1.24x."""
    for nc, (theirs, _, _) in LP03_TABLE2.items():
        assert 1.0 < L.k_ion(nc) / theirs < 1.4
        assert 0.85 < L.k_ion(nc, substitute="malloci") / theirs < 1.3


@pytest.mark.xfail(strict=True, reason=(
    "With pahchem's own k_ion the LP03 Fig. 4 fractions are off by up to 0.07 with their sigma (Paper I "
    "Fig. 11; 0.13 with Li & Draine), against 0.015 with their Table 2 k_ion. The rest is k_ion 1.06-1.34x "
    "high: the yield units (um^-1 as printed; the eV form gives 0.08) and the unstated IPs."))
def test_lp03_fig4_with_pahchem_k_ion():
    for nt, rows in LP03_FIG4.items():
        for nc, theirs in rows.items():
            assert np.allclose(_fig4(nt, nc, L.k_ion(nc)), theirs, atol=0.03), (nt, nc)


# --- Paper I Fig. 10: photodissociation rates with H loss vs N_C (digitised) ---

#: the compact family of LP03 Sect. 2 (C100H28: card c100_note)
FAMILY = [(6, 6), (10, 8), (14, 10), (16, 10), (24, 12), (32, 14), (50, 20), (80, 24), (100, 28),
          (120, 30), (150, 40)]
#: Fig. 10 class -> (charge, N_H of a C_x H_y family member)
CLASSES = {"PAH": (0, lambda y: y), "PAH+": (1, lambda y: y), "XdPAH": (0, lambda y: 1),
           "XdPAH+": (1, lambda y: 1), "PAHH": (0, lambda y: y + 1), "PAHH+": (1, lambda y: y + 1)}


def _lp01_fig10(cls):
    import pandas as pd
    from pycalima import _paths

    d = pd.read_csv(_paths.get_external_data_path("pahchem/lepage01_fig10_kph.csv"), comment="#")
    d = d[d.species_class == cls].sort_values("n_c")
    return d.n_c.to_numpy(), d.k_ph_s.to_numpy()


def _fig10_dex(cls, nc, y):
    x, k = _lp01_fig10(cls)
    if not x[0] - 1 <= nc <= x[-1] + 1:
        return None
    theirs = 10 ** np.interp(nc, x, np.log10(k))
    z, nh_of = CLASSES[cls]
    return float(np.log10(L.k_photodissociation(nc, nh_of(y), y, z)["H"] / theirs))


@pytest.mark.parametrize("cls", ["PAH", "PAH+", "PAHH", "PAHH+"])
def test_paper_i_fig10_rates(cls):
    """k_ph (H loss) at chi = 1 against Paper I Fig. 10 over the family (6-150
    C), with their sigma (Fig. 11), the whole field and (1 - Y_ion) for
    neutrals. Measured max |dex|: PAH 0.10, PAH+ 0.21, PAHH 0.17 (6 C), PAHH+
    0.15."""
    dex = [d for nc, y in FAMILY if (d := _fig10_dex(cls, nc, y)) is not None]
    assert len(dex) >= 7 and max(map(abs, dex)) < 0.25, np.round(dex, 2)


@pytest.mark.parametrize("cls", ["XdPAH", "XdPAH+"])
def test_paper_i_fig10_extremely_dehydrogenated(cls):
    """One H left: within 0.15 dex up to 24 C (measured 0.13), and the size
    where k_ph falls through 1e-12 s^-1 within 2.5 C (measured 44.7 against
    43.6 neutral, 48.0 against 46.2 cation). The rate falls ~0.27 dex per C
    there, so a 1-2 C shift is +0.3-0.5 dex at 32-50 C."""
    assert all(abs(_fig10_dex(cls, nc, y)) < 0.15 for nc, y in FAMILY if nc <= 24)
    z = CLASSES[cls][0]

    def crossing(nc, k):
        lk = np.log10(k)
        i = int(np.argmax(lk < -12))
        return float(np.interp(-12, [lk[i], lk[i - 1]], [nc[i], nc[i - 1]]))

    ncs = np.arange(30, 61)
    ours = crossing(ncs, np.array([L.k_photodissociation(int(n), 1, 20, z)["H"] for n in ncs]))
    assert ours == pytest.approx(crossing(*_lp01_fig10(cls)), abs=2.5)


# --- LP03 Figs. 6-7: H and carbon loss against size (digitised) ---

def _lp03(fig, channel):
    import pandas as pd
    from pycalima import _paths

    name = {6: "lepage03_fig6_dehydrogenated_rates.csv", 7: "lepage03_fig7_normal_rates.csv"}[fig]
    d = pd.read_csv(_paths.get_external_data_path(f"pahchem/{name}"), comment="#")
    d = d[d.channel == channel].sort_values("n_c")
    return d.n_c.to_numpy(), d.k_s.to_numpy()


def test_lp03_fig7_h_loss_is_the_cation_rate():
    """Fig. 7 H loss of the normal PAHs (C14H10-C32H14) is the cation rate:
    measured +0.02 to +0.13 dex for Z = +1, -0.37 to -0.67 for the neutral."""
    x, k = _lp03(7, "H")
    for nc, y in ((14, 10), (16, 10), (24, 12), (32, 14)):
        theirs = 10 ** np.interp(nc, x, np.log10(k))
        assert abs(np.log10(L.k_photodissociation(nc, y, y, 1)["H"] / theirs)) < 0.15
        assert np.log10(L.k_photodissociation(nc, y, y, 0)["H"] / theirs) < -0.3


def test_lp03_fig6_c2_loss_sets_the_minimum_skeleton_size():
    """C2 loss of the bare neutral skeleton (8 eV, 5 cal/K, TS of
    carbon_loss_ts) equals C+ insertion (3.5e-12 s^-1) at 16.2 C, against
    18.1 C on Fig. 6 and '18-20 C' in the text (LP03 Sect. 4.3). With one H
    left, H loss wins and C2 loss never reaches the insertion rate: Fig. 6 is
    the bare skeleton."""
    def crossing(nc, k, level=np.log10(L.C_INSERTION_FIG6)):
        lk = np.log10(k)
        i = int(np.argmax(lk < level))
        return float(np.interp(level, [lk[i], lk[i - 1]], [nc[i], nc[i - 1]]))

    ncs = np.arange(8, 31)
    ours = crossing(ncs, np.array([L.k_carbon_loss(int(n), 0, 20, 0, *L.C2_LOSS) for n in ncs]))
    assert ours == pytest.approx(crossing(*_lp03(6, "C2")), abs=2.5)


# --- The hydrogenation network (LP03 Fig. 9, Sect. 4.1) ---

_FIG9 = {}


def _standard(nc, nh0):
    if nc not in _FIG9:
        _FIG9[nc] = L.LePage01(nc, nh0).steady_state(L.STANDARD)
    return _FIG9[nc]


def test_lp03_fig9_small_pahs_are_extremely_dehydrogenated():
    """C24Hn at nt = 100, chi = 1: measured 0.86 of the population at N_H <= 2
    (Fig. 9: N_H ~ 0-2; Sect. 4.1: 20-30 C 'extremely dehydrogenated'). It was
    0.96 with eq. 11 as printed (from IP, no 1 - Y_ion); Paper I Fig. 10 fixes
    both (k_photodissociation)."""
    f = _standard(24, 12)
    assert f[:, :3].sum() > 0.8


def test_lp03_fig9_mid_and_large_pahs():
    """Against the card's reading of the 3-D bars (+-0.03; low confidence):
    C32H14+ 0.45 / C32H14 0.37 -> measured 0.56 / 0.30; C50H20+ 0.6 -> 0.64;
    C80H24+ 0.6 / C80H25+ 0.27 -> 0.59 / 0.26. The C32 split follows the
    k_ion excess of the sigma substitute (test_pahchem_k_ion_against_lp03_table2)."""
    f32, f50, f80 = _standard(32, 14), _standard(50, 20), _standard(80, 24)
    assert f32[1, 14] == pytest.approx(0.45, abs=0.12) and f32[0, 14] == pytest.approx(0.37, abs=0.1)
    assert f50[1, 20] == pytest.approx(0.6, abs=0.08)
    assert f80[1, 24] == pytest.approx(0.6, abs=0.05) and f80[1, 25] == pytest.approx(0.27, abs=0.05)
