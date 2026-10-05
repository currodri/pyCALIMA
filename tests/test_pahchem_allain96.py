"""Allain, Leach & Sedlmayr (1996, I and II) through pahchem.allain96.

These gates run from the cheapest to the most composite; each adds one input
to the previous one:
- level 0, analytic: the RRK convention, the yield law, recombination, the
  field;
- level 1: photoionisation, which tests sigma x yield x field;
- level 2: the photodissociation rates, which add k_IR and the first-event
  branching;
- level 3: the paper II network.

The measured numbers are recorded in each docstring and in
diagnostics/pahchem/check_allain96.py. Substitutes for inputs the papers do
not state (Malloci sigma, PAHdb Einstein A, Set 2 = Set 1/1.6) are listed in
the module docstring of pahchem.allain96.
"""

import numpy as np
import pytest

from pycalima.pahchem import allain96 as A
from pycalima.pahchem import fields
from pycalima.pahchem.cards import load_card

TARGETS = load_card("allain96")["targets"]
_FAMILIES = {}


@pytest.fixture
def states(model_data):
    d = model_data / "PAH_states"
    if not (d / "C16H10_0.dat").exists():
        pytest.skip("model_data/PAH_states/ has no PAHdb modes for the Allain species (see pahchem.allain96)")
    return d


def _dex(ours, theirs):
    return float(np.log10(ours / theirs))


# --- level 0: analytic ------------------------------------------------------

@pytest.mark.parametrize("nc, nh, log_k", [(6, 6, 12.0), (10, 8, 10.1), (14, 10, 8.2), (16, 10, 7.6), (24, 12, 4.5)])
def test_rrk_convention_reproduces_paper_i_fig2(nc, nh, log_k):
    """C2H2 loss at 13.6 eV with exponent 3N - 7 over all atoms (card fig2_check)."""
    assert np.log10(A.rrk_rate(13.6, nc, nh, "C2H2")) == pytest.approx(log_k, abs=0.05)


def test_rrk_coronene_at_6ev():
    assert np.log10(A.rrk_rate(6.0, 24, 12, "C2H2")) == pytest.approx(-14.0, abs=0.05)


def test_c2h2_over_h2_is_a_tenth_at_every_energy():
    """H2 and C2H2 share E0 and the exponent, so k_C2H2/k_H2 = 1e15/1e16."""
    E = A.E_GRID[A.E_GRID > 3.0]
    for nc, nh in ((10, 8), (24, 12), (50, 20)):
        assert np.allclose(A.rrk_rate(E, nc, nh, "C2H2") / A.rrk_rate(E, nc, nh, "H2"), 0.1)


def test_ionisation_yield_is_continuous_and_reaches_078():
    y = lambda e: float(A.ion_yield(e, 7.29))
    for e in (8.9, 11.0):
        assert y(e - 1e-9) == pytest.approx(y(e + 1e-9), abs=2e-3)
    assert y(13.6) == pytest.approx(0.78, abs=0.005)
    assert y(7.0) == 0.0


def test_recombination_eq9_gives_the_benzene_cation_rate():
    """II Sect. 3.1.1: P = 0.3 fitted to 1e-6 cm^3 s^-1 at 300 K (Abouelaziz+93); measured 0.997e-6."""
    assert A.recombination_rate(6, 300.0) == pytest.approx(1.0e-6, rel=0.02)


def test_set1_field_is_draine_above_5ev():
    """I eq. 18 x 4 pi is fields.draine1978 to 0.13 % (5-13.6 eV)."""
    d = fields.draine1978()
    m = d.energy_ev > 5.01
    ratio = A.set1_field().flux_at(d.energy_ev[m]) / d.photon_flux[m]
    assert np.all(np.abs(ratio - 1.0) < 3e-3)


# --- level 1: photoionisation (sigma x yield x field) ------------------------

@pytest.mark.parametrize("name, tol", [("C14H10", 0.15), ("C16H10", 0.15), ("C24H12", 0.15), ("C32H14", 0.15),
                                       ("PAH 50 C", 0.25)])
def test_paper_i_table5_photoionisation_set1(states, name, tol):
    """Measured dex: C14H10 -0.08, C16H10 +0.03, C24H12 -0.11, C32H14 -0.07 (Malloci sigma),
    PAH 50 C +0.20 (Li & Draine sigma: a bigger tolerance for a cruder substitute)."""
    i = TARGETS["paperI_table5"]["species"].index(name)
    r = A.AllainState(name, A.load_modes(name, states)).rates(A.set1_field())
    assert abs(_dex(r["ion"], TARGETS["paperI_table5"]["set1"][i])) < tol


# --- level 2: photodissociation (adds k_IR and branching) -------------------

def _table4(name, channel):
    t = TARGETS["paperI_table4"]
    i = t["species"].index(name)
    if name == "C16H10" and channel == "C2H2":
        return 0.1 * t["H2_set1"][i]          # printed 1.29e-11: misprint (card invariant_check)
    return t[f"{channel}_set1"][i]


@pytest.mark.parametrize("name", ["C14H10", "C16H10", "C24H12"])
def test_paper_i_table4_rates_small_pahs(states, name):
    """N_C <= 24, occupation factor kept. Measured dex (H, H2, C2H2):
    C14H10 -0.00/-0.03/-0.02, C16H10 +0.09/+0.07/+0.07, C24H12 -0.06/-0.06/-0.06."""
    r = A.AllainState(name, A.load_modes(name, states)).rates(A.set1_field())
    for c in A.CHANNELS:
        assert abs(_dex(r[c], _table4(name, c))) < 0.15, (c, _dex(r[c], _table4(name, c)))


def test_paper_i_table4_rates_ovalene(states):
    """k_IR = sum A_i (occupation neglected, card). Measured -0.22 dex on all three
    channels; with the occupation kept, +0.20."""
    r = A.AllainState("C32H14", A.load_modes("C32H14", states)).rates(A.set1_field())
    for c in A.CHANNELS:
        assert abs(_dex(r[c], _table4("C32H14", c))) < 0.3


@pytest.mark.xfail(strict=True, reason=(
    "PAH 50 C is -1.35 dex below I Table 4. Its N_H0 is not stated (each 2 H "
    "change the rate ~4x through the RRK exponent), and neither is its k_IR "
    "(keeping the occupation factor raises it 10x). PAHdb C50H20 is a choice."))
def test_paper_i_table4_rates_pah50(states):
    r = A.AllainState("PAH 50 C", A.load_modes("PAH 50 C", states)).rates(A.set1_field())
    assert abs(_dex(r["H"], _table4("PAH 50 C", "H"))) < 0.3


def _set2_or_skip():
    try:
        return A.set2_field(proxy_ok=False)
    except FileNotFoundError:
        pytest.skip("Set 2 (Gondhalekar+80, I Fig. 5) not digitised: with Set 1/1.6 the small PAHs "
                    "agree with II Tables 2-3 to 0.2 dex (check_allain96.py)")


@pytest.mark.parametrize("name", ["C14H10", "C16H10", "C24H12"])
def test_paper_ii_tables_2_3_small_pahs(states, name):
    """Cations, alpha_H = 0.5 and 0.25 neutrals, bare skeletons, in the Set 2 field
    (I Fig. 5). Measured |dex| <= 0.18 over these columns (0.22 with the Set 1/1.6
    proxy)."""
    field = _set2_or_skip()
    t2, t3 = TARGETS["paperII_table2"], TARGETS["paperII_table3"]
    i, nh0 = t2["species"].index(name), A.SPECIES[name][1]
    modes = A.load_modes(name, states)
    r = lambda z, n: {k: v * A.YR for k, v in A.AllainState(name, modes, z, n).rates(field).items()}
    cat, h50, h0 = r(1, nh0), r(0, int(0.5 * nh0)), r(0, 0)
    pairs = [(cat["H"], t2["cation_H"][i]), (cat["H2"], t2["cation_H2"][i]), (h50["H"], t2["aH050_H"][i]),
             (cat["C2H2"], t3["cation"][i]), (h0["C2H2"], t3["aH000"][i])]
    assert all(abs(_dex(a, b)) < 0.25 for a, b in pairs), [round(_dex(a, b), 2) for a, b in pairs]


# --- level 3: the paper II network -------------------------------------------

def _family(name, states):
    if name not in _FAMILIES:
        _FAMILIES[name] = A.AllainFamily(name, A.load_modes(name, states))
    return _FAMILIES[name]


def test_network_is_normalised_and_charges_respond_to_ne_over_chi(states):
    fam = _family("C24H12", states)
    s = {m: fam.steady(m) for m in ("NGC 7027", "HI region")}
    for v in s.values():
        assert v["f"].sum() == pytest.approx(1.0) and np.all(v["f"] >= 0)
    # n_e/chi: 1.5e-4 (NGC 7027) vs 2e-2 (HI): the PN is ionised, the HI cloud neutral (II Sect. 3.2.2)
    assert s["NGC 7027"]["alpha_ion"] > 0.9 and s["HI region"]["alpha_ion"] < 0.1


@pytest.mark.parametrize("name, key", [("C16H10", "pyrene"), ("C32H14", "ovalene")])
def test_paper_ii_table5_single_state_destruction_times(states, name, key):
    """tau of the normal neutral and the bare cation, in all six media (Set 2).
    These are II Table 3 rates scaled by chi, so they test that the media are
    read right. Measured |dex| <= 0.21 (0.37 with the Set 1/1.6 proxy); the Orion Bar pyrene entries are skipped
    (they break the 1/chi scaling: card check)."""
    t5 = TARGETS["paperII_table5"]
    fam = _family(name, states)
    for j, med in enumerate(t5["medium"]):
        if key == "pyrene" and med == "Orion Bar":
            continue
        s = fam.steady(med)
        assert abs(_dex(s["tau_neutral_yr"], t5[f"{key}_neutral"][j])) < 0.45
        assert abs(_dex(s["tau_bare_cation_yr"], t5[f"{key}_dehydrogenated_cation"][j])) < 0.45


@pytest.mark.xfail(strict=True, reason=(
    "Network-mean tau (II Table 5) is 1-2 dex shorter than theirs for pyrene (NGC 7023, M82, HI, diffuse) "
    "and ovalene (NGC 7023, diffuse): our steady state is more dehydrogenated. Their own text has pyrene "
    "alpha_H < 1 % at n_H/chi < 2 (II Sect. 3.2.3, as we find), which the Table 5 mean (13x the bare-cation "
    "tau) does not match; suspects are II eq. 24's weighting, the H+ density, and the cation abstraction scaling."))
def test_paper_ii_table5_network_mean_destruction_time(states):
    t5 = TARGETS["paperII_table5"]
    fam = _family("C16H10", states)
    j = t5["medium"].index("NGC 7023")
    assert abs(_dex(fam.steady("NGC 7023")["tau_uv_yr"], t5["pyrene_mean"][j])) < 0.5


@pytest.mark.parametrize("name", ["PAH 50 C", "PAH 100 C"])
def test_large_pahs_fully_hydrogenated_in_uir_media(states, name):
    """II Sect. 3.2.3: in UIR media, PAHs larger than 50 C are fully hydrogenated, smaller
    ones partially dehydrogenated. Measured alpha_H = 1.00 (50 and 100 C) and 0.03-0.46
    (C14H10, C16H10) in all four media."""
    for med in ("NGC 7027", "Orion Bar", "NGC 7023", "M82"):
        assert _family(name, states).steady(med)["alpha_H"] > 0.99
        assert _family("C16H10", states).steady(med)["alpha_H"] < 0.9


def test_paper_ii_critical_sizes(states):
    """II Sect. 4: survivors > 40 C everywhere, smallest in the Orion Bar and HI cloud,
    50 C in NGC 7027, 55 C in the warm diffuse medium. Measured (Set 2, log k
    interpolated linearly in N_C between our species): Orion Bar 39.2, HI 37.9,
    M82 42.0, NGC 7023 44.4, NGC 7027 46.6, diffuse warm 64.2."""
    names = ["C10H8", "C14H10", "C16H10", "C24H12", "C32H14", "PAH 50 C", "PAH 100 C"]
    nc = [A.SPECIES[n][0] for n in names]
    crit = {}
    for med, (n_h, *_rest) in A.MEDIA.items():
        crit[med] = A.critical_size(nc, [_family(n, states).steady(med)["k_C_uv"] for n in names], n_h)
    assert min(crit, key=crit.get) in ("Orion Bar", "HI region")
    assert max(crit, key=crit.get) == "diffuse warm medium"
    assert all(35.0 < v < 70.0 for v in crit.values()), crit
    assert crit["NGC 7027"] == pytest.approx(50.0, abs=5.0)
