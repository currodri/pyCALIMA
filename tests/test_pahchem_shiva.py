"""SHIVA's single-molecule photo-rates (pahchem.shiva), gated one piece at a time.

- DL01 modes and T(E) against Draine & Li 2001 Fig. 7 (vector-extracted);
- the WD01a yield against Murga+2019 Fig. 2;
- the Arrhenius law against pahchem.rates on the same modes;
- p(E): normalisation, energy balance, the single-photon limit;
- the rates: Y_sp <= 1, R_m20 <= absorption, growth with chi;
- Murga & Wiebe 2022 Fig. 1 (vector-extracted), form by form.
Measured numbers are in the docstrings and in diagnostics/pahchem/check_shiva.py.
"""

import numpy as np
import pandas as pd
import pytest

from pycalima._paths import get_external_data_path
from pycalima.pahchem import dos, rates
from pycalima.pahchem import shiva as S

YR = 3.15576e7
EV = dos.CM_PER_EV
_MOL = {}


def _csv(name):
    return pd.read_csv(get_external_data_path(f"pahchem/{name}"), comment="#")


def _mol(nc, nh, z=0, temperature="dl01"):
    key = (nc, nh, z, temperature)
    if key not in _MOL:
        _MOL[key] = S.ShivaMolecule(nc, nh, z, temperature=temperature)
    return _MOL[key]


def _mw22(product, method, chi):
    g = _csv("murga_wiebe22_fig1.csv")
    g = g[(g["product"] == product) & (g.method == method)].sort_values("chi")
    return float(np.interp(np.log10(chi), np.log10(g.chi), np.log10(g.rate_per_yr)))


# --- Draine & Li 2001 --------------------------------------------------------

@pytest.mark.parametrize("nc", [14, 24, 54, 96, 150])
def test_dl01_mode_count(nc):
    nh = S.dl01_nh(nc)
    m = S.dl01_modes(nc, nh)
    assert m.size == 3 * (nc + nh - 2) and np.all(m > 0)


def test_dl01_sequence_contains_the_coronene_family():
    """DL01 Sect. 2.1: coronene, circumcoronene, dicircumcoronene follow eq. 8."""
    assert [S.dl01_nh(n) for n in (24, 54, 96)] == [12, 18, 24]


@pytest.mark.parametrize("nc", [14, 24, 50, 100])
def test_dl01_temperature_reproduces_dl01_fig7(nc):
    """T(E) above 2000 cm^-1: measured median |dex| <= 0.0015, max 0.004."""
    t = _csv("shiva_dl01_fig7_temperature.csv")
    g = t[(t.n_c == nc) & (t.energy_cm > 2e3)]
    T = S.Temperature(S.dl01_modes(nc, S.dl01_nh(nc)))(g.energy_cm.values / EV)
    assert np.abs(np.log10(T / g.temperature_k.values)).max() < 0.01


# --- WD01a ionisation yield --------------------------------------------------

@pytest.mark.parametrize("z", [0, 1, 2, 3])
def test_wd01_yield_reproduces_murga2019_fig2(z):
    """a = 3.3 A: measured max |diff| 0.0012 (Z = 0) to 0.0004 (Z = 3)."""
    y = _csv("murga2019_fig2_yion.csv")
    g = y[y.z == z]
    assert np.abs(S.y_ion_wd01(g.energy_eV.values, 3.3e-8, z) - g.y_ion.values).max() < 0.005


# --- the rate law -------------------------------------------------------------

def test_arrhenius_matches_pahchem_rates_on_the_same_modes():
    """M20 eqs. 7-9 with T_m from the DL01 modes equal rates.arrhenius_effective_temperature
    on those modes (e exp(dS/R) = exp(1 + dS/R)), up to the T(E) interpolation."""
    modes = S.dl01_modes(54, 18)
    E = np.array([6.0, 9.0, 13.0])
    ours = S.k_arrhenius_te(E, 4.3, 11.8, S.Temperature(modes))
    ref = rates.arrhenius_effective_temperature(modes, E * EV, 4.3 * EV, 11.8 * 4.184)
    assert np.allclose(ours, ref, rtol=0.02)


@pytest.mark.parametrize("nc, z", [(16, 0), (66, 1), (150, 0)])
def test_ir_rate_splines_equal_the_planck_integral(nc, z):
    """ir_rates tabulates the Planck integral per (N_C, Z) as ln-ln cubic splines:
    measured <= 5e-10 relative against the direct integral over 2-1e5 K."""
    T = np.exp(np.random.default_rng(nc).uniform(np.log(2.0), np.log(1e5), 2000))
    k, p = S.ir_rates(nc, z, T)
    ke, pe = S._ir_rates_exact(nc, z, T)
    assert np.allclose(k, ke, rtol=1e-8) and np.allclose(p, pe, rtol=1e-8)


def test_stationary_distribution_equals_the_dense_solve():
    """The banded p(E) solve against np.linalg.solve on the same generator,
    including the extreme-field corner (C150H30, chi = 1e6) where a normalisation
    row of empty bins would lose 8 digits, and a 20 A grain (N_C = 3688) at
    chi = 1e4, where the first-pass row 0 overflows."""
    for nc, nh, chi in ((16, 10, 1.0), (150, 30, 1e6), (3688, 40, 1e4)):
        m = S.ShivaMolecule(nc, nh, 0)
        f = S.mmp83(chi)
        E_ph, h = m.heating(f, False)
        n, de = m.energy_ev.size, m.de_ev
        edges = (np.arange(n + 1) - 0.5) * de
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(E_ph))])
        jump = np.diff(np.interp(np.clip(edges, E_ph[0], E_ph[-1]), E_ph, cum))
        jump[0] = 0.0
        G = np.zeros((n, n))
        for i in range(n):
            G[i + 1:, i] += jump[1:n - i]
            G[n - 1, i] += jump[n - i:].sum()
        cool = m.p_ir / de
        cool[0] = 0.0
        G[np.arange(n - 1), np.arange(1, n)] += cool[1:]
        G[np.arange(n), np.arange(n)] -= G.sum(axis=0)
        A = G.copy()
        A[0] = 1.0
        b = np.zeros(n)
        b[0] = 1.0
        dense = np.clip(np.linalg.solve(A, b), 0.0, None)
        dense /= dense.sum()
        p, _ = m.distribution(f)
        assert np.max(np.abs(p - dense)) < 1e-10 * dense.max()


# --- p(E) ---------------------------------------------------------------------

@pytest.mark.parametrize("chi", [1.0, 1e3])
def test_distribution_is_normalised_and_conserves_energy(chi):
    """Stationary p: absorbed power = radiated power (no overflow at these chi)."""
    m = _mol(54, 18)
    f = S.mmp83(chi)
    p, post = m.distribution(f)
    assert p.sum() == pytest.approx(1.0) and post.sum() == pytest.approx(1.0)
    E_ph, h = m.heating(f)
    absorbed = np.trapezoid(h * E_ph, E_ph)
    assert p @ m.p_ir == pytest.approx(absorbed, rel=0.03)


def test_low_field_is_the_single_photon_limit():
    """At chi = 1 the time-averaged Arrhenius rate equals the single-photon-from-cold
    integral int h(eps) int_0^eps k/P_IR dE deps (measured within 0.05 dex)."""
    m = _mol(66, 20)
    f = S.mmp83(1.0)
    k = m.k_diss("H")
    p, _ = m.distribution(f)
    E_ph, h = m.heating(f)
    tau = np.where(m.p_ir > 0, 1.0 / np.maximum(m.p_ir, 1e-300), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * ((k * tau)[1:] + (k * tau)[:-1]) * m.de_ev)])
    single = np.trapezoid(h * np.interp(E_ph, m.energy_ev, cum), E_ph)
    assert np.log10(p @ k / single) == pytest.approx(0.0, abs=0.1)


# --- rates --------------------------------------------------------------------

def test_m20_rates_are_bounded_and_grow_with_chi():
    m = _mol(24, 12)
    ks = {c: m.k_diss(c) for c in ("H", "H2", "C2H2")}
    prev = None
    for chi in (1.0, 1e2, 1e4):
        f = S.mmp83(chi)
        r = m.rates(f, ks)
        r_abs = m.absorption_rate(f)
        assert all(v["m20"] <= r_abs * (1 + 1e-9) for v in r.values())
        if prev:
            assert all(r[c]["m20"] > prev[c]["m20"] for c in ks)
        prev = r


# --- Murga & Wiebe 2022 Fig. 1 (C66H20) --------------------------------------

def _rate(m, k, chi):
    return float(np.log10(m.distribution(S.mmp83(chi), ionising_photons_heat=True)[0] @ k * YR))


@pytest.mark.parametrize("chi", [1e2, 1e4])
def test_mw22_rrk_h_loss(chi):
    """RRK has no T(E) and no free convention, so it tests p(E) and the absorption:
    measured -0.15 (chi = 1e2) and +0.33 (1e4) dex."""
    m = _mol(66, 20)
    k = rates.rrk(m.energy_ev * EV, m.modes.size, 1e16, 2.8 * EV)
    assert _rate(m, k, chi) == pytest.approx(_mw22("H", "RRK", chi), abs=0.4)


@pytest.mark.xfail(strict=True, reason=(
    "MW22 RRK H loss at chi = 1 is 1.4 dex above ours: there RRK needs two-photon events, so it measures "
    "the far tail of p(E), which our continuous GD89 cooling (DL01 thermal P_IR, LD01 C_abs) makes lighter "
    "than their Pavlyuchenkov+12 Monte Carlo. Within 0.33 dex at chi >= 100."))
def test_mw22_rrk_h_loss_at_chi_1():
    m = _mol(66, 20)
    k = rates.rrk(m.energy_ev * EV, m.modes.size, 1e16, 2.8 * EV)
    assert _rate(m, k, 1.0) == pytest.approx(_mw22("H", "RRK", 1.0), abs=0.5)


ARRH = {("H", "Arrh. 1"): (3.3, 5.0), ("H", "Arrh. 2"): (4.3, 11.8),
        ("C", "Arrh. 1"): (3.65, 5.0), ("C", "Arrh. 2"): (4.6, 10.0)}


@pytest.mark.parametrize("key", sorted(ARRH))
def test_mw22_arrhenius_with_tielens_temperature(key):
    """With T_m = 2000 K (E/N_C)^0.4 the Arrhenius curves are reproduced at
    chi = 1e2 (chi = 1 too where plotted): measured H +0.58 / +0.55, C -0.08 /
    +0.04 dex (Arrh. 1 / 2)."""
    m = _mol(66, 20, temperature="tielens")
    k = S.k_arrhenius_te(m.energy_ev, *ARRH[key], m.t_m)
    assert _rate(m, k, 1e2) == pytest.approx(_mw22(*key, 1e2), abs=0.7)


@pytest.mark.xfail(strict=True, reason=(
    "With the DL01 T(E) that M20 states for SHIVA, MW22's Arrhenius curves are 1.6-3.4 dex below ours at "
    "chi <= 100. DL01 runs 5-10 % hotter than Tielens' 2000 K (E/N_C)^0.4 at 10-14 eV, which the Arrhenius "
    "exponent (E0/kT ~ 40-50) turns into 2-3 dex; MW22 evidently used Tielens' relation."))
@pytest.mark.parametrize("key", sorted(ARRH))
def test_mw22_arrhenius_with_dl01_temperature(key):
    m = _mol(66, 20)
    k = S.k_arrhenius_te(m.energy_ev, *ARRH[key], m.t_m)
    assert _rate(m, k, 1e2) == pytest.approx(_mw22(*key, 1e2), abs=0.7)


def test_mw22_laplace_h_loss(model_data):
    """MW22 take the Laplace frequencies from PAHdb (C66H20, uid 115). Measured
    -4.1 / -2.9 / -1.9 dex at chi = 1 / 1e2 / 1e4: their Laplace k(E) behaves
    as if rho had fewer modes (unresolved; recorded, so this is a bound)."""
    path = model_data / "PAH_states" / "C66H20_0.dat"
    if not path.exists():
        pytest.skip("C66H20_0.dat not in model_data/PAH_states (extract PAHdb uid 115)")
    from pycalima.pahchem.molecule import PAHSpecies

    m = _mol(66, 20)
    sp = PAHSpecies.from_states_file(path)
    Eg, counts = dos.beyer_swinehart(sp.freqs_cm, m.energy_ev[-1] * EV + 200.0)
    k = rates.inverse_laplace_arrhenius(Eg, counts, 1.4e16, 4.3 * EV)[np.rint(m.energy_ev * EV).astype(int)]
    d = _rate(m, k, 1e2) - _mw22("H", "Laplace", 1e2)
    assert -3.5 < d < -2.3
