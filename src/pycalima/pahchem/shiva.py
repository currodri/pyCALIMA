"""The single-molecule photo-rates of SHIVA (Murga+2016, 2019, 2020) in pahchem.

Method card: ``method_cards/murga2020.toml`` ([photophysics], [heating],
[dissociation], [cooling]). The pieces, as the card reads them:

- **T_m(E)**: the Draine & Li (2001) synthetic mode spectrum of a PAH with
  N_C, N_H (DL01 eqs. 4-8: two 2-D Debye C-C sets, 600 and 1740 cm^-1,
  plus N_H C-H modes at 886, 1161, 3030 cm^-1), and the DL01 eq. 32
  temperature: the T at which the harmonic mean energy of those modes is E
  (T = hbar omega_1 / k ln 2 below the 20th mode).
- **k_diss**: Tielens' Arrhenius law at T_e = T_m (1 - 0.2 E0/E),
  k = (k T_e / h) exp(1 + dS/R) exp(-E0 / k T_e) (M20 eqs. 7-9).
- **k_IR**: the thermal photon-emission rate int C_abs 4 pi B_lambda(T_m) /
  (hc/lambda) d lambda (M16 eq. 7); the energy-loss rate P_IR the same
  without 1/(hc/lambda).
- **C_abs**: Draine & Li (2007) PAH cross sections, neutral or ionised; the
  bundled Li & Draine (2001) tables stand in for them (DL07 changed band
  strengths in the IR, not the UV). Radius a = 10 A (N_C/468)^1/3 (DL01).
- **Y_ion**: Weingartner & Draine (2001a) for carbonaceous grains, from
  ``models.dust_charge`` (graphite constants, IP_V of WD01 eq. 2).
- **p(E)**: the stationary energy distribution of the molecule under
  photon heating (C_abs (1 - Y_ion) F: ionising photons heat nothing, M20
  Sect. 2.1) and continuous IR cooling P_IR(E) (Guhathakurta & Draine 1989;
  Pavlyuchenkov+2012 use a Monte Carlo of the same process). Two readings
  are returned, because the card notes M20 does not say which it used:
  ``p_time``, the time average, and ``p_post``, the distribution just
  after an absorption.
- **Rates**:
  - ``mw22``: R = int k_diss p_time dE (Murga & Wiebe 2022 eq. 1), the
    sink-free time-averaged rate;
  - ``m20``: R = <k_diss / (k_diss + k_IR)>_{p_post} x int_5^13.6 C_abs F dE
    (M20 eqs. 4-6), the yield per absorption times the absorption rate,
    per fragment, the channels not competing (eq. 4 as printed).
  R is in fragment-loss events per molecule per second; the M20 factor-2
  ambiguity for molecules (card [dissociation].units) is not applied.

Choices where the card says "not stated" (confidence E): the radius law
(DL01); LD01 for DL07; the DL01 eq. 32 low-energy rule; p_post for the M20
yield; continuous rather than Monte Carlo cooling; the energy grid.

Gates (tests/test_pahchem_shiva.py): the DL01 T(E) reproduces DL01 Fig. 7
to 0.004 dex; the WD01a yield reproduces Murga+2019 Fig. 2 to 0.001. Murga &
Wiebe 2022 Fig. 1 is reproduced only with ``temperature="tielens"`` for the
Arrhenius forms (``tielens_temperature``); RRK, which has no T(E), agrees
within 0.33 dex for chi >= 100.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pycalima.pahchem import dos, fields

H_ERG_S = 6.62607015e-27
C_CM_S = 2.99792458e10
KB_ERG_K = 1.380649e-16
EV_ERG = 1.602176634e-12
HC_EV_UM = 1.23984193
R_CAL = 1.98720425864       # cal K^-1 mol^-1
KB_EV_K = 8.617333262e-5

#: DL01 Sect. 2.1
THETA_OP_CM, THETA_IP_CM = 600.0, 1740.0
CH_MODES_CM = (886.0, 1161.0, 3030.0)

#: M20 Table 1: (E0 [eV], dS [cal K^-1 mol^-1]); "superh" applies for X_H > X_H0
CHANNELS = {"H": (4.3, 11.8), "H_superh_neg": (1.4, 13.3), "H_superh_pos": (1.55, 13.3),
            "H2": (3.52, -12.69), "C2H2": (4.6, 10.0), "C2H2_superh": (2.0, 10.0)}
E_DESTRUCTION_MIN = 5.0      # eV, lower limit of M20 eq. 6


# ---------------------------------------------------------------------------
# Draine & Li (2001): modes, radius, temperature
# ---------------------------------------------------------------------------

def dl01_nh(nc):
    """DL01 eq. 8: the H count of their 'astronomical PAH' of N_C carbons."""
    if nc <= 25:
        return int(0.5 * nc + 0.5)
    if nc <= 100:
        return int(2.5 * np.sqrt(nc) + 0.5)
    return int(0.25 * nc + 0.5)


def radius_cm(nc):
    """DL01: a = 10 A (N_C / 468)^1/3."""
    return 1e-7 * (nc / 468.0) ** (1.0 / 3.0)


def _debye_modes(nm, theta_cm, nc):
    """DL01 eqs. 4-7 with n = 2: nm modes of a 2-D Debye set [cm^-1]."""
    if nc <= 54:
        beta = 0.0
    elif nc <= 102:
        beta = (nc - 54) / 52.0 / (2 * nm - 1)
    else:
        beta = ((nc - 2) / 52.0 * (102.0 / nc) ** (2.0 / 3.0) - 1.0) / (2 * nm - 1)
    j = np.arange(1, nm + 1, dtype=float)
    delta = np.where((j == 2) | (j == 3), 1.0, 0.5)
    return theta_cm * np.sqrt((1.0 - beta) / nm * (j - delta) + beta)


def dl01_modes(nc, nh):
    """The DL01 synthetic mode spectrum [cm^-1]: 3 (N_C + N_H - 2) modes."""
    op = _debye_modes(nc - 2, THETA_OP_CM, nc)
    ip = _debye_modes(2 * (nc - 2), THETA_IP_CM, nc)
    ch = np.repeat(CH_MODES_CM, nh)
    return np.sort(np.concatenate([op, ip, ch]))


def _stationary_banded(jump, cool):
    """Stationary p of G p = 0, sum p = 1, for up-jumps i -> i + k at jump[k]
    (k >= 1; jumps past the top bin land in it) and continuous cooling
    i -> i - 1 at cool[i].

    G is banded (lower bandwidth = the largest k with jump[k] > 0, upper 1)
    and solved by ``energy_me.stationary_banded``; same solution as the
    dense solve it replaces, without the O(n^3).
    """
    from pycalima.pahchem.energy_me import stationary_banded

    n = cool.size
    nz = np.nonzero(jump[1:])[0]
    lo = int(max(1, nz.max() + 1 if nz.size else 1))
    up = 1
    ab = np.zeros((up + lo + 1, n))
    for k in range(1, lo + 1):                       # Q[j + k, j] = jump[k]
        ab[up + k, : n - k] += jump[k]
    tail = np.concatenate([np.cumsum(jump[::-1])[::-1], [0.0]])      # tail[m] = sum_{k >= m} jump[k]
    j = np.arange(max(0, n - 1 - lo), n)
    ab[up + (n - 1 - j), j] += tail[np.minimum(n - j, tail.size - 1)]   # clamped into the top bin
    ab[up - 1, 1:] += cool[1:]                       # Q[i - 1, i] = cool[i]
    ab[up] -= jump[1:].sum() + cool                  # diagonal: all losses
    p = stationary_banded(ab, lo, up)
    if p is None:
        raise np.linalg.LinAlgError("singular stationary system")
    return p


_T_GRID = np.logspace(0.0, 5.0, 4000)


@lru_cache(maxsize=512)
def _u_of_set(freqs_bytes):
    """U(T) [cm^-1] of a set of modes on _T_GRID (keyed on the frequency bytes)."""
    return dos.thermal_energy(np.frombuffer(freqs_bytes, dtype=float), _T_GRID)


class Temperature:
    """T_m(E) of DL01 eq. 32 for one mode set, by inverting U(T) on a log grid."""

    def __init__(self, modes_cm):
        self.modes = np.sort(np.asarray(modes_cm, float))
        self._T = _T_GRID
        # U(T) = sum over distinct frequencies of multiplicity x U_f(T). The DL01
        # C-H modes are 3 frequencies x N_H and the C-C set depends on N_C only,
        # so both parts are cached: a rate table over N_H at fixed N_C pays for
        # its C-C modes once.
        freqs, mult = np.unique(self.modes, return_counts=True)
        single = freqs[mult == 1]
        U = _u_of_set(single.tobytes()) if single.size else np.zeros(_T_GRID.size)
        for f, m in zip(freqs[mult > 1], mult[mult > 1]):
            U = U + m * _u_of_set(np.array([f]).tobytes())
        self._U = U / dos.CM_PER_EV                                                  # eV
        self.e20_ev = self.modes[min(19, self.modes.size - 1)] / dos.CM_PER_EV
        self.t_low = self.modes[0] / (dos.KB_CM * np.log(2.0))

    def __call__(self, energy_ev):
        E = np.asarray(energy_ev, float)
        T = np.interp(np.log(np.maximum(E, 1e-30)), np.log(self._U), self._T)
        return np.where(E <= self.e20_ev, self.t_low, T)


def tielens_temperature(nc):
    """T_m(E) = 2000 K (E[eV] / N_C)^0.4 (Tielens 2005), as a callable E -> T.

    Not SHIVA's stated relation (M20 cites DL01), but the one that reproduces
    Murga & Wiebe 2022 Fig. 1: with it their Arrh. 1 and Arrh. 2 curves are
    matched within 0.1-0.6 dex, against 1.2-3.4 dex with the DL01 T(E)
    (check_shiva.py). DL01 runs ~5-10 % hotter at 10-14 eV, which the
    Arrhenius exponent (E0/kT ~ 40-50) turns into 2-3 dex.
    """
    return lambda energy_ev: 2000.0 * (np.maximum(np.asarray(energy_ev, float), 0.0) / nc) ** 0.4


# ---------------------------------------------------------------------------
# Rate laws (M20 eqs. 7-9; MW22 Table 1)
# ---------------------------------------------------------------------------

def k_arrhenius_te(energy_ev, E0_ev, dS_cal, t_m, te_correction=True):
    """(k T_e/h) exp(1 + dS/R) exp(-E0/k T_e) [s^-1]; T_e = T_m(E)(1 - 0.2 E0/E).

    ``t_m`` is a callable E [eV] -> T [K]. Zero for E <= E0.
    ``te_correction=False`` uses T_e = T_m (a diagnostic).
    """
    E = np.atleast_1d(np.asarray(energy_ev, float))
    k = np.zeros_like(E)
    up = E > E0_ev
    T = t_m(E[up]) * ((1.0 - 0.2 * E0_ev / E[up]) if te_correction else 1.0)
    k[up] = KB_ERG_K * T / H_ERG_S * np.exp(1.0 + dS_cal / R_CAL) * np.exp(-E0_ev / (KB_EV_K * T))
    return k if np.ndim(energy_ev) else k[0]


# ---------------------------------------------------------------------------
# Optics: C_abs (LD01 tables for DL07), k_IR, P_IR, Y_ion (WD01a)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _ld01(nc, charge):
    from pycalima.models.PAH_radiation.pah_oppacity import interpolate_pah_cross_sections_2d
    from pycalima.pahchem.photo import _li_draine_table

    kind = "iPAH" if charge != 0 else "nPAH"
    _, wav_cm, _, c_abs, _ = interpolate_pah_cross_sections_2d(kind, radius_cm(nc) * 1e4,
                                                               data_table=_li_draine_table(kind))
    order = np.argsort(wav_cm)
    return np.asarray(wav_cm)[order], np.asarray(c_abs)[order]


def c_abs(nc, charge, energy_ev):
    """C_abs [cm^2] at photon energies [eV] (LD01 tables standing in for DL07)."""
    wav_cm, c = _ld01(nc, charge)
    lam_cm = HC_EV_UM * 1e-4 / np.asarray(energy_ev, float)
    return np.interp(lam_cm, wav_cm, c)


def _planck_lambda(T, wav_cm):
    x = H_ERG_S * C_CM_S / (wav_cm[None, :] * KB_ERG_K * np.maximum(T, 1e-3)[:, None])
    with np.errstate(over="ignore"):
        return 2.0 * H_ERG_S * C_CM_S**2 / wav_cm[None, :] ** 5 / np.expm1(np.minimum(x, 700.0))


def _ir_rates_exact(nc, charge, T):
    wav_cm, c = _ld01(nc, charge)
    B = _planck_lambda(T, wav_cm) * 4.0 * np.pi * c[None, :]                       # erg s^-1 cm^-1
    p = np.trapezoid(B, wav_cm, axis=1) / EV_ERG
    k = np.trapezoid(B * wav_cm[None, :] / (H_ERG_S * C_CM_S), wav_cm, axis=1)
    return k, p


_IR_LOG_T = np.linspace(np.log(2.0), np.log(1e5), 3000)


@lru_cache(maxsize=None)
def _ir_spline(nc, charge):
    """Cubic splines of ln k_IR and ln P_IR in ln T, 2-1e5 K, for one (N_C, Z)."""
    from scipy.interpolate import CubicSpline

    k, p = _ir_rates_exact(nc, charge, np.exp(_IR_LOG_T))
    return CubicSpline(_IR_LOG_T, np.log(k)), CubicSpline(_IR_LOG_T, np.log(p))


def ir_rates(nc, charge, temperature_k):
    """(k_IR [photons s^-1], P_IR [eV s^-1]) of the thermal approximation at T.

    The Planck integral over the LD01 C_abs depends on (N_C, Z, T) only, so it
    is tabulated once per (N_C, Z) and evaluated by cubic splines in
    ln k - ln T (measured relative error 5e-10 against the direct integral;
    gate in tests/test_pahchem_shiva.py); temperatures outside 2-1e5 K use
    the direct integral.
    """
    T = np.atleast_1d(np.asarray(temperature_k, float))
    inside = (T >= 2.0) & (T <= 1e5)
    k, p = np.empty_like(T), np.empty_like(T)
    if inside.any():
        sk, sp = _ir_spline(nc, charge)
        lt = np.log(T[inside])
        k[inside], p[inside] = np.exp(sk(lt)), np.exp(sp(lt))
    if (~inside).any():
        k[~inside], p[~inside] = _ir_rates_exact(nc, charge, T[~inside])
    return k, p


@lru_cache(maxsize=None)
def _graphite_im():
    from pycalima._paths import get_optical_props_path
    from pycalima.models.dust_radiation.dust_oppacity import read_dielectric_file

    d = str(get_optical_props_path("draine_lee_1984"))
    perp = read_dielectric_file(f"{d}/callindex.out_CpeD03_0.10")["table"]
    par = read_dielectric_file(f"{d}/callindex.out_CpaD03_0.10")["table"]
    return (np.asarray(perp["wavelength_um"])[::-1], np.asarray(perp["Im_n"])[::-1],
            np.asarray(par["wavelength_um"])[::-1], np.asarray(par["Im_n"])[::-1])


def y_ion_wd01(energy_ev, a_cm, charge):
    """Weingartner & Draine (2001a) photoelectric yield of a carbonaceous grain."""
    from pycalima.models.dust_charge import dust_charging as DC

    E = np.atleast_1d(np.asarray(energy_ev, float))
    wav_um = HC_EV_UM / E
    wp, ip, wa, ia = _graphite_im()
    y = DC.photoelectric_yield_graphite_vec(DC.graphite_work_function, [charge], a_cm, DC.electron_escape_length,
                                            E, wav_um * 1e-4, np.interp(wav_um, wp, ip), np.interp(wav_um, wa, ia))
    return y[:, 0] if np.ndim(energy_ev) else y[0, 0]


# ---------------------------------------------------------------------------
# The molecule
# ---------------------------------------------------------------------------

class ShivaMolecule:
    """One PAH (N_C, N_H, Z) in SHIVA's description."""

    def __init__(self, nc, nh=None, charge=0, de_ev=0.05, e_max_ev=50.0, temperature="dl01"):
        """``temperature``: "dl01" (M20, the default) or "tielens" (MW22 Fig. 1)
        for the T_m of k_diss; k_IR and P_IR always use the DL01 T(E)."""
        self.nc, self.charge = nc, charge
        self.nh = dl01_nh(nc) if nh is None else nh
        self.a_cm = radius_cm(nc)
        self.modes = dl01_modes(nc, self.nh)
        self.t_dl01 = Temperature(self.modes)
        self.t_m = {"dl01": self.t_dl01, "tielens": tielens_temperature(nc)}[temperature]
        self.energy_ev = np.arange(0.0, e_max_ev + 0.5 * de_ev, de_ev)
        self.de_ev = de_ev
        self.k_ir, self.p_ir = ir_rates(nc, charge, self.t_dl01(np.maximum(self.energy_ev, de_ev)))
        self.k_ir[0] = self.p_ir[0] = 0.0

    def k_diss(self, channel, te_correction=True):
        E0, dS = CHANNELS[channel]
        return k_arrhenius_te(self.energy_ev, E0, dS, self.t_m, te_correction)

    def heating(self, field, ionising_photons_heat=False):
        """(E_ph, absorption rate per eV of heating photons) [s^-1 eV^-1]."""
        E = field.energy_ev
        h = c_abs(self.nc, self.charge, E) * field.photon_flux
        if not ionising_photons_heat:
            h = h * (1.0 - y_ion_wd01(E, self.a_cm, self.charge))
        return E, h

    def distribution(self, field, ionising_photons_heat=False):
        """(p_time, p_post) on ``energy_ev``: stationary GD89 distribution with
        continuous cooling, and the distribution just after an absorption."""
        E_ph, h = self.heating(field, ionising_photons_heat)
        n, de = self.energy_ev.size, self.de_ev
        # integrated absorption rate per energy jump bin k (jump in [(k-1/2), (k+1/2)) de)
        edges = (np.arange(n + 1) - 0.5) * de
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(E_ph))])
        jump = np.diff(np.interp(np.clip(edges, E_ph[0], E_ph[-1]), E_ph, cum))
        jump[0] = 0.0
        cool = self.p_ir / de
        cool[0] = 0.0
        p = _stationary_banded(jump, cool)
        post = np.convolve(p, jump)[:n]              # p_post[i + k] += p[i] jump[k], truncated at the top
        post /= post.sum()
        return p, post

    def absorption_rate(self, field, e_min=E_DESTRUCTION_MIN):
        """int_{e_min}^{13.6} C_abs F dE [s^-1] (M20 eq. 6)."""
        E = field.energy_ev
        m = E >= e_min
        return float(np.trapezoid(c_abs(self.nc, self.charge, E[m]) * field.photon_flux[m], E[m]))

    def rates(self, field, k_channels, ionising_photons_heat=False):
        """{channel: {"mw22", "m20", "m20_time"}} [s^-1] for k(E) arrays on ``energy_ev``."""
        p_time, p_post = self.distribution(field, ionising_photons_heat)
        r_abs = self.absorption_rate(field)
        out = {}
        for name, k in k_channels.items():
            y = np.where(k > 0, k / (k + self.k_ir + 1e-300), 0.0)
            out[name] = {"mw22": float(p_time @ k), "m20": float(p_post @ y) * r_abs,
                         "m20_time": float(p_time @ y) * r_abs}
        return out


def mmp83(chi):
    """The MMP83 field scaled by chi (M20 footnote 3)."""
    f = fields.mathis1983()
    return fields.RadiationField(f.energy_ev, f.photon_flux * chi, f"MMP83 x {chi:g}")
