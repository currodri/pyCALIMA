"""Band positions and profiles given to DFT modes: named conventions, per mode class, optionally per emission T.

A convention turns each mode (its PAHdb v4.00 frequency, already scaled) into a line: a shift from the DFT position,
a FWHM and a profile (Lorentzian, or a pseudo-Voigt with Lorentzian fraction eta). Three are defined:

- ``LEGACY``: every mode a Lorentzian of FWHM 15 cm^-1, shifted -15 cm^-1 (pahchem's convention until 2026-10-26).
  It is not what PAHdb uses: the 15 cm^-1 redshift is Bauschlicher+10's (4-31G era; "may be somewhat
  overestimated"; Boersma+13 Sect. 3: experiments only at N_C <= 32), the 15 cm^-1 FWHM Boersma+13's Gaussian. PAHdb
  v4.00 applies no shift (Ricca+25, pyPAHdb). On v4.00 positions it puts the 11.2 um band at 11.39 um, against
  11.195 um observed (Orion Bar atomic PDR).
- ``SOURCED``: v4.00 positions plus, per class, the laws below at the class's typical emission temperature on
  pahchem's cascade (``T_REP``): constant per class, no cascade needed.
- ``SOURCED_TDEP``: each photon takes the laws at its own emission temperature, that of the cascade bin it leaves
  (Pech+02-style; ``Population`` builds the per-class temperature distributions, ``emission_temperature_shares``).

The laws, per class of the scaled frequency (``CLASSES``):

- **Shifts:** chi'_cl x T_cl, where T_cl is the classical temperature of equal vibrational energy,
  E_vib(T) / (s k_B) for the s modes of the state (energy matching). chi'_cl is a classical-MD slope:
  - out-of-plane (solo 860-1000, duo 800-860, trio and quartet 600-800 cm^-1): Mai+25's ML-MD slopes by class and
    N_C;
  - CC 6.2 (1500-1800): Mai+25, -0.056 (coronene lab -0.038 / dT_cl/dT 0.70 = -0.054);
  - 7.7 (1200-1500) and 8.6 (1000-1200): -0.034 and -0.012 (Pech+02 Table 2, coronene, / 0.70);
  - CH (2600-3300): 0.75 x Mai+25 by N_C (energy matching overshoots Joblin+95's three slopes by x1.2-1.4; the
    hot-minus-cold totals of Joblin+94 favour x1.0, so the bracket is 0.75-1.0; the 3.3 window does not see it).
  Energy matching reproduces the lab slopes (ovalene oop -0.018, ovalene solo -0.0095 against -0.0098, coronene 6.2
  -0.040 against -0.038) and the plateau below ~200-300 K (Pech+02 Sect. 3.1, Chakraborty+21).
  ``t_map=("factor", q)`` replaces T_cl by q x T, the earlier mapping (q = 0.5), kept as a bracket.
- **Widths** (quantum T, rotation removed): CH pyrene 10.03 + 0.0159 T, pseudo-Voigt eta 0.562 (Demyk+26,
  CosmicPAH-IRDB); 6.2 1.00 + 0.0230 T, 7.7 0.56 + 0.0083 T and 8.6 0.66 + 0.0114 T (Pech+02 Table 2); out-of-plane
  2.6 + chi''_cl T_cl, Lorentzian (2.6 cm^-1 the pyrene duo intercept; chi''_cl Mai+25's width slope by class and
  N_C; Mackie+22 Fig. 7 gives 5.6 cm^-1 for C96H24 at 8 eV over the whole cascade).
- **Cold positions:** the v4.00 frequencies, with the solo modes moved by ``solo_offset_cm`` = +3 cm^-1
  (jet-cooled ovalene, Lemmens+21: 884.0 against 881.0; the oop median over their four molecules is +1, range
  -7..+11). Optional ``cation_solo_offset_cm`` on cation solo modes: Ricca+21's cc-pVTZ C52H18+ puts the band ~10
  cm^-1 above 6-31G*, the matrix data 4-6 cm^-1; the bracket is +7 +- 3. It is 0 by default: the observed 11.0 band
  (908.6 cm^-1) already sits on the v4.00 cation solos.
- **Other modes** (< 600, 1800-2600 cm^-1): no shift, Lorentzian FWHM 5 cm^-1 (no law; outside every band window).

Assumed, not sourced: the CH factor 0.75, the neutral laws for ions (Oomens+03's naphthalene+ is within ~30 %), the
trio/quartet mean, and Mai+25's N_C bins. Sources and measurements: ``results/pahchem/band_profiles_2026-10-26``
(README Sects. 5 and 7, conventions.toml).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

#: mode classes by scaled frequency [cm^-1]: (name, lo, hi); the rest is "other"
CLASSES = (("ch", 2600.0, 3300.0), ("cc62", 1500.0, 1800.0), ("cc77", 1200.0, 1500.0), ("cc86", 1000.0, 1200.0),
           ("solo", 860.0, 1000.0), ("duo", 800.0, 860.0), ("trioquartet", 600.0, 800.0))
OTHER = len(CLASSES)
N_CLASSES = OTHER + 1
SOLO = 4
OOP = (4, 5, 6)

#: Mai+25 N_C bins (lower edges) and the medians of the repo's class fits (classical T; ``mai25_class_laws.log``)
NC_EDGES = (0.0, 30.0, 50.0, 80.0, 130.0)
MAI_SLOPE = {"ch": (-0.062, -0.065, -0.089, -0.096, -0.108),
             "solo": (-0.022, -0.024, -0.013, -0.011, -0.011), "duo": (-0.026, -0.025, -0.012, -0.006, -0.006),
             "trioquartet": (-0.012, -0.0035, -0.0085, 0.0, -0.002)}
MAI_WIDTH_SLOPE = {"solo": (0.015, 0.013, 0.007, 0.006, 0.005), "duo": (0.010, 0.013, 0.006, 0.005, 0.005),
                   "trioquartet": (0.013, 0.017, 0.006, 0.005, 0.007)}
CH_MAI_FACTOR = 0.75
OOP_W0 = 2.6
#: classical slope chi'_cl [cm^-1/K] and width w0 + w1 T (quantum) and eta of the non-oop classes
FIXED = {"ch": (None, 10.03, 0.0159, 0.562), "cc62": (-0.056, 1.00, 0.0230, 1.0), "cc77": (-0.034, 0.56, 0.0083, 1.0),
         "cc86": (-0.012, 0.66, 0.0114, 1.0)}
OTHER_FWHM = 5.0
#: typical emission T on the cascade (photon-weighted, strongest mode, 8 eV): CH ~840 K, CC ~720 K, oop 690 K at
#: C32 falling to 460 K at C96 (``tdep_profiles.log``); used by the constant ``SOURCED``
T_REP = {"ch": 840.0, "cc": 720.0, "oop_small": 690.0, "oop_large": 550.0}
#: emission-temperature bins of ``SOURCED_TDEP`` [K]
T_EDGES = np.arange(0.0, 6000.0 + 1.0, 25.0)
T_MID = 0.5 * (T_EDGES[1:] + T_EDGES[:-1])
_FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))


def mode_classes(freqs_cm):
    """Class index of every mode (``CLASSES`` order; ``OTHER`` for the rest)."""
    f = np.asarray(freqs_cm, dtype=float)
    c = np.full(f.shape, OTHER, dtype=int)
    for i, (_, lo, hi) in enumerate(CLASSES):
        c[(f >= lo) & (f < hi)] = i
    return c


def _nc_bin(nc):
    return int(np.searchsorted(NC_EDGES, float(nc), side="right") - 1)


def classical_temperature(freqs_cm, temperature_k):
    """T_cl [K] with the vibrational energy of the modes at T: E_vib(T) / (s k_B) (energy matching)."""
    from pycalima.pahchem import dos

    f = np.asarray(freqs_cm, dtype=float)
    T = np.asarray(temperature_k, dtype=float)
    return dos.thermal_energy(f, np.maximum(T, 1e-3)) / (f.size * dos.KB_CM)


@dataclass(frozen=True)
class BandConvention:
    """A band convention (module docstring). ``kind``: "legacy", "constant" (laws at ``T_REP``) or "tdep";
    ``t_map``: "energy" (T_cl by energy matching) or ("factor", q) (T_cl = q T)."""

    name: str
    kind: str = "tdep"
    t_map: object = "energy"
    ch_factor: float = CH_MAI_FACTOR
    solo_offset_cm: float = 3.0
    cation_solo_offset_cm: float = 0.0

    def __post_init__(self):
        if self.kind not in ("legacy", "constant", "tdep"):
            raise ValueError(f"kind must be 'legacy', 'constant' or 'tdep', not {self.kind!r}")
        if not (self.t_map == "energy" or (isinstance(self.t_map, tuple) and len(self.t_map) == 2
                                           and self.t_map[0] == "factor")):
            raise ValueError(f"t_map must be 'energy' or ('factor', q), not {self.t_map!r}")

    # -- the laws ------------------------------------------------------------------------------------
    def class_law(self, c, nc):
        """(chi'_cl [cm^-1/K], w0, w1, width on T_cl (else T), eta) of class index ``c`` at ``nc``."""
        if c == OTHER:
            return 0.0, OTHER_FWHM, 0.0, False, 1.0
        name, b = CLASSES[c][0], _nc_bin(nc)
        if c in OOP:
            return MAI_SLOPE[name][b], OOP_W0, MAI_WIDTH_SLOPE[name][b], True, 1.0
        chi, w0, w1, eta = FIXED[name]
        if chi is None:
            chi = self.ch_factor * MAI_SLOPE[name][b]
        return chi, w0, w1, False, eta

    def t_classical(self, freqs_all, temperature_k):
        """T_cl of the state (all its modes ``freqs_all``) at quantum T (``t_map``)."""
        if self.t_map == "energy":
            return classical_temperature(freqs_all, temperature_k)
        return self.t_map[1] * np.asarray(temperature_k, dtype=float)

    def t_rep(self, c, nc):
        if c == OTHER:
            return 0.0
        if c == 0:
            return T_REP["ch"]
        if c in OOP:
            return T_REP["oop_small"] if nc < 50 else T_REP["oop_large"]
        return T_REP["cc"]

    def lines_at(self, freqs_cm, nc, z, temperature_k, t_cl):
        """(shift, FWHM, eta) per mode [cm^-1] at quantum T ``temperature_k`` and classical ``t_cl`` (arrays
        broadcast with the modes)."""
        f = np.asarray(freqs_cm, dtype=float)
        T = np.broadcast_to(np.asarray(temperature_k, dtype=float), f.shape)
        Tc = np.broadcast_to(np.asarray(t_cl, dtype=float), f.shape)
        cls = mode_classes(f)
        shift, fwhm, eta = np.zeros(f.shape), np.zeros(f.shape), np.ones(f.shape)
        for c in np.unique(cls):
            m = cls == c
            chi, w0, w1, on_cl, e = self.class_law(c, nc)
            shift[m] = chi * Tc[m]
            fwhm[m] = w0 + w1 * (Tc[m] if on_cl else T[m])
            eta[m] = e
            if c == SOLO:
                shift[m] += self.solo_offset_cm + (self.cation_solo_offset_cm if z > 0 else 0.0)
        return shift, np.maximum(fwhm, 0.5), eta

    def constant_lines(self, freqs_cm, nc, z):
        """(shift, FWHM, eta) per mode of the constant convention: each class at its ``t_rep`` (``freqs_cm``: all the
        state's modes, which set T_cl)."""
        f = np.asarray(freqs_cm, dtype=float)
        if f.size == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        T = np.array([self.t_rep(c, nc) for c in mode_classes(f)])
        reps = np.unique(T)
        tc = dict(zip(reps.tolist(), np.atleast_1d(self.t_classical(f, reps)).tolist(), strict=True))
        return self.lines_at(f, nc, z, T, np.array([tc[t] for t in T.tolist()]))


LEGACY = BandConvention("legacy", kind="legacy")
SOURCED = BandConvention("sourced", kind="constant")
SOURCED_TDEP = BandConvention("sourced_tdep", kind="tdep")
CONVENTIONS = {c.name: c for c in (LEGACY, SOURCED, SOURCED_TDEP)}


def get(name_or_convention):
    """A ``BandConvention`` from its name (``CONVENTIONS``) or itself."""
    if isinstance(name_or_convention, BandConvention):
        return name_or_convention
    try:
        return CONVENTIONS[name_or_convention]
    except KeyError:
        raise ValueError(f"unknown band convention {name_or_convention!r}; one of {sorted(CONVENTIONS)}") from None


# -- profiles -------------------------------------------------------------------------------------------
def _cdf(x, fwhm, eta):
    g = 0.5 * fwhm
    s = fwhm * _FWHM_TO_SIGMA
    return eta * (np.arctan(x / g) / np.pi + 0.5) + (1.0 - eta) * 0.5 * (1.0 + erf(x / (s * np.sqrt(2.0))))


def _pdf(x, fwhm, eta):
    g = 0.5 * fwhm
    s = fwhm * _FWHM_TO_SIGMA
    return eta * g / np.pi / (x**2 + g**2) + (1.0 - eta) * np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def window_matrix(freqs_cm, windows_um, local, shift_cm, fwhm_cm, eta):
    """M[i, b]: fraction of line i (centre freqs + shift, ``fwhm``, pseudo-Voigt ``eta``) inside window b.

    The general form of ``population._band_matrix`` (per-mode shift, width and profile): the bands in ``local``
    take off the line's own values at the window edges as a baseline linear in wavelength (F_nu against lambda),
    integrated over nu, as the observed measurement does.
    """
    nu = (np.asarray(freqs_cm, dtype=float) + np.asarray(shift_cm, dtype=float))[:, None]
    w = np.broadcast_to(np.asarray(fwhm_cm, dtype=float), nu.shape[:1])[:, None]
    e = np.broadcast_to(np.asarray(eta, dtype=float), nu.shape[:1])[:, None]
    lam1 = np.array([v[0] for v in windows_um.values()])[None, :]
    lam2 = np.array([v[1] for v in windows_um.values()])[None, :]
    nu1, nu2 = 1e4 / lam1, 1e4 / lam2
    M = _cdf(nu1 - nu, w, e) - _cdf(nu2 - nu, w, e)
    is_local = np.array([b in local for b in windows_um])[None, :]
    if is_local.any():
        L1, L2 = _pdf(nu1 - nu, w, e), _pdf(nu2 - nu, w, e)
        base = L1 * (nu1 - nu2) + (L2 - L1) / (lam2 - lam1) * (1e4 * np.log(nu1 / nu2) - lam1 * (nu1 - nu2))
        M = np.where(is_local, M - base, M)
    return M


def line_spectrum(freqs_cm, strength, grid_cm, shift_cm, fwhm_cm, eta, chunk=4096):
    """Sum of lines (per-mode shift, FWHM, pseudo-Voigt eta) of ``strength`` per cm^-1 on ``grid_cm``; each line
    integrates to its strength over an infinite grid."""
    f = np.asarray(freqs_cm, dtype=float) + np.asarray(shift_cm, dtype=float)
    s = np.asarray(strength, dtype=float)
    w = np.broadcast_to(np.asarray(fwhm_cm, dtype=float), f.shape)
    e = np.broadcast_to(np.asarray(eta, dtype=float), f.shape)
    grid = np.asarray(grid_cm, dtype=float)
    out = np.zeros(grid.size)
    on = s != 0.0
    f, s, w, e = f[on], s[on], w[on], e[on]
    for i in range(0, f.size, chunk):
        sl = slice(i, i + chunk)
        out += s[sl] @ _pdf(grid[None, :] - f[sl, None], w[sl, None], e[sl, None])
    return out


# -- emission temperatures ----------------------------------------------------------------------------
def bin_temperatures(freqs_cm, energy_cm, grain_cm=2.0, window_cm=200.0):
    """T(E) [K] of the cascade bins: microcanonical (1/kT = d ln rho / dE, rho smoothed over ``window_cm``), canonical
    where the count overflows or is empty."""
    from pycalima.pahchem import dos

    e = np.asarray(energy_cm, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        E, counts = dos.beyer_swinehart(freqs_cm, e[-1] + window_cm, grain_cm)
        T = dos.microcanonical_temperature(E, dos.smoothed_log_density(E, counts, window_cm))
    T = np.interp(e, E, np.where(np.isfinite(T), T, -1.0))
    bad = ~(T > 0.0)
    if bad.any():
        T[bad] = dos.canonical_temperature(freqs_cm, e[bad])
    T[e <= 0.0] = 0.0
    return T


def class_temperature_shares(freqs_cm, photons_by_bin, bin_temperature_k):
    """W[c, t]: the share of class c's IR power emitted in temperature bin t (``T_EDGES``); rows sum to 1 or 0.

    ``photons_by_bin`` is X[i, j] of ``emission.emission_by_energy``; power = nu_i X[i, j]."""
    f = np.asarray(freqs_cm, dtype=float)
    power = f[:, None] * np.asarray(photons_by_bin, dtype=float)
    cls = mode_classes(f)
    by_class = np.zeros((N_CLASSES, power.shape[1]))
    np.add.at(by_class, cls, power)
    t = np.clip(np.searchsorted(T_EDGES, bin_temperature_k, side="right") - 1, 0, T_MID.size - 1)
    W = np.zeros((N_CLASSES, T_MID.size))
    np.add.at(W.T, t, by_class.T)
    tot = W.sum(axis=1, keepdims=True)
    return np.divide(W, tot, out=np.zeros_like(W), where=tot > 0)


def tdep_window_matrix(convention, freqs_cm, nc, z, windows_um, local, shares):
    """M[i, b] of ``window_matrix`` averaged over each mode's class emission-temperature distribution ``shares``
    (``freqs_cm``: all the state's modes, which set T_cl)."""
    f = np.asarray(freqs_cm, dtype=float)
    cls = mode_classes(f)
    t_cl = convention.t_classical(f, T_MID)
    M = np.zeros((f.size, len(windows_um)))
    for c in np.unique(cls):
        m = cls == c
        for t in np.flatnonzero(shares[c] > 0.0):
            sh, fw, et = convention.lines_at(f[m], nc, z, T_MID[t], t_cl[t])
            M[m] += shares[c, t] * window_matrix(f[m], windows_um, local, sh, fw, et)
    return M


def tdep_spectrum(convention, freqs_cm, strength, nc, z, grid_cm, shares, min_share=1e-4):
    """``line_spectrum`` averaged over each mode's class emission-temperature distribution (bins below
    ``min_share`` of a class are folded into its nearest kept bin, conserving the power)."""
    f = np.asarray(freqs_cm, dtype=float)
    s = np.asarray(strength, dtype=float)
    cls = mode_classes(f)
    t_cl = convention.t_classical(f, T_MID)
    out = np.zeros(np.size(grid_cm))
    for c in np.unique(cls):
        m = cls == c
        w = shares[c]
        keep = np.flatnonzero(w >= min_share * w.max()) if w.max() > 0 else np.zeros(0, int)
        if keep.size == 0:
            continue
        ww = np.zeros(keep.size)
        np.add.at(ww, np.abs(np.arange(w.size)[:, None] - keep[None, :]).argmin(axis=1), w)
        for t, share in zip(keep, ww, strict=True):
            sh, fw, et = convention.lines_at(f[m], nc, z, T_MID[t], t_cl[t])
            out += line_spectrum(f[m], share * s[m], grid_cm, sh, fw, et)
    return out
