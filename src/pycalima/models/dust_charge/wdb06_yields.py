"""Photoelectric yields for EUV/X-ray photons: Weingartner, Draine & Barr (2006).

WDB06 extend the WD01 yield to hnu >> 20 eV. Each absorbed photon can release:

- a primary photoelectron from the band structure (secs. 4.1-4.2);
- a primary photoelectron from an inner shell (i, s) (sec. 4.2);
- Auger electrons after an inner-shell vacancy (sec. 5, Dwek & Smith 1996);
- secondary electrons excited by any of these (sec. 6, Draine & Salpeter 1979).

Every channel k returns a yield Y_k(hnu, Z) and the mean energy at infinity of
the escaping electrons <E>_k(hnu, Z), so that the charging current (eqs. 22-25)
and the gas heating (eq. 27) come from the same numbers. Secondary emission by
impinging gas electrons (J_sec,gas, eq. 26) is at the end of the module.

Below 20 eV the band yield is WD01's, so for soft fields ``yield_model='WDB06'``
differs from WD01 only through the van Hoof et al. (2004) E_min (eq. 3), the
photo-excited secondaries (a few per cent of the yield) and J_sec,gas.

Material ``'cm20'``: THEMIS amorphous carbon, charged as DustEM v4.3 charges
its CM20 grains (``CHRG_CM20.DAT``; ``PE_CURRENT`` and ``GAS_CURRENTS`` of
``DM_zdist.f90``), with WDB06's channels above 20 eV:

- band yield below 20 eV: Kimura (2016) eq. A12 (``cm20_band_yield``), with
  beta = 3/4 Q_ext from the THEMIS optics and the Seah & Dench escape length
  l_e(E) at the electron's mean energy inside the grain; blended in log to
  WDB06's eq. 9 between 20 and 50 eV, as WDB06 blend WD01's yield;
- thresholds: W = 4.75 eV and DustEM's aromatic correction A_C =
  (e^2/2a)(pi a/2a_R - 1), a_R = a/2, the energy to take the electron off the
  largest aromatic domain (a conducting disk; DustEM user guide, App. B), added
  to the ionisation potential in place of WD01's (Z+2)(0.3 A/a) e^2/a and
  subtracted from the electron affinity; and the size-dependent band gap of Jones (2012),
  E_bg(a) = 0.2 (50 A/a - 1) eV within [0.1, 2.5], in the electron affinity;
- electron sticking 0.5 (1 - exp(-a/l_e)) with no PAH factor, l_e averaged
  as <1/l_e> over photoemission in the Mathis ISRF (``cm20_sticking``);
- inner shells, Auger and secondary electrons, and the hot-gas terms: carbon
  as for graphite (WDB06 eq. 12 l_e, eps = 117 eV, DS79 graphite constants),
  at the CM20 density of 1.6 g/cm^3 and its carbon number density. These are
  graphite's constants, used for want of a-C values.

``'cm20'`` is the faithful DustEM port, kept for benchmarks. Material ``'ac'`` is
pyCALIMA's default for amorphous carbon: the same density and channels, with the
choices that measurements support (docs/physics/charging-cm20.md):

- the ionisation potential of WD01 eq. 2 with W = 4.75 eV, without DustEM's
  aromatic correction, which puts small-grain thresholds 2.6-3.6 eV above the
  measured IEs of pyrene, coronene and coronene clusters (WD01 eq. 2 is within
  0.3 eV of them, and of the second IEs of C30-C48 cations);
- the electron affinity W - E_lim(a) + (Z - 1/2) e^2/a, without DustEM's aromatic
  correction, with the size-limited gap the optics use, E_lim = 0.75 eV (1 nm/a)^1.06
  (``jones2012_optics.size_limited_gap``, the law that makes Jones's constants reproduce
  DustEM's Q_CM20 at 1-10 nm), in place of DustEM's E_bg(a) = 0.2 (5 nm/a - 1) eV; below
  1 nm, where neither the fit nor the optics' size scheme applies, DustEM's E_bg(a)
  (``ac_band_gap``, ``ac_electron_affinity``);
- the band yield of the extended Kimura model, eK16, of Hrodmarsson et al. (2025),
  validated on the photoionisation of PAHs (``ek16_band_yield``);
- WD01's electron sticking (eqs. 27-30), not DustEM's photoelectron length;
- the CM20 optics with the large grains recomputed (``themis_optics.mie_efficiencies``).

Material ``'ach'``: H-rich a-C:H (E_g = 2.5 eV), as 'ac' but with the bulk IE of
4.6 eV and electron affinity 0 that Schafer, Ristein & Ley (1993) measure on
polymeric a-C:H (work function 2.6 eV, valence-band top 2 eV below E_F, conduction
band at the vacuum level), the Jones (2012) optics and 1.33 g/cm^3. Their defect
bands in the gap, ~1e-3 of the atoms, which set the 2.6 eV photoyield onset, are
left out.

Materials ``'acx<E_g>'`` (e.g. ``'acx1.2500'``, ``acx_label``): the a-C(:H) of tracked
hydrogen (rung M1), one per bulk band gap between the 'ac' (0.1 eV) and 'ach'
(2.5 eV) ends, as 'ac' otherwise. No measurement fixes the materials in between
(Schafer+93's films vary sp2 at a fixed ~40 at% H), so these are labelled
assumptions: the bulk IE linear in E_g from 4.75 to 4.6 eV; the electron affinity
IE - E_mob, with a mobility gap rising linearly in E_g from the size-limited optical
gap max(E_g, E_lim(a)) (``ac_band_gap``) at the a-C end to the
IE at the a-C:H end (affinity 0, as 'ach'); the Jones (2012) optics at E_g; density
and X_H of Jones (2012). At 2.5 eV it is 'ach'; at 0.1 eV it differs from 'ac' in the
optics (Jones's tables, not Q_CM20).

Deliberate departures from WDB06:

- Auger heating uses E_max = E_A - (Z+1) e^2/a, the same upper limit as their
  y2 for Auger electrons; their sec. 8 writes I_{i,s} - (Z+1) e^2/a, which
  gives the electron more energy than the transition releases (up to +7% of
  the total heating for charged silicate grains in a quasar spectrum).
- J_sec,gas averages the DS79 yield over the thermal impact-energy
  distribution, with no secondaries below the work function or below the
  escape barrier (which energy conservation forbids), instead of
  evaluating it at the mean energy <E0> (DS79 eq. 16). The two agree to ~15%
  for T >~ 1e5 K; at 1e4 K the mean-energy form extrapolates DS79's fit to
  <E0> ~ 2 eV < W and overstates the yield 10-100 times.

- Secondary yields (eq. 18) carry an extra 1 - exp(-a / l_e(E_exc)), read off
  their Figs. 3-5 (see ``_iter_components``). Their Fig. 4 curve for
  a = 0.001 um carbonaceous grains below 291 eV is NOT reproduced: its band
  secondaries are exactly 2 l_e(E) / 10 A times eq. 18, i.e. l_e fixed at 10 A
  and no a^-1 term, so the primary both escapes with its energy and spends
  all of it on secondaries. We keep eq. 18; our yield there is ~2.5x lower.

Additions beyond WDB06 (hot gas, Draine & Salpeter 1979): fast electrons that
pass through the grain are not captured (``electron_transmission``; DS79 range
law, since the CSDA electron stopping fit of dust_collisional_cooling fails
below ~100 eV), and fast ions that leave it still ionised transfer no charge
(``ion_charge_transfer``; ion stopping as in the sputtering and collisional
cooling tables). Both vanish in H II gas; electrons matter for a <~ 5 nm at
T >~ 1e6 K, ions only for T >~ 3e7 K.

Energies are in eV and lengths in cm unless stated otherwise.
"""

import numpy as np

from pycalima.models.dust_charge import wdb06_atomic as atomic
from pycalima.models.dust_charge.shared_physics import (
    E_STATC,
    EV2ERG,
    GRAPHITE_WORK_FUNCTION,
    SILICATE_WORK_FUNCTION,
    TINY,
    BT94_y0_graphite_vec,
    Watson73_y1_vec,
    y0_silicate_vec,
)

ANGSTROM = 1e-8

E_TILDE2 = 8.0  # eV^2, secondary energy scale of eq. 19
L_E_LOW = 10.0 * ANGSTROM  # l_e at low energy (WD01), eqs. 12-13

# WDB06 secs. 4.1.3-4.1.4 and 6.
MATERIALS = {
    "graphite": {
        "W": GRAPHITE_WORK_FUNCTION,
        "n": {"C": 1.12e23},
        "band": [("C", "2s"), ("C", "2p")],
        "inner": [("C", "1s")],
        "le_break": 164.0, "le_coeff": 4.78e-3,  # eq. 12
        "eps": 117.0,
        "y0_wd01": BT94_y0_graphite_vec,
    },
    "silicate": {
        "W": SILICATE_WORK_FUNCTION,
        "n": {"Mg": 1.22e22, "Fe": 1.22e22, "Si": 1.22e22, "O": 4.88e22},
        "band": [("O", "2s"), ("O", "2p"), ("Si", "3s"), ("Si", "3p"),
                 ("Mg", "3s"), ("Fe", "3d"), ("Fe", "4s")],
        "inner": [("O", "1s"), ("Mg", "1s"), ("Mg", "2s"), ("Mg", "2p"),
                  ("Si", "1s"), ("Si", "2s"), ("Si", "2p"),
                  ("Fe", "1s"), ("Fe", "2s"), ("Fe", "2p"), ("Fe", "3s"), ("Fe", "3p")],
        "le_break": 211.0, "le_coeff": 3.27e-3,  # eq. 13
        "eps": 155.0,
        "y0_wd01": y0_silicate_vec,
    },
}

# THEMIS a-C, DustEM v4.3 CHRG_CM20.DAT (its WD01 Y0 and electron-affinity
# constants are not used by the Kimura branch of DM_zdist)
CM20 = {
    "W": 4.75,                            # bulk work function [eV]
    "uait": (3.9, 0.12, 2.0),             # U_ait = -(p1 + p2 a + p3/a) V, a in A: WD01's carbon values
    "aromatic": (1.0, 2.0),               # p_6: A_C = p1 (pi p2/2 - 1) e^2/2a, p2 = a/a_R
    "stick": (0.5, 1.0, 0.0),             # p_se: s_e = p1 (1 - exp(-p2 a/l_e)); p3 = 0: no PAH factor
    "le_nm": (143.0, -2.0, 0.054, 0.5),   # p_le: l_e = p1 E^p2 + p3 E^p4 nm, E inside the grain [eV]
    "rho": 1.6,                           # GRAIN_J13.DAT
    "x_h": 0.1 / 4.3,                     # H atomic fraction of a-C (E_g = 0.1 eV, Jones 2012)
}
_AMU = 1.66053906660e-24
_N_ATOMS_CM20 = CM20["rho"] / (((1.0 - CM20["x_h"]) * 12.011 + CM20["x_h"] * 1.008) * _AMU)
MATERIALS["cm20"] = dict(MATERIALS["graphite"], W=CM20["W"],
                         n={"C": (1.0 - CM20["x_h"]) * _N_ATOMS_CM20},
                         y0_wd01=None, rho=CM20["rho"])
# pyCALIMA's a-C: W = 4.75 eV (Feuerbacher & Fitton 1972, via Kimura 2016; soot 4.05-4.77 eV,
# Ouf et al. 2016), WD01 thresholds, the recomputed CM20 optics
MATERIALS["ac"] = dict(MATERIALS["cm20"])
# pyCALIMA's H-rich a-C:H (E_g = 2.5 eV, the aliphatic carrier): the polymeric a-C:H of
# Schafer, Ristein & Ley (1993): work function 2.6 eV, valence-band top 2 eV below E_F
# (bulk IE = 4.6 eV), conduction band at the vacuum level (chi = 0); Jones (2012) optics
ACH = {"W": 2.6 + 2.0, "chi": 0.0, "e_gap": 2.5, "rho": 1.3 + 0.4 * np.exp(-2.7), "x_h": 2.5 / 4.3}
_N_ATOMS_ACH = ACH["rho"] / (((1.0 - ACH["x_h"]) * 12.011 + ACH["x_h"] * 1.008) * _AMU)
MATERIALS["ach"] = dict(MATERIALS["cm20"], W=ACH["W"], n={"C": (1.0 - ACH["x_h"]) * _N_ATOMS_ACH},
                        rho=ACH["rho"])
# the intermediate a-C(:H) family of tracked hydrogen (module docstring)
ACX_PREFIX = "acx"
ACX_RANGE = (0.1, 2.5)


def acx_label(e_gap):
    """Material label of the intermediate a-C(:H) with bulk band gap ``e_gap`` [eV]."""
    e = float(e_gap)
    if not ACX_RANGE[0] - 1e-9 <= e <= ACX_RANGE[1] + 1e-9:
        raise ValueError(f"acx band gap {e} eV outside {ACX_RANGE}")
    return f"{ACX_PREFIX}{e:.4f}"


def is_acx(material):
    return isinstance(material, str) and material.startswith(ACX_PREFIX)


def acx_e_gap(material):
    return float(material[len(ACX_PREFIX):])


def acx_ionisation_energy(e_gap):
    """Bulk IE [eV] of the intermediate a-C(:H): linear in E_g from 'ac' to 'ach'."""
    w = (float(e_gap) - ACX_RANGE[0]) / (ACX_RANGE[1] - ACX_RANGE[0])
    return (1.0 - w) * CM20["W"] + w * ACH["W"]


def acx_mobility_gap(e_gap, a):
    """E_mob [eV] of a grain of radius ``a`` [cm]: from the size-limited optical gap at the
    a-C end to the IE at the a-C:H end, linear in E_g (a labelled assumption)."""
    e = float(e_gap)
    w = (e - ACX_RANGE[0]) / (ACX_RANGE[1] - ACX_RANGE[0])
    e_eff = np.maximum(e, ac_band_gap(a))
    return (1.0 - w) * e_eff + w * acx_ionisation_energy(e)


def acx_electron_affinity(Z, a, e_gap):
    """EA(Z, a) [eV] of the intermediate a-C(:H): IE - E_mob + (Z - 1/2) e^2/a."""
    Z = np.asarray(Z, dtype=float)
    return acx_ionisation_energy(e_gap) - acx_mobility_gap(e_gap, a) + (Z - 0.5) * _e2_over_a(a)


def _acx_entry(e_gap):
    x_h = float(e_gap) / 4.3
    rho = 1.3 + 0.4 * np.exp(-(float(e_gap) + 0.2))
    n_atoms = rho / (((1.0 - x_h) * 12.011 + x_h * 1.008) * _AMU)
    return dict(MATERIALS["cm20"], W=acx_ionisation_energy(e_gap), n={"C": (1.0 - x_h) * n_atoms}, rho=rho)


class _Materials(dict):
    """MATERIALS, creating an 'acx<E_g>' entry on first use."""

    def __missing__(self, key):
        if is_acx(key):
            return self.setdefault(key, _acx_entry(acx_e_gap(key)))
        raise KeyError(key)


class _Family(tuple):
    """A tuple of labels that also contains every 'acx<E_g>' label."""

    def __contains__(self, material):
        return tuple.__contains__(self, material) or is_acx(material)


MATERIALS = _Materials(MATERIALS)
KIMURA_MATERIALS = _Family(("cm20", "ac", "ach"))       # band yield of Kimura (2016) or eK16
EK16_MATERIALS = _Family(("ac", "ach"))


def _label(material):
    """WDB06 material of a label: 'graphite', 'cm20', 'ac', 'ach', 'acx<E_g>', and silicate
    for any other."""
    return material if material in ("graphite", "cm20", "ac", "ach") or is_acx(material) else "silicate"


def _material(material):
    return MATERIALS[_label(material)]


def electron_escape_length(E_e, material):
    """l_e(E_e) [cm], WDB06 eqs. 12-13."""
    m = _material(material)
    E_e = np.asarray(E_e, dtype=float)
    return np.where(E_e <= m["le_break"], L_E_LOW,
                    m["le_coeff"] * ANGSTROM * np.maximum(E_e, 0.0) ** 1.5)


def _n_sigma(material, element, shell, E):
    return _material(material)["n"][element] * atomic.photoionisation_cross_section(element, shell, E)


def absorption_coefficient(E, material, shells=None):
    """sum_{i,s} n_i sigma_{i,s}(E) [cm^-1] over ``shells`` (default: all)."""
    m = _material(material)
    if shells is None:
        shells = m["band"] + m["inner"]
    E = np.asarray(E, dtype=float)
    return sum(_n_sigma(material, el, sh, E) for el, sh in shells)


def photon_attenuation_length(E, material, la_wd01=None):
    """l_a(hnu) [cm]: atomic (eq. 10) above 20 eV, ``la_wd01`` (dielectric) below."""
    E = np.asarray(E, dtype=float)
    la_atomic = 1.0 / np.maximum(absorption_coefficient(E, material), TINY)
    if la_wd01 is None:
        return la_atomic
    return np.where(E > 20.0, la_atomic, np.broadcast_to(la_wd01, E.shape))


def _slab_escape(n_sigma, le, la):
    """n sigma l_e [1 - (l_e/l_a) ln(1 + l_a/l_e)], WDB06 eqs. 9 and 16."""
    r = la / le
    return n_sigma * le * (1.0 - np.log1p(r) / r)


# ---------------------------------------------------------------------------
# Charge-dependent thresholds and energy ranges (WDB06 sec. 2)
# ---------------------------------------------------------------------------

def _e2_over_a(a):
    return E_STATC ** 2 / a / EV2ERG


def min_energy_vH04(Z, a):
    """E_min [eV] for Z < -1 (WDB06 eq. 3; van Hoof et al. 2004 eq. 1).

    theta_nu is the DS87 eq. 3.6 fit in units of e^2/a.
    """
    Z = np.asarray(Z, dtype=float)
    nu = np.abs(Z + 1.0)
    nu_safe = np.maximum(nu, TINY)
    theta = nu_safe / (1.0 + 1.0 / np.sqrt(nu_safe))
    corr = 1.0 - 0.3 * (a / (10.0 * ANGSTROM)) ** -0.45 * nu_safe ** -0.26
    return np.where(Z < -1, theta * corr * _e2_over_a(a), 0.0)


def ionisation_potential(I, Z, a, material="graphite"):
    """IP(Z, a) = I + (Z + 1/2) e^2/a + (Z + 2) (e^2/a)(0.3 A / a), eq. 2;
    for 'cm20' the last term is the aromatic correction (``aromatic_correction``)."""
    e2a = _e2_over_a(a)
    if material == "cm20":
        return I + (Z + 0.5) * e2a + aromatic_correction(a)
    return I + (Z + 0.5) * e2a + (Z + 2.0) * e2a * (0.3 * ANGSTROM / a)


def threshold_energy(I, Z, a, material="graphite"):
    """hnu_pet (eq. 1) with W -> I for inner shells."""
    IP = ionisation_potential(I, Z, a, material)
    return np.where(Z < -1, IP + min_energy_vH04(Z, a), IP)


# ---------------------------------------------------------------------------
# THEMIS a-C (material 'cm20'): DustEM's CHRG_CM20 constants
# ---------------------------------------------------------------------------

def aromatic_correction(a):
    """A_C = p1 (pi p2/2 - 1) e^2/2a [eV], p2 = a/a_R: DustEM's correction for
    taking the electron off the largest aromatic domain, a conducting disk of
    radius a_R = a/2 (capacitance 2 a_R/pi), added to the IP and subtracted from
    the EA (DustEM user guide, App. B, eq. 9; DM_zdist.f90 cites Brus 1988)."""
    p1, p2 = CM20["aromatic"]
    return p1 * (0.5 * np.pi * p2 - 1.0) * 0.5 * _e2_over_a(a)


def cm20_band_gap(a):
    """E_bg(a) [eV] of a-C(:H) grains, DustEM's size law (DM_inout, nbg = -1)."""
    return np.clip(0.2 * (50.0 * ANGSTROM / np.asarray(a, dtype=float) - 1.0), 0.1, 2.5)


def cm20_electron_affinity(Z, a):
    """EA(Z, a) [eV]: W - E_bg + (Z - 1/2) e^2/a - A_C (DustEM user guide, App. B)."""
    Z = np.asarray(Z, dtype=float)
    return CM20["W"] - cm20_band_gap(a) + (Z - 0.5) * _e2_over_a(a) - aromatic_correction(a)


#: radius [cm] above which 'ac' takes the fitted size-limited gap (jones2012_optics.EFFECTIVE_GAP_MIN_NM)
AC_GAP_MIN_CM = 1e-7


def ac_band_gap(a):
    """Size-limited band gap [eV] of 'ac' grains: the optics' E_lim(a) = 0.75 eV (1 nm/a)^1.06
    within [0.1, 2.5] from 1 nm, DustEM's E_bg(a) below (0.05 eV above E_lim at 1 nm)."""
    from pycalima.models.dust_radiation.jones2012_optics import size_limited_gap

    a = np.asarray(a, dtype=float)
    return np.where(a >= AC_GAP_MIN_CM, np.clip(size_limited_gap(a * 1e4), 0.1, 2.5), cm20_band_gap(a))


def ac_electron_affinity(Z, a):
    """EA(Z, a) [eV] of pyCALIMA's a-C: W - E_lim(a) + (Z - 1/2) e^2/a (``ac_band_gap``),
    without DustEM's aromatic correction. Below 1 nm (DustEM's E_bg) it is the closest of
    the published forms to measured small-PAH EAs, though still above them (pyrene 0.86
    vs 0.41 eV, coronene 1.37 vs 0.47 eV): the size law underestimates molecular gaps."""
    Z = np.asarray(Z, dtype=float)
    return MATERIALS["ac"]["W"] - ac_band_gap(a) + (Z - 0.5) * _e2_over_a(a)


def cm20_escape_length(E):
    """Photoelectron escape length l_e [cm] at energy E inside the grain [eV]
    (Seah & Dench 1979 form with DustEM's p_le)."""
    p1, p2, p3, p4 = CM20["le_nm"]
    E = np.maximum(np.asarray(E, dtype=float), 1e-3)
    return 1e-7 * (p1 * E ** p2 + p3 * E ** p4)


def most_negative_charge_cm20(a):
    """Zmin = floor(U_ait a / e) + 1 with DustEM's U_ait (here WD01's carbon one)."""
    p1, p2, p3 = CM20["uait"]
    aA = np.asarray(a, dtype=float) / ANGSTROM
    return np.floor(-(p1 + p2 * aA + p3 / aA) / 14.4 * aA) + 1.0


def theta_parameter(E, hnu_pet, Z, a):
    """Theta, eq. 5."""
    return E - hnu_pet + np.where(Z >= 0, (Z + 1.0) * _e2_over_a(a), 0.0)


def energy_limits(hnu_minus_pet, Z, a):
    """(E_low, E_high) of attempting primary photoelectrons, WDB06 sec. 2:
    [E_min, E_min + hnu - hnu_pet] for Z < 0, [-(Z+1) e^2/a, hnu - hnu_pet]
    for Z >= 0."""
    E_min = min_energy_vH04(Z, a)
    E_low = np.where(Z < 0, E_min, -(Z + 1.0) * _e2_over_a(a))
    E_high = hnu_minus_pet + np.where(Z < 0, E_min, 0.0)
    return E_low, E_high


def _parabolic_y2_and_mean(E_low, E_high):
    """Escape fraction y2 (eq. 8) and mean energy at infinity of the escaping
    electrons for the parabolic distribution (eq. 7)."""
    d = np.maximum(E_high - E_low, 1e-12)  # cubed below; TINY would underflow
    lo = np.maximum(E_low, 0.0)
    ok = E_high > lo
    # integrals of f0 and E f0 from lo to E_high, f0 = 6 (E - El)(Eh - E)/d^3
    def F(E):
        u = E - E_low
        return 6.0 * (0.5 * d * u ** 2 - u ** 3 / 3.0) / d ** 3

    def G(E):  # int E f0 dE = int (u + El) f0 du
        u = E - E_low
        return 6.0 * (d * u ** 3 / 3.0 - u ** 4 / 4.0) / d ** 3 + E_low * F(E)

    y2 = np.where(ok, F(E_high) - F(lo), 0.0)
    num = np.where(ok, G(E_high) - G(lo), 0.0)
    mean = np.where(y2 > 0, num / np.maximum(y2, TINY), 0.0)
    return np.clip(y2, 0.0, 1.0), mean


def _secondary_y2_and_mean(E_low, E_high):
    """Escape fraction (eq. 21) and mean energy at infinity for the secondary
    distribution of eq. 19 on [E_low, E_high]."""
    Et = np.sqrt(E_TILDE2)
    D = np.maximum(E_high - E_low, 0.0)
    alpha = 1.0 / np.maximum(1.0 - (1.0 + D ** 2 / E_TILDE2) ** -0.5, TINY)
    lo = np.maximum(E_low, 0.0)
    ok = (E_high > lo) & (D > 0)

    def F(E):  # int_El^E f0
        u = E - E_low
        return -alpha * (1.0 + u ** 2 / E_TILDE2) ** -0.5

    def G(E):  # int E f0 = int (u + El) f0 du
        x = (E - E_low) / Et
        return alpha * Et * (np.arcsinh(x) - x / np.sqrt(1.0 + x * x)) + E_low * F(E)

    y2 = np.where(ok, F(E_high) - F(lo), 0.0)
    num = np.where(ok, G(E_high) - G(lo), 0.0)
    mean = np.where(y2 > 0, num / np.maximum(y2, TINY), 0.0)
    return np.clip(y2, 0.0, 1.0), mean


def _w3(x):
    """x^2 - 2x + 2 (1 - e^-x), by its series below 1e-2 (where it cancels to x^3/3)."""
    x = np.asarray(x, dtype=float)
    xs = np.maximum(x, 1e-2)
    return np.where(x < 1e-2, x ** 3 / 3.0 - x ** 4 / 12.0 + x ** 5 / 60.0,
                    xs * xs - 2.0 * xs - 2.0 * np.expm1(-xs))


def cm20_band_yield(E, Zs, a, beta, material="cm20"):
    """Band yield of CM20 grains, Kimura (2016) eq. A12 as in DustEM's PE_CURRENT.

    E : (N_E,) photon energies [eV]; Zs : (N_Z,) charges; beta : (N_E,) 3/4 Q_ext.
    Returns (Y, e2, l_e), each (N_E, N_Z): Y = y2 min(y0 y1, 1), the mean energy at
    infinity e2 of the escaping electrons [eV] and their escape length [cm], taken
    at their mean energy inside the grain e_t = e2 + hnu - Theta.
    """
    E = np.asarray(E, dtype=float)[:, None]
    Z = np.atleast_1d(np.asarray(Zs, dtype=float))[None, :]
    beta = np.asarray(beta, dtype=float)[:, None]
    hnu_pet = threshold_energy(MATERIALS[material]["W"], Z, a, material)
    theta = theta_parameter(E, hnu_pet, Z, a)
    E_low, E_high = energy_limits(E - hnu_pet, Z, a)
    y2, e2 = _parabolic_y2_and_mean(E_low, E_high)
    et = np.maximum(e2 + E - theta, 1e-3)
    le = cm20_escape_length(et)
    alpha = beta + a / le
    s = np.sqrt(np.clip(1.0 - e2 / et, 0.0, 1.0))
    num = (1.0 + s) * alpha ** 2 * (-np.expm1(-alpha)) + _w3(alpha) * (alpha - beta)
    y1 = num * (beta / alpha) ** 3 / (alpha - beta + 1.0) / np.maximum(_w3(beta), TINY)
    y0 = 0.5 * (1.0 - s)
    Y = np.where(E > hnu_pet, y2 * np.minimum(y0 * y1, 1.0), 0.0)
    return Y, e2, le


EK16_CX = 0.15     # c_x of the electron-grain interaction, Hrodmarsson et al. (2025)


def ek16_energy_limits(hnu_minus_pet, Z, a, c_x=EK16_CX):
    """(E_low, E_high) at infinity of the eK16 photoelectrons: WD01's, with the
    interaction energy E_x = -(Z + 1 + c_x) e^2/a of Hrodmarsson et al. (2025, eq. 2)
    as E_low of a non-negative grain."""
    E_min = min_energy_vH04(Z, a)
    E_low = np.where(Z < 0, E_min, -(Z + 1.0 + c_x) * _e2_over_a(a))
    E_high = hnu_minus_pet + np.where(Z < 0, E_min, 0.0)
    return E_low, E_high


def ek16_band_yield(E, Zs, a, beta, material="ac", c_x=EK16_CX):
    """Band yield of the extended Kimura model, eK16 (Hrodmarsson et al. 2025, A&A
    698, A202, Sect. 4), shape (N_E, N_Z): (Y, e2, l_e).

    As ``cm20_band_yield``, except that (i) the electron's total energy inside the
    grain is E_t = E_2i + IE_x, with the nascent mean kinetic energy E_2i = Theta/2 of
    a parabolic distribution on [0, Theta] (their eq. 3; DustEM 4.3 uses the mean
    energy of the escaping electrons), and (ii) the electron-grain interaction is
    E_x = -(Z + 1 + c_x) e^2/a, c_x = 0.15, which enters IE_x = IE + E_x, Theta, the
    escape fraction Y2 and the energy at infinity e2. Theta = hnu - IE_x is the range
    of kinetic energy at the surface; IE is the material's threshold (WD01 eq. 2 for
    'ac', as in their model).
    """
    E = np.asarray(E, dtype=float)[:, None]
    Z = np.atleast_1d(np.asarray(Zs, dtype=float))[None, :]
    beta = np.asarray(beta, dtype=float)[:, None]
    hnu_pet = threshold_energy(MATERIALS[material]["W"], Z, a, material)
    E_low, E_high = ek16_energy_limits(E - hnu_pet, Z, a, c_x)
    y2, e2 = _parabolic_y2_and_mean(E_low, E_high)
    theta = np.maximum(E_high - E_low, 0.0)
    ie_x = E - theta
    et = np.maximum(0.5 * theta + ie_x, 1e-3)
    le = cm20_escape_length(et)
    alpha = beta + a / le
    s = np.sqrt(np.clip(ie_x / et, 0.0, 1.0))
    num = (1.0 + s) * alpha ** 2 * (-np.expm1(-alpha)) + _w3(alpha) * (alpha - beta)
    y1 = num * (beta / alpha) ** 3 / (alpha - beta + 1.0) / np.maximum(_w3(beta), TINY)
    y0 = 0.5 * (1.0 - s)
    Y = np.where(E > hnu_pet, y2 * np.minimum(y0 * y1, 1.0), 0.0)
    return Y, e2, le


def _cm20_band_y01(E1, Z, a, theta, mu_band, la, material="cm20"):
    """y0 y1 of the band channel of a Kimura material: Kimura's (``cm20_band_yield``)
    below 20 eV, WDB06 eq. 9 times the Watson factor above 50 eV, log-linear between."""
    W = MATERIALS[material]["W"]
    E = E1[:, None]
    lo = E1 < 50.0
    y_lo = np.zeros(np.broadcast(E, Z).shape)
    if lo.any():
        beta = 0.75 * cm20_extinction_efficiency(E1[lo], a, material)
        hnu_pet = threshold_energy(W, Z, a, material)
        Elo = E[lo]
        limits = ek16_energy_limits if material in EK16_MATERIALS else energy_limits
        E_low, E_high = limits(Elo - hnu_pet, Z, a)
        y2, _ = _parabolic_y2_and_mean(E_low, E_high)
        band = ek16_band_yield if material in EK16_MATERIALS else cm20_band_yield
        Y, _, _ = band(E1[lo], Z.ravel(), a, beta, material)
        y_lo[lo] = np.where(y2 > 0, Y / np.maximum(y2, TINY), 0.0)
    le = electron_escape_length(np.maximum(theta, 0.0), "cm20")
    y_atom = np.where(theta > 0, _slab_escape(mu_band, le, la), 0.0)
    y_hi = y_atom * Watson73_y1_vec(a, la, electron_escape_length(E - W, "cm20"))
    w = np.clip(np.log(np.maximum(E, 20.0) / 20.0) / np.log(2.5), 0.0, 1.0)
    blend = np.exp((1.0 - w) * np.log(np.maximum(y_lo, TINY)) + w * np.log(np.maximum(y_hi, TINY)))
    return np.where(w <= 0.0, y_lo, np.where(w >= 1.0, y_hi, blend))


# ---------------------------------------------------------------------------
# Yield channels
# ---------------------------------------------------------------------------

def band_bulk_yield(theta, material, a=None, la_wd01_of_E=None):
    """Bulk band-structure yield y0(band, Theta), WDB06 secs. 4.1.3-4.1.4.

    WD01 below hnu = Theta + W = 20 eV, eq. 9 above 50 eV, and log y0 linear
    in log hnu between (read off WDB06 Figs. 1-2: the paper only says
    "interpolate"; a linear-in-hnu blend is up to 75% too high at 35 eV). For carbonaceous grains with a <= 13 A, eq. 15 blends back to
    WD01 (PAHs keep Y -> 1 at high energy).
    """
    m = _material(material)
    W = m["W"]
    theta = np.asarray(theta, dtype=float)
    hnu = theta + W
    y_wd01 = m["y0_wd01"](theta, W)

    E_eval = np.maximum(hnu, W)
    le = electron_escape_length(np.maximum(theta, 0.0), material)
    la = 1.0 / np.maximum(absorption_coefficient(E_eval, material), TINY)
    y_atom = _slab_escape(absorption_coefficient(E_eval, material, m["band"]), le, la)
    y_atom = np.where(theta > 0, y_atom, 0.0)

    w = np.clip(np.log(np.maximum(hnu, 20.0) / 20.0) / np.log(2.5), 0.0, 1.0)
    blend = (w > 0.0) & (w < 1.0)
    y0 = np.where(w <= 0.0, y_wd01, y_atom)
    y0 = np.where(blend, np.exp((1.0 - w) * np.log(np.maximum(y_wd01, TINY))
                                + w * np.log(np.maximum(y_atom, TINY))), y0)

    if material == "graphite" and a is not None and a <= 13.0 * ANGSTROM:
        f = np.clip((a - 6.0 * ANGSTROM) / (7.0 * ANGSTROM), 0.0, 1.0)  # eq. 15
        y0 = (1.0 - f) * y_wd01 + f * y0
    return y0


def _iter_components(E, Zs, a, material, la_wd01=None, include_secondaries=True):
    """Yield (name, Y, E_mean) per channel, one at a time, so that callers
    summing over channels hold only one channel's arrays."""
    m = _material(material)
    W = m["W"]
    E1 = np.asarray(E, dtype=float)
    E = E1[:, None]
    Z = np.asarray(Zs, dtype=float)[None, :]
    e2a = _e2_over_a(a)
    size_factor = 1.0 / (1.0 / a + 1.0 / L_E_LOW)  # eq. 18, last factor times l_e

    la = photon_attenuation_length(E1, material, la_wd01)[:, None]
    mu_tot = absorption_coefficient(E1, material)[:, None]

    def secondaries(Y, y2, E_low, E_high, E_exc):
        """Eq. 18, Y_sec = Y (y2_sec / y2) (E_exc / eps) l_e^-1 / (a^-1 + 10 A^-1),
        times 1 - exp(-a / l_e): an exciting electron with l_e >~ a leaves the
        grain before losing its energy. The factor is not in the printed eq. 18
        but reproduces WDB06's own Figs. 3-5 above 2 keV (0.01-1 um, within 3%);
        without it the secondary yield of a 0.1 um grain at 7 keV is 2x theirs."""
        y2s, mean_s = _secondary_y2_and_mean(E_low, E_high)
        ratio = np.where(y2 > 0, y2s / np.maximum(y2, TINY), 0.0)
        E_exc = np.maximum(E_exc, 0.0)
        le = electron_escape_length(E_exc, material)
        Ys = Y * ratio * (E_exc / m["eps"]) * size_factor / le * (-np.expm1(-a / le))
        return np.where(Y > 0, Ys, 0.0), mean_s

    # --- band structure (eqs. 4-8, 14) ---
    hnu_pet = threshold_energy(W, Z, a, _label(material))
    theta = theta_parameter(E, hnu_pet, Z, a)
    mu_band = absorption_coefficient(E1, material, m["band"])[:, None]
    if m["y0_wd01"] is None:            # cm20
        y01 = _cm20_band_y01(E1, Z, a, theta, mu_band, la, _label(material))
    else:
        y01 = (band_bulk_yield(theta, material, a)
               * Watson73_y1_vec(a, la, electron_escape_length(E - W, material)))
    P_band = np.where(E > 20.0, np.clip(mu_band / np.maximum(mu_tot, TINY), 0.0, 1.0), 1.0)
    E_low, E_high = (ek16_energy_limits if _label(material) in EK16_MATERIALS else energy_limits)(E - hnu_pet, Z, a)
    y2, mean = _parabolic_y2_and_mean(E_low, E_high)
    Y = np.where(E >= hnu_pet, y2 * np.minimum(y01, P_band), 0.0)
    yield ("band", Y, mean)
    if include_secondaries:
        yield ("sec:band",) + secondaries(Y, y2, E_low, E_high, E - W)

    # --- inner shells and their Auger electrons ---
    for el, sh in m["inner"]:
        I = atomic.shell_threshold_eV(el, sh)
        if E1.max() < I:
            continue
        tag = f"{el}{sh}"
        n_i = m["n"][el]
        sig_hnu = atomic.photoionisation_cross_section(el, sh, E1)[:, None]
        P_is = np.clip(n_i * sig_hnu / np.maximum(mu_tot, TINY), 0.0, 1.0)

        # primary (sec. 4.2: W -> I)
        hnu_pet = threshold_energy(I, Z, a, _label(material))
        theta = theta_parameter(E, hnu_pet, Z, a)
        E_eval = np.maximum(theta, 0.0) + I
        le0 = electron_escape_length(np.maximum(theta, 0.0), material)
        la0 = 1.0 / np.maximum(absorption_coefficient(E_eval, material), TINY)
        y0 = np.where(theta > 0, _slab_escape(
            n_i * atomic.photoionisation_cross_section(el, sh, E_eval), le0, la0), 0.0)
        y1 = Watson73_y1_vec(a, la, electron_escape_length(E - I, material))
        E_low, E_high = energy_limits(E - hnu_pet, Z, a)
        y2, mean = _parabolic_y2_and_mean(E_low, E_high)
        Y = np.where(E >= hnu_pet, y2 * np.minimum(y0 * y1, P_is), 0.0)
        yield ("inner:" + tag, Y, mean)
        if include_secondaries:
            yield ("sec:inner:" + tag,) + secondaries(Y, y2, E_low, E_high, E - I)

        # Auger (sec. 5): threshold I, Theta_A = hnu - I, l_e at E_A
        EA, pA = atomic.auger_spectrum(el, sh)
        if EA.size == 0:
            continue
        above = E >= I
        E_low = np.where(Z < 0, min_energy_vH04(Z, a), -(Z + 1.0) * e2a)
        YA_tot = np.zeros(np.broadcast(E, Z).shape)
        num_tot = np.zeros_like(YA_tot)
        Ys_tot = np.zeros_like(YA_tot)
        num_s = np.zeros_like(YA_tot)
        for Ej, pj in zip(EA, pA):
            leA = electron_escape_length(Ej, material)
            y0A = _slab_escape(pj * n_i * sig_hnu, leA, la)
            y1A = Watson73_y1_vec(a, la, leA)
            E_high = Ej - (Z + 1.0) * e2a
            y2A, meanA = _parabolic_y2_and_mean(E_low, E_high)
            y2A = np.where(E_high > 0, y2A, 0.0)
            YA = np.where(above, y2A * np.minimum(y0A * y1A, P_is * pj), 0.0)
            YA_tot += YA
            num_tot += YA * meanA
            if include_secondaries:
                Ys, mean_s = secondaries(YA, y2A, E_low, E_high, Ej)
                Ys_tot += Ys
                num_s += Ys * mean_s
        yield ("auger:" + tag, YA_tot,
               np.where(YA_tot > 0, num_tot / np.maximum(YA_tot, TINY), 0.0))
        if include_secondaries:
            yield ("sec:auger:" + tag, Ys_tot,
                   np.where(Ys_tot > 0, num_s / np.maximum(Ys_tot, TINY), 0.0))


def yield_components(E, Zs, a, material, la_wd01=None, include_secondaries=True):
    """Per-channel yields and mean escaping-electron energies.

    Parameters
    ----------
    E : (N_E,) photon energies [eV]
    Zs : (N_Z,) grain charges
    a : grain radius [cm]
    material : 'graphite', 'silicate', 'cm20' or 'ac'
    la_wd01 : (N_E,) dielectric attenuation length [cm], used below 20 eV
        (as in WD01); if None, the atomic l_a is used everywhere.

    Returns
    -------
    dict
        name -> (Y, E_mean), each (N_E, N_Z). Names: 'band', 'inner:<El><shell>',
        'auger:<El><shell>', and 'sec:<parent>' for each channel above. Holds every
        channel at once: use ``total_yield`` / ``yield_energy_product`` on large grids.
    """
    return {name: (Y, mean) for name, Y, mean in
            _iter_components(E, Zs, a, material, la_wd01, include_secondaries)}


# elements per (N_E, N_Z) temporary; a channel holds ~10 of them
_MAX_CHUNK_ELEMENTS = 1_000_000


def _sum_over_channels(E, Zs, a, material, la_wd01, include_secondaries, with_energy):
    E = np.asarray(E, dtype=float)
    Zs = np.atleast_1d(np.asarray(Zs, dtype=float))
    out = np.zeros((E.size, Zs.size))
    step = max(1, _MAX_CHUNK_ELEMENTS // max(E.size, 1))
    for i in range(0, Zs.size, step):
        sl = slice(i, i + step)
        for _, Y, mean in _iter_components(E, Zs[sl], a, material, la_wd01, include_secondaries):
            out[:, sl] += Y * mean if with_energy else Y
    return out


def total_yield(E, Zs, a, material, la_wd01=None, include_secondaries=True):
    """Electrons emitted per absorbed photon, summed over all channels."""
    return _sum_over_channels(E, Zs, a, material, la_wd01, include_secondaries, False)


def yield_energy_product(E, Zs, a, material, la_wd01=None, include_secondaries=True):
    """sum_k Y_k <E>_k [eV per absorbed photon]: the integrand of eq. 27."""
    return _sum_over_channels(E, Zs, a, material, la_wd01, include_secondaries, True)


def yield_wdb06_vectorized(nu, Zs, a, params):
    """Adapter with the ``yield_func(nu, Zs, a, params)`` signature used by
    ``compute_Rpe_vectorized``. ``params`` needs 'material' and 'E_eV'; 'la'
    (the dielectric l_a on the same grid) is used below 20 eV if present."""
    return total_yield(params["E_eV"], Zs, a, params.get("material", "graphite"),
                       la_wd01=params.get("la"),
                       include_secondaries=params.get("include_secondaries", True))


# ---------------------------------------------------------------------------
# Photodetachment threshold with the van Hoof E_min (WD01 eq. 18, WDB06 sec. 7)
# ---------------------------------------------------------------------------

def photodetachment_energy(Z, a, material):
    """hnu_pdt = EA(Z+1, a) + E_min(Z, a), with E_min from eq. 3."""
    Z = np.asarray(Z, dtype=float)
    return electron_affinity(Z + 1.0, a, material) + min_energy_vH04(Z, a)


def electron_affinity(Z, a, material):
    """EA(Z, a) [eV] of a grain of charge Z (WD01 eq. 18 for graphite and silicate)."""
    from pycalima.models.dust_charge.shared_physics import (
        electron_affinity_graphite_vec,
        electron_affinity_silicate_vec,
    )
    label = _label(material)
    if label == "cm20":
        return cm20_electron_affinity(Z, a)
    if label == "ac":
        return ac_electron_affinity(Z, a)
    if label == "ach":
        return ACH["chi"] + (np.asarray(Z, dtype=float) - 0.5) * _e2_over_a(a)
    if is_acx(label):
        return acx_electron_affinity(Z, a, acx_e_gap(label))
    return (electron_affinity_graphite_vec if label == "graphite" else electron_affinity_silicate_vec)(Z, a)


# ---------------------------------------------------------------------------
# Secondary emission by impinging gas electrons (Draine & Salpeter 1979)
# ---------------------------------------------------------------------------

# DS79 Table 5 rows matching the eps adopted in WDB06 sec. 6.
_DS79_MATERIAL = {"graphite": "graphite", "silicate": "lunar_dust", "cm20": "graphite", "ac": "graphite",
                  "ach": "graphite"}


def _ds79(material):
    """DS79 Table 5 constants of ``material``, at its own density if it has one."""
    label = _label(material)
    p = dict(atomic.secondary_emission_parameters(_DS79_MATERIAL["ac" if is_acx(label) else label]))
    p["rho_gcm3"] = MATERIALS[label].get("rho", p["rho_gcm3"])
    return p

# Gauss-Laguerre nodes for the thermal average over impact energies.
_LAGUERRE_T, _LAGUERRE_W = np.polynomial.laguerre.laggauss(48)


def _electron_range_cm(E0_eV, rho):
    """R(E0) = 300 A (rho / g cm^-3)^-0.85 (E0 / keV)^1.5, DS79 eqs. 11 and A2."""
    return 300.0 * ANGSTROM * rho ** -0.85 * (np.asarray(E0_eV, dtype=float) / 1e3) ** 1.5


def _escape_barrier(Z, a):
    """Energy [eV] an electron leaving a grain of charge Z must climb, (Z+1) e^2/a
    if positive (as for photoelectrons, WD01)."""
    return np.maximum((np.asarray(Z, dtype=float) + 1.0) * _e2_over_a(a), 0.0)


def gas_secondary_yield_at(E0, E_s, a, material):
    """Secondaries per electron impinging with energy E0 [eV] that escape with
    energy > E_s [eV]: DS79's yield delta_inf(E0, 0) (eqs. 14-15) times the
    small-particle factor delta_a / delta_inf of eq. A11, times the escape
    fraction of the eq. 19 secondary spectrum above E_s.

    Zero unless E0 > max(W, E_s). Below the work function, by choice: DS79's fit,
    made for E0 >~ 10^2 eV, would otherwise be extrapolated to a few eV. Below E_s,
    by energy conservation: the primary arrives with E0 + W above the Fermi level,
    and a secondary taken from it needs W + E_s to escape, so it cannot clear a
    barrier higher than E0. The second condition binds for small positive grains,
    whose barrier (Z+1) e^2/a exceeds W: e.g. Z >= 4 at a = 1.3 nm.

    The escape fraction is that of the spectrum on [0, E0]
    (``_secondary_y2_and_mean``), the spectrum ``gas_secondary_mean_energy``
    uses. DS79's factor 1/sqrt(1 + E_s^2/8) is the untruncated spectrum's, its
    limit for E0 >> E_s; near E0 = E_s it kept ~ 1/sqrt(1 + E_s^2/8) of the yield
    where energy conservation leaves none (e.g. 17x too much at E0 = 13 eV,
    E_s = 12.3 eV). That carried the charge chain's tail of nm grains in ~1e4 K
    gas (CALIMA_GSD_dev studies/coulomb_explosion_20261128)."""
    m = _material(material)
    p = _ds79(material)
    E0 = np.asarray(E0, dtype=float)
    x = E0 / p["E_m_eV"]
    E_s = np.asarray(E_s, dtype=float)
    escape = _secondary_y2_and_mean(-E_s, E0 - E_s)[0]
    delta_inf = p["delta_m"] * 4.0 * x / (1.0 + x) ** 2 * escape
    lam = p["lam_A"] * ANGSTROM
    xr = 4.0 * a / (3.0 * _electron_range_cm(np.maximum(E0, TINY), p["rho_gcm3"]))
    f1 = (1.6 + 1.4 * xr ** 2 + 0.54 * xr ** 4) / (1.0 + 0.54 * xr ** 4)  # eq. A7
    r = lam / a
    f2 = (1.0 + 2.0 * r ** 2 + r ** 4) / (1.0 + r ** 4)  # eq. A10
    delta = delta_inf * 2.0 * (-np.expm1(-4.0 * a / (3.0 * lam))) * f1 * f2  # eq. A11
    return np.where(E0 > np.maximum(m["W"], E_s), delta, 0.0)


def _thermal_nodes(Z, a, T, material):
    """Impact energies E [eV] and normalised weights for the flux of thermal
    electrons reaching a grain of charge Z.

    With Coulomb focusing the impact-energy distribution is proportional to
    E exp(-E/kT) for E > max(0, eU) (repelled: E = E_inf - |eU|; attracted:
    E = E_inf + eU), whose mean is DS79 eq. 16. Only E > max(W, E_s) can produce
    escaping secondaries (``gas_secondary_yield_at``), so the nodes start at
    max(eU, W, E_s), with E_s the escape barrier (``_escape_barrier``).
    Returns E (N_Z, N_q) and w (N_Z, N_q) with sum(w) = fraction above that.
    """
    from pycalima.models.dust_charge.shared_physics import KB_CGS

    kT = KB_CGS * T / EV2ERG
    Z = np.atleast_1d(np.asarray(Z, dtype=float))
    L = np.maximum(Z * _e2_over_a(a), 0.0)[:, None]          # flux lower bound
    Lw = np.maximum(np.maximum(L, _material(material)["W"]),
                    _escape_barrier(Z, a)[:, None])         # secondary threshold
    E = Lw + kT * _LAGUERRE_T[None, :]
    # int_Lw^inf E e^{-E/kT} dE = kT e^{-Lw/kT} sum w_i (Lw + kT t_i); normalise by
    # int_L^inf E e^{-E/kT} dE = kT e^{-L/kT} (L + kT)
    w = np.exp(-(Lw - L) / kT) * _LAGUERRE_W[None, :] * E / (L + kT)
    return E, w


def gas_secondary_yield(Z, a, T, material):
    """<delta>: secondaries per impinging thermal electron for a grain of charge
    Z, averaged over the impact-energy distribution (see ``_thermal_nodes``)."""
    Zs = np.atleast_1d(np.asarray(Z, dtype=float))
    E, w = _thermal_nodes(Zs, a, T, material)
    out = np.sum(w * gas_secondary_yield_at(E, _escape_barrier(Zs, a)[:, None], a, material), axis=1)
    return out if np.ndim(Z) else out[0]


def gas_secondary_rate(Zs, a, n_e, T, material):
    """J_sec,gas(Z) [s^-1] (WDB06 eq. 26): the collisional electron arrival rate
    pi a^2 n_e v_e J~(Z, -1) times <delta>. Unlike J_e it carries no sticking
    coefficient: every impinging electron can excite secondaries."""
    from pycalima.models.dust_charge.shared_physics import (
        DS87_J_function_vec,
        KB_CGS,
        ME_CGS,
    )
    Zs = np.atleast_1d(np.asarray(Zs, dtype=float))
    v_e = np.sqrt(8.0 * KB_CGS * T / (np.pi * ME_CGS))
    arrival = np.pi * a ** 2 * n_e * v_e * DS87_J_function_vec(Zs, -1.0, a, T)
    return arrival * gas_secondary_yield(Zs, a, T, material)


def gas_secondary_mean_energy(Zs, a, T, material):
    """Mean energy at infinity [eV] of the escaping gas-excited secondaries.

    WDB06 eq. 28 takes the eq. 19 spectrum on [0, <E0>]; here it is on
    [0, E] for each impact energy E, weighted by delta(E) over the thermal
    distribution. The escaping electron climbs the barrier (Z+1) e^2/a.
    """
    Zs = np.atleast_1d(np.asarray(Zs, dtype=float))
    E, w = _thermal_nodes(Zs, a, T, material)
    E_s = _escape_barrier(Zs, a)[:, None]
    ephi = (Zs + 1.0)[:, None] * _e2_over_a(a)
    _, mean = _secondary_y2_and_mean(-ephi, E - ephi)
    wd = w * gas_secondary_yield_at(E, E_s, a, material)
    norm = np.sum(wd, axis=1)
    return np.where(norm > 0, np.sum(wd * mean, axis=1) / np.maximum(norm, TINY), 0.0)


# ---------------------------------------------------------------------------
# Transmission of fast gas electrons through small grains (Draine & Salpeter 1979)
# ---------------------------------------------------------------------------

def penetration_energy(a, material):
    """E_pen [eV] with R(E_pen) = 4a/3, the mean chord of a sphere (DS79 sec.
    III.a.i): an electron hitting the grain with more energy passes through."""
    p = _ds79(material)
    R_keV = 300.0 * ANGSTROM * p["rho_gcm3"] ** -0.85  # R at 1 keV
    return 1e3 * (4.0 * a / (3.0 * R_keV)) ** (2.0 / 3.0)


def electron_transmission(Zs, a, T, material):
    """(P_tr, f_cool) for thermal electrons reaching a grain of charge Z.

    Straight paths through a thickness 4a/3 with R proportional to E^1.5
    (DS79 eqs. 11-12): an electron entering with E > E_pen leaves with
    E_2 = (E^1.5 - E_pen^1.5)^(2/3) and escapes if E_2 > eU (U = Z e / a), i.e. if
    E > E_c = (E_pen^1.5 + max(eU, 0)^1.5)^(2/3).

    P_tr is the fraction of arriving electrons that pass through and escape
    (they are not captured). f_cool is the energy they and the captured ones
    remove from the gas (E_inf for captured, E - E_2 for transmitted) over the
    energy all arriving electrons carry; it scales the recombination cooling.
    Arrivals follow the impact-energy distribution E exp(-E/kT), E > max(0, eU)
    (see ``_thermal_nodes``); E_inf = E - eU.
    """
    from pycalima.models.dust_charge.shared_physics import KB_CGS

    kT = KB_CGS * T / EV2ERG
    Zs = np.atleast_1d(np.asarray(Zs, dtype=float))
    eU = Zs * _e2_over_a(a)
    L = np.maximum(eU, 0.0)
    Ep = penetration_energy(a, material)
    Ec = np.maximum((Ep ** 1.5 + L ** 1.5) ** (2.0 / 3.0), L)

    # moments of E^n exp(-E/kT) above x, in units of kT^2 exp(-L/kT)
    def m1(x):
        return np.exp(-(x - L) / kT) * (x + kT) / kT
    def m2(x):
        return np.exp(-(x - L) / kT) * (x * x + 2 * kT * x + 2 * kT * kT) / kT

    n_all = m1(L)
    P_tr = m1(Ec) / n_all

    e_all = m2(L) - eU * m1(L)                     # int (E - eU) E e^{-E/kT}
    e_stop = e_all - (m2(Ec) - eU * m1(Ec))        # captured part, E < E_c
    E = Ec[:, None] + kT * _LAGUERRE_T[None, :]    # transmitted part, E > E_c
    E2 = np.maximum(E ** 1.5 - Ep ** 1.5, 0.0) ** (2.0 / 3.0)
    wt = np.exp(-(Ec - L) / kT)[:, None] * _LAGUERRE_W[None, :] * E / kT  # same units as m1, m2
    e_dep = np.sum(wt * (E - E2), axis=1)
    f_cool = np.clip((e_stop + e_dep) / np.maximum(e_all, TINY), 0.0, 1.0)
    return P_tr, f_cool


# Ion stopping as in the sputtering / collisional-cooling tables (nuclear plus
# electronic stopping along the 4a/3 chord, dust_collisional_cooling), with the
# material constants of export_collisional_cooling_bins.
_ION_E_LOOKUP = {}
I_H_EV = 13.605693
M_E_AMU = 5.48579909e-4


def _ion_exit_energy_table(a, material, Z_atomic, mass_amu):
    """(E, E_out) [eV]: kinetic energy of an ion leaving the far side of the
    grain after entering with E; E_out = 0 where it stops inside."""
    key = (_label(material), float(a), int(Z_atomic), float(mass_amu))
    if key not in _ION_E_LOOKUP:
        import pycalima.models.dust_gas_collisions.dust_collisional_cooling as dcc
        from pycalima.models.dust_gas_collisions.export_collisional_cooling_bins import (
            COMPOSITION_PROPERTIES,
        )
        props = COMPOSITION_PROPERTIES["graphite" if key[0] in KIMURA_MATERIALS else key[0]]   # a-C(:H): graphite's
        am, an = props["atomic_mass"], props["atomic_number"]
        rho = MATERIALS[key[0]].get("rho", props["density"])
        n_dust = rho / (am * dcc.au2cgs_m) * an
        ne_val = 4.0 if an <= 6.5 else (2 + 8 + 4 + 4 * 6) / 7.0  # as export_collisional_cooling
        E, E_imp = dcc._build_deposited_energy_lookup_ion(
            n_dust, ne_val, a, Z_atomic, an, am * dcc.au2cgs_m, mass_amu * dcc.au2cgs_m,
            0.05, 1.0, 1e9, nE_lookup=256)
        E_out = E - E_imp
        E_out = np.where(E_out > 1e-3 * E, E_out, 0.0)  # the march stops at 1e-3 E
        _ION_E_LOOKUP[key] = (E, E_out)
    return _ION_E_LOOKUP[key]


def ion_charge_transfer(Zs, a, T, material, z, Z_atomic, mass_amu):
    """Probability that an impinging ion of charge z transfers its charge to a
    grain of charge Z (DS79 sec. III.b, eqs. 19): it does unless it passes
    through and leaves still ionised, i.e. with more than E_B = (M/m_e) I_H
    (faster than ~ the Bohr speed, DS79) and enough to escape an attracting
    grain. Arrivals follow E exp(-E/kT) above max(0, z Z e^2/a).
    """
    from pycalima.models.dust_charge.shared_physics import KB_CGS

    kT = KB_CGS * T / EV2ERG
    Zs = np.atleast_1d(np.asarray(Zs, dtype=float))
    zeU = z * Zs * _e2_over_a(a)
    L = np.maximum(zeU, 0.0)
    E_B = mass_amu / M_E_AMU * I_H_EV
    thr = np.maximum(E_B, -zeU)                      # exit energy needed
    E, E_out = _ion_exit_energy_table(a, material, Z_atomic, mass_amu)
    # lowest entry energy whose exit energy exceeds thr (E_out rises with E)
    E_star = np.array([E[np.argmax(E_out > t)] if np.any(E_out > t) else np.inf for t in thr])
    Ec = np.maximum(E_star, L)
    fail = np.where(np.isfinite(Ec), np.exp(-(Ec - L) / kT) * (Ec + kT) / (L + kT), 0.0)
    return 1.0 - fail


def _atomic_number_from_mass(mass_amu):
    from pycalima.models.dust_gas_collisions.export_collisional_cooling_bins import ION_SPECIES

    best = min(ION_SPECIES, key=lambda sp: abs(sp["mass"] - mass_amu))
    return best["Z"]


def with_ion_transmission(ion_species, a, material):
    """Copy of ``ion_species`` whose entries carry a DS79 charge-transfer
    probability (see ``ion_charge_transfer``). Ion dicts may give 'Z_atomic';
    otherwise it is taken from the nearest mass in the collisional tables."""
    from pycalima.models.dust_gas_collisions.dust_collisional_cooling import au2cgs_m

    out = []
    for ion in ion_species or []:
        ion = dict(ion)
        m_g = float(ion.get("m", 1.0))
        m_g = m_g * 1e3 if m_g < 1e-25 else m_g        # kg -> g, as collisional_rates_ions
        mass_amu = m_g / au2cgs_m
        Z_at = int(ion.get("Z_atomic", _atomic_number_from_mass(mass_amu)))
        z, T = float(ion.get("z", 1.0)), float(ion.get("T", 1.0))
        ion["transfer"] = (lambda Zs, z=z, T=T, Z_at=Z_at, m=mass_amu:
                           ion_charge_transfer(Zs, a, T, material, z, Z_at, m))
        out.append(ion)
    return out


# ---------------------------------------------------------------------------
# Absorption efficiency to X-ray energies (WDB06 sec. 2)
# ---------------------------------------------------------------------------

# Draine (2003) dielectric functions; graphite in the 1/3-2/3 approximation.
_D03_FILES = {
    "graphite": [("callindex.out_CpaD03_0.10", 1.0 / 3.0), ("callindex.out_CpeD03_0.10", 2.0 / 3.0)],
    "silicate": [("callindex.out_silD03", 1.0)],
}
_HC_EV_UM = 1.23984193
X_ADT = 2e4  # Mie below, anomalous diffraction above (WDB06 sec. 2)


def _adt_qabs(x, k):
    """Anomalous-diffraction absorption efficiency of a sphere (van de Hulst):
    1 + 2 e^-w / w + 2 (e^-w - 1) / w^2, w = 4 x Im(m)."""
    w = 4.0 * np.asarray(x, dtype=float) * np.asarray(k, dtype=float)
    ws = np.maximum(w, 1e-3)
    direct = 1.0 + 2.0 * np.exp(-ws) / ws + 2.0 * np.expm1(-ws) / ws ** 2
    series = w * (2.0 / 3.0 - w / 4.0 + w ** 2 / 15.0)
    return np.where(w < 1e-3, series, direct)


def dielectric_attenuation_length(E, material):
    """WD01 photon attenuation length l_a [cm] from the Draine (2003) graphite
    (1/3-2/3) or the smoothed-UV astrosilicate dielectric functions, the
    choice made by ``equilibrium_charge_for_grain``; used below 20 eV.
    None for 'cm20' and 'ac', whose band yield takes Q_ext instead (Kimura 2016)."""
    if material in KIMURA_MATERIALS:
        return None
    from pycalima._paths import get_optical_props_path
    from pycalima.models.dust_charge.shared_physics import (
        photon_attenuation_length_graphite_vec,
        photon_attenuation_length_silicate_vec,
    )
    from pycalima.models.dust_radiation.dust_oppacity import read_dielectric_file

    wav_um = _HC_EV_UM / np.asarray(E, dtype=float)

    def im_n(fname):
        t = read_dielectric_file(str(get_optical_props_path("draine_lee_1984", fname)))["table"]
        w = np.asarray(t["wavelength_um"])
        o = np.argsort(w)
        return np.interp(wav_um, w[o], np.asarray(t["Im_n"])[o])

    if material == "graphite":
        return photon_attenuation_length_graphite_vec(
            wav_um * 1e-4, im_n("callindex.out_CpeD03_0.10"), im_n("callindex.out_CpaD03_0.10"))
    return photon_attenuation_length_silicate_vec(wav_um * 1e-4, im_n("eps_suvSil"))


def _refractive_index(fname, wav_um):
    from pycalima._paths import get_optical_props_path
    from pycalima.models.dust_radiation.dust_oppacity import read_dielectric_file

    t = read_dielectric_file(str(get_optical_props_path("draine_lee_1984", fname)))["table"]
    w = np.asarray(t["wavelength_um"], dtype=float)
    order = np.argsort(w)
    w = w[order]
    n1 = np.asarray(t["Re_n_minus_1"], dtype=float)[order]
    k = np.asarray(t["Im_n"], dtype=float)[order]
    lw = np.log(w)
    re = 1.0 + np.interp(np.log(wav_um), lw, n1)
    im = np.exp(np.interp(np.log(wav_um), lw, np.log(np.maximum(k, 1e-300))))
    return re, im


_QABS_CACHE = {}


def absorption_efficiency(E, a, material, n_table=400):
    """Q_abs(hnu) for a sphere of radius ``a`` [cm] between 1 eV and 20 keV
    (the Draine 2003 range), tabulated once per (material, a); for 'cm20' and
    'ac', the THEMIS optics (``_cm20_efficiencies``)."""
    import miepython

    if material in KIMURA_MATERIALS:
        lE, lQ, _ = _cm20_efficiencies(a, n_table, optics=_optics_of(material))
        return np.exp(np.interp(np.log(np.asarray(E, dtype=float)), lE, lQ))
    key = ("graphite" if material == "graphite" else "silicate", float(a), int(n_table))
    if key not in _QABS_CACHE:
        E_tab = np.geomspace(1.0, 2e4, n_table)
        wav_um = _HC_EV_UM / E_tab
        x = 2.0 * np.pi * a * 1e4 / wav_um
        Q = np.zeros_like(E_tab)
        for fname, weight in _D03_FILES[key[0]]:
            re, im = _refractive_index(fname, wav_um)
            for i in range(E_tab.size):
                if x[i] > X_ADT:
                    Q[i] += weight * float(_adt_qabs(x[i], im[i]))
                else:
                    qext, qsca, _, _ = miepython.efficiencies(re[i] - 1j * im[i], 2.0 * a * 1e4, wav_um[i])
                    Q[i] += weight * max(float(qext - qsca), 0.0)
        _QABS_CACHE[key] = (np.log(E_tab), np.log(np.maximum(Q, 1e-300)))
    lE, lQ = _QABS_CACHE[key]
    return np.exp(np.interp(np.log(np.asarray(E, dtype=float)), lE, lQ))


_CM20_Q_CACHE = {}


_OPTICS = {"cm20": "tables", "ac": "recomputed", "ach": "jones2012"}


def _optics_of(material):
    """Optics key of a Kimura material; 'jones2012@<E_g>' for the intermediate a-C(:H)."""
    return f"jones2012@{acx_e_gap(material):.4f}" if is_acx(material) else _OPTICS[material]


def _cm20_efficiencies(a, n_table=400, optics="tables"):
    """(ln E, ln Q_abs, ln Q_ext) of a CM20 grain of radius ``a`` [cm] on n_table
    energies from 1 eV to 20 keV: ``themis_optics`` (the DustEM tables, Henke-Mie
    below 0.04 um; with ``recomputed``, the large grains by coated-sphere Mie,
    ``themis_optics.mie_efficiencies``; 'jones2012': a-C:H with E_g = 2.5 eV of
    ``jones2012_optics``; 'jones2012@<E_g>': the same at another band gap), and anomalous diffraction on the Henke index above
    x = X_ADT, where Q_ext = 2 (reached only above ~60 eV, where no yield uses Q_ext)."""
    key = (float(a), int(n_table), optics)
    if key not in _CM20_Q_CACHE:
        from pycalima.models.dust_radiation import themis_optics

        E_tab = np.geomspace(1.0, 2e4, n_table)
        lam_um = _HC_EV_UM / E_tab
        a_um = a * 1e4
        x = 2.0 * np.pi * a_um / lam_um
        mie = x <= X_ADT
        qabs = np.empty_like(E_tab)
        qext = np.full_like(E_tab, 2.0)
        e_jones = (ACH["e_gap"] if optics == "jones2012" else float(optics.split("@")[1])
                   if optics.startswith("jones2012@") else None)
        if e_jones is not None:
            from pycalima.models.dust_radiation import jones2012_optics

            qa, qs, _ = jones2012_optics.efficiencies(a_um, lam_um[mie], e_jones)
        else:
            q_of = themis_optics.mie_efficiencies if optics == "recomputed" else themis_optics.efficiencies
            qa, qs, _ = q_of(a_um, lam_um[mie])
        qabs[mie], qext[mie] = qa[:, 0], qa[:, 0] + qs[:, 0]
        if (~mie).any():
            if e_jones is not None:
                from pycalima.models.dust_radiation import jones2012_optics

                m = jones2012_optics._henke_index(e_jones, lam_um[~mie])
            else:
                m = themis_optics.henke_index(a_um, lam_um[~mie])[:, 0]
            qabs[~mie] = _adt_qabs(x[~mie], np.abs(m.imag))
        _CM20_Q_CACHE[key] = (np.log(E_tab), np.log(np.maximum(qabs, 1e-300)), np.log(qext))
    return _CM20_Q_CACHE[key]


def cm20_extinction_efficiency(E, a, material="cm20"):
    """Q_ext(hnu) of a grain of a Kimura material (``_cm20_efficiencies``)."""
    lE, _, lQ = _cm20_efficiencies(a, optics=_optics_of(material))
    return np.exp(np.interp(np.log(np.asarray(E, dtype=float)), lE, lQ))


def cm20_sticking(Zs, a):
    """Electron sticking coefficient s_e(Z) of CM20 grains, as DustEM's GAS_CURRENTS.

    s_e = p1 (1 - exp(-p2 a / l_e)) for Z > Zmin, 0 at Zmin, where l_e is the
    photoelectron escape length averaged as <1/l_e> over the band photoemission
    rate (DM_zdist PE_CURRENT) in the Mathis et al. (1983) ISRF; the average
    depends on the spectral shape only, so it is fixed per grain here. As in
    DustEM's recursion, the sticking at charge Z takes the l_e of charge Z - 1,
    and of the last charge with photoemission where Z - 1 has none. Z - 1 is
    rounded to an integer, so that a continuous Z (the solver's root search, the
    table's U nodes) sees steps at half-integers, not at the integers.
    """
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    Zs = np.atleast_1d(np.asarray(Zs, dtype=float))
    E = np.geomspace(1.0, 13.6, 400)
    n_E = np.array([Mathis83_radiation_field(e) for e in E]) / E   # photons per eV, any units
    w = absorption_efficiency(E, a, "cm20") * n_E
    Y, _, le = cm20_band_yield(E, np.round(Zs) - 1.0, a, 0.75 * cm20_extinction_efficiency(E, a))
    rate = np.trapezoid(w[:, None] * Y, E, axis=0)
    inv = np.trapezoid(w[:, None] * Y / le, E, axis=0)
    ok = rate > 0
    le_bar = np.where(ok, rate / np.maximum(inv, TINY), np.nan)
    if not ok.any():
        le_bar[:] = cm20_escape_length(CM20["W"])
    else:
        order = np.argsort(Zs)
        last = np.nan
        for k in order:                   # carry the last value up to the charges without photoemission
            if ok[k]:
                last = le_bar[k]
            elif not np.isnan(last):
                le_bar[k] = last
        le_bar = np.where(np.isnan(le_bar), le_bar[order][ok[order]][0], le_bar)
    p1, p2, _ = CM20["stick"]
    s = p1 * (-np.expm1(-p2 * a / le_bar))
    return np.where(Zs <= most_negative_charge_cm20(a), 0.0, s)
