"""Photon absorption by PAHs: cross sections, ionisation yields, heating.

An absorbed photon of energy E either ionises the PAH, with probability
Y_ion(E), or heats it. The rates are

  r_abs = int sigma F dE,   k_ion = int sigma Y F dE,
  heating spectrum h(E) = sigma (1 - Y) F   [s^-1 eV^-1],

where h is what the internal-energy master equation (``energy_me``) receives.
An ionising photon is taken to leave no internal energy, as in Montillaud et
al. (2013). Andrews et al. (2016) do not say; this is the one place where the
energy given to the electron would enter.

Cross sections [cm^2] on an energy grid in eV:

- ``sigma_li_draine``: Li & Draine (2001), neutral tables for Z <= 0 and
  ionised tables for Z > 0, at the LD01 radius a = 10 A (N_C/468)^(1/3). The
  legacy PAH_photophysics code used the disc radius 0.9 sqrt(N_C) A, which
  overestimates sigma by 1.7-3.3x for C24-C96.
- ``sigma_malloci_coronene``: Malloci et al. (2007) TD-DFT for C24H12 at
  Z = 0, +1, from the fetched ``malloci2007-spectra`` (the served anion file is a C54H18- spectrum,
  ``malloci2007.INCONSISTENT``).
- ``sigma_malloci_average``: Berne+22's strategy, pahchem's default since 2026-10-26 (``Params.uv_cross_section``).
  - sigma / N_C is averaged per charge over the Malloci+07 spectra of nine compact PAHs, N_C 32-66
    (``MALLOCI_AVERAGE_MOLECULES``), times N_C.
  - Berne+22 use it at 5-13.6 eV, where their field is. Here it covers 5-30 eV, Malloci's range; Li & Draine is used
    below 5 eV and above 30 eV.
  - Below 5 eV, Malloci's spectra carry a floor from their exponential damping window, 60-300x Li & Draine at
    1-1.5 eV. In the old-star light of the ISRF it would triple the absorption below 5 eV
    (results/pahchem/uv_sigma_2026-10-26).
  - At 5-13.6 eV the average absorbs 0.69-0.87x Li & Draine's photons; at 17 eV it is 0.66x.
  - Z = +3, which Malloci did not compute, takes the dication average.
- ``sigma_uv``: the cross section of a ``Params.uv_cross_section`` name.
- ``sigma_malloci_molecule``: one Malloci+07 molecule, optionally scaled to another N_C, for the reproductions of
  papers that used it (Andrews+16, Montillaud+13).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


HC_EV_UM = 1.23984193

# no anion: the served coronene anion file is a C54H18- spectrum (malloci2007.INCONSISTENT)
_MALLOCI_CHARGES = (0, 1)


def li_draine_radius_um(nc: int) -> float:
    """Li & Draine (2001) PAH radius: 10 A (N_C/468)^(1/3), in micron."""
    return 1e-3 * (nc / 468.0) ** (1.0 / 3.0)


@lru_cache(maxsize=None)
def _li_draine_table(kind):
    """The parsed Li & Draine (2001) efficiency table (parsing it takes ~0.3 s)."""
    import os

    from pycalima.models.PAH_radiation.pah_oppacity import PATH_OPTICS, pah_efficiencies

    name = "PAHion_30" if kind == "iPAH" else "PAHneu_30"
    return pah_efficiencies(os.path.join(PATH_OPTICS, "li_draine_2001", name))


_SIGMA_CACHE: dict = {}


def ld01_radius_and_volume_factor(radius_um, kind="nPAH"):
    """(radius at which to read the Li & Draine table, factor on C_abs).

    The table starts at a = 3.548 A (N_C = 20.9). The 2-D interpolation holds
    Q_abs at that radius below it while C_abs = Q pi a^2 still scales as a^2,
    so sigma per C rose as (20.9 / N_C)^(1/3): 1.28 for C10, 1.14 for C14
    (pahchem-audit, C4). Below the table, C_abs is taken at a_min and scaled
    by volume, (a / a_min)^3, i.e. Q proportional to a, as LD01's per-C cross
    section implies.
    """
    a_min = min(float(r) for r in _li_draine_table(kind)[1])
    if radius_um >= a_min:
        return float(radius_um), 1.0
    return a_min, (float(radius_um) / a_min) ** 3


def sigma_li_draine(nc, charge, energy_ev) -> np.ndarray:
    """Li & Draine (2001) absorption cross section [cm^2].

    Memoised on (N_C, neutral/ionised, energy grid): the 2-D interpolation
    costs ~0.2 s and callers ask for the same grid for every state. Below the
    table's smallest radius (N_C < 21) it scales with N_C
    (``ld01_radius_and_volume_factor``).
    """
    from pycalima.models.PAH_radiation.pah_oppacity import interpolate_pah_cross_sections_2d

    E = np.asarray(energy_ev, dtype=float)
    kind = "iPAH" if charge > 0 else "nPAH"
    key = (float(nc), kind, E.shape, E.tobytes())
    if key not in _SIGMA_CACHE:
        if len(_SIGMA_CACHE) > 256:
            _SIGMA_CACHE.clear()
        radius, factor = ld01_radius_and_volume_factor(li_draine_radius_um(nc), kind)
        _, _, _, c_abs, _ = interpolate_pah_cross_sections_2d(
            kind, radius, target_wavelengths=HC_EV_UM / E, data_table=_li_draine_table(kind))
        _SIGMA_CACHE[key] = factor * np.asarray(c_abs, dtype=float)
    return _SIGMA_CACHE[key].copy()


def rf_levels_from_cross_section(energy_ev, sigma_cm2, e_max_ev=3.1, e_min_ev=0.5, f_min=1e-4):
    """Recurrent-fluorescence levels [(E_peak [eV], f)] read off an absorption cross section.

    Each band between two successive minima of sigma below ``e_max_ev`` becomes
    one level, at its peak, with f = (m_e c / pi e^2) int sigma dnu. A band cut
    by ``e_max_ev`` is dropped, and so are bands with f < ``f_min`` or a peak
    below ``e_min_ev``. Below 0.5 eV (4000 cm^-1) an electronic state sits among
    the fundamental vibrations, where a broadened TD-DFT spectrum is not
    reliable. Taking one level per band keeps the line broadening from moving
    strength to low E_n, where rho(E - E_n)/rho(E) makes RF most sensitive.
    A weak low-lying band still matters: in coronene+ the 0.6 eV band
    (f = 5e-4) carries most of the RF power at low internal energy.
    """
    E = np.asarray(energy_ev, dtype=float)
    s = np.clip(np.asarray(sigma_cm2, dtype=float), 0.0, None)
    keep = E <= e_max_ev
    E, s = E[keep], s[keep]
    minima = [0] + [i for i in range(1, E.size - 1) if s[i] <= s[i - 1] and s[i] < s[i + 1]]
    f_per = 9.1093837e-28 * 2.99792458e10 / (np.pi * 4.80320471e-10**2) / 4.135667696e-15   # per cm^2 eV
    levels = []
    for a, b in zip(minima[:-1], minima[1:]):
        seg = slice(a, b + 1)
        f = f_per * np.trapezoid(s[seg], E[seg])
        peak = float(E[seg][np.argmax(s[seg])])
        if f >= f_min and peak >= e_min_ev:
            levels.append((peak, float(f)))
    return levels


def malloci_rf_levels(name, charge, **kwargs):
    """``rf_levels_from_cross_section`` of a Malloci+07 spectrum (``malloci2007.sigma_of``, e.g. "coronene", 1)."""
    from pycalima.pahchem import malloci2007

    return rf_levels_from_cross_section(*malloci2007.sigma_of(name, charge), **kwargs)


def sigma_amespahdb(nc, charge, energy_ev) -> np.ndarray:
    """Absorption cross section of the AmesPAHdb suites' ISRF/stellar convolution [cm^2].

    ``Transitions.absorption_cross_section`` of the AmesPAHdb Python suite,
    transcribed: the two UV Drude terms and exponential of Draine & Li (2007),
    times the visible/near-IR cutoff of Mattioda et al. (2005, eq. 4; Salama et
    al. 1996), plus, for any Z != 0, the ion near-IR continuum and Gaussian
    bands of Mattioda et al. (2005). It is the cross section Maragkoudakis+20
    state ("Draine & Li 2007 and Mattioda 2005"). As in the suite, it is not
    multiplied by N_C: N_C enters only through the cutoff.
    """
    wave = HC_EV_UM / np.asarray(energy_ev, dtype=float)               # um
    A = np.array([7.97e-17, 1.23e-17, 20e-21, 14e-21, 80e-24, 84e-24, 46e-24, -322e-24])
    W = np.array([0.195, 0.217, 0.0805, 0.20, 0.0370, 0.0450, 0.0150, 0.135])
    C = np.array([0.0722, 0.2175, 1.05, 1.23, 1.66, 1.745, 1.885, 1.90])
    y = 1.0 / (0.889 + 2.282 / np.sqrt(0.4 * nc)) / wave
    w = wave[None, :]
    drude = np.sum((W[:2] * C[:2] * A[:2])[:, None]
                   / ((w / C[:2, None] - C[:2, None] / w) ** 2 + W[:2, None] ** 2), axis=0)
    sigma = (np.arctan(1e3 * (y - 1.0) ** 3 / y) / np.pi + 0.5) * (3458e-20 * 10.0 ** (-3.431 * wave)
                                                                    + 2.0 / np.pi * drude)
    if charge != 0:
        sigma = sigma + np.exp(-0.1 / wave**2) * 1.5e-19 * 10.0 ** (-wave) + np.sqrt(2.0 / np.pi) * np.sum(
            (A[2:] / W[2:])[:, None] * np.exp(-2.0 * (w - C[2:, None]) ** 2 / W[2:, None] ** 2), axis=0)
    return sigma


def sigma_malloci_coronene(charge, energy_ev) -> np.ndarray:
    """Malloci et al. (2007) C24H12 cross section [cm^2] (fetched ``malloci2007-spectra``); zero outside 0-30 eV."""
    from pycalima.pahchem import malloci2007

    if charge not in _MALLOCI_CHARGES:
        raise ValueError(f"Malloci coronene sigma exists for Z = 0, 1, not {charge}")
    E, s = malloci2007.sigma_of("coronene", charge)
    return np.interp(energy_ev, E, s, left=0.0, right=0.0)


#: Berne+22's molecules for Z = 0, +1, +2 (their anion set adds coronene, whose served anion file is C54H18-'s and
#: is left out: ``malloci2007.INCONSISTENT``), N_C 32-66
MALLOCI_AVERAGE_MOLECULES = ("ovalene", "tetrabenzocoronene", "circumbiphenyl", "circumanthracene", "circumpyrene",
                             "hexabenzocoronene", "dicoronylene", "circumcoronene", "circumovalene")
#: where the Malloci average is used [eV]; Li & Draine outside (module docstring)
MALLOCI_AVERAGE_RANGE_EV = (5.0, 30.0)
UV_CROSS_SECTIONS = ("malloci_average", "li_draine")


@lru_cache(maxsize=None)
def _malloci_per_carbon(charge):
    """(E [eV], <sigma / N_C> [cm^2]) over ``MALLOCI_AVERAGE_MOLECULES`` on Malloci's 0-30 eV grid."""
    from pycalima.pahchem import malloci2007

    z = min(int(charge), 2)
    spec = malloci2007.spectra()
    E = None
    rows = []
    for name in MALLOCI_AVERAGE_MOLECULES:
        e, sig = spec[name]["sigma"][z]
        if E is None:
            E = e
        rows.append(np.interp(E, e, sig / spec[name]["nc"]))
    return E, np.mean(rows, axis=0)


def sigma_malloci_average(nc, charge, energy_ev) -> np.ndarray:
    """Berne+22's cross section [cm^2]: N_C x the Malloci+07 average per C of ``MALLOCI_AVERAGE_MOLECULES`` at the
    state's charge (Z >= +2: the dication) within ``MALLOCI_AVERAGE_RANGE_EV``, Li & Draine outside."""
    E = np.asarray(energy_ev, dtype=float)
    lo, hi = MALLOCI_AVERAGE_RANGE_EV
    inside = (E >= lo) & (E <= hi)
    out = sigma_li_draine(nc, charge, E)
    if inside.any():
        em, per_c = _malloci_per_carbon(max(-1, min(int(charge), 2)))
        out[inside] = nc * np.interp(E[inside], em, per_c)
    return out


def sigma_malloci_molecule(name, charge, energy_ev, nc=None) -> np.ndarray:
    """Malloci+07 cross section [cm^2] of molecule ``name`` at ``charge``, times nc / N_C(name) if ``nc`` is given; zero
    outside 0-30 eV. A charge Malloci did not compute (or whose served file is ``malloci2007.INCONSISTENT``) takes the
    nearest computed charge of the same molecule."""
    from pycalima.pahchem import malloci2007

    spec = malloci2007.spectra()[name]
    z = min(spec["sigma"], key=lambda q: (abs(q - int(charge)), -q))
    e, s = spec["sigma"][z]
    scale = 1.0 if nc is None else nc / spec["nc"]
    return scale * np.interp(np.asarray(energy_ev, dtype=float), e, s, left=0.0, right=0.0)


def sigma_uv(nc, charge, energy_ev, model="malloci_average") -> np.ndarray:
    """The UV absorption cross section [cm^2] of ``model`` (``UV_CROSS_SECTIONS``; ``Params.uv_cross_section``)."""
    if model == "malloci_average":
        return sigma_malloci_average(nc, charge, energy_ev)
    if model == "li_draine":
        return sigma_li_draine(nc, charge, energy_ev)
    raise ValueError(f"uv_cross_section must be one of {UV_CROSS_SECTIONS}, not {model!r}")


def yield_jochims1996(energy_ev, ip_ev) -> np.ndarray:
    """Linear rise from IP to unity at IP + 9.2 eV (Jochims et al. 1996)."""
    return np.clip((np.asarray(energy_ev, dtype=float) - ip_ev) / 9.2, 0.0, 1.0)


def yield_wenzel2020(energy_ev, ip_ev, nc, alpha=0.3) -> np.ndarray:
    """Cation -> dication yield of Wenzel et al. (2020), eqs. 4-5.

    Zero below IP2, a linear rise to the plateau alpha = 0.3 at 11.3 eV, flat
    to 12.9 eV, a linear rise to beta(N_C) at 15 eV, then flat; beta =
    0.59 + 8.1e-3 N_C (32 <= N_C < 50), 1 above. Fitted to their SOLEIL
    yields of C30H14+ to C48H20+ and stated for N_C >= 32 (extendable to ~100,
    their conclusions); undefined for IP2 >= 11.3 eV. Both raise.
    """
    if nc < 32:
        raise ValueError(f"Wenzel+20 eq. 5 is stated for N_C >= 32, not {nc}")
    if ip_ev >= 11.3:
        raise ValueError(f"Wenzel+20 eq. 4 needs IP2 < 11.3 eV, not {ip_ev}")
    E = np.asarray(energy_ev, dtype=float)
    beta = 1.0 if nc >= 50 else 0.59 + 8.1e-3 * nc
    return np.select([E < ip_ev, E < 11.3, E < 12.9, E < 15.0],
                     [0.0, alpha * (E - ip_ev) / (11.3 - ip_ev), alpha,
                      alpha + (beta - alpha) * (E - 12.9) / 2.1], beta)


#: hc [eV um]: Le Page+01 eq. 14 takes every energy in um^-1
EV_UM = 1.239842


def yield_lepage2001(energy_ev, ip_ev, ip_coronene_ev=7.02, form="montillaud13") -> np.ndarray:
    """Y = 0.8 exp{-b [c (E - d)]^4}, c = (d - IP_cor)/(d - IP); zero below IP.

    ``form="montillaud13"``: Montillaud et al. (2013) eq. 1, E in eV with
    d = 14.89 eV and b = 0.00128. ``form="lepage01"``: Le Page et al. (2001)
    eq. 14 as printed, with every energy in um^-1 (d = 12 um^-1 = 14.88 eV,
    b = 0.00128 um^4). Montillaud's transcription kept b while converting d
    to eV, which lowers Y by ~9x at 7.5 eV and ~1.5x at 10 eV (method card
    lepage01.toml, yield_unit_discrepancy).

    Fitted below 13.6 eV only. The quartic is symmetric about d, so above d
    the expression falls back to zero; ``population._yield`` replaces it
    there by a ramp to unity.
    """
    E = np.asarray(energy_ev, dtype=float)
    if form == "montillaud13":
        x, ip, ip_cor, d = E, ip_ev, ip_coronene_ev, 14.89
    elif form == "lepage01":
        x, ip, ip_cor, d = E / EV_UM, ip_ev / EV_UM, ip_coronene_ev / EV_UM, 12.0
    else:
        raise ValueError(f"form must be 'montillaud13' or 'lepage01', not {form!r}")
    c = (d - ip_cor) / (d - ip)
    return np.where(E >= ip_ev, 0.8 * np.exp(-0.00128 * (c * (x - d)) ** 4), 0.0)


def ionisation_rate(field, sigma_cm2, yield_, energy_ev) -> float:
    """k_ion = int sigma Y F dE [s^-1]."""
    return field.photon_rate(np.asarray(sigma_cm2) * np.asarray(yield_), energy_ev)


#: mean share of E - IP carried by the photoelectron for Z >= 0: Berné+22 (App. C.4), the unweighted average over
#: 7.1-10.1 eV of the coronene photoelectron spectra of Bréchignac+14; 0.5 is the classical value (V90, BT94)
PHOTOELECTRON_PARTITION = 0.46


#: (hv [eV], <E_k>/(hv - IP)) above 13.6 eV: the means over 8 PAHs (C13-C20) of the Hrodmarsson+25 (A&A 698, A202)
#: photoelectron matrices (Zenodo 10.5281/zenodo.15271348, deposited by Verstraete et al.; their 7-22 eV SOLEIL
#: data, reduced 2026-10-22; ``pahchem.hrodmarsson2025``): 0.43-0.48 at 13.6, 0.47-0.52 at 15,
#: 0.53-0.59 at 18, 0.55-0.61 at 20, 0.54-0.62 at 21.9 eV
PARTITION_NODES = ((13.6, 0.46), (15.0, 0.49), (18.0, 0.56), (20.0, 0.58), (21.9, 0.58))


def photoelectron_energy(energy_ev, threshold_ev, partition=PHOTOELECTRON_PARTITION):
    """(<E_k>, E_int) [eV] of the photoelectron and of the ion after an ionisation at ``energy_ev``.

    - E <= 13.6 eV: <E_k> = ``partition`` (E - threshold).
    - 13.6-21.9 eV: the measured share (``PARTITION_NODES``, scaled by ``partition`` / 0.46 so that the two join).
    - Above 21.9 eV: the ion keeps the internal energy it has at 21.9 eV (the hole energy, ~6 eV for these
      PAHs), and the electron takes the rest, E - threshold - E_int. No photoelectron spectra of PAHs exist
      above 22 eV; this is the molecular limit (KE = hv - binding energy), not the solid-state WDB06 parabola.
    Zero below the threshold. A share above 1 is capped (the electron cannot take more than the excess).
    """
    E = np.asarray(energy_ev, dtype=float)
    excess = np.clip(E - threshold_ev, 0.0, None)
    e_n, f_n = np.array(PARTITION_NODES).T
    scale = partition / PHOTOELECTRON_PARTITION
    share = np.minimum(np.where(E <= e_n[0], partition, scale * np.interp(E, e_n, f_n)), 1.0)
    e_int_top = (1.0 - min(scale * f_n[-1], 1.0)) * max(e_n[-1] - threshold_ev, 0.0)
    e_int = np.where(E <= e_n[-1], (1.0 - share) * excess, np.minimum(e_int_top, excess))
    return excess - e_int, e_int


def photoelectron_power(field, sigma_cm2, yield_, energy_ev, threshold_ev, partition=PHOTOELECTRON_PARTITION,
                        electron_energy_ev=None) -> float:
    """Kinetic energy given to the gas by photoelectrons, int sigma Y F <E_k> dE [eV s^-1].

    The same sigma, yield and threshold as ``ionisation_rate``, so the heating and the charge balance share
    one rate. <E_k> = ``partition`` (E - threshold), the share of the excess the electron carries (the default
    for photoionisation; 1 if the electron takes all of it), unless ``electron_energy_ev`` gives the mean
    energy per event on ``energy_ev`` directly (e.g. ``population.statistical_detachment``).
    """
    E = np.asarray(energy_ev, dtype=float)
    if electron_energy_ev is None:
        ek = partition * np.clip(E - threshold_ev, 0.0, None)
    else:
        ek = np.asarray(electron_energy_ev, dtype=float)
    return field.photon_rate(np.asarray(sigma_cm2) * np.asarray(yield_) * ek, E)


def heating_spectrum(field, sigma_cm2, yield_, energy_ev) -> np.ndarray:
    """h(E) = sigma (1 - Y) F [s^-1 eV^-1]: absorptions that heat, per photon energy."""
    return np.asarray(sigma_cm2) * (1.0 - np.asarray(yield_)) * field.flux_at(energy_ev)
