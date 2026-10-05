"""The WD01 charge distribution of Murga et al. (2020, 2022), on pyCALIMA's WD01 physics.

Murga+2020 (Sect. 3) and Murga+2022 (Sects. 3.1, 4.3.3) take f(a, Z) from
Weingartner & Draine (2001a) (method cards murga2020.toml, murga2022.toml).
``f_wd01`` solves the WD01 birth-death chain on Z_min..Z_max from the rate
functions of ``models.dust_charge`` (WD01 carbonaceous photoemission and
photodetachment yields, IP and EA laws, DS87 Coulomb factors).

The M22 Figshare f(a, Z) (158 locations x 10 sizes, with chi, T and n_e
tabulated per location) fix the choices Murga do not state.
``MURGA2022`` is the configuration that reproduces them
(check_murga_charge.py):
- **Field shape: MMP83** (``field="Mathis"``), scaled by chi. chi is
  therefore in MMP83 units, although MARION's is Draine (card
  [field].chi_definition_sect2). The Draine shape gives <Z> +0.18 in PDR 2.
- **Electron sticking s_e = 1** for every Z. With the WD01 law
  (0.5 (1 - exp(-a/l_e)), and the small-N_C factor for Z <= 0) <Z> is
  +0.4 to +0.95 too high in the PDR.
- **No ion collisions.** With C+ at n(C+) of Fig. 6, the DF and MC are
  +0.03 to +0.08 too positive.
- **Z_min of WD01 eq. 24 as printed** (``z_min="int"``): no anions below
  6 A, Z >= -1 to 7.8 A, Z >= -2 from 8.7 A, exactly as in the files.
- **The Z = -2..9 grid truncates the chain** (``fold=False``), at 50 A.
- **Absorption cross section:** Li & Draine (2001) neutral PAH Q_abs at
  radius a (``cabs="nPAH"``; ``"iPAH"`` is identical to 0.001 in <Z>).
  Draine & Lee graphite Mie lowers <Z> by up to 1.5 in PDR 2.
The HII points are excluded: M22 add EUV photons there from an O6 star with
no stated spectrum.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

AMU_G = 1.66054e-24
C_CGS, H_CGS, EV_ERG = 2.99792458e10, 6.62607015e-27, 1.602176634e-12


def radius_cm(nc):
    """WD01 / Li & Draine radius: a = 10 A (N_C / 468)^(1/3)."""
    return 1e-7 * (np.asarray(nc, float) / 468.0) ** (1.0 / 3.0)


@lru_cache(maxsize=None)
def _field(model):
    """(nu [Hz], photons cm^-2 s^-1 Hz^-1 at chi = 1, wavelength [cm]), increasing nu."""
    from pycalima.models.dust_charge.dust_photoelectric_heating import get_radiation_field

    rad, _ = get_radiation_field(model)
    rad = np.asarray(rad, float)
    if rad.shape[1] >= 3:                                   # [E_eV, lambda_nm, I_E erg s^-1 cm^-2 eV^-1]
        order = np.argsort(rad[:, 0])
        E, lam_nm, I_E = rad[order, 0], rad[order, 1], rad[order, 2]
        lam = lam_nm * 1e-7
        nu = C_CGS / lam
        J = I_E / (nu * EV_ERG)
    else:                                                   # [lambda_nm, I_lambda erg s^-1 cm^-2 nm^-1 sr^-1]
        lam = rad[::-1, 0] * 1e-7
        I_lam = rad[::-1, 1] * 1e7
        nu = C_CGS / lam
        J = 4.0 * np.pi * I_lam * lam ** 3 / (H_CGS * C_CGS ** 2)
        order = np.argsort(nu)
        nu, J, lam = nu[order], J[order], lam[order]
    return nu, J, lam


@lru_cache(maxsize=None)
def _optics(a_cm, cabs, lam_key):
    from pycalima.models.dust_charge.dust_charging import PATH_OPTICS
    from pycalima.models.dust_charge.dust_photoelectric_heating import read_dielectric_file
    from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections

    lam = np.asarray(lam_key)
    _, wav_cs, _, C_abs, _ = interpolate_cross_sections(cabs, float(a_cm) * 1e4)
    order = np.argsort(wav_cs)
    C = np.interp(lam, wav_cs[order], C_abs[order])
    perp = read_dielectric_file(f"{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10")["table"]
    par = read_dielectric_file(f"{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10")["table"]
    um = lam * 1e4
    im_perp = np.interp(um, perp["wavelength_um"][::-1], perp["Im_n"][::-1])
    im_par = np.interp(um, par["wavelength_um"][::-1], par["Im_n"][::-1])
    return C, im_perp, im_par


def z_min_wd01(a_cm, rounding="int"):
    """Most negative charge, WD01 eq. 24: Z_min = int(U_ait a / 14.4 V A) + 1,
    U_ait = -(3.9 + 0.12 a/A + 2 A/a) V (carbonaceous, eq. 23).

    ``rounding="int"`` takes the integer part (truncation toward zero), as
    eq. 24 is printed; it reproduces the Murga+2022 Figshare limits exactly
    (Z_min = 0 at 3.3-5 A, -1 at 6-7.8 A, -2 at 8.7 A). ``"floor"`` is what
    ``models.dust_charge.shared_physics.most_negative_allowed_charge_graphite``
    computes: one charge more negative for every grain below ~100 A.
    """
    a_A = np.asarray(a_cm, float) * 1e8
    x = -(3.9 + 0.12 * a_A + 2.0 / a_A) / 14.4 * a_A
    return (np.trunc(x) if rounding == "int" else np.floor(x)).astype(int) + 1


def _sticking(kind):
    """Electron sticking s_e(Z, a): ``"wd01"`` (the WD01 law of pyCALIMA, with
    0.5 (1 - exp(-a/l_e)) and the small-N_C factor for Z <= 0) or a constant
    for every Z (s_e = 0 at Z_min is kept by the chain truncation)."""
    from pycalima.models.dust_charge.shared_physics import electron_sticking_coefficient_graphite

    if kind == "wd01":
        return electron_sticking_coefficient_graphite
    value = float(kind)
    return lambda Z, a: np.full(np.shape(np.asarray(Z, float)), value)


def f_wd01(a_cm, chi, temperature_k, n_e, n_cplus=0.0, field="Draine", cabs="nPAH", z_range=None,
           z_min="int", sticking="wd01", fold=True):
    """WD01 equilibrium charge distribution of a carbonaceous grain of radius ``a_cm``.

    Returns (Z, f) on Z_min..Z_max (WD01 eqs. 22-24), f normalised.
    - The birth-death chain is solved here, in log space, from pyCALIMA's
      rate functions (``compute_Rpe_vectorized`` with the WD01 graphite yield
      and photodetachment, ``collisional_rates_electrons_vector``,
      ``collisional_rates_ions_vector``). The solver's own bounds finder
      overflows at chi <~ 0.01 and T ~ 13 K (deep MC) and returns a spurious
      Z_min; the fixed WD01 range avoids it.
    - ``z_min``: ``"int"`` is WD01 eq. 24 as printed (``z_min_wd01``);
      ``"floor"`` is pyCALIMA's ``most_negative_allowed_charge_graphite``.
    - ``sticking``: see ``_sticking``.
    - ``z_range=(lo, hi)`` restricts the result to a fixed Z grid (M22:
      Z = -2..9). ``fold=True`` moves the probability outside it into the end
      states; ``fold=False`` drops it and renormalises (the chain truncated
      at the grid edges).
    """
    from pycalima.models.dust_charge.dust_charging import (compute_Rpe_vectorized, electron_escape_length,
                                                           graphite_work_function, yield_graphite_vectorized)
    from pycalima.models.dust_charge.shared_physics import (collisional_rates_electrons_vector,
                                                            collisional_rates_ions_vector,
                                                            most_positive_allowed_charge)

    nu, J1, lam = _field(field)
    C, im_perp, im_par = _optics(float(a_cm), cabs, tuple(lam))
    params = {"material": "graphite", "W": graphite_work_function, "le": electron_escape_length,
              "Imperp": im_perp, "Impar": im_par, "wav": lam}
    a = float(a_cm)
    hnu_max = float(H_CGS * nu.max() / EV_ERG)
    Z = np.arange(int(z_min_wd01(a, z_min)), int(most_positive_allowed_charge(a, graphite_work_function,
                                                                               hnu_max)) + 1)
    up = compute_Rpe_vectorized(nu, J1 * float(chi), C, a, Z, yield_graphite_vectorized, params)
    if n_cplus > 0:
        ions = [{"n": float(n_cplus), "T": float(temperature_k), "m": 12.0 * AMU_G, "z": 1, "name": "C+"}]
        up = up + collisional_rates_ions_vector(np.array([a]), Z, ions)
    down = collisional_rates_electrons_vector(np.array([a]), Z, float(n_e), float(temperature_k), _sticking(sticking))
    with np.errstate(divide="ignore"):
        step = np.log(np.asarray(up[:-1], float)) - np.log(np.asarray(down[1:], float))   # ln P(Z+1)/P(Z)
    logP = np.concatenate([[0.0], np.cumsum(np.nan_to_num(step, nan=-np.inf, posinf=700.0))])
    logP = np.where(np.isfinite(logP), logP, -np.inf)
    P = np.exp(logP - logP.max())
    if z_range is not None:
        lo, hi = z_range
        grid = np.arange(lo, hi + 1)
        f = np.array([P[Z == z].sum() for z in grid])
        if fold:
            f[0] += P[Z < lo].sum()
            f[-1] += P[Z > hi].sum()
        Z, P = grid, f
    return Z, P / P.sum()


#: the configuration that reproduces the Murga+2022 Figshare f(a, Z) (module docstring)
MURGA2022 = dict(field="Mathis", cabs="nPAH", sticking=1.0, z_min="int", fold=False)
#: the Figshare charge grid
Z_GRID_2022 = (-2, 9)


def f_murga2022(a_cm, chi, temperature_k, n_e):
    """f(Z) on Z = -2..9 in the ``MURGA2022`` configuration (no ions)."""
    return f_wd01(a_cm, chi, temperature_k, n_e, 0.0, z_range=Z_GRID_2022, **MURGA2022)


def f_bt94_ladder(nc, chi, temperature_k, n_e, z_range=(-1, 3)):
    """Cross-check, not Murga's model: pahchem's own charge chain for one N_C.

    Photoionisation int sigma Y F dE with the Li & Draine sigma, the Jochims+96
    yield and the WD01 IP(Z) ladder in the MMP83 field (G0 = 1.13 chi);
    capture ``chemistry.electron_capture`` (BT94, s_e = 1); photodetachment
    of the anion with Y = 1 above EA. Returns (Z, f).
    """
    from pycalima.pahchem import chemistry, fields, photo
    from pycalima.models.dust_charge.shared_physics import (electron_affinity_graphite_vec,
                                                            ionisation_potential_valence_vec)

    E = np.linspace(0.6, 13.6, 600)
    F = fields.mathis1983().scaled(1.13 * float(chi))
    a = radius_cm(nc)
    Z = np.arange(z_range[0], z_range[1] + 1)
    up = np.zeros(Z.size)
    for i, z in enumerate(Z):
        if z >= 0:                                         # WD01 eq. 2 IP; Jochims yield above it
            ip = float(ionisation_potential_valence_vec(4.4, z, a))
            y = photo.yield_jochims1996(E, ip)
        else:                                              # detachment above EA(Z + 1), WD01 eq. 4; Y = 1
            ip = float(electron_affinity_graphite_vec(z + 1, a))
            y = (E >= ip).astype(float)
        up[i] = F.photon_rate(photo.sigma_li_draine(nc, min(int(z), 1), E) * np.clip(y, 0, 1), E)
    down = np.array([chemistry.electron_capture(nc, int(z), temperature_k, radius_cm=a) * n_e for z in Z])
    with np.errstate(divide="ignore"):
        logP = np.concatenate([[0.0], np.cumsum(np.log(up[:-1]) - np.log(down[1:]))])
    P = np.exp(logP - logP[np.isfinite(logP)].max())
    return Z, np.where(np.isfinite(P), P, 0.0) / np.nansum(np.where(np.isfinite(P), P, 0.0))


def mean_and_mode(Z, f):
    """<Z> and the most probable Z."""
    Z, f = np.asarray(Z), np.asarray(f)
    return float(Z @ f), int(Z[np.argmax(f)])


#: M22 regions by distance from theta1 Ori C [arcsec] (card [environment].locations: IF 113, DF 130)
REGIONS_2022 = {"HII": (0.0, 113.0), "PDR 1": (113.0, 128.0), "PDR 2": (128.0, 130.0),
                "DF": (130.0, 131.0), "MC": (131.0, 1e9)}


def region_of(distance_arcsec):
    for name, (lo, hi) in REGIONS_2022.items():
        if lo <= distance_arcsec < hi:
            return name
    return "MC"
