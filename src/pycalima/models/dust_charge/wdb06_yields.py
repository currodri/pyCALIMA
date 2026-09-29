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

Deliberate departures from WDB06:

- Auger heating uses E_max = E_A - (Z+1) e^2/a, the same upper limit as their
  y2 for Auger electrons; their sec. 8 writes I_{i,s} - (Z+1) e^2/a, which
  gives the electron more energy than the transition releases (up to +7% of
  the total heating for charged silicate grains in a quasar spectrum).
- J_sec,gas averages the DS79 yield over the thermal impact-energy
  distribution, with no secondaries below the work function, instead of
  evaluating it at the mean energy <E0> (DS79 eq. 16). The two agree to ~15%
  for T >~ 1e5 K; at 1e4 K the mean-energy form extrapolates DS79's fit to
  <E0> ~ 2 eV < W and overstates the yield 10-100 times.

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


def _material(material):
    return MATERIALS["graphite" if material == "graphite" else "silicate"]


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


def ionisation_potential(I, Z, a):
    """IP(Z, a) = I + (Z + 1/2) e^2/a + (Z + 2) (e^2/a)(0.3 A / a), eq. 2."""
    e2a = _e2_over_a(a)
    return I + (Z + 0.5) * e2a + (Z + 2.0) * e2a * (0.3 * ANGSTROM / a)


def threshold_energy(I, Z, a):
    """hnu_pet (eq. 1) with W -> I for inner shells."""
    IP = ionisation_potential(I, Z, a)
    return np.where(Z < -1, IP + min_energy_vH04(Z, a), IP)


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


# ---------------------------------------------------------------------------
# Yield channels
# ---------------------------------------------------------------------------

def band_bulk_yield(theta, material, a=None, la_wd01_of_E=None):
    """Bulk band-structure yield y0(band, Theta), WDB06 secs. 4.1.3-4.1.4.

    WD01 below hnu = Theta + W = 20 eV, eq. 9 above 50 eV, linear in hnu
    between. For carbonaceous grains with a <= 13 A, eq. 15 blends back to
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

    w = np.clip((hnu - 20.0) / 30.0, 0.0, 1.0)
    y0 = (1.0 - w) * y_wd01 + w * y_atom

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
        """Eq. 18: Y_sec = Y (y2_sec / y2) (E_exc / eps) l_e^-1 / (a^-1 + 10 A^-1)."""
        y2s, mean_s = _secondary_y2_and_mean(E_low, E_high)
        ratio = np.where(y2 > 0, y2s / np.maximum(y2, TINY), 0.0)
        E_exc = np.maximum(E_exc, 0.0)
        Ys = Y * ratio * (E_exc / m["eps"]) * size_factor / electron_escape_length(E_exc, material)
        return np.where(Y > 0, Ys, 0.0), mean_s

    # --- band structure (eqs. 4-8, 14) ---
    hnu_pet = threshold_energy(W, Z, a)
    theta = theta_parameter(E, hnu_pet, Z, a)
    y0 = band_bulk_yield(theta, material, a)
    y1 = Watson73_y1_vec(a, la, electron_escape_length(E - W, material))
    mu_band = absorption_coefficient(E1, material, m["band"])[:, None]
    P_band = np.where(E > 20.0, np.clip(mu_band / np.maximum(mu_tot, TINY), 0.0, 1.0), 1.0)
    E_low, E_high = energy_limits(E - hnu_pet, Z, a)
    y2, mean = _parabolic_y2_and_mean(E_low, E_high)
    Y = np.where(E >= hnu_pet, y2 * np.minimum(y0 * y1, P_band), 0.0)
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
        hnu_pet = threshold_energy(I, Z, a)
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
    material : 'graphite' or 'silicate'
    la_wd01 : (N_E,) dielectric attenuation length [cm], used below 20 eV
        (as in WD01); if None, the atomic l_a is used everywhere.

    Returns
    -------
    dict name -> (Y, E_mean), each (N_E, N_Z). Names: 'band', 'inner:<El><shell>',
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
    from pycalima.models.dust_charge.shared_physics import (
        electron_affinity_graphite_vec,
        electron_affinity_silicate_vec,
    )
    EA = electron_affinity_graphite_vec if material == "graphite" else electron_affinity_silicate_vec
    Z = np.asarray(Z, dtype=float)
    return EA(Z + 1.0, a) + min_energy_vH04(Z, a)


# ---------------------------------------------------------------------------
# Secondary emission by impinging gas electrons (Draine & Salpeter 1979)
# ---------------------------------------------------------------------------

# DS79 Table 5 rows matching the eps adopted in WDB06 sec. 6.
_DS79_MATERIAL = {"graphite": "graphite", "silicate": "lunar_dust"}

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
    energy > E_s [eV]: delta_inf(E0, E_s) (DS79 eqs. 14-15) times the
    small-particle factor delta_a / delta_inf of eq. A11. Zero below the work
    function, where the impinging electron cannot lift one out (DS79's fit has
    no threshold, being made for E0 >~ 10^2 eV)."""
    m = _material(material)
    p = atomic.secondary_emission_parameters(_DS79_MATERIAL["graphite" if material == "graphite" else "silicate"])
    E0 = np.asarray(E0, dtype=float)
    x = E0 / p["E_m_eV"]
    delta_inf = p["delta_m"] * 4.0 * x / (1.0 + x) ** 2 / np.sqrt(1.0 + 0.125 * np.asarray(E_s) ** 2)
    lam = p["lam_A"] * ANGSTROM
    xr = 4.0 * a / (3.0 * _electron_range_cm(np.maximum(E0, TINY), p["rho_gcm3"]))
    f1 = (1.6 + 1.4 * xr ** 2 + 0.54 * xr ** 4) / (1.0 + 0.54 * xr ** 4)  # eq. A7
    r = lam / a
    f2 = (1.0 + 2.0 * r ** 2 + r ** 4) / (1.0 + r ** 4)  # eq. A10
    delta = delta_inf * 2.0 * (-np.expm1(-4.0 * a / (3.0 * lam))) * f1 * f2  # eq. A11
    return np.where(E0 > m["W"], delta, 0.0)


def _thermal_nodes(Z, a, T, material):
    """Impact energies E [eV] and normalised weights for the flux of thermal
    electrons reaching a grain of charge Z.

    With Coulomb focusing the impact-energy distribution is proportional to
    E exp(-E/kT) for E > max(0, eU) (repelled: E = E_inf - |eU|; attracted:
    E = E_inf + eU), whose mean is DS79 eq. 16. Only E > W can produce
    secondaries, so the nodes start at max(eU, W).
    Returns E (N_Z, N_q) and w (N_Z, N_q) with sum(w) = fraction above W.
    """
    from pycalima.models.dust_charge.shared_physics import KB_CGS

    kT = KB_CGS * T / EV2ERG
    Z = np.atleast_1d(np.asarray(Z, dtype=float))
    L = np.maximum(Z * _e2_over_a(a), 0.0)[:, None]          # flux lower bound
    Lw = np.maximum(L, _material(material)["W"])            # secondary threshold
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
    (the Draine 2003 range), tabulated once per (material, a)."""
    import miepython

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
