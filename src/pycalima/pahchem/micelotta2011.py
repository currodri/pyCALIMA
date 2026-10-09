"""Micelotta, Jones & Tielens (2011): PAH destruction by cosmic rays, re-implemented from their stated inputs.

The equations and numbers are those of ``method_cards/micelotta2011.toml``.

- **Ions** (H, He, CNO, Fe-Co-Ni) cross a thick disc. They deposit the
  Bethe-Bloch stopping power times the chord, and a fraction f of it becomes
  vibrational energy. An Arrhenius competition between C2 loss and IR emission
  gives the dissociation probability.
- **Electrons** knock out a carbon atom when they transfer more than T0 to the
  nucleus (McKinley-Feshbach).

Both rates are integrated over their interstellar spectra (eqs. 22 and 25).
Nothing here is shared with ``impact``; it is the reference that the modern
calculation is compared with.
"""

from __future__ import annotations

import numpy as np

from pycalima.pahchem.cards import load_card

CARD = load_card("micelotta2011")
YR_S = 3.15576e7
K_B_EV = 8.617333262e-5
ME_EV = 0.51099895e6
AMU_GEV = 0.931494
KAPPA = CARD["stopping"]["kappa"]["value"]
Z2 = CARD["stopping"]["z2"]["value"]
I_EV = CARD["stopping"]["mean_ionisation"]["value"]
D_A = CARD["species"]["thickness"]["value"]
K0 = CARD["dissociation"]["k0"]["value"]
K_IR = CARD["dissociation"]["k_ir"]["value"]
DEPS = CARD["dissociation"]["photon_energy"]["value"]
#: Table 1: name -> (Z1, M1 [amu], C [(cm^2 s sr GeV^-1.7)^-1])
IONS = {"H": (1, 1.0, 1.45), "He": (2, 4.0, 0.90), "CNO": (7, 14.0, 0.36), "Fe-Co-Ni": (26, 58.0, 0.24)}
E0_STAR_H_GEV = CARD["spectra"]["e0_star_h"]["value"]


def radius_a(nc) -> float:
    return 0.9 * np.sqrt(nc)


def n_max(nc) -> int:
    """IR photons in eq. 13: N_C / 5 (Sect. 6.1)."""
    return int(round(nc / 5))


def ion_spectrum(name: str, e_gev):
    """Eq. 22 [particles cm^-2 s^-1 sr^-1 GeV^-1], E the kinetic energy per ion [GeV]."""
    _, m1, c = IONS[name]
    E = np.asarray(e_gev, dtype=float)
    return c * E**0.3 / (E + E0_STAR_H_GEV * m1) ** 3


def electron_spectrum(e_gev):
    """Eq. 25 (Cummings 1973) [particles cm^-2 s^-1 sr^-1 GeV^-1]."""
    E = np.asarray(e_gev, dtype=float)
    return np.where(E <= 2.0, 0.0254e5 * (E * 1e3) ** -1.8, 5.19e5 * (E * 1e3) ** -2.5)


def beta2(e_gev, m1):
    """Eq. 6."""
    return 1.0 - 1.0 / (1.0 + np.asarray(e_gev, dtype=float) / (AMU_GEV * m1)) ** 2


def stopping_ev_per_a(name: str, e_gev):
    """Eqs. 8-10: S [eV/A] in amorphous carbon."""
    z1, m1, _ = IONS[name]
    b2 = beta2(e_gev, m1)
    f = np.log(2.0 * ME_EV * b2 / (1.0 - b2)) - b2
    s_p = KAPPA * Z2 / b2 * (f - np.log(I_EV))
    if z1 == 1:
        return s_p
    zstar = lambda z: z * (1.0 - np.exp(-125.0 * np.sqrt(b2) * z ** (-2.0 / 3.0)))  # noqa: E731
    return s_p * (zstar(z1) / zstar(1)) ** 2


def temperature_k(e_prime, nc, e0):
    """Eq. 14."""
    e = np.maximum(np.asarray(e_prime, dtype=float), 1e-30)
    return 2000.0 * (e / nc) ** 0.4 * (1.0 - 0.2 * e0 / e)


def dissociation_probability(t_ev, nc, e0, f):
    """Eqs. 12-13 for transferred energy T [eV]."""
    nm = n_max(nc)
    e1 = f * np.asarray(t_ev, dtype=float)
    e2 = e1 - nm * DEPS
    t1, t2 = temperature_k(e1, nc, e0), temperature_k(e2, nc, e0)
    ok = (t1 > 0) & (t2 > 0)
    t_av = np.sqrt(np.where(ok, t1 * t2, 1.0))
    with np.errstate(over="ignore", under="ignore"):
        k = K0 * np.exp(-e0 / (K_B_EV * t_av))
    return np.where(ok, k / (K_IR / (nm + 1) + k), 0.0)


def sigma_diss(name: str, e_gev, nc, e0, f, n_theta=400):
    """Eq. 27: Sigma(E) = int_0^(pi/2) sigma_g(theta) P(E, theta) sin theta dtheta [cm^2]."""
    R, d = radius_a(nc), D_A
    th = np.linspace(0.0, 0.5 * np.pi, n_theta)
    tan_alpha = 2.0 * R / d
    with np.errstate(divide="ignore"):
        l = np.where(np.abs(np.tan(th)) < tan_alpha, d / np.abs(np.cos(th)), 2.0 * R / np.abs(np.sin(th)))
    sg = (np.pi * R * R * np.cos(th) + 2.0 * R * d * np.sin(th)) * 1e-16
    S = np.atleast_1d(stopping_ev_per_a(name, e_gev))
    P = dissociation_probability(S[:, None] * l[None, :], nc, e0, f)
    return np.trapezoid(sg * P * np.sin(th), th, axis=1)


def ion_ejection_rate(name: str, nc, e0, f, e_max_per_nucleon=True, n=400):
    """Eq. 30: C atoms ejected per second, 2 x 4 pi int I Sigma dE."""
    _, m1, _ = IONS[name]
    e_lo = 5e-3 * m1
    e_hi = 10.0 * m1 if e_max_per_nucleon else 10.0
    E = np.geomspace(e_lo, e_hi, n)
    return 2.0 * 4.0 * np.pi * np.trapezoid(ion_spectrum(name, E) * sigma_diss(name, E, nc, e0, f), E)


def mckinley_feshbach(e_gev, t0_ev):
    """Eq. 18: C knock-out cross section per atom [cm^2] for an electron of kinetic energy E."""
    E = np.asarray(e_gev, dtype=float) * 1e9
    a0, e_r, alpha = 0.529177e-8, 13.6057, 6.0 / 137.036
    mc2 = 12.0 * AMU_GEV * 1e9
    t_max = 2.0 * E * (E + 2.0 * ME_EV) / mc2
    b2 = 1.0 - (ME_EV / (E + ME_EV)) ** 2
    b = np.sqrt(b2)
    x = t0_ev / t_max
    s = (4.0 * Z2**2 * e_r**2 / ME_EV**2) * np.pi * a0**2 * (1.0 - b2) / b2**2 / x * (
        1.0 + 2.0 * np.pi * alpha * b * np.sqrt(x) - x * (1.0 + 2.0 * np.pi * alpha * b + (b2 + np.pi * alpha * b) * np.log(1.0 / x)))
    return np.where(t_max > t0_ev, np.clip(s, 0.0, None), 0.0)


ORIENTATION = CARD["electrons"]["orientation_factor"]["value"]


def electron_lifetime_yr(t0_ev, e_max_gev=10.0, orientation=ORIENTATION, n=600) -> float:
    """Eq. 37 with eq. 31: ln(3/2) N_C / R_e, independent of N_C.

    ``orientation`` = 0.5 as printed in eq. 31; 1 reproduces Table 2
    (card: electrons.orientation_discrepancy).
    """
    E = np.geomspace(5e-3, e_max_gev, n)
    rate_per_c = orientation * 4.0 * np.pi * np.trapezoid(electron_spectrum(E) * mckinley_feshbach(E, t0_ev), E)
    return float(np.log(1.5) / rate_per_c / YR_S)


def ion_lifetime_yr(nc, e0, f, ions=tuple(IONS), **kwargs) -> float:
    """Eq. 34: N_C / (3 R_i), R_i summed over ``ions``."""
    r = sum(ion_ejection_rate(name, nc, e0, f, **kwargs) for name in ions)
    return float(nc / (3.0 * r) / YR_S) if r > 0 else np.inf


def total_lifetime_yr(nc, e0=4.6, t0=7.5, f=1.0, **kwargs) -> float:
    """Ions and electrons together (Fig. 8): 1/tau = 1/tau_ions + 1/tau_e."""
    return 1.0 / (1.0 / ion_lifetime_yr(nc, e0, f, **kwargs) + 1.0 / electron_lifetime_yr(t0, orientation=1.0))


def crossover_nc(tau_yr, e0=4.6, t0=7.5, f=1.0, nc_grid=None) -> float:
    """N_C at which the total CR lifetime equals ``tau_yr`` (log interpolation)."""
    nc = np.geomspace(50, 1000, 25) if nc_grid is None else np.asarray(nc_grid, dtype=float)
    tau = np.array([total_lifetime_yr(n, e0, t0, f) for n in nc])
    return float(np.exp(np.interp(np.log(tau_yr), np.log(tau), np.log(nc))))
