"""Heat capacity of amorphous carbon a-C(:H) grains: a harmonic sum over their atoms.

THEMIS (Jones et al. 2013, App. B) builds the heat capacity of a-C(:H) as the
abundance-weighted sum of its atoms' contributions; DustEM's C_CM20.DAT, which
follows it, exceeds the Dulong-Petit limit by 35-50% above ~1000 K (its sp3 term
is reproduced by the liquid-polyethylene Cp extrapolated far beyond its data).
This module keeps the THEMIS recipe with harmonic terms only, so that every atom
carries at most 3 k:

- composition from the band gap (Jones 2012, paper I): X_H = E_g / 4.3 eV, and
  X_sp3 = X_H / 2 for the defected-graphite network (0 <= X_H <= 0.35, eqs. 64-66)
  or X_sp3 / X_sp2 = (8 X_H - 3)/(8 - 13 X_H) for the random covalent network
  (0.38 <= X_H <= 0.62, eq. 7, Angus & Jansen 1988); in both, H is bonded to sp3 C;
- sp2 C: graphite, two 2-D Debye spectra with Theta = 863 and 2504 K
  (Draine & Li 2001, eq. 9, with f_n normalised by n, not 1/n as printed);
- sp3 C: polyethylene's skeletal modes, 2 Tarasov modes (Theta_1 = 519 K,
  Theta_3 = 158 K) and the C-C stretch (Wunderlich 1995; 1990, Fig. 9: ATHAS);
- H on sp3 C: half of polyethylene's six CH2 group vibrations per CH2 (ATHAS);
- ``aromatic_h=True`` puts the H on aromatic C instead: Draine & Li's C-H
  modes at 886, 1161 and 3030 cm^-1;
- a grain of N atoms has 3N - 6 modes.

Anharmonic and C_p - C_v terms are left out (<= 2% for graphite to 3500 K,
Butland & Maddison 1973), as is the excess of amorphous solids below ~15 K (two-level
systems; C_CM20.DAT is 2.5-3.3 times this model there). H loss above ~600 K is
chemistry, not heat capacity. ``themis_optics.heat_capacity`` keeps DustEM's table
for benchmarks.
"""
from __future__ import annotations

import numpy as np

K_B = 1.380649e-16
AMU = 1.66053906660e-24
CM_TO_K = 1.4387769                      # h c / k [K cm]

THETA_GRAPHITE = (863.0, 2504.0)         # K, out-of-plane (1 mode) and in-plane (2 modes), DL01
THETA_TARASOV = (519.0, 158.0)           # K, polyethylene skeletal Theta_1, Theta_3 (ATHAS)
CH_AROMATIC_CM = (886.0, 1161.0, 3030.0)  # cm^-1, DL01 C-H out-of-plane, in-plane, stretch

# polyethylene group vibrations per CH2 (Wunderlich 1990, Fig. 9): (Theta_low, Theta_high [K], N);
# Theta_low = Theta_high is an Einstein mode, otherwise a box (uniform) spectrum
PE_H_MODES = ((4097.7, 4097.7, 1.00), (4148.1, 4148.1, 1.00),            # CH2 sym., asym. stretch
              (2074.7, 2074.7, 1.00),                                     # CH2 bending
              (1698.3, 1976.6, 0.65), (1976.6, 1976.6, 0.35),            # CH2 wagging
              (1689.6, 1874.3, 0.48), (1874.3, 1874.3, 0.52),            # CH2 twisting
              (1494.1, 1494.1, 0.04), (1038.0, 1494.1, 0.59), (1079.1, 1079.1, 0.37))  # CH2 rocking
PE_CC_STRETCH = ((1377.6, 1637.5, 0.34), (1377.6, 1525.4, 0.35), (1525.4, 1525.4, 0.31))

_Y, _WY = np.polynomial.legendre.leggauss(96)
_Y, _WY = 0.5 * (_Y + 1.0), 0.5 * _WY          # nodes and weights on [0, 1]


def _einstein_z(z):
    """z^2 e^z / (e^z - 1)^2, the heat capacity of one mode in units of k (z = Theta/T)."""
    z = np.asarray(z, float)
    zs = np.minimum(z, 700.0)
    em = np.exp(-zs)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = zs * zs * em / (-np.expm1(-zs)) ** 2
    return np.where(z < 1e-8, 1.0, np.where(z > 700.0, 0.0, v))


def einstein(theta, T):
    return _einstein_z(theta / np.asarray(T, float))


def debye(n, theta, T):
    """n-dimensional Debye heat capacity per mode [k]: n (T/Theta)^n int_0^(Theta/T) u^(n-1) E(u) du,
    integrated in u up to min(Theta/T, 60) (beyond, E(u) < 1e-23)."""
    x = theta / np.atleast_1d(np.asarray(T, float))
    top = np.minimum(x, 60.0)
    u = _Y[None, :] * top[:, None]
    return n / x ** n * top * np.sum(_WY * u ** (n - 1) * _einstein_z(u), axis=1)


def box(theta_lo, theta_hi, T):
    """A uniform spectrum of modes between theta_lo and theta_hi, per mode [k]: the
    Einstein heat capacity averaged over it."""
    T = np.atleast_1d(np.asarray(T, float))
    if theta_hi == theta_lo:
        return einstein(theta_hi, T)
    th = theta_lo + (theta_hi - theta_lo) * _Y
    return np.sum(_WY * _einstein_z(th[None, :] / T[:, None]), axis=1)


def tarasov(T):
    """Polyethylene skeletal modes per mode [k] (ATHAS; Wunderlich 1995, eq. 12)."""
    t1, t3 = THETA_TARASOV
    return debye(1, t1, T) - (t3 / t1) * (debye(1, t3, T) - debye(3, t3, T))


def _modes(spec, T):
    return sum(n * box(lo, hi, T) for lo, hi, n in spec)


def per_sp2_carbon(T):
    t_op, t_ip = THETA_GRAPHITE
    return debye(2, t_op, T) + 2.0 * debye(2, t_ip, T)


def per_sp3_carbon(T):
    return 2.0 * tarasov(T) + _modes(PE_CC_STRETCH, T)


def per_aliphatic_h(T):
    return 0.5 * _modes(PE_H_MODES, T)


def per_aromatic_h(T):
    T = np.atleast_1d(np.asarray(T, float))
    return sum(einstein(c * CM_TO_K, T) for c in CH_AROMATIC_CM)


def composition(e_gap):
    """(X_H, X_sp2, X_sp3) atomic fractions of a-C(:H) with band gap ``e_gap`` [eV]."""
    x_h = float(e_gap) / 4.3
    if 0.0 <= x_h <= 0.35:
        x_sp3 = 0.5 * x_h                                  # defected graphite
    elif 0.38 <= x_h <= 0.62:
        x_sp3 = (1.0 - x_h) * (8 * x_h - 3) / ((8 * x_h - 3) + (8 - 13 * x_h))   # random network
    else:
        raise ValueError(f"X_H = {x_h:.3f} (E_g = {e_gap} eV) is outside the networks' "
                         "ranges (0-0.35 defected graphite, 0.38-0.62 random network)")
    return x_h, 1.0 - x_h - x_sp3, x_sp3


def mean_atomic_mass(e_gap):
    x_h = float(e_gap) / 4.3
    return (1.0 - x_h) * 12.011 + x_h * 1.008


def per_atom(T, e_gap, aromatic_h=False):
    """Heat capacity per atom [k] of bulk a-C(:H) with band gap ``e_gap``."""
    x_h, x_sp2, x_sp3 = composition(e_gap)
    h = per_aromatic_h(T) if aromatic_h else per_aliphatic_h(T)
    return x_sp2 * per_sp2_carbon(T) + x_sp3 * per_sp3_carbon(T) + x_h * h


def grain_heat_capacity(n_atoms, T, e_gap, aromatic_h=False):
    """Heat capacity [erg/K] of a grain of ``n_atoms`` atoms, with 3N - 6 modes."""
    n = np.asarray(n_atoms, float)
    return K_B * per_atom(T, e_gap, aromatic_h)[:, None] * np.maximum(n - 2.0, 0.0)[None, :]


def heat_capacity_cm20(a_um, T, mantle_um=0.020, rho=1.6, e_gap_mantle=0.1, e_gap_core=2.5,
                       rho_core=None):
    """Heat capacity per grain [erg/K] of THEMIS CM20 grains, shape (nT, na): a-C
    (E_g 0.1) below ``mantle_um``, an a-C:H core (E_g 2.5) in an a-C mantle above,
    each layer at its own atom density (the core at Jones's rho(E_g) unless given)."""
    from pycalima.models.dust_radiation.jones2012_optics import density

    a = np.atleast_1d(np.asarray(a_um, float)) * 1e-4
    T = np.atleast_1d(np.asarray(T, float))
    rho_core = float(density(e_gap_core)) if rho_core is None else rho_core
    r_core = np.clip(a - mantle_um * 1e-4, 0.0, None)
    v_core = 4.0 / 3.0 * np.pi * r_core ** 3
    v_mantle = 4.0 / 3.0 * np.pi * a ** 3 - v_core
    n_m = v_mantle * rho / (mean_atomic_mass(e_gap_mantle) * AMU)
    n_c = v_core * rho_core / (mean_atomic_mass(e_gap_core) * AMU)
    n = n_m + n_c
    per = (per_atom(T, e_gap_mantle)[:, None] * n_m + per_atom(T, e_gap_core)[:, None] * n_c) / n
    return K_B * per * np.maximum(n - 2.0, 0.0)
