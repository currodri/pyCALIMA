"""
pah_hydrogen_chemistry.py — Hydrogen addition rate coefficients for PAH molecules.

Implements the Andrews (2016) prescription:
  - Cation:  k = 1.4e-10 cm³/s (all hydrogenation steps)
  - Anion:   k = 7.8e-10 cm³/s (associative detachment)
  - Neutral: k(T) = k_coll(T) × P_reac(T)
      k_coll = π a_pah² √(8 kB T / π mH)
      P_reac per Demarais et al. (2014):
        dehydrogenated     → P = 0.07 (no barrier)
        1st extra H step   → P = exp(−0.06 eV / kB T)
        2nd extra H step   → P = 1
        3rd extra H step   → P = exp(−0.03 eV / kB T)
        4th–8th extra step → P = 1
  - Superhydrogenated H2 abstraction (Eley-Rideal):
      k_ER = 8.7e-13 * sqrt(Tgas/100) * n(H)  [s^-1]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
KB_CGS = 1.380649e-16    # erg K⁻¹
EV2ERG = 1.60218e-12     # erg eV⁻¹
MH_CGS = 1.6735575e-24   # g (hydrogen atom mass)

# ---------------------------------------------------------------------------
# Andrews (2016) ion rate constants
# ---------------------------------------------------------------------------
K_CATION_CM3S = 1.4e-10  # cm³/s — cation H-addition (all steps)
K_ANION_CM3S  = 7.8e-10  # cm³/s — anion associative detachment

# Reaction efficiency for dehydrogenated neutrals (Demarais et al. 2014)
EFF_DEHYDROGENATED = 0.07

# Barriers (eV) indexed by n_extra_H already on molecule:
#   index 0 → adding 1st extra H  (barrier = 0.06 eV)
#   index 1 → adding 2nd extra H  (no barrier, None)
#   index 2 → adding 3rd extra H  (barrier = 0.03 eV)
#   index ≥ 3 → no barrier (up to 8 extra H)
_BARRIER_EV = [0.06, None, 0.03]

# Eley-Rideal H2-abstraction coefficient for superhydrogenated PAHs
K_ER_COEFF_CM3S = 8.7e-13  # cm³/s, at reference T = 100 K


def collisional_rate(a_pah_cm, T):
    """
    Classical Langevin collision rate k_coll = π a² √(8 kB T / π mH) [cm³/s].

    Parameters
    ----------
    a_pah_cm : float   PAH effective radius [cm] (use afromNc from pah_charge_utils).
    T : float|array    Gas temperature [K].
    """
    return np.pi * a_pah_cm**2 * np.sqrt(
        8.0 * KB_CGS * np.asarray(T, dtype=float) / (np.pi * MH_CGS)
    )


def reaction_efficiency_neutral(n_extra_H, T):
    """
    Reaction efficiency P_reac(T) for H addition to a neutral PAH.

    Parameters
    ----------
    n_extra_H : int
        H atoms already on the molecule beyond the normal hydrogenation count.
        Negative values denote a dehydrogenated state.
    T : float|array [K]

    Returns
    -------
    P : same shape as T, dimensionless.
    """
    T = np.asarray(T, dtype=float)
    if n_extra_H < 0:
        # Dehydrogenated: no barrier, efficiency from Demarais+2014
        return np.full_like(T, EFF_DEHYDROGENATED)
    if n_extra_H < len(_BARRIER_EV):
        e = _BARRIER_EV[n_extra_H]
        if e is None:
            return np.ones_like(T)
        return np.exp(-e * EV2ERG / (KB_CGS * T))
    # 3 or more extra H already present → no barrier
    return np.ones_like(T)


def h_addition_rate(charge, a_pah_cm, n_extra_H, T):
    """
    H-addition rate coefficient k(T) [cm³/s] for a PAH molecule.

    Parameters
    ----------
    charge : int        Charge state (>0 cation, <0 anion, 0 neutral).
    a_pah_cm : float    PAH effective radius [cm].
    n_extra_H : int     H atoms beyond normal hydrogenation count
                        (<0 = dehydrogenated, 0 = normally hydrogenated,
                        >0 = superhydrogenated).
    T : float|array     Gas temperature [K].

    Returns
    -------
    k : same shape as T [cm³/s].
    """
    T = np.asarray(T, dtype=float)
    if charge > 0:
        return np.full_like(T, K_CATION_CM3S)
    if charge < 0:
        return np.full_like(T, K_ANION_CM3S)
    return collisional_rate(a_pah_cm, T) * reaction_efficiency_neutral(n_extra_H, T)


def h2_abstraction_rate_coefficient(T):
    """
    Eley-Rideal H2-abstraction rate coefficient k_ER(T) = 8.7e-13 sqrt(T/100) [cm³/s].

    Applies to superhydrogenated PAH states (n_extra_H > 0).
    """
    return K_ER_COEFF_CM3S * np.sqrt(np.asarray(T, dtype=float) / 100.0)


def h2_abstraction_rate(T, n_H):
    """
    Eley-Rideal H2-abstraction rate [s^-1] for a superhydrogenated PAH:

        k_ER = 8.7e-13 * sqrt(Tgas/100) * n(H)

    Parameters
    ----------
    T : float|array     Gas temperature [K].
    n_H : float|array   Atomic hydrogen number density [cm^-3].

    Returns
    -------
    rate : broadcast of T and n_H [s^-1].
    """
    return h2_abstraction_rate_coefficient(T) * np.asarray(n_H, dtype=float)


# ---------------------------------------------------------------------------
# Quick diagnostic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from pycalima.models.PAH_photophysics.pah_charge_utils import afromNc

    Nc = 54
    a_cm = afromNc(Nc)
    T_grid = np.array([10, 50, 100, 300, 500, 1000])

    print(f"C{Nc}H18  a = {a_cm:.3e} cm\n")
    print(f"{'Charge':>8}  {'n_extra':>7}  " + "  ".join(f"T={T:4g}K" for T in T_grid))
    print("-" * 80)

    cases = [
        (+1, 0, "cation"),
        (-1, 0, "anion"),
        (0, -3, "neutral (dehydrogenated)"),
        (0,  0, "neutral (normal, +1st extra)"),
        (0,  1, "neutral (+2nd extra)"),
        (0,  2, "neutral (+3rd extra)"),
        (0,  4, "neutral (+5th extra)"),
    ]
    for charge, n_extra, label in cases:
        k = h_addition_rate(charge, a_cm, n_extra, T_grid)
        vals = "  ".join(f"{v:.2e}" for v in k)
        print(f"  {label:<34} {vals}")

    print("\nEley-Rideal H2-abstraction (superhydrogenated), n(H) = 1 cm^-3:")
    k_er = h2_abstraction_rate(T_grid, n_H=1.0)
    vals = "  ".join(f"{v:.2e}" for v in k_er)
    print(f"  {'k_ER [s^-1]':<34} {vals}")
