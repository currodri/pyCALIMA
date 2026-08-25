"""
shiva_dissociation.py — Arrhenius C₂H₂ loss rates in the SHIVA model.

Implements the microcanonical Gibbs dissociation rate (Tielens 2005,
§6.3.4) for C₂H₂ ejection from a PAH at microcanonical temperature T_m:

    k_dis(T_m) = (k_B T_m / h) × exp(ΔS†/R) × exp(−E₀ / k_B T_m)

where the activation parameters from Murga+2020 (their Table 1 / text)
for C₂H₂ loss from a dehydrogenated PAH are:
    E₀  = 4.6 eV
    ΔS† = 10.0 cal mol⁻¹ K⁻¹   (Tielens 2005, their Table 6.2)

The average rate per molecule is obtained by integrating k_dis(T_m(T))
over the GD89 temperature probability distribution P(T):

    ⟨k_dis⟩ = Σ_Z f(Z) × ∫ P_Z(T) × k_dis(T_m(T, Nc)) dT

The microcanonical temperature T_m is evaluated from the DL01 relation
U(T_m) = U(T), i.e. T_m ≡ T in this formulation (since P(T) already uses
the DL01 energy grid).

References
----------
Tielens, A.G.G.M. 2005, "The Physics and Chemistry of the Interstellar
    Medium", Cambridge University Press, §6.3.4
Murga, M.S. et al. 2020, A&A, 644, A89 (Table 1)
Guhathakurta, P. & Draine, B.T. 1989, ApJ, 345, 230 (GD89)
Draine, B.T. & Li, A. 2001, ApJ, 551, 807 (DL01)
"""

import numpy as np

from .gd89_heating import compute_PT_shiva
from .shiva_charge import steady_state_charges

# ── Physical constants ────────────────────────────────────────────────────
k_B_erg = 1.380649e-16    # Boltzmann constant [erg K⁻¹]
k_B_eV  = 8.61733326e-5   # Boltzmann constant [eV K⁻¹]
h_erg   = 6.62607015e-27  # Planck constant [erg s]
eV_erg  = 1.602176634e-12 # [erg eV⁻¹]
R_cal   = 1.98720425864   # gas constant [cal mol⁻¹ K⁻¹]

# ── Default Arrhenius parameters for C₂H₂ loss ───────────────────────────
# Dehydrogenated PAH cation/neutral (Murga+2020 Table 1 / Tielens 2005)
C2H2_E0_EV   = 4.6    # activation energy [eV]
C2H2_DS_CAL  = 10.0   # activation entropy ΔS† [cal mol⁻¹ K⁻¹]


def k_C2H2_arrhenius(T_m, E0_eV=C2H2_E0_EV, dS_cal_molK=C2H2_DS_CAL):
    """
    Arrhenius (Gibbs microcanonical) C₂H₂ loss rate [s⁻¹] at temperature T_m.

    k_dis(T_m) = (k_B T_m / h) × exp(ΔS†/R) × exp(−E₀ / k_B T_m)

    Parameters
    ----------
    T_m : float
        Microcanonical temperature [K].  Values below ~500 K give negligible
        rates; the function clips the exponent to avoid underflow.
    E0_eV : float
        Activation energy [eV].  Default: 4.6 eV (Murga+2020 C₂H₂ channel).
    dS_cal_molK : float
        Activation entropy [cal mol⁻¹ K⁻¹].  Default: 10.0 (Murga+2020).

    Returns
    -------
    k : float  [s⁻¹]  (≥ 0)
    """
    if T_m < 10.0:
        return 0.0

    prefactor = k_B_erg * T_m / h_erg              # ~2.08e10 × T_m  [s⁻¹]
    entropy_factor = np.exp(dS_cal_molK / R_cal)    # exp(ΔS†/R)
    E0_erg = E0_eV * eV_erg
    exponent = -E0_erg / (k_B_erg * T_m)
    if exponent < -700.0:
        return 0.0
    return prefactor * entropy_factor * np.exp(exponent)


def C2H2_loss_rate_per_molecule(Nc, Z, u_E_fn, G0=1.0,
                                 N_T=150, E0_eV=C2H2_E0_EV,
                                 dS_cal_molK=C2H2_DS_CAL):
    """
    C₂H₂ loss rate ⟨k_C2H2⟩ [s⁻¹ per molecule] for a PAH with charge Z.

    Integrates k_dis(T_m(T)) over the GD89 temperature distribution P_Z(T):

        ⟨k⟩ = ∫ P(T) × k_C2H2(T) dT

    In the DL01 / GD89 framework, the microcanonical temperature T_m equals
    the bin temperature T (both parameterise the same internal energy via
    U_dl01), so the integral is simply over the P(T) distribution.

    Parameters
    ----------
    Nc : int or float
        Number of carbon atoms.
    Z : int
        PAH charge state.
    u_E_fn : callable
        Unit-G0 spectral energy density [erg cm⁻³ eV⁻¹].
    G0 : float
        Habing field scaling.
    N_T : int
        Number of temperature bins for GD89.
    E0_eV : float
        C₂H₂ activation energy [eV].
    dS_cal_molK : float
        C₂H₂ activation entropy [cal mol⁻¹ K⁻¹].

    Returns
    -------
    rate : float  [s⁻¹ per molecule]
    """
    T_grid, P_T, dT = compute_PT_shiva(Nc, Z, u_E_fn, G0=G0, N_T=N_T)
    # Use the same bin widths dT that were used in the GD89 normalisation
    # (i.e. Σ P_T[i] × dT[i] = 1 by construction).
    k_arr = np.array([k_C2H2_arrhenius(T, E0_eV, dS_cal_molK) for T in T_grid])
    rate = np.sum(P_T * k_arr * dT)
    return rate


def C2H2_loss_rate_charge_averaged(Nc, T_gas, ne, u_E_fn, G0=1.0,
                                    N_T=150, E0_eV=C2H2_E0_EV,
                                    dS_cal_molK=C2H2_DS_CAL):
    """
    Charge-averaged C₂H₂ loss rate [s⁻¹ per molecule].

    ⟨k_C2H2⟩ = Σ_Z f(Z) × ⟨k_C2H2⟩_Z

    Parameters
    ----------
    Nc : int or float
        Number of carbon atoms.
    T_gas : float
        Gas temperature [K] (used for charge-balance recombination rates).
    ne : float
        Electron number density [cm⁻³].
    u_E_fn : callable
        Unit-G0 spectral energy density [erg cm⁻³ eV⁻¹].
    G0 : float
        Habing field scaling.
    N_T : int
        Temperature bins for GD89.
    E0_eV, dS_cal_molK : float
        Arrhenius parameters.

    Returns
    -------
    rate_avg : float  [s⁻¹ per molecule]
    """
    fracs = steady_state_charges(Nc, T_gas, ne, u_E_fn, G0=G0)
    rate_avg = 0.0
    for Z, f_Z in fracs.items():
        if f_Z < 1e-10:
            continue
        rate_Z = C2H2_loss_rate_per_molecule(
            Nc, Z, u_E_fn, G0=G0, N_T=N_T,
            E0_eV=E0_eV, dS_cal_molK=dS_cal_molK
        )
        rate_avg += f_Z * rate_Z
    return rate_avg
