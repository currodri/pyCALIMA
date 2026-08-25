"""
shiva_charge.py — SHIVA charge-balance solver for PAHs.

Computes steady-state charge fractions f(Z) for Z ∈ {−1, 0, +1, +2} and
the mean charge ⟨Z⟩ using WD01a photoionisation yields and DS87
recombination rates.

The four-state charge balance (WD01a §3, Murga+2020 §2):

    df(-1)/dt = k_att × ne × f(0)  − k_pi(-1) × f(-1)    = 0
    df(0)/dt  = k_pi(-1) × f(-1) + k_rec(+1) × ne × f(+1)
              − [k_att × ne + k_pi(0)] × f(0)             = 0
    df(+1)/dt = k_pi(0) × f(0) + k_rec(+2) × ne × f(+2)
              − [k_pi(+1) + k_rec(+1) × ne] × f(+1)      = 0
    df(+2)/dt = k_pi(+1) × f(+1) − k_rec(+2) × ne × f(+2) = 0
    Σ f(Z) = 1

This is a linear system; we solve it via scipy.linalg.lstsq.

References
----------
Weingartner, J.C. & Draine, B.T. 2001a, ApJS, 134, 263 (WD01a)
Draine, B.T. & Sutin, B. 1987, ApJ, 320, 803 (DS87)
Murga, M.S. et al. 2020, A&A, 644, A89
"""

import numpy as np
from scipy.linalg import lstsq

from .wd01a_yields import (
    photoion_rate_wd01a,
    recombination_rate_ds87,
    attachment_rate_wd01a,
)

# Charge states tracked
_CHARGES = [-1, 0, 1, 2]


def compute_charge_rates(Nc, T, ne, u_E_fn, G0=1.0):
    """
    Evaluate all charge-transition rate coefficients.

    Parameters
    ----------
    Nc : int or float
        Number of carbon atoms.
    T : float
        Gas temperature [K].
    ne : float
        Electron number density [cm⁻³].
    u_E_fn : callable
        u_E_fn(E_eV) → u_E [erg cm⁻³ eV⁻¹] for unit-G0 ISRF.
    G0 : float
        Habing field scaling.

    Returns
    -------
    rates : dict
        k_pi   : array shape (4,)  — photoionisation rate [s⁻¹] for Z = −1,0,+1,+2
        k_rec  : array shape (4,)  — recombination rate coeff [cm³ s⁻¹] for Z = −1,0,+1,+2
                 (k_rec[Z] is the rate for grain at charge Z to capture an electron,
                  i.e. Z → Z−1; defined only for Z ≥ 1)
        k_att  : float             — attachment rate coeff [cm³ s⁻¹] for Z=0 → Z=−1
    """
    k_pi  = np.zeros(4)
    k_rec = np.zeros(4)

    for iz, Z in enumerate(_CHARGES):
        k_pi[iz] = photoion_rate_wd01a(Z, Nc, u_E_fn, G0=G0)
        if Z >= 1:
            k_rec[iz] = recombination_rate_ds87(Z, Nc, T)

    k_att = attachment_rate_wd01a(Nc, T)
    return dict(k_pi=k_pi, k_rec=k_rec, k_att=k_att)


def steady_state_charges(Nc, T, ne, u_E_fn, G0=1.0):
    """
    Steady-state charge fractions f(Z) for Z ∈ {−1, 0, +1, +2}.

    Solves the 4×4 linear system (charge balance + normalisation).

    Parameters
    ----------
    Nc : int or float
        Number of carbon atoms.
    T : float
        Gas temperature [K].
    ne : float
        Electron number density [cm⁻³].
    u_E_fn : callable
        Unit-G0 radiation field energy density [erg cm⁻³ eV⁻¹].
    G0 : float
        Habing field scaling.

    Returns
    -------
    fractions : dict
        {-1: f_{-1}, 0: f_0, +1: f_{+1}, +2: f_{+2}}
        All fractions ≥ 0 and sum to 1.
    """
    r = compute_charge_rates(Nc, T, ne, u_E_fn, G0=G0)
    k_pi  = r['k_pi']   # k_pi[i] for Z = _CHARGES[i]
    k_rec = r['k_rec']
    k_att = r['k_att']

    # Unpack for readability (index: 0→Z=-1, 1→Z=0, 2→Z=+1, 3→Z=+2)
    kp_m1, kp_0, kp_p1, kp_p2 = k_pi
    # k_rec[iz] = rate for grain at Z=_CHARGES[iz] → Z-1; only Z>=1 has nonzero
    kr_p1 = k_rec[2] * ne   # Z=+1 captures electron: s⁻¹
    kr_p2 = k_rec[3] * ne   # Z=+2 captures electron: s⁻¹
    ka    = k_att * ne       # neutral captures electron: s⁻¹

    # Steady-state equations (rows 0..2 from balance, row 3 = normalisation):
    # Variables: [f(-1), f(0), f(+1), f(+2)]
    #
    # d f(-1)/dt = ka × f(0) − kp_m1 × f(-1) = 0
    # d f(0)/dt  = kp_m1×f(-1) + kr_p1×f(+1) − (ka + kp_0)×f(0) = 0
    # d f(+1)/dt = kp_0×f(0) + kr_p2×f(+2) − (kp_p1 + kr_p1)×f(+1) = 0
    # Σ f(Z) = 1

    A = np.array([
        [-kp_m1,      ka,          0.0,       0.0  ],
        [ kp_m1, -(ka + kp_0),   kr_p1,       0.0  ],
        [ 0.0,       kp_0,  -(kp_p1 + kr_p1), kr_p2],
        [ 1.0,        1.0,         1.0,         1.0  ],
    ])
    b = np.array([0.0, 0.0, 0.0, 1.0])

    # Solve overdetermined (first 3 eqs + normalisation) via least squares
    f, *_ = lstsq(A, b)
    f = np.clip(f, 0.0, 1.0)
    norm = f.sum()
    if norm > 0.0:
        f /= norm

    return {Z: f[i] for i, Z in enumerate(_CHARGES)}


def mean_charge(Nc, T, ne, u_E_fn, G0=1.0):
    """
    Mean charge ⟨Z⟩ = Σ_Z Z × f(Z).

    Parameters
    ----------
    (same as steady_state_charges)

    Returns
    -------
    Z_mean : float
    """
    frac = steady_state_charges(Nc, T, ne, u_E_fn, G0=G0)
    return sum(Z * frac[Z] for Z in _CHARGES)
