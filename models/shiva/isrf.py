"""
isrf.py — Interstellar radiation field definitions for the SHIVA module.

Provides unit-G0 spectral energy densities u_E(E) [erg cm⁻³ eV⁻¹] and
a convenience wrapper that scales by G0.

Two fields are available:
    mathis83   — Mathis, Mezger & Panagia (1983) ISRF (three blackbodies + UV)
    draine78   — Draine (1978) UV field (matches Habing with G0=1 at 6–13.6 eV)

The Mathis+83 field is used as the default because it includes both UV and
the optical/IR stellar contributions, and it is the field normalised to
G0 = 1 by the existing CALIMA code.

References
----------
Mathis, J.S., Mezger, P.G. & Panagia, N. 1983, A&A, 128, 212 (MMP83)
Draine, B.T. 1978, ApJS, 36, 595
Habing, H.J. 1968, BAN, 19, 421
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from models.tools.radiation_fields import Mathis83_radiation_field


# ── Habing normalisation ──────────────────────────────────────────────────
# G0 = 1 corresponds to u_total(6–13.6 eV) = 5.33e-14 erg cm⁻³
# (Habing 1968; Draine 1978 gives a slightly different normalisation of
#  u = 8.93e-14 erg cm⁻³ and the factor 1.7 converts between the two)

c_cgs = 2.99792458e10      # cm s⁻¹
habing_u = 5.33e-14        # erg cm⁻³ (integrated 6–13.6 eV, G0 = 1)


def mathis83_uE(E_eV):
    """
    Mathis+83 spectral energy density at G0 = 1.

    Parameters
    ----------
    E_eV : float
        Photon energy [eV].

    Returns
    -------
    u_E : float  [erg cm⁻³ eV⁻¹]  (0 for E > 13.6 eV or E ≤ 0)
    """
    if E_eV <= 0.0 or E_eV > 13.6:
        return 0.0
    return float(Mathis83_radiation_field(E_eV))


def scaled_uE(E_eV, G0=1.0, field='mathis83'):
    """
    Radiation field energy density scaled by G0.

    Parameters
    ----------
    E_eV : float
        Photon energy [eV].
    G0 : float
        Habing field strength.
    field : str
        'mathis83' (default).

    Returns
    -------
    u_E : float  [erg cm⁻³ eV⁻¹]
    """
    if field == 'mathis83':
        return G0 * mathis83_uE(E_eV)
    raise ValueError(f"Unknown radiation field '{field}'")


def make_isrf_callable(G0=1.0, field='mathis83'):
    """
    Return a callable u_E_fn(E_eV) → float [erg cm⁻³ eV⁻¹] for unit G0,
    suitable for passing into the SHIVA charge/dissociation modules.

    The G0 scaling is applied by the individual rate integrals, not here.
    """
    if field == 'mathis83':
        return mathis83_uE
    raise ValueError(f"Unknown field '{field}'")
