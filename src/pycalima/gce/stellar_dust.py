"""Dust condensation in stellar ejecta (after Dwek 1998 and Dubois+2024).

Two species, matching the solver's grain families:

- **carbon**: in AGB winds with C/O > 1 (by number) the CO-free carbon,
  ``δ (M_C - (μ_C/μ_O) M_O)``; in SNe ``δ M_C``.
- **silicate**: in AGB winds with C/O <= 1 and in SNe, limited by the
  key element of the silicate composition, ``δ min_X(M_X / f_X)`` where
  ``f_X`` are the mass fractions of the compound.

Efficiencies per channel are selected with a preset.
"""

from __future__ import annotations

import numpy as np

from .stellar_yields import ELEMENTS

MU = {"C": 12.0107, "O": 15.9994, "Mg": 24.305, "Si": 28.0855, "Fe": 55.845}

# Olivine MgFeSiO4, by mass
OLIVINE = {el: n * MU[el] for el, n in (("Mg", 1), ("Fe", 1), ("Si", 1), ("O", 4))}
_tot = sum(OLIVINE.values())
OLIVINE = {el: v / _tot for el, v in OLIVINE.items()}

# (carbon, silicate) condensation efficiency per channel
PRESETS = {
    # Dwek (1998) as in cmp_yield_release.pro /dwek (0.99 instead of 1)
    "dwek98": {"agb": (0.99, 0.99), "snii": (0.5, 0.8), "snia": (0.0, 0.0)},
    # The IDL default ("dustcpopping17" files)
    "popping17": {"agb": (0.2, 0.2), "snii": (0.15, 0.15), "snia": (0.0, 0.0)},
    # Parente+2023/2026 L-Galaxies
    "parente23": {"agb": (0.1, 0.1), "snii": (0.1, 0.1), "snia": (0.0, 0.0)},
}

_IDX = {el: ELEMENTS.index(el) for el in ELEMENTS}


def condense(ejecta: np.ndarray, channel: str, preset: str = "dwek98",
             silicate_composition: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Carbon and silicate dust masses formed in ``ejecta``.

    ``ejecta`` has shape (..., len(ELEMENTS)) of gross element masses.
    Returns ``(carbon, silicate)`` arrays of shape (...).
    """
    try:
        d_carb, d_sil = PRESETS[preset][channel]
    except KeyError:
        raise ValueError(f"unknown preset/channel {preset!r}/{channel!r}") from None
    comp = silicate_composition or OLIVINE
    ej = np.asarray(ejecta, dtype=float)
    m_c, m_o = ej[..., _IDX["C"]], ej[..., _IDX["O"]]
    sil_max = np.min(np.stack([ej[..., _IDX[el]] / f for el, f in comp.items()]), axis=0)

    if channel == "agb":
        carbon_rich = m_c * MU["O"] > m_o * MU["C"]
        carbon = np.where(carbon_rich, d_carb * (m_c - MU["C"] / MU["O"] * m_o), 0.0)
        silicate = np.where(carbon_rich, 0.0, d_sil * sil_max)
    else:
        carbon = d_carb * m_c
        silicate = d_sil * sil_max
    return carbon, silicate
