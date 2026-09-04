"""
Dust and molecular self-shielding factor implementations.

Companion to rtz/molecules_module.f90 in RAMSES.

Functions
---------
comp_SH2          H2 Lyman-Werner self-shielding (Draine & Bertoldi 1996)
comp_Sd_old       Dust LW shielding with fixed MW cross-section (current RAMSES)
comp_Sd_new       Dust LW shielding using per-bin optical properties
comp_SCO          CO self-shielding (Lee 1996 / Visser et al. 2009 tables)
comp_G0_selfshield  Dust self-shielding of the radiation field G0

Utilities
---------
compute_kappa_LW_draine  κ_LW [cm²/g] from Draine Q-tables for arbitrary (size, composition)
compute_kappa_LW_precomp κ_LW [cm²/g] from pre-computed pyCALIMA cross-section tables
"""

from .shielding_functions import (
    comp_SH2,
    comp_Sd_old,
    comp_Sd_new,
    comp_SCO,
    comp_G0_selfshield,
    compute_kappa_LW_draine,
    compute_kappa_LW_precomp,
    SDEFF_MW,
    DGR_MW,
)

__all__ = [
    "comp_SH2",
    "comp_Sd_old",
    "comp_Sd_new",
    "comp_SCO",
    "comp_G0_selfshield",
    "compute_kappa_LW_draine",
    "compute_kappa_LW_precomp",
    "SDEFF_MW",
    "DGR_MW",
]
