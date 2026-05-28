"""Dataclasses describing the state of the dust chemistry ODE system.

Mirrors the Fortran DustChemistryInfo, DustBin, and PAHBin types from
RAMSES-CALIMA (dustbin_types.f90 / dust_commons.f90).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Standard element registry (matches RAMSES element ordering)
# ---------------------------------------------------------------------------

ELEMENT_NAMES: List[str] = ["H", "He", "C", "N", "O", "Mg", "Si", "S", "Fe"]

#: Atomic numbers
ELEMENT_ATOMIC_NUMBERS: List[int] = [1, 2, 6, 7, 8, 12, 14, 16, 26]

#: Atomic masses in unified atomic mass units (1 u = AU2G grams)
ELEMENT_ATOMIC_MASSES_AU: List[float] = [
    1.008, 4.003, 12.011, 14.007, 15.999, 24.305, 28.086, 32.065, 55.845
]

#: Conversion: 1 atomic mass unit → grams
AU2G: float = 1.66053906660e-24

#: Atomic masses in grams
ELEMENT_ATOMIC_MASSES_G: List[float] = [m * AU2G for m in ELEMENT_ATOMIC_MASSES_AU]

N_ELEMENTS: int = len(ELEMENT_NAMES)


# ---------------------------------------------------------------------------
# Dust bin parameters
# ---------------------------------------------------------------------------

@dataclass
class DustBinParams:
    """Physical parameters for one non-PAH dust grain bin.

    Attributes
    ----------
    bin_id : str
        Human-readable label, e.g. ``'DustBin_01'``.
    composition : str
        Grain material: ``'graphite'`` or ``'silicate'``.
    bin_index : int
        0-based index of this bin in the dust-only arrays (i.e. the index
        into ``y_dust[state.npah:]``).
    asize_micron : float
        Characteristic grain radius [µm].
    asize_cm : float
        Characteristic grain radius [cm].
    sgrain : float
        Grain material (bulk) density [g cm⁻³].
    mgrain : float
        Mass of a single grain [g].
    el_names : list of str
        Names of elements that make up this grain (subset of ELEMENT_NAMES).
    el_indices : list of int
        Corresponding indices into ELEMENT_NAMES / y_gas.
    el_mfractions : list of float
        Mass fraction of each element in the grain composition (sums to 1).
    k0_acc : float
        Accretion rate constant [cm³ g⁻⁰·⁵ K⁻⁰·⁵ s⁻¹].
        Pre-computed from grain geometry and LeBourlot et al. (2012).
    k0_coa : float
        Coagulation rate constant [cm³ g⁻¹ s⁻¹].
        Pre-computed from grain geometry (Aoyama et al. 2017).
    coag_partner_index : int or None
        0-based dust-bin index of the larger bin into which this bin
        coagulates.  ``None`` means no coagulation for this bin.
    sputtering_interps : dict
        Mapping ``element_name → callable(T, phi=0) → rate [µm yr⁻¹ cm³]``.
        Built at initialisation from the pre-computed CALIMA tables.
    nhmax_acc : float
        Maximum nH for accretion [cm⁻³].  Above this density the
        accretion timescale is clamped (subgrid depletion model).
    nh_coa : float
        Minimum nH for coagulation to be active [cm⁻³].
    smallr_dust : float
        Minimum dust density below which a bin is considered inactive [g cm⁻³].
    """

    bin_id: str
    composition: str
    bin_index: int

    asize_micron: float
    asize_cm: float
    sgrain: float
    mgrain: float

    el_names: List[str]
    el_indices: List[int]
    el_mfractions: List[float]

    k0_acc: float = 0.0
    k0_coa: float = 0.0

    coag_partner_index: Optional[int] = None
    sputtering_interps: Dict[str, Any] = field(default_factory=dict)

    nhmax_acc: float = 1.0e4
    nh_coa: float = 0.1
    smallr_dust: float = 1.0e-40

    # --- Shattering parameters ---
    catastrophic_spec_energy: float = 1.0e7
    """Catastrophic specific energy Q* [erg g⁻¹]."""
    mgrain_min: float = 0.0
    """Min grain mass in size distribution [g] (for fragment distribution)."""
    mgrain_max: float = 0.0
    """Max grain mass in size distribution [g]."""
    interact_pah: bool = False
    """Whether shattering fragments of this bin can populate PAH bins."""

    # --- Turbulent coagulation ---
    vthresh_coag: float = 1.0e4
    """Coagulation threshold velocity [cm s⁻¹] for self-collisions."""


# ---------------------------------------------------------------------------
# PAH bin parameters
# ---------------------------------------------------------------------------

@dataclass
class PAHBinParams:
    """Physical parameters for one PAH size bin.

    Attributes
    ----------
    bin_id : str
        Human-readable label, e.g. ``'PAHbin_01'``.
    bin_index : int
        0-based index of this bin in the PAH-only part of y_dust (indices
        0 … npah-1).
    nc : int
        Characteristic number of carbon atoms per PAH molecule.
    nc_min : int
        Minimum nc for this bin (lower edge of the size distribution).
    mpah : float
        Mass of a single PAH molecule [g].
    spah : float
        Effective PAH material density [g cm⁻³].
    fpah : float
        Initial fraction of total carbon locked in PAHs (used to set
        initial conditions when ``initial_mass_density_gcm3`` is not given).
    sputtering_interps : dict
        Same convention as DustBinParams.
    smallr_pah : float
        Minimum density threshold [g cm⁻³].
    """

    bin_id: str
    bin_index: int

    nc: int
    nc_min: int = 0
    mpah: float = 0.0
    spah: float = 2.24
    fpah: float = 0.1

    sputtering_interps: Dict[str, Any] = field(default_factory=dict)
    smallr_pah: float = 1.0e-40

    # --- Geometry and size range ---
    apah_cm: float = 0.0
    """Effective PAH radius [cm] (derived from mass and density)."""
    nc_max: int = 0
    """Maximum nc for this bin (upper edge of size distribution)."""
    mpah_min: float = 0.0
    """Min PAH mass in bin [g] (nc_min × 12 u)."""
    mpah_max: float = 0.0
    """Max PAH mass in bin [g] (nc_max × 12 u)."""

    # --- Photolysis ---
    C_index: int = 2
    """Index of carbon in ELEMENT_NAMES (= 2, hardcoded like RAMSES)."""
    dissociation_interp: Optional[Any] = None
    """Callable ``(log10_G0, log10_nH) → log10(rate [s⁻¹])`` for photolysis."""

    # --- Cluster evaporation ---
    is_cluster: bool = False
    """True if this bin represents PAH clusters (not monomers)."""

    # --- PAH freezing onto dust ---
    dust_index_interact: int = -1
    """First dust-bin index (into dust_bins list) for PAH–dust interaction."""
    nd_bins: int = 0
    """Number of dust bins for PAH freezing."""


# ---------------------------------------------------------------------------
# Full chemistry state
# ---------------------------------------------------------------------------

@dataclass
class DustChemistryState:
    """Complete description of a dust-chemistry ODE system at a fixed gas cell.

    This bundles the local gas environment (temperature, density, etc.) with
    the grain-bin parameter objects and solver settings.  It is the Python
    analogue of the Fortran ``DustChemistryInfo`` type together with the
    ``dustbins_props`` and ``pahbins_props`` arrays from
    ``dust_commons.f90``.

    ODE state layout
    ----------------
    ``y_gas``  : 1-D array, shape (n_elements,)  — element mass densities [g cm⁻³]
    ``y_dust`` : 1-D array, shape (npah + ndust,) — PAH bins first, then dust bins
    """

    # ---- Fixed gas environment ----
    local_Tk: float
    """Gas kinetic temperature [K]."""
    local_nH: float
    """Hydrogen number density [H cm⁻³]."""
    local_rho: float
    """Total gas mass density [g cm⁻³]."""
    local_ne: float
    """Electron number density [cm⁻³]."""
    local_G0: float
    """Far-UV radiation field strength in Habing units."""
    local_mu: float = 1.4
    """Mean molecular weight [m_H]."""

    # ---- Element registry ----
    n_elements: int = N_ELEMENTS
    el_names: List[str] = field(default_factory=lambda: list(ELEMENT_NAMES))
    el_atomic_mass_g: List[float] = field(
        default_factory=lambda: list(ELEMENT_ATOMIC_MASSES_G)
    )

    # ---- Grain bins ----
    ndust: int = 0
    npah: int = 0
    dust_bins: List[DustBinParams] = field(default_factory=list)
    pah_bins: List[PAHBinParams] = field(default_factory=list)

    # ---- Solver settings (mirror of dust_commons.f90 parameters) ----
    errmax: float = 0.1
    """Max allowed relative change per RK4 step."""
    countmax: int = 10000
    """Max number of ODE solver iterations before bailing out."""

    # ---- Physics process flags ----
    dust_accretion: bool = False
    dust_sputtering: bool = False
    dust_coagulation: bool = False
    dust_shattering: bool = False
    pah_accretion: bool = False
    pah_photolysis: bool = False
    pah_sputtering: bool = False
    pah_coalescence: bool = False
    pah_cluster_evaporation: bool = False
    pah_freezing: bool = False

    # ---- Model selection strings (match RAMSES physics flags) ----
    coagulation_model: str = "Aoyama2017"
    """Choices: ``'Aoyama2017'`` | ``'turbulent'`` | ``'turbulent_all'``."""
    shattering_model: str = "turbulent"
    """Choices: ``'turbulent'`` (self-only) | ``'turbulent_all'`` (all pairs)."""
    dust_velocity_model: str = "Ormel2007"
    """Choices: ``'Ormel2007'`` | ``'Hirashita2019'``."""
    coalescence_model: str = "Totton2012"
    """Choices: ``'Totton2012'`` | ``'Tielens2021'``."""
    photolysis_model: str = "RM2026"
    pah_sputtering_model: str = "RM2026"
    cluster_evaporation_model: str = "Montillaud2014"

    # ---- Turbulent environment ----
    local_sigma: float = 0.0
    """Turbulent 1-D velocity dispersion [cm s⁻¹]."""
    local_dx: float = 0.0
    """Turbulence injection scale [cm]."""

    # ---- Shattering parameters ----
    slope_frag_func: float = 1.3 / 3.0
    """Power-law slope of the fragment mass distribution."""

    # ---- Optional metadata ----
    radiation_field_model: str = "habing"

    @property
    def ntot(self) -> int:
        """Total number of ODE variables (gas elements + dust + PAH bins)."""
        return self.n_elements + self.npah + self.ndust
