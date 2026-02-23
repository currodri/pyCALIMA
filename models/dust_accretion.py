"""
DUST ACCRETION MODULE

This module contains functions and classes for computing dust accretion rates
in various astrophysical environments. The accretion process depends on:
  - Grain composition and surface area
  - Ion/element number densities in the gas phase
  - Gas temperature and thermal velocities
  - Grain charge state (Coulomb enhancement/suppression)
  - Sticking coefficients for different species

The key concept is that different ions/elements collide with grain surfaces
at different rates determined by their mass, density, and Coulomb interactions.
The accretion rate is typically limited by the species with the fastest
collision rate (most abundant or lightest ion).

Physical models referenced:
  - Tielens (2010): ISM dust properties and evolution
  - Draine (2011): Physics of the Interstellar and Intergalactic Medium
  - Grain accretion onto dust surfaces

By: Curro Rodriguez Montero (currodri@gmail.com)
"""

# Import libraries
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dust_pah_evolution import (
    GrainComposition,
    SILICATE_COMPOSITION,
    GRAPHITE_COMPOSITION,
    OLIVINE_COMPOSITION,
    CARBON_COMPOSITION,
    ATOMIC_MASSES_AMU,
    MOLECULAR_MASSES_AMU,
    ATOMIC_MASS_UNIT,
    get_atomic_mass_grams,
    infer_ion_charge,
)

# Physical constants (CGS units)
from dust_pah_evolution import BOLTZMANN_CONSTANT as KB
from dust_pah_evolution import ELEMENTARY_CHARGE

SPEED_OF_LIGHT = 2.99792458e10  # Speed of light [cm/s]
PERMITTIVITY_CONSTANT = 4.0 * np.pi * 8.8541878188e-21  # 4πε₀ in CGS


# ============================================================================
# COMPOSITION STRUCTURES IMPORTED FROM DUST_PAH_EVOLUTION
# ============================================================================
# See dust_pah_evolution.py for GrainComposition class and standard compositions


# ============================================================================
# ION COLLISION RATE FUNCTIONS
# ============================================================================


def thermal_velocity(mass_grams: float, temperature_K: float) -> float:
    """
    Compute mean thermal velocity of a particle.
    
    Uses 1D root-mean-square velocity: v = sqrt(3*kB*T/m)
    
    Parameters
    ----------
    mass_grams : float
        Particle mass [g]
    temperature_K : float
        Gas temperature [K]
    
    Returns
    -------
    velocity : float
        Thermal velocity [cm/s]
    """
    if temperature_K <= 0 or mass_grams <= 0:
        return 0.0
    
    velocity = np.sqrt(3.0 * KB * temperature_K / mass_grams)
    return float(velocity)


def collision_cross_section_geometric(grain_radius_cm: float) -> float:
    """
    Geometric collision cross-section for a spherical grain.
    
    σ = π * a²
    
    Parameters
    ----------
    grain_radius_cm : float
        Grain radius [cm]
    
    Returns
    -------
    cross_section : float
        Cross-section [cm²]
    """
    return np.pi * grain_radius_cm ** 2


def coulomb_enhancement_factor(
    grain_charge: int,
    ion_charge: int,
    grain_radius_cm: float,
    temperature_K: float,
) -> float:
    """
    Compute the Coulomb enhancement factor for ion-grain collisions.
    
    Accounts for long-range Coulomb interactions between ions and charged grains.
    For neutral grains, returns 1.0.
    
    Based on the approach in Draine (2003), considering the Coulomb potential
    around a charged grain. For a grain with charge Z_g and ion with charge Z_i,
    the enhancement factor accounts for attractive/repulsive interactions.
    
    Parameters
    ----------
    grain_charge : int
        Grain charge (in units of elementary charge)
    ion_charge : int
        Ion charge (typically +1 for singly ionized)
    grain_radius_cm : float
        Grain radius [cm]
    temperature_K : float
        Gas temperature [K]
    
    Returns
    -------
    enhancement_factor : float
        Multiplicative factor for collision cross-section (≥ 0)
        
    Notes
    -----
    The Coulomb enhancement is given by:
        η = (1 + Z_i * Z_g * e² / (a * k_B * T)) if attractive (Z_i * Z_g < 0)
        η ≈ exp(-Z_i * Z_g * e² / (a * k_B * T)) if repulsive (Z_i * Z_g > 0)
    
    For a neutral grain (Z_g = 0), this returns 1.0.
    """
    if grain_charge == 0 or grain_radius_cm <= 0 or temperature_K <= 0:
        return 1.0
    
    # Coulomb parameter
    coulomb_param = (
        grain_charge * ion_charge * ELEMENTARY_CHARGE ** 2 / 
        (grain_radius_cm * KB * temperature_K)
    )
    
    # For attractive interaction (opposite charges): enhancement
    if grain_charge * ion_charge < 0:
        # Attractive: η ≈ 1 + |coulomb_param| for modest enhancement
        # More accurate: use the focusing factor
        enhancement = 1.0 + np.abs(coulomb_param)
    else:
        # Repulsive interaction: suppression
        # η ≈ exp(-coulomb_param)
        enhancement = np.exp(-coulomb_param)
    
    return float(np.maximum(enhancement, 1e-10))  # Prevent negative/zero


def collision_rate_single_species(
    species_name: str,
    species_density_cm3: float,
    species_charge: int,
    grain_radius_cm: float,
    grain_charge: int,
    temperature_K: float,
) -> float:
    """
    Compute collision rate of a single ion/atom species with a grain.
    
    The collision rate is:
        Γ = n * v_th * σ_eff
    
    where:
    - n: number density of species [cm⁻³]
    - v_th: thermal velocity [cm/s]
    - σ_eff: effective collisional cross-section [cm²] with Coulomb enhancement
    
    Parameters
    ----------
    species_name : str
        Species name (e.g., 'H', 'He⁺', 'C⁺', 'O')
    species_density_cm3 : float
        Number density of species [cm⁻³]
    species_charge : int
        Charge of species (0 for neutral, +1 for singly ionized, etc.)
    grain_radius_cm : float
        Grain radius [cm]
    grain_charge : int
        Grain charge (electrons)
    temperature_K : float
        Gas temperature [K]
    
    Returns
    -------
    collision_rate : float
        Collision rate [s⁻¹] (per unit grain, per cm³ of space)
        Actually: rate density [cm⁻³ s⁻¹]
    """
    if species_density_cm3 <= 0:
        return 0.0
    
    # Get mass of species
    mass_grams = get_atomic_mass_grams(species_name)
    
    # Compute thermal velocity
    v_th = thermal_velocity(mass_grams, temperature_K)
    if v_th <= 0:
        return 0.0
    
    # Geometric cross-section
    sigma_geom = collision_cross_section_geometric(grain_radius_cm)
    
    # Coulomb enhancement factor
    coulomb_factor = coulomb_enhancement_factor(
        grain_charge, species_charge, grain_radius_cm, temperature_K
    )
    
    # Effective cross-section
    sigma_eff = sigma_geom * coulomb_factor
    
    # Collision rate (per unit grain cross-section)
    collision_rate = species_density_cm3 * v_th * sigma_eff
    
    return float(collision_rate)


def collision_rate_from_densities(
    ion_densities: Dict[str, float],
    grain_radius_cm: float,
    grain_charge: int,
    temperature_K: float,
    ion_charges: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Compute collision rates for all ions/species with a grain.
    
    Parameters
    ----------
    ion_densities : dict
        Maps species name → density [cm⁻³]
        Example: {'H': 0.5, 'H⁺': 0.001, 'He': 0.1, 'C⁺': 1e-4}
    grain_radius_cm : float
        Grain radius [cm]
    grain_charge : int
        Grain charge
    temperature_K : float
        Gas temperature [K]
    ion_charges : dict, optional
        Maps species name → charge. If not provided, infers from name.
    
    Returns
    -------
    rates : dict
        Maps species name → collision rate [cm⁻³ s⁻¹]
    """
    if ion_charges is None:
        ion_charges = {}
    
    rates = {}
    
    for species, density in ion_densities.items():
        if density <= 0:
            continue
        
        # Determine charge
        if species in ion_charges:
            charge = ion_charges[species]
        else:
            # Infer from name (e.g., 'H⁺' → +1, 'C' → 0)
            charge = infer_ion_charge(species)
        
        # Compute collision rate
        rate = collision_rate_single_species(
            species, density, charge, grain_radius_cm, grain_charge, temperature_K
        )
        
        rates[species] = rate
    
    return rates


def limiting_collision_rate(
    ion_densities: Dict[str, float],
    grain_radius_cm: float,
    grain_charge: int,
    temperature_K: float,
    ion_charges: Optional[Dict[str, int]] = None,
    return_limiting_species: bool = False,
) -> Union[float, Tuple[float, str]]:
    """
    Find the limiting collision rate (largest collision rate among all species).
    
    The limiting species determines the accretion rate because it has the
    fastest collision rate with the grain.
    
    Parameters
    ----------
    ion_densities : dict
        Ion/species densities [cm⁻³]
    grain_radius_cm : float
        Grain radius [cm]
    grain_charge : int
        Grain charge
    temperature_K : float
        Gas temperature [K]
    ion_charges : dict, optional
        Ion charges (inferred from names if not provided)
    return_limiting_species : bool
        If True, also return name of limiting species
    
    Returns
    -------
    limit_rate : float
        Largest collision rate [cm⁻³ s⁻¹]
    limiting_species : str (optional)
        Name of species with largest collision rate
    """
    rates = collision_rate_from_densities(
        ion_densities, grain_radius_cm, grain_charge, temperature_K, ion_charges
    )
    
    if not rates:
        return (0.0, '') if return_limiting_species else 0.0
    
    limiting_species = max(rates, key=rates.get)
    limiting_rate = rates[limiting_species]
    
    if return_limiting_species:
        return limiting_rate, limiting_species
    else:
        return limiting_rate


# ============================================================================
# ACCRETION RATE COMPUTATION
# ============================================================================


def accretion_rate_da_dt(
    grain_radius_cm: float,
    grain_density_gcm3: float,
    ion_densities: Dict[str, float],
    grain_charge: int,
    temperature_K: float,
    ion_charges: Optional[Dict[str, int]] = None,
    sticking_coefficient: float = 1.0,
) -> float:
    """
    Compute grain radius growth rate from accretion.
    
    The radius changes according to:
        da/dt = Γ * m_species / (4π * a² * ρ_grain)
    
    where Γ is the limiting collision rate (fastest species).
    
    Parameters
    ----------
    grain_radius_cm : float
        Grain radius [cm]
    grain_density_gcm3 : float
        Grain material density [g/cm³]
    ion_densities : dict
        Number densities of accreting species [cm⁻³]
    grain_charge : int
        Grain charge state
    temperature_K : float
        Gas temperature [K]
    ion_charges : dict, optional
        Ion charges (inferred if not provided)
    sticking_coefficient : float
        Probability that particle sticks upon collision (0-1)
    
    Returns
    -------
    da_dt : float
        Radius growth rate [cm/s]
        After conversion: can be in [µm/year] with appropriate scaling
    """
    # Get limiting collision rate and species
    limit_rate, limiting_species = limiting_collision_rate(
        ion_densities, grain_radius_cm, grain_charge, temperature_K,
        ion_charges, return_limiting_species=True
    )
    
    if limit_rate <= 0 or grain_radius_cm <= 0:
        return 0.0
    
    # Mass of accreting material (single atom/ion)
    m_species = get_atomic_mass_grams(limiting_species)
    
    # Surface area of grain
    a_surface = 4.0 * np.pi * grain_radius_cm ** 2
    
    # Growth rate: da/dt = (sticking) * Γ * m / (4πa² * ρ)
    da_dt = (sticking_coefficient * limit_rate * m_species / 
             (a_surface * grain_density_gcm3))
    
    return float(da_dt)


def accretion_rate_dn_dt(
    grain_population_cm3: float,
    grain_radius_cm: float,
    grain_density_gcm3: float,
    ion_densities: Dict[str, float],
    grain_charge: int,
    temperature_K: float,
    ion_charges: Optional[Dict[str, int]] = None,
    sticking_coefficient: float = 1.0,
) -> float:
    """
    Compute accretion rate as change in grain number density (dn/dt).
    
    In accretion, grain number doesn't change (grains grow but don't multiply),
    so dn/dt = 0. This function is provided for completeness in the evolution
    framework, but the actual growth is tracked via change in grain radius.
    
    In the framework, use accretion_rate_da_dt() instead.
    
    Parameters
    ----------
    grain_population_cm3 : float
        Number density of grains [cm⁻³]
    grain_radius_cm : float
        Grain radius [cm]
    grain_density_gcm3 : float
        Grain density [g/cm³]
    ion_densities : dict
        Ion number densities [cm⁻³]
    grain_charge : int
        Grain charge
    temperature_K : float
        Temperature [K]
    ion_charges : dict, optional
        Ion charges
    sticking_coefficient : float
        Sticking probability
    
    Returns
    -------
    dn_dt : float
        Change in number density (always 0 for pure accretion)
    """
    # Accretion doesn't change the number of grains, only their size
    return 0.0


def accretion_timescale(
    grain_radius_cm: float,
    grain_density_gcm3: float,
    ion_densities: Dict[str, float],
    grain_charge: int,
    temperature_K: float,
    ion_charges: Optional[Dict[str, int]] = None,
    sticking_coefficient: float = 1.0,
) -> float:
    """
    Compute characteristic timescale for grain growth by accretion.
    
    Timescale τ = a / (da/dt) is the time for grain to grow by one e-folding.
    
    Parameters
    ----------
    grain_radius_cm : float
        Current grain radius [cm]
    grain_density_gcm3 : float
        Grain density [g/cm³]
    ion_densities : dict
        Ion densities [cm⁻³]
    grain_charge : int
        Grain charge
    temperature_K : float
        Temperature [K]
    ion_charges : dict, optional
        Ion charges
    sticking_coefficient : float
        Sticking coefficient
    
    Returns
    -------
    timescale : float
        Timescale [seconds]
    """
    da_dt = accretion_rate_da_dt(
        grain_radius_cm, grain_density_gcm3, ion_densities,
        grain_charge, temperature_K, ion_charges, sticking_coefficient
    )
    
    if da_dt <= 0:
        return np.inf
    
    timescale = grain_radius_cm / da_dt
    return float(timescale)


# ============================================================================
# COMPOSITION-AWARE ACCRETION
# ============================================================================


def accretion_rate_from_composition(
    grain_radius_cm: float,
    grain_composition: GrainComposition,
    ion_densities: Dict[str, float],
    grain_charge: int,
    temperature_K: float,
    ion_charges: Optional[Dict[str, int]] = None,
    sticking_coefficient: float = 1.0,
) -> float:
    """
    Compute accretion rate accounting for grain composition.
    
    For a grain of specific composition, different ions/atoms accrete at
    different rates. This function computes the volume-weighted accretion
    rate based on the grain material density.
    
    Parameters
    ----------
    grain_radius_cm : float
        Grain radius [cm]
    grain_composition : GrainComposition
        Grain composition object with density and elemental abundances
    ion_densities : dict
        Ion denities [cm⁻³]
    grain_charge : int
        Grain charge
    temperature_K : float
        Temperature [K]
    ion_charges : dict, optional
        Ion charges
    sticking_coefficient : float
        Sticking coefficient
    
    Returns
    -------
    da_dt : float
        Radius growth rate [cm/s]
    """
    return accretion_rate_da_dt(
        grain_radius_cm,
        grain_composition.density,
        ion_densities,
        grain_charge,
        temperature_K,
        ion_charges,
        sticking_coefficient
    )


def composition_update_from_accretion(
    grain_composition: GrainComposition,
    ion_densities: Dict[str, float],
    grain_charge: int,
    temperature_K: float,
    ion_charges: Optional[Dict[str, int]] = None,
    time_step_seconds: float = 1e12,
) -> Dict[str, float]:
    """
    Update grain composition if accreting material has different composition.
    
    If accreting species differ from grain composition, the grain composition
    will gradually change over time.
    
    Parameters
    ----------
    grain_composition : GrainComposition
        Current grain composition
    ion_densities : dict
        Ion densities [cm⁻³]
    grain_charge : int
        Grain charge
    temperature_K : float
        Temperature
    ion_charges : dict, optional
        Ion charges
    time_step_seconds : float
        Time step [seconds]
    
    Returns
    -------
    updated_composition : dict
        Updated element abundances
    """
    # Find limiting species (what's being accreted)
    rates = collision_rate_from_densities(
        ion_densities, 1e-5, grain_charge, temperature_K, ion_charges
    )
    
    if not rates:
        return grain_composition.elements.copy()
    
    limiting_species = max(rates, key=rates.get)
    
    # Extract element from species name (e.g., 'C' from 'C⁺')
    accreting_element = ''.join([c for c in limiting_species if c.isalpha()])
    
    # Update composition
    new_composition = grain_composition.elements.copy()
    
    # Simple model: increase accreting element slightly
    if accreting_element in new_composition:
        # Weight change by collision rate relative to grain surface
        new_composition[accreting_element] *= 1.001
    else:
        new_composition[accreting_element] = 0.001
    
    # Renormalize
    total = sum(new_composition.values())
    if total > 0:
        for elem in new_composition:
            new_composition[elem] /= total
    
    return new_composition


# ============================================================================
# DETAILED COLLISION ANALYSIS
# ============================================================================


def collision_rate_analysis(
    grain_radius_cm: float,
    grain_charge: int,
    temperature_K: float,
    ion_densities: Dict[str, float],
    ion_charges: Optional[Dict[str, int]] = None,
    return_df: bool = False,
):
    """
    Detailed analysis of collision rates for all species.
    
    Returns a breakdown of how much each species contributes to total accretion.
    
    Parameters
    ----------
    grain_radius_cm : float
        Grain radius [cm]
    grain_charge : int
        Grain charge
    temperature_K : float
        Temperature [K]
    ion_densities : dict
        Species densities [cm⁻³]
    ion_charges : dict, optional
        Species charges
    return_df : bool
        If True, return as pandas DataFrame
    
    Returns
    -------
    analysis : dict or DataFrame
        Maps species → dict with keys:
        - 'density': number density [cm⁻³]
        - 'mass': atomic mass [g]
        - 'thermal_velocity': mean velocity [cm/s]
        - 'coulomb_factor': Coulomb enhancement
        - 'collision_rate': collision rate [cm⁻³ s⁻¹]
    """
    if ion_charges is None:
        ion_charges = {}
    
    analysis = {}
    
    for species, density in ion_densities.items():
        if density <= 0:
            continue
        
        # Get charge
        if species in ion_charges:
            charge = ion_charges[species]
        else:
            charge = infer_ion_charge(species)
        
        # Mass
        mass = get_atomic_mass_grams(species)
        
        # Thermal velocity
        v_th = thermal_velocity(mass, temperature_K)
        
        # Coulomb factor
        coulomb = coulomb_enhancement_factor(grain_charge, charge, grain_radius_cm, temperature_K)
        
        # Collision rate
        rate = collision_rate_single_species(
            species, density, charge, grain_radius_cm, grain_charge, temperature_K
        )
        
        analysis[species] = {
            'density': density,
            'mass': mass,
            'thermal_velocity': v_th,
            'coulomb_factor': coulomb,
            'collision_rate': rate,
        }
    
    if return_df:
        try:
            import pandas as pd
            df = pd.DataFrame(analysis).T
            return df
        except ImportError:
            return analysis
    
    return analysis


# ============================================================================
# INTEGRATION WITH EVOLUTION FRAMEWORK
# ============================================================================


class AccretionProcess:
    """
    Accretion process for use with dust evolution framework.
    
    This class wraps the accretion computations to work with the
    DustEvolutionSystem from dust_pah_evolution.py.
    """
    
    def __init__(
        self,
        ion_densities_function=None,
        ion_charges: Optional[Dict[str, int]] = None,
        sticking_coefficient: float = 1.0,
    ):
        """
        Initialize accretion process.
        
        Parameters
        ----------
        ion_densities_function : callable, optional
            Function f(state) → dict of ion densities [cm⁻³]
            If None, uses fixed ion_densities dict
        ion_charges : dict, optional
            Explicit ion charges
        sticking_coefficient : float
            Sticking coefficient (0-1)
        """
        self.ion_densities_function = ion_densities_function
        self.ion_charges = ion_charges or {}
        self.sticking_coefficient = sticking_coefficient
    
    def compute_accretion_rates(
        self,
        state,  # DustEvolutionState
        grain_pop,  # GrainPopulation
        return_details: bool = False,
    ):
        """
        Compute accretion rates for all grain bins.
        
        Returns da/dt for each bin (radius growth rate).
        
        Parameters
        ----------
        state : DustEvolutionState
            Current system state
        grain_pop : GrainPopulation
            Grain population
        return_details : bool
            If True, also return limiting species and rates dict
        
        Returns
        -------
        da_dt_array : ndarray
            Radius growth rate [cm/s] for each bin
        (optional) limiting_species_list : list
        (optional) rates_list : list of dicts
        """
        da_dt_array = np.zeros(grain_pop.nbins)
        limiting_species_list = []
        rates_list = []
        
        # Get ion densities
        if self.ion_densities_function is not None:
            ion_densities = self.ion_densities_function(state)
        else:
            ion_densities = getattr(state.env, 'ion_densities', {})
        
        for i, bin_ in enumerate(grain_pop.bins):
            # Get grain charge
            grain_charge = bin_.metadata.get('charge', 0)
            
            # Compute growth rate
            da_dt = accretion_rate_da_dt(
                bin_.radius_cm,
                bin_.density,
                ion_densities,
                grain_charge,
                state.env.temperature_K,
                self.ion_charges,
                self.sticking_coefficient
            )
            
            da_dt_array[i] = da_dt
            
            if return_details:
                _, lim_species = limiting_collision_rate(
                    ion_densities, bin_.radius_cm, grain_charge,
                    state.env.temperature_K, self.ion_charges,
                    return_limiting_species=True
                )
                limiting_species_list.append(lim_species)
                rates = collision_rate_from_densities(
                    ion_densities, bin_.radius_cm, grain_charge,
                    state.env.temperature_K, self.ion_charges
                )
                rates_list.append(rates)
        
        if return_details:
            return da_dt_array, limiting_species_list, rates_list
        
        return da_dt_array
