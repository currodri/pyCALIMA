"""
CORE DATA STRUCTURES FOR DUST AND PAH EVOLUTION

This module defines the fundamental data structures used across all dust
and PAH evolution processes:
  - GrainComposition: Elemental composition of dust grains
  - Atomic/molecular mass data
  - Helper functions for composition and stoichiometry

All dust processes (accretion, sputtering, coagulation, etc.) import
these structures to ensure consistency in composition handling.

By: Curro Rodriguez (currodri@gmail.com)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


# ============================================================================
# GRAIN COMPOSITION DATA STRUCTURES
# ============================================================================


@dataclass
class GrainComposition:
    """
    Represents the composition of a dust grain by element abundance.
    
    Attributes
    ----------
    grain_type : str
        Type of grain ('silicate', 'graphite', 'carbon', 'olivine', etc.)
    elements : dict
        Element symbols → fraction by mass (or by number)
        Example: {'Si': 0.33, 'O': 0.38, 'Mg': 0.24, 'Fe': 0.05}
    density : float
        Grain material density [g/cm³]
    composition_type : str
        'mass' if abundances are by mass, 'number' if by number of atoms
    """
    
    grain_type: str
    elements: Dict[str, float]
    density: float
    composition_type: str = 'mass'
    
    def normalize(self) -> None:
        """Normalize element abundances to sum to 1.0."""
        total = sum(self.elements.values())
        if total > 0:
            for elem in self.elements:
                self.elements[elem] /= total
    
    def get_element_fraction(self, element: str) -> float:
        """Get abundance fraction of a specific element."""
        return self.elements.get(element, 0.0)
    
    def copy(self):
        """Return a copy of this composition."""
        return GrainComposition(
            grain_type=self.grain_type,
            elements=self.elements.copy(),
            density=self.density,
            composition_type=self.composition_type
        )


# ============================================================================
# STANDARD GRAIN COMPOSITIONS
# ============================================================================

# Silicate composition (astronomical silicate, Draine & Lee 1984)
# Typical ISM silicate with Mg, Fe, Si, O
SILICATE_COMPOSITION = GrainComposition(
    grain_type='silicate',
    elements={
        'Si': 0.335,  # Silicon
        'Mg': 0.240,  # Magnesium
        'Fe': 0.050,  # Iron
        'O': 0.375,   # Oxygen
    },
    density=3.3,
    composition_type='mass'
)

# Pure graphite composition
GRAPHITE_COMPOSITION = GrainComposition(
    grain_type='graphite',
    elements={
        'C': 1.0,
    },
    density=2.2,
    composition_type='mass'
)

# Olivine composition (Mg-rich silicate)
OLIVINE_COMPOSITION = GrainComposition(
    grain_type='olivine',
    elements={
        'Mg': 0.363,
        'Fe': 0.044,
        'Si': 0.183,
        'O': 0.410,
    },
    density=3.7,
    composition_type='mass'
)

# Amorphous carbon composition
CARBON_COMPOSITION = GrainComposition(
    grain_type='carbon',
    elements={
        'C': 1.0,
    },
    density=1.8,
    composition_type='mass'
)


# ============================================================================
# ATOMIC AND MOLECULAR MASSES
# ============================================================================

# Atomic masses in atomic mass units (AMU)
# Source: NIST, https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-nist
ATOMIC_MASSES_AMU = {
    'H': 1.008,
    'He': 4.003,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'Si': 28.085,
    'S': 32.06,
    'Fe': 55.845,
    'Mg': 24.305,
}

# Molecular masses [AMU]
MOLECULAR_MASSES_AMU = {
    'H2': 2.0158,
    'CO': 28.010,
    'CO2': 44.009,
    'H2O': 18.015,
    'O2': 31.998,
    'N2': 28.014,
    'SiO': 44.085,
}

# Physical constants (CGS units)
ATOMIC_MASS_UNIT = 1.66053906660e-24  # [g]
BOLTZMANN_CONSTANT = 1.380649e-16  # [erg/K]
ELEMENTARY_CHARGE = 4.8032047e-10  # [esu/statC]


# ============================================================================
# MASS LOOKUP FUNCTIONS
# ============================================================================


def get_atomic_mass_amu(element_or_species: str) -> float:
    """
    Get atomic or molecular mass in AMU.
    
    Parameters
    ----------
    element_or_species : str
        Element symbol (e.g., 'H', 'O', 'Si') or molecule (e.g., 'H2', 'CO')
    
    Returns
    -------
    mass_amu : float
        Mass in atomic mass units (AMU)
    """
    # Try molecular mass first
    if element_or_species in MOLECULAR_MASSES_AMU:
        return MOLECULAR_MASSES_AMU[element_or_species]
    
    # Then atomic mass
    if element_or_species in ATOMIC_MASSES_AMU:
        return ATOMIC_MASSES_AMU[element_or_species]
    
    # Unknown species - return hydrogen mass as default
    return ATOMIC_MASSES_AMU.get('H', 1.008)


def get_atomic_mass_grams(element_or_species: str) -> float:
    """
    Get atomic or molecular mass in grams.
    
    Parameters
    ----------
    element_or_species : str
        Element symbol (e.g., 'H', 'O', 'Si') or molecule (e.g., 'H2', 'CO')
    
    Returns
    -------
    mass_grams : float
        Mass in grams
    """
    mass_amu = get_atomic_mass_amu(element_or_species)
    return mass_amu * ATOMIC_MASS_UNIT


# ============================================================================
# COMPOSITION UTILITIES
# ============================================================================


def get_mean_molecular_weight(composition: GrainComposition) -> float:
    """
    Compute mean molecular weight of a grain composition.
    
    For a composition given by mass fractions, the mean atomic mass is:
        <m> = 1 / sum(f_i / M_i)
    where f_i is mass fraction and M_i is atomic mass of element i.
    
    Parameters
    ----------
    composition : GrainComposition
        Grain composition with mass fractions
    
    Returns
    -------
    mean_mass : float
        Mean atomic mass in grams
    """
    if composition.composition_type != 'mass':
        raise ValueError("Mean molecular weight calculation requires mass fractions")
    
    inverse_sum = 0.0
    for element, fraction in composition.elements.items():
        if fraction > 0:
            m_element = get_atomic_mass_grams(element)
            inverse_sum += fraction / m_element
    
    if inverse_sum <= 0:
        return ATOMIC_MASS_UNIT  # Default to hydrogen
    
    return 1.0 / inverse_sum


# ============================================================================
# CHARGE STATE INFORMATION
# ============================================================================


def infer_ion_charge(species_name: str) -> int:
    """
    Infer charge of an ion/atom from its name.
    
    Parameters
    ----------
    species_name : str
        Species name (e.g., 'H', 'H+', 'C+', 'O2-', 'Fe2+')
    
    Returns
    -------
    charge : int
        Electric charge (0 for neutral, +1 for singly ionized, etc.)
    """
    charge = 0
    
    # Count + symbols
    if '+' in species_name:
        charge += species_name.count('+')
    
    # Check for explicit charge numbers
    if '2+' in species_name or '²⁺' in species_name:
        charge = 2
    elif '3+' in species_name or '³⁺' in species_name:
        charge = 3
    
    # Count - symbols
    if '-' in species_name:
        charge -= species_name.count('-')
    
    if '2-' in species_name or '²⁻' in species_name:
        charge = -2
    elif '3-' in species_name or '³⁻' in species_name:
        charge = -3
    
    return charge


# ============================================================================
# COMPOSITION OPERATIONS
# ============================================================================


def mix_compositions(comp1: GrainComposition, comp2: GrainComposition, 
                     weight1: float = 0.5) -> GrainComposition:
    """
    Create a mixture of two grain compositions.
    
    Parameters
    ----------
    comp1 : GrainComposition
        First composition
    comp2 : GrainComposition
        Second composition
    weight1 : float
        Weight of first composition (0-1)
    
    Returns
    -------
    mixture : GrainComposition
        Mixed composition
    """
    weight2 = 1.0 - weight1
    
    all_elements = set(comp1.elements.keys()) | set(comp2.elements.keys())
    
    mixed_elements = {}
    for elem in all_elements:
        mixed_elements[elem] = (
            weight1 * comp1.get_element_fraction(elem) +
            weight2 * comp2.get_element_fraction(elem)
        )
    
    # Normalize
    total = sum(mixed_elements.values())
    if total > 0:
        mixed_elements = {k: v/total for k, v in mixed_elements.items()}
    
    # Mix densities by weighted average
    mixed_density = weight1 * comp1.density + weight2 * comp2.density
    
    mixed_name = f"mixed_{comp1.grain_type}_{comp2.grain_type}"
    
    return GrainComposition(
        grain_type=mixed_name,
        elements=mixed_elements,
        density=mixed_density,
        composition_type=comp1.composition_type
    )
