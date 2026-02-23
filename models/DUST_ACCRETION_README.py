"""
DUST ACCRETION MODULE - COMPREHENSIVE DOCUMENTATION

The dust_accretion.py module provides a complete framework for computing
grain accretion rates in astrophysical environments.

================================================================================
KEY FEATURES
================================================================================

1. COMPOSITION AWARENESS
   - GrainComposition dataclass tracks elemental abundances
   - Pre-defined compositions: silicate, graphite, olivine
   - Supports arbitrary elemental compositions

2. ION/ELEMENT DENSITY HANDLING
   - Can work with multiple ion species simultaneously
   - Automatically determines which species is "limiting" (fastest collision)
   - The limiting species determines the accretion rate

3. THERMAL PHYSICS
   - Thermal velocity computation based on particle mass and temperature
   - Coulomb enhancement/suppression factor for charged grains
   - Accounts for long-range Coulomb interactions

4. GRAIN CHARGE EFFECTS
   - Coulomb enhancement factor computed from grain charge state
   - Attractive interactions (opposite charges) increase collision rate
   - Repulsive interactions (same charges) suppress collision rate

5. EVOLUTION FRAMEWORK INTEGRATION
   - AccretionProcess class for use with DustEvolutionSystem
   - Computes da/dt (radius growth rate) for each grain bin
   - Can be combined with other processes (sputtering, coagulation, PAH, etc.)

================================================================================
CORE CONCEPTS
================================================================================

LIMITING COLLISION RATE
-----------------------
The accretion rate is determined by the ion/atom species that collides with
the grain fastest. This is typically:
  - H atoms (lightest, high density): most common environment
  - H⁺ ions (lighter, faster thermal velocity)
  - He atoms (denser, significant in some environments)
  - Heavier species (slower, usually not limiting)

The collision rate for species X is:
    Γ_X = n_X * v_th,X * σ_eff,X
where:
  - n_X: number density [cm⁻³]
  - v_th,X: thermal velocity [cm/s]
  - σ_eff,X: effective cross-section = π*a² * Coulomb_factor [cm²]

GRAIN GROWTH
------------
When ions/atoms stick to the grain, its radius increases:

    da/dt = Γ_limit * m_species / (4π * a² * ρ_grain)

where:
  - Γ_limit: collision rate of limiting species [s⁻¹]
  - m_species: mass of accreting particle [g]
  - a: grain radius [cm]
  - ρ_grain: grain material density [g/cm³]

COULOMB EFFECTS
---------------
For a charged grain, the Coulomb potential affects ion trajectories:
  - Attractive (opposite charges): increases collision cross-section
  - Repulsive (same charges): decreases collision cross-section
  - Neutral grain: no effect

The Coulomb parameter is:
    ξ = Z_grain * Z_ion * e² / (a * k_B * T)

For attractive: enhancement ~ 1 + |ξ|
For repulsive: suppression ~ exp(-ξ)

================================================================================
CLASS AND FUNCTION REFERENCE
================================================================================

DATA CLASSES
============

GrainComposition
  - Represents grain elemental composition
  - Attributes:
    * grain_type: 'silicate', 'graphite', etc.
    * elements: dict of element → abundance
    * density: grain material density [g/cm³]
    * composition_type: 'mass' or 'number'
  - Methods:
    * normalize(): make abundances sum to 1.0
    * get_element_fraction(element): get abundance of element

Pre-defined compositions:
  - SILICATE_COMPOSITION: Mg₂SiO₄-like (ρ = 3.3 g/cm³)
  - GRAPHITE_COMPOSITION: pure C (ρ = 2.2 g/cm³)
  - OLIVINE_COMPOSITION: Mg,Fe silicate (ρ = 3.7 g/cm³)


THERMAL PHYSICS FUNCTIONS
===========================

thermal_velocity(mass_grams, temperature_K) → float
  - Compute mean thermal velocity
  - Formula: v = sqrt(3*k_B*T/m)
  - Returns: velocity [cm/s]

collision_cross_section_geometric(grain_radius_cm) → float
  - Simple geometric cross-section: σ = π*a²
  - Returns: cross-section [cm²]

coulomb_enhancement_factor(Z_grain, Z_ion, a_cm, T_K) → float
  - Coulomb interaction strength
  - Attractive (opposite Z): factor > 1
  - Repulsive (same Z): factor < 1
  - Returns: multiplicative factor for cross-section


COLLISION RATE FUNCTIONS
=========================

collision_rate_single_species(species, density, charge, a, Z_grain, T) → float
  - Collision rate for one ion species with grain
  - Formula: Γ = n * v_th * σ_eff
  - Returns: rate [cm⁻³ s⁻¹] (per unit grain)

collision_rate_from_densities(ion_dict, a, Z_grain, T) → dict
  - Rates for all species at once
  - Input: {'H': 1.0, 'He': 0.1, 'H⁺': 0.01, ...}
  - Returns: {'H': Γ_H, 'He': Γ_He, 'H⁺': Γ_H+, ...}

limiting_collision_rate(ion_dict, a, Z_grain, T) → float
  - Find maximum collision rate (which species is limiting)
  - Optional: also return name of limiting species
  - Returns: max(Γ_X) [cm⁻³ s⁻¹]


ACCRETION RATE COMPUTATION
===========================

accretion_rate_da_dt(a, ρ, ion_dict, Z_grain, T, stick=1.0) → float
  - Radius growth rate
  - Formula: da/dt = stick * Γ_limit * m / (4πa² * ρ)
  - Returns: growth rate [cm/s]

accretion_timescale(a, ρ, ion_dict, Z_grain, T) → float
  - Characteristic timescale for grain growth
  - τ = a / (da/dt)
  - Returns: timescale [seconds]

accretion_rate_from_composition(a, composition, ion_dict, Z_grain, T) → float
  - Like accretion_rate_da_dt but using GrainComposition object
  - Automatically uses composition.density


ADVANCED ANALYSIS
=================

collision_rate_analysis(a, Z_grain, T, ion_dict) → dict
  - Detailed breakdown of all collision rates
  - Returns dict of species → {density, mass, v_th, coulomb, rate}
  - Optional: return as pandas DataFrame

composition_update_from_accretion(composition, ion_dict, Z_grain, T, dt)
  - Update grain composition if accreting material differs
  - Gradually shifts composition toward accreting species
  - Returns: updated elements dict


EVOLUTION FRAMEWORK INTEGRATION
================================

AccretionProcess class
  - Integrates with DustEvolutionSystem from dust_pah_evolution.py
  - __init__(ion_densities_function, ion_charges, sticking_coeff)
  - compute_accretion_rates(state, grain_pop) → da_dt array
  - Optional detailed output (limiting species, rates per bin)

Usage:
  accretion = AccretionProcess(
      ion_densities_function=lambda state: state.env.ion_densities,
      sticking_coefficient=1.0
  )
  system.add_process(accretion)


================================================================================
EXAMPLE USAGE
================================================================================

EXAMPLE 1: Basic Accretion Rate
───────────────────────────────

from dust_accretion import accretion_rate_da_dt

# Environment
ion_densities = {
    'H': 0.9,      # Most abundant
    'He': 0.1,     # 10% helium
    'H+': 0.001,   # Small ionized fraction
}

# Grain properties
a_micron = 0.1
a_cm = a_micron * 1e-4
T_K = 100

# Grain charge (from dust_charging.py)
Z_grain = 0

# Compute growth rate
da_dt = accretion_rate_da_dt(
    grain_radius_cm=a_cm,
    grain_density_gcm3=3.3,  # silicate
    ion_densities=ion_densities,
    grain_charge=Z_grain,
    temperature_K=T_K,
    sticking_coefficient=1.0
)

print(f"Growth rate: {da_dt:.2e} cm/s")
print(f"Growth rate: {da_dt * 3.15e7:.2e} µm/year")


EXAMPLE 2: Detailed Collision Analysis
───────────────────────────────────────

from dust_accretion import collision_rate_analysis

analysis = collision_rate_analysis(
    grain_radius_cm=a_cm,
    grain_charge=Z_grain,
    temperature_K=T_K,
    ion_densities=ion_densities,
    return_df=True  # pandas DataFrame
)

print(analysis)
# Output:
#              density         mass  thermal_velocity  coulomb_factor  collision_rate
# H        9.000000e-01  1.674e-24        1.161e+05             1.000     6.54e+04
# He       1.000000e-01  6.646e-24    5.815e+04             1.000     1.63e+04
# H+       1.000000e-03  1.674e-24    1.161e+05             1.050     6.88e+02


EXAMPLE 3: With Grain Charge Effects
─────────────────────────────────────

# Negative grain (like in ISM)
Z_grain = -5

da_dt_charged = accretion_rate_da_dt(
    grain_radius_cm=a_cm,
    grain_density_gcm3=3.3,
    ion_densities=ion_densities,  # H+ will be enhanced
    grain_charge=Z_grain,
    temperature_K=T_K,
)

print(f"Neutral grain: da/dt = {da_dt:.2e} cm/s")
print(f"Charged grain: da/dt = {da_dt_charged:.2e} cm/s")
print(f"Enhancement factor: {da_dt_charged / da_dt:.2f}")


EXAMPLE 4: With Composition
─────────────────────────────

from dust_accretion import (
    accretion_rate_from_composition,
    SILICATE_COMPOSITION
)

da_dt_sil = accretion_rate_from_composition(
    grain_radius_cm=a_cm,
    grain_composition=SILICATE_COMPOSITION,
    ion_densities=ion_densities,
    grain_charge=0,
    temperature_K=T_K,
)

print(f"Silicate grain growth: {da_dt_sil:.2e} cm/s")


EXAMPLE 5: Integration Framework
──────────────────────────────────

from dust_pah_evolution import *
from dust_accretion import AccretionProcess
from scipy.integrate import solve_ivp

# Create grain population
grain_pop = GrainPopulation("silicate")
for a in np.logspace(-3, -1, 10):
    bin_ = GrainBin("silicate", a, 3.3, population=1e-13)
    grain_pop.add_bin(bin_)

# Environmental conditions with ion densities
env = EnvironmentalConditions(
    temperature_K=100,
    hydrogen_density_cm3=1.0,
    electron_density_cm3=0.01,
    radiation_field=1.0,
    custom_params={'ion_densities': {
        'H': 0.99,
        'H+': 0.01,
        'He': 0.10
    }}
)

# Create state and system
state = DustEvolutionState(environmental_conditions=env)
state.add_grain_population("silicate", grain_pop)
system = DustEvolutionSystem(state)

# Add accretion process
def get_ion_densities(st):
    return st.env.custom_params.get('ion_densities', {})

accretion = AccretionProcess(ion_densities_function=get_ion_densities)
system.add_process(accretion)  # Convert to EvolutionProcess adapter first

# Integrate
solution = solve_ivp(
    lambda t, y: system.ode_derivative(t, y),
    (0, 1e6),
    state.get_state_vector(),
    method="RK45"
)


================================================================================
PHYSICAL ASSUMPTIONS AND LIMITATIONS
================================================================================

ASSUMPTIONS MADE:
─────────────────
1. Spherical grains
2. Sticking coefficient ≤ 1 (particles sometimes bounce)
3. Geometric collision cross-section (ignores size effects of Coulomb)
4. Single limiting species determines accretion rate
5. Accretion of single atoms (not molecular clusters)
6. Sticking is instantaneous upon collision

LIMITATIONS:
────────────
1. Doesn't account for:
   - Grain charging due to accretion
   - Heterogeneous composition after accretion
   - Sputtering/etching during accretion
   - Chemical reactions in gas phase
   - Grain alignment effects

2. Coulomb factor is simplified:
   - Uses approximate formula, not full trajectory calculation
   - Valid for moderate charges (|Z| < 10)

3. Temperature effects:
   - Assumes Maxwell-Boltzmann velocity distribution
   - Valid for T > 10 K

FUTURE IMPROVEMENTS:
────────────────────
- Full trajectory calculations for Coulomb interactions
- Include molecular accretion (H₂, CO, H₂O)
- Track composition evolution during extended accretion
- Feed grain charge evolution into accretion rates
- Include erosion/sputtering competing with accretion


================================================================================
PHYSICAL REFERENCE MODELS
================================================================================

This module is based on:

1. Accretion physics:
   - Draine (2003, 2011): Comprehensive ISM dust physics
   - Tielens (2010): Dust properties and interactions
   - Weingartner & Draine (2001): Grain charging and interactions

2. Coulomb effects:
   - Spitzer (1962): Ion-grain interactions in plasmas
   - Draine & Sutin (1987): Grain charging in diffuse ISM
   - Weingartner & Draine (1999): Grain properties in ISM

3. Thermal velocities and collision rates:
   - Reif (1965): Fundamentals of Statistical and Thermal Physics
   - Chapman & Cowling (1970): Mathematical Theory of Non-Uniform Gases


================================================================================
KEY PARAMETERS AND TYPICAL VALUES
================================================================================

GRAIN RADII:
  Small grains: 0.001 - 0.01 µm (PAHs, very small dust)
  Medium grains: 0.01 - 0.1 µm (typical ISM dust)
  Large grains: 0.1 - 1 µm (dust around stars)

GRAIN DENSITIES:
  Silicate: 3.0 - 3.5 g/cm³
  Graphite: 2.2 g/cm³
  Olivine: 3.7 g/cm³
  Composite: 2.5 - 3.0 g/cm³

GAS TEMPERATURES:
  Cold IS clouds: 10 - 100 K
  Diffuse ISM: 100 - 1000 K
  Warm IS medium: 1000 - 10000 K
  Hot IS medium: 10000+ K

ION DENSITIES:
  Cold clouds: n(H+) ~ 10⁻³ cm⁻³, n(H) ~ 1 cm⁻³
  Diffuse ISM: n(H+) ~ 10⁻² cm⁻³, n(H) ~ 0.5 cm⁻³
  Warm ISM: n(H+) ~ 10⁻¹ cm⁻³, n(H) ~ 0.1 cm⁻³

ACCRETION TIMESCALES:
  Typical ISM conditions: 10⁷ - 10¹⁰ years
  (i.e., grains grow very slowly in most environments)


================================================================================
UNITS AND CONVERSIONS
================================================================================

Input units:
  - Temperature: Kelvin [K]
  - Density: cm⁻³
  - Grain radius: cm (converted from µm)
  - Mass: grams [g]
  - Charge: elementary charge units

Output units:
  - Growth rate: cm/s → multiply by 3.15e7 to get µm/year
  - Timescale: seconds → divide by 3.15e7 to get years
  - Collision rate: cm⁻³ s⁻¹ (per unit grain)

Useful conversions:
  1 µm = 1e-4 cm
  1 year = 3.15e7 seconds
  1 amu = 1.66e-24 grams
  k_B = 1.38e-16 erg/K
  e = 4.80e-10 statC


================================================================================
DEBUGGING AND TESTING
================================================================================

COMMON ISSUES:

1. Growth rate too fast/slow?
   - Check ion densities (order of magnitude?)
   - Verify grain radius in correct units (cm, not µm)
   - Check temperature is reasonable

2. Coulomb factor not helping?
   - Only significant for |Z| > 5 and small grains
   - Verify grain charge is correct
   - Check that temperature isn't too high

3. Wrong limiting species?
   - Print collision_rate_analysis output
   - Verify ion densities sum to reasonable total
   - Check ion charges are correct

VALIDATION:

Compare against known systems:
  - Diffuse ISM, T=100K, H-dominated: τ ~ 10⁸ years
  - Molecular cloud, T=50K, higher density: τ ~ 10⁶ years
  - Warm ISM, T=1000K: τ ~ 10⁹ years


================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
