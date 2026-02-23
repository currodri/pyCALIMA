"""
INTEGRATION GUIDE: Connecting Existing Models to Evolution Framework

This module provides guidelines and helper functions for integrating the existing
models (from dust_model.py, dust_sputtering.py, PAHs_model.py, etc.) into the
dust_pah_evolution framework.

The main strategy:
1. Wrap existing model functions in EvolutionProcess subclasses
2. Pass environmental conditions from DustEvolutionState to model functions
3. Convert bin populations to/from the formats expected by each model
4. Implement rate calculations that respect physical constraints

Key integration patterns are shown below with templates and examples.

By: Curro Rodriguez (currodri@gmail.com)
"""

import numpy as np
from typing import Optional, Tuple
from dust_pah_evolution import (
    EvolutionProcess,
    GrainPopulation,
    PAHPopulation,
    DustEvolutionState,
)


# ============================================================================
# TEMPLATE 1: Wrapping grain sputtering models
# ============================================================================


class SputteringProcessTemplate(EvolutionProcess):
    """
    Template for integrating thermal/non-thermal sputtering models.

    Integration points with existing code:
    - dust_sputtering.sputtering_yield()
    - dust_model.Tielens_rate()
    - dust_model.thermal_spu_nozawa06 (coefficient data)

    The workflow is:
    1. For each grain bin, compute sputtering yield based on size and composition
    2. Get sputtering rate from temperature-dependent fits
    3. Compute mass/radius removal rate
    4. Track destruction time and bin population changes
    """

    def __init__(
        self,
        name: str = "sputtering",
        composition: str = "silicate",
        sputtering_model: str = "Nozawa2006",
    ):
        """
        Initialize sputtering process.

        Parameters
        ----------
        name : str
            Name of this process instance
        composition : str
            Grain composition ('silicate', 'graphite', etc.)
        sputtering_model : str
            Which model to use ('Nozawa2006', 'Tielens1994', etc.)
        """
        super().__init__(name, process_type="grain")
        self.composition = composition
        self.sputtering_model = sputtering_model

    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> np.ndarray:
        """
        Compute sputtering destruction rates.

        The typical workflow:
        1. Compute sputtering timescale t_sputter from dust_model.t_sputtering()
        2. Rate = -n / t_sputter (simple exponential decay assumption)
        3. Account for size/temperature dependence

        For grains that completely sputter away, track partial destruction
        by monitoring radius change and moving material to smaller bins.

        Returns
        -------
        rates : ndarray
            Destruction rates dn/dt [cm⁻³ s⁻¹] for each bin
        """
        if grain_pop is None:
            return np.array([])

        rates = np.zeros(grain_pop.nbins)

        # Example: get sputtering rates from temperature-dependent formula
        # This would be:
        # from dust_sputtering import sputtering_yield, ...
        # from dust_model import Tielens_rate, thermal_spu_nozawa06, ...

        for i, bin_ in enumerate(grain_pop.bins):
            # 1. Get sputtering timescale based on environment
            # t_sputter = compute_sputtering_timescale(
            #     grain_radius=bin_.radius_cm,
            #     temperature=state.env.temperature_K,
            #     H_density=state.env.hydrogen_density_cm3,
            #     composition=self.composition
            # )

            # 2. Compute destruction rate
            # rates[i] = -bin_.population / t_sputter

            # Simple placeholder for now:
            # Sputtering accelerates with temperature
            T_factor = (state.env.temperature_K / 1000.0) ** 2
            rates[i] = -(bin_.population / 1e6) * T_factor  # Timescale in years

        return rates


# ============================================================================
# TEMPLATE 2: Wrapping coagulation models
# ============================================================================


class CoagulationProcessTemplate(EvolutionProcess):
    """
    Template for grain coagulation (sticking collisions).

    Integration points with existing code:
    - dust_model.t_coagulation()
    - dust_model.grain_charge_dist()
    - dust_model.grain_mean_charge()

    The workflow:
    1. Compute collision rates between grain populations
    2. Account for charge effects (Coulomb enhancement/suppression)
    3. Track mass conservation when grains merge
    4. Split material between merging and fragmentation

    Complexity: typically requires matrix formulation (collision kernel).
    For N bins, you have N×N collisions creating effects in multiple bins.
    """

    def __init__(
        self,
        name: str = "coagulation",
        stick_probability: float = 1.0,
        include_coulomb: bool = True,
    ):
        """
        Initialize coagulation process.

        Parameters
        ----------
        name : str
            Process name
        stick_probability : float
            Probability of sticking (0-1) for collisions
        include_coulomb : bool
            Whether to include Coulomb enhancement effects
        """
        super().__init__(name, process_type="grain")
        self.stick_probability = stick_probability
        self.include_coulomb = include_coulomb

    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> np.ndarray:
        """
        Compute coagulation rates (complex - requires collision kernel).

        IMPORTANT: Coagulation is inherently a multi-bin process:
        - Smaller bins collide to form larger bins
        - Requires tracking of mass/number conservation
        - Results in an N×N collision matrix (expensive for large N)

        Strategy:
        1. Compute relative velocities for all grain pairs
        2. Compute collision cross-section (including Coulomb effects)
        3. Build collision kernel K_ij = n_i * n_j * (collision rate)
        4. Account for mass conservation when grains merge
        5. Update population rates accordingly

        Returns
        -------
        rates : ndarray
            Rate of change dn/dt for each bin from coagulation
        """
        if grain_pop is None:
            return np.array([])

        rates = np.zeros(grain_pop.nbins)

        # from dust_model import relative_velocity, t_coagulation
        # from dust_charging import equilibrium_charge_for_grain

        # Placeholder showing the pattern:
        for i in range(grain_pop.nbins):
            for j in range(grain_pop.nbins):
                if i == j:
                    # Same-size collision: particles are destroyed
                    # (combine into larger size)
                    continue
                # Relative velocity
                # v_rel = relative_velocity(...)

                # Collision cross-section
                # sigma_coll = pi * (a_i + a_j)^2

                # Collision rate (accounting for Coulomb effects if enabled)
                # if self.include_coulomb:
                #     coulomb_factor = cmp_D_WD99(...)
                # else:
                #     coulomb_factor = 1.0

                # collision_rate = sigma_coll * v_rel * coulomb_factor

                pass

        # This is necessarily more complex than other processes
        # A complete implementation would need to handle:
        # - Output size distribution after collision
        # - Mass conservation
        # - Multi-bin scattering matrix approach

        return rates


# ============================================================================
# TEMPLATE 3: Wrapping grain charging models
# ============================================================================


class ChargingEquilibriumProcess(EvolutionProcess):
    """
    Template for computing equilibrium grain charge distributions.

    Integration points with existing code:
    - dust_charging.equilibrium_charge_for_grain()
    - dust_charging.charge_equilibrium_from_rates()
    - dust_model.grain_mean_charge()
    - dust_model.grain_charge_dist()

    This process is somewhat different: it doesn't directly change populations,
    but computes the charge state of grains which affects other processes
    (charging affects collision cross-sections, photoelectric heating, etc.).

    Workflow:
    1. For each grain bin, compute equilibrium charge distribution
    2. Store mean charge and charge distribution in bin metadata
    3. Other processes can access this information
    """

    def __init__(self, name: str = "charging_equilibrium", radiation_model: str = "Mathis"):
        """
        Initialize charging process.

        Parameters
        ----------
        name : str
            Process name
        radiation_model : str
            Radiation field model ('Mathis', 'Draine', etc.)
        """
        super().__init__(name, process_type="grain")
        self.radiation_model = radiation_model

    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> np.ndarray:
        """
        Compute equilibrium charges (no population change - returns zeros).

        Instead, store charge info in grain metadata for use by other processes.

        The typical pattern:
        1. Call dust_charging.equilibrium_charge_for_grain() for each bin
        2. Store Zmean, charge distribution in bin.metadata['charge_dist']
        3. Return zero rates (this process doesn't change populations directly)

        Returns
        -------
        rates : ndarray
            All zeros (charging equilibrium doesn't change bin populations)
        """
        if grain_pop is None:
            return np.array([])

        rates = np.zeros(grain_pop.nbins)

        # from dust_charging import equilibrium_charge_for_grain

        for i, bin_ in enumerate(grain_pop.bins):
            # Compute equilibrium charge for this grain size
            # Zs, P, rates_charging, Zmean, Zsigma = equilibrium_charge_for_grain(
            #     G0=state.env.radiation_field,
            #     ne=state.env.electron_density_cm3,
            #     T=state.env.temperature_K,
            #     grain_type=bin_.grain_type,
            #     a_micron=bin_.radius_micron,
            #     radiation_model=self.radiation_model,
            #     ...
            # )

            # Store in metadata for use by other processes
            # bin_.metadata['charge_dist'] = P
            # bin_.metadata['Zmean'] = Zmean
            # bin_.metadata['Zsigma'] = Zsigma

            pass

        # Return zero rates - this is information-gathering, not population-changing
        return rates


# ============================================================================
# TEMPLATE 4: Wrapping PAH photodissociation models
# ============================================================================


class PAHPhotoDestructionProcess(EvolutionProcess):
    """
    Template for PAH destruction by UV photons.

    Integration points with existing code:
    - PAHs_model.py (various dissociation rate functions)
    - References: Galliano, Allain, Micelotta models

    Workflow:
    1. Look up dissociation rate for each PAH size (Nc value)
    2. Rate depends on radiation field, temperature, grain properties
    3. Implement as exponential decay: n(t) = n0 * exp(-t/tau_dissoc)
    4. For fragmentation, track where destroyed carbon goes
    """

    def __init__(
        self,
        name: str = "pah_photodissociation",
        dissociation_model: str = "Galliano2008",
    ):
        """
        Initialize PAH dissociation process.

        Parameters
        ----------
        name : str
            Process name
        dissociation_model : str
            Which model to use ('Galliano2008', 'Allain1996', 'Micelotta2010', etc.)
        """
        super().__init__(name, process_type="pah")
        self.dissociation_model = dissociation_model

    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> np.ndarray:
        """
        Compute PAH destruction rates.

        Pattern:
        1. For each PAH bin (identified by Nc), compute dissociation rate
        2. Rate is typically: tau_dissoc(Nc, G0, T) from model data
        3. dn/dt = -n / tau_dissoc
        4. Can optionally track fragmentation into smaller PAHs

        Returns
        -------
        rates : ndarray
            Destruction rates dn/dt [cm⁻³ s⁻¹] for each PAH size
        """
        if pah_pop is None:
            return np.array([])

        rates = np.zeros(pah_pop.nbins)

        # from PAHs_model import size_from_Nc, mass_from_Nc, ...

        for i, bin_ in enumerate(pah_pop.bins):
            # Get PAH radius from Nc
            # a_angstrom = size_from_Nc(bin_.Nc)

            # Compute dissociation timescale
            # tau_diss = compute_dissociation_timescale(
            #     Nc=bin_.Nc,
            #     G0=state.env.radiation_field,
            #     T=state.env.temperature_K,
            #     model=self.dissociation_model
            # )

            # Simple exponential decay
            # rates[i] = -bin_.abundance / tau_diss

            # More sophisticated: track fragmentation
            # - Dissociation might produce smaller PAHs
            # - Would need to distribute destroyed PAH carbon to other bins

            pass

        return rates


# ============================================================================
# TEMPLATE 5: Grain-PAH coupling process (e.g., H2 formation, accretion)
# ============================================================================


class GrainPAHCouplingTemplate(EvolutionProcess):
    """
    Template for processes coupling grain and PAH evolution.

    Examples:
    - H2 formation on grain surfaces and PAHs
    - Destruction of PAHs by collisions with grains
    - Carbon accretion onto/off grains

    This process returns a tuple (grain_rates, pah_rates) affecting both populations.

    Integration points:
    - dust_h2_formation.py (grain surface H2 formation rates)
    - dust_charging.py (collision cross-sections)
    """

    def __init__(self, name: str = "grain_pah_coupling"):
        """Initialize coupling process."""
        super().__init__(name, process_type="coupling")

    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rates affecting both grain and PAH populations.

        Returns
        -------
        grain_rates : ndarray or None
            Rates for grain population
        pah_rates : ndarray or None
            Rates for PAH population
        """
        grain_rates = None
        pah_rates = None

        if grain_pop is not None:
            grain_rates = np.zeros(grain_pop.nbins)
            # Compute grain rate changes due to PAH interactions
            # e.g., carbon from destroyed PAHs deposited on grains

        if pah_pop is not None:
            pah_rates = np.zeros(pah_pop.nbins)
            # Compute PAH rate changes
            # e.g., destruction by collisions with grains

        return grain_rates, pah_rates


# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================


def convert_to_bin_distributions(
    population_array: np.ndarray, grain_type: str, size_grid: np.ndarray
) -> GrainPopulation:
    """
    Convert a size distribution array to a GrainPopulation object.

    Useful for initializing populations from observational data or
    output from other modeling codes.

    Parameters
    ----------
    population_array : ndarray
        Population densities at each size point [cm⁻³]
    grain_type : str
        Type of grain ('silicate', 'graphite', etc.)
    size_grid : ndarray
        Grain sizes in micrometers

    Returns
    -------
    population : GrainPopulation
        Discretized population in bins
    """
    if len(population_array) != len(size_grid):
        raise ValueError("Array and size grid must have same length")

    pop = GrainPopulation(grain_type)

    # Density of silicate
    density_map = {"silicate": 3.3, "graphite": 2.2}
    density = density_map.get(grain_type, 2.5)

    for size, n in zip(size_grid, population_array):
        from dust_pah_evolution import GrainBin
        
        bin_ = GrainBin(grain_type, size, density, population=n)
        pop.add_bin(bin_)

    return pop


def compute_total_carbon_content(pah_pop: PAHPopulation) -> float:
    """
    Compute total carbon content from PAH population.

    Parameters
    ----------
    pah_pop : PAHPopulation
        PAH population

    Returns
    -------
    carbon_content : float
        Total carbon [cm⁻³ or scaled by unit]
    """
    total = 0.0
    for bin_ in pah_pop.bins:
        total += bin_.Nc * bin_.abundance

    return total


def apply_survival_probability(
    process: EvolutionProcess, survival_prob: float
) -> EvolutionProcess:
    """
    Wrapper to reduce process effects by a survival probability.

    Useful for testing process interactions and relative importance.

    Parameters
    ----------
    process : EvolutionProcess
        The process to wrap
    survival_prob : float
        Probability of process occurring (0-1)

    Returns
    -------
    wrapped_process : EvolutionProcess
        Same process but with reduced rates
    """
    original_compute_rates = process.compute_rates

    def wrapped_compute_rates(state, grain_pop=None, pah_pop=None):
        rates = original_compute_rates(state, grain_pop, pah_pop)
        return rates * survival_prob

    process.compute_rates = wrapped_compute_rates
    return process


# ============================================================================
# CHECKLIST FOR INTEGRATING A NEW PROCESS
# ============================================================================

"""
When integrating a new evolution process:

1. IDENTIFY the physical model
   - Locate the function in the existing codebase
   - Understand inputs (size, T, n_H, composition, etc.)
   - Understand outputs (rates, timescales, etc.)

2. CREATE a new EvolutionProcess subclass
   - Inherit from EvolutionProcess
   - Choose process_type ('grain', 'pah', 'coupling', 'ion', 'electron', etc.)
   - Implement compute_rates()

3. IMPLEMENT compute_rates()
   - Extract relevant state variables from DustEvolutionState
   - For each bin, compute individual rates
   - Return array(s) of dn/dt values
   - Units: typically [cm⁻³ s⁻¹] for number density rates
   - Respect mass and charge conservation

4. TEST the process
   - Check units (especially time conversions: years → seconds)
   - Verify against known timescales
   - Test with simple cases (e.g., single bin)
   - Verify rate signs (destruction should be negative)

5. VALIDATE physics
   - Compare results against observational timescales
   - Check that processes compete appropriately
   - Verify that state remains physical (n ≥ 0, etc.)

6. DOCUMENT
   - Add docstring explaining physical model
   - Cite paper or reference
   - List parameters with units
   - Give example usage

7. OPTIMIZE (if needed)
   - Profile for bottlenecks
   - Consider caching precomputed values
   - Use vectorized operations (numpy, not loops)

8. INTEGRATE into system
   - Add to DustEvolutionSystem via add_process()
   - Test with other processes enabled/disabled
   - Check for unexpected interactions
"""

if __name__ == "__main__":
    print("Integration guide module loaded.")
    print("\nKey templates provided:")
    print("  - SputteringProcessTemplate")
    print("  - CoagulationProcessTemplate")
    print("  - ChargingEquilibriumProcess")
    print("  - PAHPhotoDestructionProcess")
    print("  - GrainPAHCouplingTemplate")
    print("\nHelper functions:")
    print("  - convert_to_bin_distributions()")
    print("  - compute_total_carbon_content()")
    print("  - apply_survival_probability()")
