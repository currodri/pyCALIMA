"""
DUST AND PAH EVOLUTION FRAMEWORK

This module provides a flexible class-based framework for modeling the evolution
of dust grain and PAH populations in different environments. The structure allows
for composable evolution processes that can be combined with various ODE solvers
to track how grain growth, shattering, coagulation, and other processes affect
dust and PAH distributions over time.

The main design features:
- GrainBin / PAHBin: represent discrete populations at specific sizes
- GrainPopulation / PAHPopulation: manage collections of bins
- EvolutionProcess: base class for all evolution mechanisms
- DustEvolutionState: snapshot of system state at a time
- DustEvolutionSystem: high-level orchestration and ODE solver interface

By: Curro Rodriguez (currodri@gmail.com)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Union
from copy import deepcopy


class GrainBin:
    """
    Represents a single grain size bin with fixed properties and evolving population.

    Attributes
    ----------
    grain_type : str
        Type of grain material ('silicate', 'graphite', etc.)
    radius_micron : float
        Grain radius in micrometers
    density : float
        Grain material density in g/cm³
    population : float
        Number density of grains in this bin [cm⁻³]
    metadata : dict
        Optional additional properties (mass, cross-section, etc.)
    """

    def __init__(
        self,
        grain_type: str,
        radius_micron: float,
        density: float,
        population: float = 0.0,
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize a grain size bin.

        Parameters
        ----------
        grain_type : str
            Type of grain material
        radius_micron : float
            Grain radius in micrometers
        density : float
            Grain material density in g/cm³
        population : float
            Initial number density [cm⁻³]
        metadata : dict, optional
            Additional properties (will be computed if not provided)
        """
        self.grain_type = grain_type
        self.radius_micron = radius_micron
        self.radius_cm = radius_micron * 1e-4
        self.density = density  # g/cm³
        self.population = float(population)

        # Compute basic grain properties
        self.mass_gram = (4.0 / 3.0) * np.pi * (self.radius_cm ** 3) * density
        self.cross_section_cm2 = np.pi * (self.radius_cm ** 2)
        self.volume_cm3 = (4.0 / 3.0) * np.pi * (self.radius_cm ** 3)

        # Optional metadata
        self.metadata = metadata if metadata else {}

    def get_mass_density(self) -> float:
        """Get mass density contribution of this bin [g/cm³]"""
        return self.population * self.mass_gram

    def get_number_density(self) -> float:
        """Get number density of grains in this bin [cm⁻³]"""
        return self.population

    def copy(self) -> "GrainBin":
        """Create a deep copy of this bin."""
        return GrainBin(
            grain_type=self.grain_type,
            radius_micron=self.radius_micron,
            density=self.density,
            population=self.population,
            metadata=deepcopy(self.metadata),
        )

    def __repr__(self) -> str:
        return (
            f"GrainBin({self.grain_type}, a={self.radius_micron:.2e}µm, "
            f"n={self.population:.2e}cm⁻³)"
        )


class PAHBin:
    """
    Represents a single PAH population with fixed carbon content and evolving abundance.

    Attributes
    ----------
    Nc : int
        Number of carbon atoms in the PAH molecule
    charge : int
        Current charge state of the PAH
    abundance : float
        Number density of this PAH species [cm⁻³]
    metadata : dict
        Optional properties (mass, radius, etc.)
    """

    def __init__(
        self,
        Nc: int,
        charge: int = 0,
        abundance: float = 0.0,
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize a PAH population bin.

        Parameters
        ----------
        Nc : int
            Number of carbon atoms in the PAH
        charge : int
            Charge state of the PAH (default 0)
        abundance : float
            Initial abundance [cm⁻³]
        metadata : dict, optional
            Additional properties
        """
        self.Nc = int(Nc)
        self.charge = int(charge)
        self.abundance = float(abundance)
        self.metadata = metadata if metadata else {}

    def get_abundance(self) -> float:
        """Get abundance of this PAH species [cm⁻³]"""
        return self.abundance

    def copy(self) -> "PAHBin":
        """Create a deep copy of this bin."""
        return PAHBin(
            Nc=self.Nc,
            charge=self.charge,
            abundance=self.abundance,
            metadata=deepcopy(self.metadata),
        )

    def __repr__(self) -> str:
        return f"PAHBin(C{self.Nc}⁺{self.charge}, n={self.abundance:.2e}cm⁻³)"


class GrainPopulation:
    """
    Manages a collection of grain bins representing the full size distribution.

    Attributes
    ----------
    bins : list of GrainBin
        All grain size bins in the population
    grain_type : str
        Type of grain (all bins should be the same type)
    """

    def __init__(self, grain_type: str, bins: Optional[List[GrainBin]] = None):
        """
        Initialize a grain population.

        Parameters
        ----------
        grain_type : str
            Type of grain ('silicate', 'graphite', etc.)
        bins : list of GrainBin, optional
            Initial bins; if None, an empty population is created
        """
        self.grain_type = grain_type
        self.bins = bins if bins is not None else []

    def add_bin(self, grain_bin: GrainBin) -> None:
        """Add a grain bin to the population."""
        if grain_bin.grain_type != self.grain_type:
            raise ValueError(
                f"Grain type mismatch: {grain_bin.grain_type} != {self.grain_type}"
            )
        self.bins.append(grain_bin)

    def remove_bin(self, index: int) -> None:
        """Remove a grain bin by index."""
        self.bins.pop(index)

    def get_total_mass_density(self) -> float:
        """Get total mass density of all grains [g/cm³]"""
        return sum(bin.get_mass_density() for bin in self.bins)

    def get_total_number_density(self) -> float:
        """Get total number density of all grains [cm⁻³]"""
        return sum(bin.get_number_density() for bin in self.bins)

    def get_populations(self) -> np.ndarray:
        """Get array of all grain populations."""
        return np.array([bin.population for bin in self.bins])

    def set_populations(self, populations: np.ndarray) -> None:
        """Set populations from an array (used by ODE solvers)."""
        if len(populations) != len(self.bins):
            raise ValueError("Population array length mismatch")
        for i, bin_ in enumerate(self.bins):
            bin_.population = float(populations[i])

    def copy(self) -> "GrainPopulation":
        """Create a deep copy of this population."""
        new_pop = GrainPopulation(self.grain_type)
        for bin_ in self.bins:
            new_pop.add_bin(bin_.copy())
        return new_pop

    @property
    def nbins(self) -> int:
        """Number of bins in this population."""
        return len(self.bins)

    def __repr__(self) -> str:
        return (
            f"GrainPopulation({self.grain_type}, "
            f"nbins={self.nbins}, M={self.get_total_mass_density():.2e}g/cm³)"
        )


class PAHPopulation:
    """
    Manages a collection of PAH bins representing different sizes (Nc values).

    Attributes
    ----------
    bins : list of PAHBin
        All PAH species in this population
    """

    def __init__(self, bins: Optional[List[PAHBin]] = None):
        """
        Initialize a PAH population.

        Parameters
        ----------
        bins : list of PAHBin, optional
            Initial bins; if None, empty population
        """
        self.bins = bins if bins is not None else []

    def add_bin(self, pah_bin: PAHBin) -> None:
        """Add a PAH bin to the population."""
        self.bins.append(pah_bin)

    def remove_bin(self, index: int) -> None:
        """Remove a PAH bin by index."""
        self.bins.pop(index)

    def get_total_abundance(self) -> float:
        """Get total PAH abundance [cm⁻³]"""
        return sum(bin.get_abundance() for bin in self.bins)

    def get_abundances(self) -> np.ndarray:
        """Get array of all PAH abundances."""
        return np.array([bin.abundance for bin in self.bins])

    def set_abundances(self, abundances: np.ndarray) -> None:
        """Set abundances from an array (used by ODE solvers)."""
        if len(abundances) != len(self.bins):
            raise ValueError("Abundance array length mismatch")
        for i, bin_ in enumerate(self.bins):
            bin_.abundance = float(abundances[i])

    def copy(self) -> "PAHPopulation":
        """Create a deep copy of this population."""
        new_pop = PAHPopulation()
        for bin_ in self.bins:
            new_pop.add_bin(bin_.copy())
        return new_pop

    @property
    def nbins(self) -> int:
        """Number of bins in this population."""
        return len(self.bins)

    def __repr__(self) -> str:
        return (
            f"PAHPopulation(nbins={self.nbins}, "
            f"total_abundance={self.get_total_abundance():.2e}cm⁻³)"
        )


@dataclass
class EnvironmentalConditions:
    """
    Stores physical conditions of the environment.

    Attributes
    ----------
    temperature_K : float
        Gas temperature [K]
    electron_density_cm3 : float
        Electron density [cm⁻³]
    hydrogen_density_cm3 : float
        Hydrogen density [cm⁻³]
    radiation_field : float
        Radiation field strength (in Habing units or similar)
    cosmic_ray_ionization : Optional[float]
        Cosmic ray ionization rate [s⁻¹ cm⁻³]
    custom_params : dict
        Any additional environmental parameters
    """

    temperature_K: float = 100.0
    electron_density_cm3: float = 1e-3
    hydrogen_density_cm3: float = 1.0
    radiation_field: float = 1.0
    cosmic_ray_ionization: Optional[float] = None
    custom_params: Dict = field(default_factory=dict)

    def copy(self) -> "EnvironmentalConditions":
        """Create a copy of environmental conditions."""
        return EnvironmentalConditions(
            temperature_K=self.temperature_K,
            electron_density_cm3=self.electron_density_cm3,
            hydrogen_density_cm3=self.hydrogen_density_cm3,
            radiation_field=self.radiation_field,
            cosmic_ray_ionization=self.cosmic_ray_ionization,
            custom_params=deepcopy(self.custom_params),
        )


class EvolutionProcess(ABC):
    """
    Abstract base class for all dust/PAH evolution mechanisms.

    Each process computes rates (dn/dt) for specific populations based on
    environmental conditions. Processes can act on grains, PAHs, or both.

    Subclasses must implement compute_rates() which returns rate arrays
    compatible with ODE solvers.
    """

    def __init__(self, name: str, process_type: str = "generic"):
        """
        Initialize an evolution process.

        Parameters
        ----------
        name : str
            Descriptive name for this process
        process_type : str
            Type of process ('grain', 'pah', 'coupling', 'ion', etc.)
        """
        self.name = name
        self.process_type = process_type
        self.enabled = True

    @abstractmethod
    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute evolution rates for this process.

        Parameters
        ----------
        state : DustEvolutionState
            Current state of the system
        grain_pop : GrainPopulation, optional
            Grain population (required for grain processes)
        pah_pop : PAHPopulation, optional
            PAH population (required for PAH processes)

        Returns
        -------
        rates : ndarray or tuple of ndarrays
            Rate array(s) matching the population structure:
            - For grain processes: array of shape (nbins_grain,)
            - For PAH processes: array of shape (nbins_pah,)
            - For coupling: tuple (grain_rates, pah_rates)
        """
        pass

    def disable(self) -> None:
        """Disable this process temporarily."""
        self.enabled = False

    def enable(self) -> None:
        """Enable this process."""
        self.enabled = True

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"EvolutionProcess({self.name}, {self.process_type}) [{status}]"


class DustEvolutionState:
    """
    Snapshot of the complete dust and PAH evolution system at a given time.

    This class handles:
    - Grain populations (multiple types if needed)
    - PAH populations
    - Environmental conditions
    - Metadata and evolution history

    It provides vectorized access to populations for ODE solver integration.
    """

    def __init__(
        self,
        time_year: float = 0.0,
        environmental_conditions: Optional[EnvironmentalConditions] = None,
    ):
        """
        Initialize a dust evolution state.

        Parameters
        ----------
        time_year : float
            Current time in years
        environmental_conditions : EnvironmentalConditions, optional
            Environmental conditions at this time
        """
        self.time_year = float(time_year)
        self.env = (
            environmental_conditions
            if environmental_conditions is not None
            else EnvironmentalConditions()
        )

        # Grain populations by type
        self.grain_populations: Dict[str, GrainPopulation] = {}

        # PAH population (typically a single population)
        self.pah_population: Optional[PAHPopulation] = None

        # Evolution history
        self.history: Dict[str, List[float]] = {
            "time": [],
            "total_grain_mass": [],
            "total_pah_abundance": [],
        }

    def add_grain_population(self, grain_type: str, population: GrainPopulation) -> None:
        """Add a grain population for a specific type."""
        if grain_type in self.grain_populations:
            raise ValueError(f"Grain population {grain_type} already exists")
        self.grain_populations[grain_type] = population

    def add_pah_population(self, population: PAHPopulation) -> None:
        """Add a PAH population."""
        self.pah_population = population

    def get_grain_population(self, grain_type: str) -> Optional[GrainPopulation]:
        """Get grain population by type."""
        return self.grain_populations.get(grain_type)

    def get_all_grain_types(self) -> List[str]:
        """Get all grain types in this state."""
        return list(self.grain_populations.keys())

    def get_total_grain_mass_density(self) -> float:
        """Get total mass density from all grain populations [g/cm³]"""
        return sum(pop.get_total_mass_density() for pop in self.grain_populations.values())

    def get_total_pah_abundance(self) -> float:
        """Get total PAH abundance [cm⁻³]"""
        if self.pah_population is None:
            return 0.0
        return self.pah_population.get_total_abundance()

    def get_state_vector(
        self, grain_types: Optional[List[str]] = None, include_pahs: bool = True
    ) -> np.ndarray:
        """
        Get complete state vector (populations array) for all populations.

        This is the interface for ODE solvers - returns a 1D array of all
        populations concatenated together.

        Parameters
        ----------
        grain_types : list of str, optional
            Which grain types to include (default: all)
        include_pahs : bool
            Whether to include PAH abundances

        Returns
        -------
        state_vector : ndarray
            Concatenated populations/abundances
        """
        if grain_types is None:
            grain_types = self.get_all_grain_types()

        vector = []

        # Add grain populations
        for gtype in grain_types:
            if gtype in self.grain_populations:
                vector.append(self.grain_populations[gtype].get_populations())

        # Add PAH abundances
        if include_pahs and self.pah_population is not None:
            vector.append(self.pah_population.get_abundances())

        if not vector:
            return np.array([])

        return np.concatenate(vector)

    def set_state_vector(
        self,
        state_vector: np.ndarray,
        grain_types: Optional[List[str]] = None,
        include_pahs: bool = True,
    ) -> None:
        """
        Set state from a complete state vector (inverse of get_state_vector).

        Used by ODE solvers to update all populations from solver output.

        Parameters
        ----------
        state_vector : ndarray
            Concatenated populations/abundances
        grain_types : list of str, optional
            Order of grain types (must match get_state_vector order)
        include_pahs : bool
            Whether PAHs are included in the vector
        """
        if grain_types is None:
            grain_types = self.get_all_grain_types()

        idx = 0

        # Set grain populations
        for gtype in grain_types:
            if gtype in self.grain_populations:
                pop = self.grain_populations[gtype]
                nbins = pop.nbins
                pop.set_populations(state_vector[idx : idx + nbins])
                idx += nbins

        # Set PAH abundances
        if include_pahs and self.pah_population is not None:
            nbins_pah = self.pah_population.nbins
            self.pah_population.set_abundances(state_vector[idx : idx + nbins_pah])

    def record_history(self) -> None:
        """Record current state in the evolution history."""
        self.history["time"].append(self.time_year)
        self.history["total_grain_mass"].append(self.get_total_grain_mass_density())
        self.history["total_pah_abundance"].append(self.get_total_pah_abundance())

    def copy(self) -> "DustEvolutionState":
        """Create a deep copy of this state."""
        new_state = DustEvolutionState(
            time_year=self.time_year,
            environmental_conditions=self.env.copy(),
        )

        # Copy grain populations
        for gtype, pop in self.grain_populations.items():
            new_state.add_grain_population(gtype, pop.copy())

        # Copy PAH population
        if self.pah_population is not None:
            new_state.add_pah_population(self.pah_population.copy())

        # Copy history
        new_state.history = deepcopy(self.history)

        return new_state

    def __repr__(self) -> str:
        ngrain_types = len(self.grain_populations)
        npah_bins = (
            self.pah_population.nbins if self.pah_population is not None else 0
        )
        return (
            f"DustEvolutionState(t={self.time_year:.2e}yr, "
            f"grain_types={ngrain_types}, pah_bins={npah_bins}, "
            f"T={self.env.temperature_K:.0f}K)"
        )


class DustEvolutionSystem:
    """
    High-level manager for dust and PAH evolution combining populations and processes.

    This class orchestrates:
    - Event-driven state updates
    - Process rate computations
    - ODE solver integration
    - Time stepping and history tracking

    It provides the interface needed to connect with scipy ODE solvers.
    """

    def __init__(self, initial_state: DustEvolutionState):
        """
        Initialize the evolution system.

        Parameters
        ----------
        initial_state : DustEvolutionState
            Initial state of the system
        """
        self.state = initial_state
        self.processes: List[EvolutionProcess] = []

        # For ODE solver tracking
        self.state_history: List[DustEvolutionState] = [initial_state.copy()]
        self.time_history: List[float] = [initial_state.time_year]

    def add_process(self, process: EvolutionProcess) -> None:
        """Add an evolution process to the system."""
        self.processes.append(process)

    def remove_process(self, name: str) -> None:
        """Remove a process by name."""
        self.processes = [p for p in self.processes if p.name != name]

    def get_process(self, name: str) -> Optional[EvolutionProcess]:
        """Get a process by name."""
        for p in self.processes:
            if p.name == name:
                return p
        return None

    def compute_total_rates(
        self,
        state: Optional[DustEvolutionState] = None,
        grain_types: Optional[List[str]] = None,
        include_pahs: bool = True,
    ) -> np.ndarray:
        """
        Compute total evolution rates from all enabled processes.

        Parameters
        ----------
        state : DustEvolutionState, optional
            State to compute rates for (default: current state)
        grain_types : list of str, optional
            Which grain types to include
        include_pahs : bool
            Whether to include PAH processes

        Returns
        -------
        rates : ndarray
            Total rates array matching state vector structure
        """
        if state is None:
            state = self.state

        if grain_types is None:
            grain_types = state.get_all_grain_types()

        # Initialize rate accumulators
        rate_dict = {}
        for gtype in grain_types:
            if gtype in state.grain_populations:
                rate_dict[gtype] = np.zeros(state.grain_populations[gtype].nbins)

        if include_pahs and state.pah_population is not None:
            rate_dict["pah"] = np.zeros(state.pah_population.nbins)

        # Accumulate rates from all enabled processes
        for process in self.processes:
            if not process.enabled:
                continue

            result = process.compute_rates(
                state,
                grain_pop=state.grain_populations.get(grain_types[0]) if grain_types else None,
                pah_pop=state.pah_population if include_pahs else None,
            )

            # Handle different return types
            if isinstance(result, tuple):
                # Coupling process returns (grain_rates, pah_rates)
                grain_rates, pah_rates = result
                if grain_rates is not None:
                    for i, gtype in enumerate(grain_types):
                        if gtype in rate_dict:
                            rate_dict[gtype] += grain_rates
                if pah_rates is not None and "pah" in rate_dict:
                    rate_dict["pah"] += pah_rates
            elif isinstance(result, np.ndarray):
                # Single array - determine which population it corresponds to
                if process.process_type == "pah":
                    rate_dict["pah"] += result
                elif process.process_type == "grain":
                    # Add to first grain type
                    if grain_types:
                        rate_dict[grain_types[0]] += result

        # Concatenate in same order as get_state_vector
        rates = []
        for gtype in grain_types:
            if gtype in rate_dict:
                rates.append(rate_dict[gtype])
        if include_pahs and "pah" in rate_dict:
            rates.append(rate_dict["pah"])

        if not rates:
            return np.array([])

        return np.concatenate(rates)

    def ode_derivative(
        self, time: float, state_vector: np.ndarray
    ) -> np.ndarray:
        """
        Compute derivatives for ODE solver (interface for scipy.integrate solvers).

        This function is called by ODE solvers like solve_ivp() or odeint().

        Parameters
        ----------
        time : float
            Current time
        state_vector : ndarray
            Current state vector (all populations)

        Returns
        -------
        derivatives : ndarray
            Time derivatives of state vector (dn/dt)
        """
        # Update state
        self.state.set_state_vector(state_vector)
        self.state.time_year = time

        # Compute rates
        rates = self.compute_total_rates()

        return rates

    def step_forward(
        self,
        time_step_year: float,
        grain_types: Optional[List[str]] = None,
    ) -> None:
        """
        Advance the system by a simple forward Euler step (for testing).

        For actual evolution, use an ODE solver via ode_derivative().

        Parameters
        ----------
        time_step_year : float
            Time step in years
        grain_types : list of str, optional
            Which grain types to advance
        """
        if grain_types is None:
            grain_types = self.state.get_all_grain_types()

        # Compute current rates
        rates = self.compute_total_rates(grain_types=grain_types)

        # Get current state
        current_state = self.state.get_state_vector(grain_types=grain_types)

        # Update with forward Euler
        new_state = current_state + rates * time_step_year

        # Clamp to non-negative
        new_state = np.maximum(new_state, 0.0)

        # Set updated state
        self.state.set_state_vector(new_state, grain_types=grain_types)
        self.state.time_year += time_step_year

        # Record history
        self.state.record_history()
        self.state_history.append(self.state.copy())
        self.time_history.append(self.state.time_year)

    def get_summary(self) -> str:
        """Get a summary of the system state."""
        lines = [
            "=== Dust Evolution System Summary ===",
            f"Current time: {self.state.time_year:.2e} years",
            f"Temperature: {self.state.env.temperature_K:.2e} K",
            f"Electron density: {self.state.env.electron_density_cm3:.2e} cm⁻³",
            f"Grain types: {', '.join(self.state.get_all_grain_types())}",
            f"Total grain mass density: {self.state.get_total_grain_mass_density():.2e} g/cm³",
        ]

        if self.state.pah_population is not None:
            lines.append(
                f"PAH population: {self.state.pah_population.nbins} bins, "
                f"total abundance: {self.state.get_total_pah_abundance():.2e} cm⁻³"
            )

        lines.append(f"Active processes: {len([p for p in self.processes if p.enabled])}/{len(self.processes)}")
        for p in self.processes:
            status = "✓" if p.enabled else "✗"
            lines.append(f"  {status} {p.name} ({p.process_type})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DustEvolutionSystem(nprocesses={len(self.processes)}, "
            f"state={repr(self.state)})"
        )


# Example concrete implementations of EvolutionProcess

class ThermalSputteringProcess(EvolutionProcess):
    """
    Simple example: thermal sputtering process for grains.

    This is a placeholder showing the pattern. Real implementations would
    use actual sputtering models from the dust_sputtering module.
    """

    def __init__(self, sputtering_rate_per_cm3: float = 1e-15):
        """
        Initialize thermal sputtering process.

        Parameters
        ----------
        sputtering_rate_per_cm3 : float
            Sputtering rate parameter
        """
        super().__init__("thermal_sputtering", process_type="grain")
        self.sputtering_rate = sputtering_rate_per_cm3

    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> np.ndarray:
        """
        Compute sputtering rates for grain bins.

        Placeholder: actual implementation would use physical models.
        """
        if grain_pop is None:
            return np.array([])

        # Simple placeholder: assume proportional to population and temperature
        rates = np.zeros(grain_pop.nbins)
        for i, bin_ in enumerate(grain_pop.bins):
            # Rate increases with temperature (very simplified)
            temp_factor = (state.env.temperature_K / 100.0) ** 1.5
            # Rate decreases with grain size (smaller grains sputter faster)
            size_factor = (0.1 / bin_.radius_micron) ** 0.5
            rates[i] = -bin_.population * self.sputtering_rate * temp_factor * size_factor

        return rates


class SimpleCoagulationProcess(EvolutionProcess):
    """
    Simple example: coagulation process combining small grains into larger ones.
    """

    def __init__(self, coagulation_efficiency: float = 0.1):
        """
        Initialize coagulation process.

        Parameters
        ----------
        coagulation_efficiency : float
            Coagulation efficiency parameter (0-1)
        """
        super().__init__("coagulation", process_type="grain")
        self.efficiency = coagulation_efficiency

    def compute_rates(
        self,
        state: "DustEvolutionState",
        grain_pop: Optional[GrainPopulation] = None,
        pah_pop: Optional[PAHPopulation] = None,
    ) -> np.ndarray:
        """
        Compute coagulation rates.

        Placeholder: actual implementation would use physical collision models.
        """
        if grain_pop is None:
            return np.array([])

        rates = np.zeros(grain_pop.nbins)

        # Very simplified placeholder: small grains coalesce into larger ones
        if grain_pop.nbins < 2:
            return rates

        # Rate of small grain destruction is proportional to their density
        # and the density of other grains to collide with
        for i in range(grain_pop.nbins - 1):
            for j in range(i + 1, grain_pop.nbins):
                # Collision rate proportional to both populations
                collision_rate = (
                    self.efficiency
                    * grain_pop.bins[i].population
                    * grain_pop.bins[j].population
                    * np.sqrt(state.env.temperature_K)
                )
                rates[i] -= collision_rate
                rates[j] -= collision_rate
                # Add to larger bins
                if j + 1 < grain_pop.nbins:
                    rates[j + 1] += collision_rate

        return rates
