"""
EXAMPLE: DUST AND PAH EVOLUTION WITH ODE SOLVER

This example demonstrates how to use the dust_pah_evolution framework
to set up a complete dust evolution system and integrate it forward in time
using scipy's ODE solvers.

The example includes:
1. Creating grain and PAH populations with realistic size distributions
2. Setting up environmental conditions
3. Defining custom evolution processes
4. Integrating with scipy.integrate.solve_ivp
5. Analyzing and plotting results

By: Curro Rodriguez (currodri@gmail.com)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dust_pah_evolution import (
    GrainBin,
    PAHBin,
    GrainPopulation,
    PAHPopulation,
    EnvironmentalConditions,
    DustEvolutionState,
    DustEvolutionSystem,
    EvolutionProcess,
)


# ============================================================================
# CUSTOM PROCESSES - implement actual dust models here
# ============================================================================


class GrainGrowthProcess(EvolutionProcess):
    """
    Example grain growth process (e.g., accretion from local gas).

    This is a simple placeholder showing how to implement custom processes.
    In practice, you would integrate your growth models from dust_model.py
    """

    def __init__(self, growth_rate_cm_per_year: float = 1e-13):
        super().__init__("grain_growth", process_type="grain")
        self.growth_rate = growth_rate_cm_per_year

    def compute_rates(self, state, grain_pop=None, pah_pop=None):
        """
        Compute grain growth rates.

        Growth rate increases with density and is size-dependent.
        """
        if grain_pop is None:
            return np.array([])

        rates = np.zeros(grain_pop.nbins)

        # Growth by accretion: dn/dt depends on available material
        # This is a simplified model
        for i, bin_ in enumerate(grain_pop.bins):
            # Accretion rate increases with hydrogen density
            H_density = state.env.hydrogen_density_cm3
            # Rate is proportional to available surface area
            surface_area = bin_.cross_section_cm2
            # Result: grains grow slightly
            growth_rate = self.growth_rate * H_density * surface_area
            # For simplicity: shift material to larger bins
            if i < grain_pop.nbins - 1:
                rates[i] -= growth_rate * bin_.population
                rates[i + 1] += growth_rate * bin_.population * 0.9  # Some loss

        return rates


class SimpleShatteringProcess(EvolutionProcess):
    """
    Example shattering process where collisions break grains apart.

    In practice, integrate shattering timescales from dust_model.t_shattering()
    """

    def __init__(self, shattering_rate: float = 1e-12):
        super().__init__("shattering", process_type="grain")
        self.rate = shattering_rate

    def compute_rates(self, state, grain_pop=None, pah_pop=None):
        """Compute shattering destruction rates."""
        if grain_pop is None:
            return np.array([])

        rates = np.zeros(grain_pop.nbins)

        # Shattering depends on collision rates (function of density and temperature)
        collision_rate = (
            self.rate * state.env.hydrogen_density_cm3 * 
            np.sqrt(state.env.temperature_K)
        )

        for i, bin_ in enumerate(grain_pop.bins):
            # Larger grains shatter into multiple smaller pieces
            shatter_loss = collision_rate * bin_.population
            rates[i] -= shatter_loss

            # Fragments go to smaller sizes (simplified: just destroy for now)
            # In a real implementation, distribute to appropriate smaller bins

        return rates


class PAHDestructionProcess(EvolutionProcess):
    """
    Example PAH destruction by UV photodissociation.

    Would integrate models from PAHs_model.py and dust_photoelectric_heating.py
    """

    def __init__(self, uv_destruction_timescale_year: float = 1e5):
        super().__init__("pah_destruction", process_type="pah")
        self.destruction_timescale = uv_destruction_timescale_year

    def compute_rates(self, state, grain_pop=None, pah_pop=None):
        """Compute PAH destruction rates by UV photons."""
        if pah_pop is None:
            return np.array([])

        rates = np.zeros(pah_pop.nbins)

        # Destruction timescale decreases with radiation field
        # (More UV = faster destruction)
        destruction_rate = 1.0 / (
            self.destruction_timescale / state.env.radiation_field
        )

        for i, bin_ in enumerate(pah_pop.bins):
            # Smaller PAHs are destroyed faster
            size_factor = float(bin_.Nc) / 100.0  # Simple size dependence
            rates[i] = -bin_.abundance * destruction_rate * size_factor

        return rates


# ============================================================================
# EXAMPLE 1: Simple silicate grain evolution
# ============================================================================


def example_basic_grain_evolution():
    """
    Basic example: evolve silicate grain distribution with growth and shattering.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Grain Evolution (Silicate)")
    print("=" * 70)

    # Create initial grain population - silicate
    grain_pop_sil = GrainPopulation("silicate")

    # Add grain bins at logarithmic spacing
    radii_micron = np.logspace(-3, -1, 10)  # 0.001 to 0.1 micron
    density_silicate = 3.3  # g/cm³

    for a in radii_micron:
        # Initial populations (number density in cm⁻³)
        # Typically determined from observational constraints
        n0 = 1e-12 / len(radii_micron)  # Distribute total density
        bin_ = GrainBin("silicate", a, density_silicate, population=n0)
        grain_pop_sil.add_bin(bin_)

    print(f"Created grain population: {grain_pop_sil}")
    print(f"  Initial total mass: {grain_pop_sil.get_total_mass_density():.2e} g/cm³")

    # Create initial state
    env = EnvironmentalConditions(
        temperature_K=100.0,
        electron_density_cm3=0.03,
        hydrogen_density_cm3=1.0,
        radiation_field=1.0,
    )

    state = DustEvolutionState(time_year=0.0, environmental_conditions=env)
    state.add_grain_population("silicate", grain_pop_sil)

    # Create evolution system
    system = DustEvolutionSystem(state)

    # Add processes
    system.add_process(GrainGrowthProcess(growth_rate_cm_per_year=1e-14))
    system.add_process(SimpleShatteringProcess(shattering_rate=1e-13))

    print(system.get_summary())

    # Solve ODE from t=0 to t=1Myr
    t_span = (0, 1e6)  # 0 to 1 million years
    t_eval = np.logspace(0, 6, 100)  # Evaluation times

    print("\nIntegrating ODE system...")

    # Create wrapper for ODE solver
    def ode_func(t, y):
        return system.ode_derivative(t, y)

    # Initial state vector
    y0 = state.get_state_vector()

    # Solve
    solution = solve_ivp(
        ode_func,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
        max_step=1e4,  # Max 10k years per step
    )

    print(f"ODE integration completed: {solution.status}")
    print(f"  Time evaluated at {len(solution.t)} points")
    print(f"  Solution shape: {solution.y.shape}")

    # Extract and plot results
    times = solution.t / 1e6  # Convert to Myr
    masses = []

    for i, t in enumerate(solution.t):
        y = solution.y[:, i]
        state.set_state_vector(y, grain_types=["silicate"])
        masses.append(state.get_total_grain_mass_density())

    masses = np.array(masses)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200)
    ax.semilogy(times, masses, "b-", linewidth=2, label="Silicate grain mass")
    ax.set_xlabel("Time [Myr]", fontsize=12)
    ax.set_ylabel("Grain mass density [g/cm³]", fontsize=12)
    ax.set_title("Silicate Grain Evolution Example")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig("example1_grain_evolution.png", dpi=200)
    print("Plot saved: example1_grain_evolution.png")

    return solution, state


# ============================================================================
# EXAMPLE 2: Multi-grain compositions and PAHs
# ============================================================================


def example_multi_composition_pah():
    """
    Advanced example: silicate + graphite grains with PAH population.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Composition Evolution (Silicate + Graphite + PAHs)")
    print("=" * 70)

    # Create silicate and graphite populations
    grain_pop_sil = GrainPopulation("silicate")
    grain_pop_gra = GrainPopulation("graphite")

    # Silicate
    radii_micron = np.logspace(-3, -1, 8)
    for a in radii_micron:
        n0 = 5e-13 / len(radii_micron)
        bin_ = GrainBin("silicate", a, 3.3, population=n0)
        grain_pop_sil.add_bin(bin_)

    # Graphite
    for a in radii_micron:
        n0 = 5e-13 / len(radii_micron)
        bin_ = GrainBin("graphite", a, 2.2, population=n0)
        grain_pop_gra.add_bin(bin_)

    # PAH population
    pah_pop = PAHPopulation()
    # PAHs with different Nc values (carbon atom counts)
    Nc_values = [20, 40, 60, 100, 150, 200, 300, 500]
    for Nc in Nc_values:
        # Initial abundance - typically ~1e-12 cm⁻³ for all PAHs combined
        abundance = 1e-12 / len(Nc_values)
        pah = PAHBin(Nc, charge=0, abundance=abundance)
        pah_pop.add_bin(pah)

    print(f"Silicate population: {grain_pop_sil}")
    print(f"Graphite population: {grain_pop_gra}")
    print(f"PAH population: {pah_pop}")

    # Environmental conditions - typical diffuse ISM
    env = EnvironmentalConditions(
        temperature_K=100.0,
        electron_density_cm3=0.05,
        hydrogen_density_cm3=0.5,
        radiation_field=1.0,
    )

    # Create state
    state = DustEvolutionState(time_year=0.0, environmental_conditions=env)
    state.add_grain_population("silicate", grain_pop_sil)
    state.add_grain_population("graphite", grain_pop_gra)
    state.add_pah_population(pah_pop)

    # Evolution system
    system = DustEvolutionSystem(state)

    # Add processes for each dust type
    system.add_process(GrainGrowthProcess(growth_rate_cm_per_year=1e-14))
    system.add_process(SimpleShatteringProcess(shattering_rate=1e-13))
    system.add_process(PAHDestructionProcess(uv_destruction_timescale_year=1e4))

    print(system.get_summary())

    # Integrate
    t_span = (0, 1e5)  # 100k years
    t_eval = np.logspace(0, 5, 50)

    print("\nIntegrating multi-component system...")

    def ode_func(t, y):
        return system.ode_derivative(t, y)

    y0 = state.get_state_vector()

    solution = solve_ivp(
        ode_func,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
        max_step=1e3,
    )

    print(f"Integration completed: {solution.status}")

    # Extract results
    times = solution.t / 1e3  # Convert to kyr
    sil_masses = []
    gra_masses = []
    pah_abundances = []

    for i, t in enumerate(solution.t):
        y = solution.y[:, i]
        state.set_state_vector(y, grain_types=["silicate", "graphite"])
        sil_masses.append(state.grain_populations["silicate"].get_total_mass_density())
        gra_masses.append(state.grain_populations["graphite"].get_total_mass_density())
        pah_abundances.append(state.pah_population.get_total_abundance())

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=200)

    ax1.semilogy(times, sil_masses, "b-", linewidth=2, label="Silicate")
    ax1.semilogy(times, gra_masses, "r-", linewidth=2, label="Graphite")
    ax1.set_ylabel("Grain mass density [g/cm³]", fontsize=11)
    ax1.set_title("Multi-Grain Evolution")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.semilogy(times, pah_abundances, "g-", linewidth=2, label="PAHs")
    ax2.set_xlabel("Time [kyr]", fontsize=11)
    ax2.set_ylabel("PAH abundance [cm⁻³]", fontsize=11)
    ax2.set_title("PAH Evolution (UV Destruction)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig("example2_multi_evolution.png", dpi=200)
    print("Plot saved: example2_multi_evolution.png")

    return solution, state


# ============================================================================
# EXAMPLE 3: Variable environmental conditions
# ============================================================================


def example_variable_environment():
    """
    Example: evolution with time-dependent environment (e.g., shock passage).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Variable Environment (Temperature Spike)")
    print("=" * 70)

    # Simple grain population
    grain_pop = GrainPopulation("silicate")
    for a in np.logspace(-3, -1, 6):
        bin_ = GrainBin("silicate", a, 3.3, population=1e-12 / 6)
        grain_pop.add_bin(bin_)

    # Initial calm environment
    env = EnvironmentalConditions(
        temperature_K=100.0,
        electron_density_cm3=0.01,
        hydrogen_density_cm3=0.1,
        radiation_field=0.3,
    )

    state = DustEvolutionState(time_year=0.0, environmental_conditions=env)
    state.add_grain_population("silicate", grain_pop)

    system = DustEvolutionSystem(state)
    system.add_process(GrainGrowthProcess(growth_rate_cm_per_year=5e-15))
    system.add_process(SimpleShatteringProcess(shattering_rate=1e-13))

    # Modified ODE function that changes environment with time
    def ode_func_variable_env(t, y):
        # Simulate temperature spike at t = 5e4 yr
        if 4e4 < t < 6e4:
            # During "shock": high temperature and density
            system.state.env.temperature_K = 1000.0
            system.state.env.hydrogen_density_cm3 = 10.0
        else:
            # Normal conditions
            system.state.env.temperature_K = 100.0
            system.state.env.hydrogen_density_cm3 = 0.1

        return system.ode_derivative(t, y)

    t_span = (0, 1e5)
    t_eval = np.linspace(0, 1e5, 200)

    print("Integrating with variable environment (shock at t=50 kyr)...")

    y0 = state.get_state_vector()
    solution = solve_ivp(
        ode_func_variable_env,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
        max_step=1e3,
    )

    print(f"Integration completed: {solution.status}")

    # Extract and plot
    times = solution.t / 1e3
    masses = []

    for i, t in enumerate(solution.t):
        y = solution.y[:, i]
        state.set_state_vector(y, grain_types=["silicate"])
        masses.append(state.get_total_grain_mass_density())

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200)
    ax.semilogy(times, masses, "b-", linewidth=2, label="Silicate grains")
    ax.axvspan(40, 60, alpha=0.2, color="red", label="Shock region")
    ax.set_xlabel("Time [kyr]", fontsize=12)
    ax.set_ylabel("Grain mass density [g/cm³]", fontsize=12)
    ax.set_title("Grain Evolution with Temperature Spike")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig("example3_variable_env.png", dpi=200)
    print("Plot saved: example3_variable_env.png")

    return solution, state


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DUST AND PAH EVOLUTION EXAMPLES")
    print("=" * 70)

    # Run examples
    sol1, st1 = example_basic_grain_evolution()
    sol2, st2 = example_multi_composition_pah()
    sol3, st3 = example_variable_environment()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - example1_grain_evolution.png")
    print("  - example2_multi_evolution.png")
    print("  - example3_variable_env.png")
