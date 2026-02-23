"""
DUST AND PAH EVOLUTION FRAMEWORK - COMPREHENSIVE DOCUMENTATION

================================================================================
OVERVIEW
================================================================================

The DustRAMSES evolution framework provides a flexible, modular system for
modeling how dust and PAH populations evolve over time under various physical
processes. It is designed to:

1. FLEXIBLY represent grain size distributions and PAH populations
2. COMPOSABLY combine different evolution processes
3. SEAMLESSLY integrate with Python ODE solvers (scipy)
4. EASILY incorporate existing physical models from the codebase
5. TRACK complete system state including environmental conditions

The framework separates concerns into distinct components:
- BINS: Individual grain/PAH populations at fixed sizes
- POPULATIONS: Collections of bins
- EVOLUTION PROCESSES: Physical mechanisms causing change
- SYSTEM: Orchestrates processes and provides ODE interface

================================================================================
KEY CLASSES AND THEIR ROLES
================================================================================

GrainBin & PAHBin
-----------------
Represent single-size populations.

GrainBin: A population of grains with identical radius and composition.
  - Tracks: grain type, radius (micron), material density, population (cm^-3)
  - Computes: mass, cross-section, volume
  - Stores: metadata (charge state, sputtering rate, optical properties, etc.)

PAHBin: A population of PAH molecules with identical carbon content.
  - Tracks: number of carbon atoms (Nc), charge state, abundance (cm^-3)
  - Stores: metadata (mass, radius, charge distribution, etc.)

Example:
    from dust_pah_evolution import GrainBin, PAHBin
    
    # Create a silicate grain bin at 0.1 µm
    grain = GrainBin(
        grain_type="silicate",
        radius_micron=0.1,
        density=3.3,  # g/cm³
        population=1e-12  # cm⁻³
    )
    
    # Create a PAH bin (C100, neutral charge)
    pah = PAHBin(Nc=100, charge=0, abundance=1e-12)


GrainPopulation & PAHPopulation
--------------------------------
Manage collections of bins representing full size distributions.

Purpose: Treat a collection of bins as a single object with bulk properties
  - get_total_mass_density(): sum mass from all bins
  - get_populations(): array interface for ODE solvers
  - set_populations(): update all bins from ODE solver output

Example:
    from dust_pah_evolution import GrainPopulation
    
    sil_pop = GrainPopulation("silicate")
    
    # Add bins at logarithmic spacing
    for radius in np.logspace(-3, -1, 10):
        bin_ = GrainBin("silicate", radius, 3.3, population=1e-13)
        sil_pop.add_bin(bin_)


EnvironmentalConditions
-----------------------
Stores the physical environment affecting evolution.

Attributes:
  - temperature_K: gas temperature [K]
  - electron_density_cm3: electron number density [cm⁻³]
  - hydrogen_density_cm3: hydrogen number density [cm⁻³]
  - radiation_field: radiation field strength (e.g., Habing units)
  - cosmic_ray_ionization: ionization rate [s⁻¹ cm⁻³]
  - custom_params: dict for additional parameters

These values are passed to processes to compute rates.

Example:
    env = EnvironmentalConditions(
        temperature_K=100.0,      # cold ISM
        electron_density_cm3=0.03,
        hydrogen_density_cm3=1.0,
        radiation_field=1.0,      # Habing field
    )


DustEvolutionState
------------------
Complete snapshot of the system at a given time.

Manages:
  - Multiple grain populations (by type: silicate, graphite, etc.)
  - One PAH population (optional)
  - Environmental conditions
  - Evolution history for tracking

Key methods:
  - get_state_vector(): extract all populations as 1D array (for ODE solvers)
  - set_state_vector(): update all populations from ODE solver output
  - record_history(): save current state to history

Example:
    from dust_pah_evolution import DustEvolutionState
    
    state = DustEvolutionState(
        time_year=0.0,
        environmental_conditions=env
    )
    state.add_grain_population("silicate", sil_pop)
    state.add_grain_population("graphite", gra_pop)
    state.add_pah_population(pah_pop)
    
    # Interface for ODE solvers:
    y = state.get_state_vector()  # → array([n_sil_bin0, n_sil_bin1, ..., n_gra_bin0, ..., n_pah_bin0, ...])


EvolutionProcess
----------------
Abstract base class for all physical evolution mechanisms.

Every specific process (sputtering, coagulation, etc.) is a subclass of
EvolutionProcess.

Key requirement: implement compute_rates() which returns dn/dt for affected bins.

Process types:
  - "grain": affects grain populations
  - "pah": affects PAH populations
  - "coupling": affects both (returns tuple)
  - Custom: "ion", "electron", etc.

Example:
    from dust_pah_evolution import EvolutionProcess
    import numpy as np
    
    class MyCustomProcess(EvolutionProcess):
        def __init__(self):
            super().__init__("my_process", process_type="grain")
        
        def compute_rates(self, state, grain_pop=None, pah_pop=None):
            if grain_pop is None:
                return np.array([])
            
            rates = np.zeros(grain_pop.nbins)
            for i, bin_ in enumerate(grain_pop.bins):
                # Compute dn/dt for this bin based on state
                rates[i] = ...
            
            return rates


DustEvolutionSystem
-------------------
High-level orchestrator that brings everything together.

Responsibilities:
  - Maintain current state
  - Manage list of processes
  - Compute total rates from all processes
  - Provide ODE solver interface via ode_derivative()

Key methods:
  - add_process(process): add an evolution mechanism
  - remove_process(name): remove by name
  - get_process(name): retrieve by name
  - compute_total_rates(): sum rates from all enabled processes
  - ode_derivative(t, y): ODE solver callback
  - step_forward(dt): simple forward Euler step (for testing)

Example:
    from dust_pah_evolution import DustEvolutionSystem
    from scipy.integrate import solve_ivp
    
    system = DustEvolutionSystem(state)
    system.add_process(MyGrainGrowthProcess())
    system.add_process(MySputteringProcess())
    system.add_process(MyPAHDestructionProcess())
    
    # Integrate with scipy
    def ode_func(t, y):
        return system.ode_derivative(t, y)
    
    solution = solve_ivp(
        ode_func,
        (0, 1e6),  # 0 to 1 million years
        state.get_state_vector(),
        method="RK45"
    )


================================================================================
WORKFLOW: From Setup to Results
================================================================================

1. CREATE BINS
   Create GrainBin and PAHBin objects representing your populations.
   
2. CREATE POPULATIONS
   Group bins into GrainPopulation and PAHPopulation objects.
   
3. CREATE STATE
   Create DustEvolutionState with initial conditions and add populations.
   
4. CREATE PROCESSES
   Implement EvolutionProcess subclasses for each physical mechanism.
   
5. CREATE SYSTEM
   Create DustEvolutionSystem and add all processes.
   
6. INTEGRATE
   Use scipy.integrate.solve_ivp (or odeint) with system.ode_derivative.
   
7. ANALYZE
   Extract results from solution and/or state.history.

Complete example:

    import numpy as np
    from scipy.integrate import solve_ivp
    from dust_pah_evolution import *
    
    # 1. Create bins
    grains = GrainPopulation("silicate")
    for a in np.logspace(-3, -1, 10):
        grains.add_bin(GrainBin("silicate", a, 3.3, 1e-13))
    
    # 2. Create populations (already done above)
    
    # 3. Create state
    env = EnvironmentalConditions(temperature_K=100, ...)
    state = DustEvolutionState(environmental_conditions=env)
    state.add_grain_population("silicate", grains)
    
    # 4. Create processes
    system = DustEvolutionSystem(state)
    system.add_process(MyGrowthProcess())
    system.add_process(MySputteringProcess())
    
    # 5. Integrate
    solution = solve_ivp(
        lambda t, y: system.ode_derivative(t, y),
        (0, 1e6),
        state.get_state_vector(),
        t_eval=np.logspace(0, 6, 100),
        method="RK45"
    )
    
    # 6. Analyze
    for i, t in enumerate(solution.t):
        state.set_state_vector(solution.y[:, i])
        mass = state.get_total_grain_mass_density()
        print(f"t={t:.2e}: M={mass:.2e}")


================================================================================
DESIGN PATTERNS
================================================================================

PATTERN 1: Variable Environmental Conditions
---------------------------------------------
Some simulations require environment to vary with time (e.g., shocks, 
changes in radiation field). Wrap the ODE function:

    def ode_func_dynamic(t, y):
        # Update environment based on time
        if t < 1e4:
            system.state.env.temperature_K = 100
        elif t < 2e4:
            system.state.env.temperature_K = 1000
        else:
            system.state.env.temperature_K = 100
        
        return system.ode_derivative(t, y)
    
    solution = solve_ivp(ode_func_dynamic, ...)


PATTERN 2: Process Activation/Deactivation
-------------------------------------------
Enable/disable processes during integration to test their effects:

    growth = system.get_process("grain_growth")
    
    def ode_func_selective(t, y):
        if t < 1e4:
            growth.disable()  # No growth initially
        else:
            growth.enable()
        
        return system.ode_derivative(t, y)


PATTERN 3: Multiple Grain Types
--------------------------------
Many simulations need separate silicate and graphite populations.
They track independently but can interact via coupling processes:

    state.add_grain_population("silicate", sil_pop)
    state.add_grain_population("graphite", gra_pop)
    
    # Access individual populations
    sil = state.get_grain_population("silicate")
    gra = state.get_grain_population("graphite")
    
    # Coupling process might move material between them
    system.add_process(GrainCompositionChangeProcess())


PATTERN 4: Metadata for Derived Quantities
-------------------------------------------
Computationally expensive quantities (charge distributions, optical properties)
can be cached in bin metadata and updated only when needed:

    # In a charging process:
    for i, bin_ in enumerate(grain_pop.bins):
        Z_dist = compute_charge_distribution(bin_, state)
        bin_.metadata['charge_dist'] = Z_dist
    
    # In a coagulation process, reuse:
    Z_dist = bin_.metadata.get('charge_dist')
    if Z_dist is not None:
        coulomb_factor = compute_enhancement(Z_dist)


PATTERN 5: Testing and Validation
----------------------------------
Develop and test processes in isolation before integration:

    # Test a single process with a simple state
    test_state = DustEvolutionState(
        environmental_conditions=EnvironmentalConditions(...)
    )
    test_state.add_grain_population("silicate", simple_pop)
    
    process = MySputteringProcess()
    rates = process.compute_rates(test_state, simple_pop, None)
    
    # Verify rates are reasonable
    assert np.all(rates <= 0), "Sputtering should destroy grains"
    assert np.max(np.abs(rates)) < 1e-10, "Rates seem too large"


================================================================================
INTEGRATION WITH EXISTING CODE
================================================================================

The framework is designed to wrap existing models from the DustRAMSES package:

dust_model.py
  - LogNormal_Distribution: use to initialize populations
  - grain_charge_dist(): integrate into ChargingProcess
  - t_sputtering(), t_coagulation(): compute rates for processes
  - relative_velocity(): needed by collision processes

dust_sputtering.py
  - sputtering_yield(): use in sputtering rate calculation
  - Tielens/Nozawa coefficients: lookup tables for temperature dependence

PAHs_model.py
  - Nc_from_size(), size_from_Nc(): PAH size conversions
  - mass_from_Nc(): PAH mass
  - dissociation models: PAH destruction processes

dust_charging.py
  - equilibrium_charge_for_grain(): store in metadata
  - cmp_D_WD99(): Coulomb enhancement factor

See integration_guide.py for detailed templates and examples.


================================================================================
COMMON PITFALLS AND SOLUTIONS
================================================================================

PITFALL 1: Units Mismatch
  Problem: ODE solver gets huge/tiny numbers because of unit inconsistency
  Solution: Be explicit about units everywhere. Use:
    - Time: years (convert to seconds only where needed)
    - Density: cm⁻³
    - Size: micrometers for grain radius
    - Mass: grams
    Document units in every function!

PITFALL 2: Negative Populations
  Problem: ODE solver can produce n < 0, which is unphysical
  Solution: Clamp to non-negative:
    rates = np.maximum(current_pop + dt * rates, 0)
  Or use an event in solve_ivp to stop before hitting zero.

PITFALL 3: Mass Not Conserved
  Problem: Material created/destroyed unexplainably in coupled processes
  Solution: Track mass carefully in coagulation/fragmentation:
    mass_before = sum(n_i * m_i)
    # ... update bins via process ...
    mass_after = sum(n_i * m_i)
    assert abs(mass_after - mass_before) < tolerance

PITFALL 4: Rates Don't Match Expectations
  Problem: Evolution seems too fast or slow
  Solution: Compare against known timescales:
    - Sputtering: typically 1e4-1e7 years
    - Coagulation: typically 1e3-1e6 years
    - PAH destruction: typically 1e3-1e5 years
  If your rates give very different timescales, debug the physics!

PITFALL 5: Expensive Calculations
  Problem: ODE solver is very slow for fine grain grids (100+ bins)
  Solution:
    - Use coarser bin spacing if possible
    - Cache expensive computations in metadata
    - Parallelize process computation if time-independent
    - Use faster ODE method (e.g., "BDF" for stiff systems)


================================================================================
PERFORMANCE TIPS
================================================================================

1. USE REALISTIC BIN COUNTS
   10-20 grain size bins is usually sufficient. More bins = slower integration.

2. CACHE EXPENSIVE COMPUTATIONS
   - Charge distributions (don't recompute every step if stable)
   - Optical properties (interpolate once, store)
   - Temperature-dependent coefficients (use lookup tables)

3. VECTORIZE OPERATIONS
   Use numpy operations instead of Python loops where possible.

4. CHOOSE APPROPRIATE ODE METHOD
   - RK45: good default, adaptive stepping
   - BDF: better for stiff systems (fast/slow process mixing)
   - DOP853: high accuracy if needed

5. USE EVENT HANDLING
   scipy's solve_ivp can detect special events:
     - Stop if all grains destroyed
     - Stop if steady state reached
     - Record times when specific thresholds crossed

6. PROFILE BEFORE OPTIMIZING
   Use cProfile to find actual bottlenecks before optimizing.


================================================================================
REFERENCES AND FURTHER READING
================================================================================

Key papers referenced in the framework:

Physical models:
  - Tielens & Barcos (1994): grain sputtering
  - Draine & Lee (1984): optical properties
  - Weingartner & Draine (2001): charging and grain properties
  - Ibanez-Mejias et al. (2019): charge distribution fitting

Evolution processes:
  - Nozawa et al. (2006): dust production/destruction
  - Galliano et al. (2005): PAH destruction
  - Micelotta et al. (2010): sputtering by thermal ions

Related resources:
  - DustRAMSES package documentation
  - scipy.integrate documentation (ODE solvers)
  - numpy documentation (array operations)

================================================================================
GETTING HELP
================================================================================

For issues or questions:

1. Check the docstrings in each class
2. Review examples in example_evolution.py
3. See integration_guide.py for wrapping existing models
4. Check dust_model.py, dust_sputtering.py, etc. for available models
5. Run with verbose output and check ODE solver messages
6. Print intermediate rates and state vectors to debug

"""

# Quick reference card
QUICK_REFERENCE = """
============================================================================
QUICK REFERENCE: Essential Classes and Methods
============================================================================

# SETUP
from dust_pah_evolution import *
import numpy as np
from scipy.integrate import solve_ivp

# Create bins
grain = GrainBin("silicate", radius_micron=0.1, density=3.3, population=1e-12)
pah = PAHBin(Nc=100, charge=0, abundance=1e-12)

# Create populations
pop_grain = GrainPopulation("silicate")
pop_grain.add_bin(grain)

pop_pah = PAHPopulation()
pop_pah.add_bin(pah)

# Create state
env = EnvironmentalConditions(temperature_K=100, ...)
state = DustEvolutionState(environmental_conditions=env)
state.add_grain_population("silicate", pop_grain)
state.add_pah_population(pop_pah)

# In state: vector for ODE solver
y = state.get_state_vector()
state.set_state_vector(y_updated)

# Create system
system = DustEvolutionSystem(state)
system.add_process(MyProcess())

# Integrate
solution = solve_ivp(lambda t, y: system.ode_derivative(t, y), ...)

# Analyze results
for i, t in enumerate(solution.t):
    state.set_state_vector(solution.y[:, i])
    mass = state.get_total_grain_mass_density()

============================================================================
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n")
    print(QUICK_REFERENCE)
