"""
DUST ACCRETION - QUICK START EXAMPLES

This file contains practical examples showing how to use the dust_accretion module.

"""

import numpy as np
from dust_accretion import (
    accretion_rate_da_dt,
    accretion_timescale,
    collision_rate_analysis,
    collision_rate_from_densities,
    limiting_collision_rate,
    thermal_velocity,
    SILICATE_COMPOSITION,
    GRAPHITE_COMPOSITION,
)


def example_1_basic_accretion():
    """Example 1: Compute accretion rate for typical ISM grain"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Accretion Rate")
    print("=" * 80)
    
    # Typical diffuse ISM properties
    env_params = {
        'temperature_K': 100,
        'hydrogen_density_cm3': 0.5,
        'ion_densities': {
            'H': 0.49,      # Neutral hydrogen (dominant)
            'He': 0.05,     # Helium
            'H+': 0.01,     # Ionized hydrogen
        }
    }
    
    # Grain properties
    grain_radius_micron = 0.1  # Typical ISM grain
    grain_radius_cm = grain_radius_micron * 1e-4
    grain_density = 3.3  # silicate
    grain_charge = 0  # neutral assumption
    
    # Compute growth rate
    da_dt = accretion_rate_da_dt(
        grain_radius_cm,
        grain_density,
        env_params['ion_densities'],
        grain_charge,
        env_params['temperature_K'],
    )
    
    print(f"Environment: Diffuse ISM")
    print(f"  Temperature: {env_params['temperature_K']} K")
    print(f"  Grain radius: {grain_radius_micron} µm")
    print(f"\nResults:")
    print(f"  Growth rate (da/dt): {da_dt:.3e} cm/s")
    print(f"  Growth rate (da/dt): {da_dt * 3.15e7:.3e} µm/year")
    
    # Compute timescale
    t_scale = accretion_timescale(
        grain_radius_cm,
        grain_density,
        env_params['ion_densities'],
        grain_charge,
        env_params['temperature_K'],
    )
    
    print(f"  Growth timescale: {t_scale:.3e} seconds")
    print(f"  Growth timescale: {t_scale / 3.15e7:.3e} years")
    print(f"  Growth timescale: {t_scale / (3.15e7 * 1e6):.3e} Myr")
    print()


def example_2_collision_analysis():
    """Example 2: Which species dominates accretion?"""
    print("=" * 80)
    print("EXAMPLE 2: Collision Rate Analysis - Which Species is Limiting?")
    print("=" * 80)
    
    ion_densities = {
        'H': 0.9,
        'He': 0.1,
        'H+': 0.01,
        'C': 1e-4,
        'C+': 1e-5,
    }
    
    a_cm = 0.1e-4
    T_K = 100
    Z_grain = 0
    
    # Detailed analysis
    analysis = collision_rate_analysis(
        a_cm, Z_grain, T_K, ion_densities, return_df=False
    )
    
    print(f"Grain: a = {a_cm*1e4:.2e} µm, T = {T_K} K, Z = {Z_grain}")
    print(f"\nCollision rates for all species:")
    print(f"{'Species':<10} {'Density':<12} {'v_thermal':<12} {'Coulomb':<10} {'Rate':<12}")
    print("-" * 60)
    
    for species in sorted(analysis.keys(), key=lambda x: analysis[x]['collision_rate'], reverse=True):
        data = analysis[species]
        print(f"{species:<10} {data['density']:<12.2e} {data['thermal_velocity']:<12.2e} "
              f"{data['coulomb_factor']:<10.3f} {data['collision_rate']:<12.2e}")
    
    # Find limiting
    limit_rate, lim_species = limiting_collision_rate(
        ion_densities, a_cm, Z_grain, T_K, return_limiting_species=True
    )
    
    print(f"\nLimiting species: {lim_species} with rate {limit_rate:.2e} cm⁻³ s⁻¹")
    print()


def example_3_charge_effects():
    """Example 3: How does grain charge affect accretion?"""
    print("=" * 80)
    print("EXAMPLE 3: Grain Charge Effects on Accretion")
    print("=" * 80)
    
    ion_densities = {
        'H': 0.9,
        'H+': 0.1,
    }
    
    a_cm = 0.1e-4
    T_K = 100
    grain_density = 3.3
    
    print(f"Grain: a = {a_cm*1e4:.3e} µm, ρ = {grain_density} g/cm³")
    print(f"Ions: {ion_densities}")
    print(f"\nAccretion rates vs. grain charge:")
    print(f"{'Z_grain':<10} {'da/dt (cm/s)':<20} {'Timescale (years)':<20} {'Relative':<10}")
    print("-" * 60)
    
    rates_list = []
    
    for Z in [-5, -3, -1, 0, 1, 3, 5]:
        da_dt = accretion_rate_da_dt(
            a_cm, grain_density, ion_densities, Z, T_K
        )
        t_scale = accretion_timescale(
            a_cm, grain_density, ion_densities, Z, T_K
        )
        
        rates_list.append(da_dt)
        
        if Z == 0:
            neutral_rate = da_dt
        
        if t_scale == np.inf:
            t_str = "∞"
        else:
            t_str = f"{t_scale/3.15e7:.2e}"
        
        relative = da_dt / neutral_rate if neutral_rate > 0 else 0
        
        print(f"{Z:<10} {da_dt:<20.3e} {t_str:<20} {relative:<10.3f}x")
    
    print("\nInterpretation:")
    print("  - Negative grain charge INCREASES H+ collision rate (attractive)")
    print("  - Positive grain charge DECREASES H+ collision rate (repulsive)")
    print("  - Effects are significant for |Z| > 3")
    print()


def example_4_temperature_dependence():
    """Example 4: Temperature effects on accretion"""
    print("=" * 80)
    print("EXAMPLE 4: Temperature Dependence")
    print("=" * 80)
    
    ion_densities = {'H': 1.0}
    a_cm = 0.1e-4
    grain_density = 3.3
    
    print(f"Grain: a = {a_cm*1e4:.3e} µm")
    print(f"Gas: pure hydrogen, n(H) = 1 cm⁻³")
    print(f"\n{'T (K)':<10} {'v_thermal':<15} {'da/dt (cm/s)':<18} {'Rate change':<15}")
    print("-" * 60)
    
    temps = [10, 50, 100, 500, 1000, 5000]
    rates = []
    
    for T in temps:
        v_th = thermal_velocity(1.67e-24, T)  # hydrogen mass
        da_dt = accretion_rate_da_dt(a_cm, grain_density, ion_densities, 0, T)
        rates.append(da_dt)
        
        if len(rates) > 1:
            change = rates[-1] / rates[-2]
        else:
            change = 1.0
        
        print(f"{T:<10} {v_th:<15.3e} {da_dt:<18.3e} {change:<15.2f}x")
    
    print("\nInterpretation:")
    print("  - da/dt increases with sqrt(T) (due to thermal velocity)")
    print("  - Warmer gas = faster accreting atoms = faster grain growth")
    print()


def example_5_size_dependence():
    """Example 5: How accretion rate depends on grain size"""
    print("=" * 80)
    print("EXAMPLE 5: Grain Size Dependence")
    print("=" * 80)
    
    ion_densities = {'H': 1.0}
    T_K = 100
    grain_density = 3.3
    
    print(f"Gas: hydrogen, n(H) = 1 cm⁻³, T = {T_K} K")
    print(f"\n{'a (µm)':<10} {'da/dt (cm/s)':<18} {'da/dt (µm/yr)':<18} {'Rel. Rate':<15}")
    print("-" * 60)
    
    radii_micron = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    rates = []
    
    for a_micron in radii_micron:
        a_cm = a_micron * 1e-4
        da_dt = accretion_rate_da_dt(a_cm, grain_density, ion_densities, 0, T_K)
        rates.append(da_dt)
        
        da_dt_micron_yr = da_dt * 3.15e7
        
        if len(rates) > 1:
            rel = rates[-1] / rates[-2]
        else:
            rel = 1.0
        
        print(f"{a_micron:<10} {da_dt:<18.3e} {da_dt_micron_yr:<18.3e} {rel:<15.3f}x")
    
    print("\nInterpretation:")
    print("  - da/dt ∝ 1/a² (inversely proportional to grain size)")
    print("  - Smaller grains grow FASTER (more surface area relative to volume)")
    print("  - This is WHY small grains are efficiently destroyed while large grains persist")
    print()


def example_6_comparison_environments():
    """Example 6: Accretion in different ISM environments"""
    print("=" * 80)
    print("EXAMPLE 6: Accretion in Different ISM Environments")
    print("=" * 80)
    
    environments = {
        'Cold Cloud': {
            'T': 50,
            'n_H': 100,
            'ions': {'H': 99.9, 'H+': 0.1},
        },
        'Diffuse ISM': {
            'T': 100,
            'n_H': 0.5,
            'ions': {'H': 0.4, 'He': 0.1, 'H+': 0.01},
        },
        'Warm ISM': {
            'T': 1000,
            'n_H': 0.1,
            'ions': {'H': 0.08, 'He': 0.01, 'H+': 0.02},
        },
        'HII Region': {
            'T': 5000,
            'n_H': 1,
            'ions': {'H': 0.5, 'He': 0.1, 'H+': 0.4},
        },
    }
    
    a_micron = 0.1
    a_cm = a_micron * 1e-4
    grain_density = 3.3
    
    print(f"Grain: a = {a_micron} µm, silicate (ρ = {grain_density} g/cm³)")
    print(f"\n{'Environment':<15} {'T (K)':<8} {'da/dt (µm/yr)':<18} {'Timescale':<15}")
    print("-" * 60)
    
    for env_name, params in environments.items():
        da_dt = accretion_rate_da_dt(
            a_cm, grain_density, params['ions'], 0, params['T']
        )
        da_dt_micron_yr = da_dt * 3.15e7
        
        t_scale = accretion_timescale(
            a_cm, grain_density, params['ions'], 0, params['T']
        )
        
        if t_scale < np.inf:
            t_str = f"{t_scale/3.15e7:.2e} yr"
        else:
            t_str = "∞"
        
        print(f"{env_name:<15} {params['T']:<8} {da_dt_micron_yr:<18.2e} {t_str:<15}")
    
    print("\nInterpretation:")
    print("  - Cold clouds: FASTEST growth (high density dominates)")
    print("  - Diffuse ISM: SLOW growth (low density)")
    print("  - Warm ISM: VERY slow (hot gas, but lower density)")
    print("  - HII regions: Variable (depends on ionization/density balance)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DUST ACCRETION MODULE - PRACTICAL EXAMPLES")
    print("=" * 80 + "\n")
    
    example_1_basic_accretion()
    example_2_collision_analysis()
    example_3_charge_effects()
    example_4_temperature_dependence()
    example_5_size_dependence()
    example_6_comparison_environments()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
