"""
Example: Basic galaxy SAM evolution with different yield models.

This script demonstrates how to use the galaxy_sam module to run 
simulations with different stellar yield models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import galaxySAM modules
from galaxySAM import galaxy_sam
from galaxySAM import plotting
from galaxySAM import constants


def example_basic_evolution():
    """Run basic galaxy evolution with Kobayashi yields."""
    
    print("Example 1: Basic Galaxy Evolution with Kobayashi Yields")
    print("-" * 60)
    
    # Create SAM with Kobayashi yields at solar metallicity
    sam = galaxy_sam.GalaxySAM(
        yield_model='kobayashi',
        metallicity=0.02,  # Solar metallicity
        imf_type='chabrier',
        tscale_infall=7.0,  # 7 Gyr infall timescale
        tscale_sfr=2.2,     # 2.2 Gyr SFR timescale
        nbint=500,          # Moderate time resolution
    )
    
    # Run evolution
    results = sam.evolve()
    
    # Print summary
    print(f"Final stellar mass: {results['mstar'][-1]:.2e} Msun")
    print(f"Final gas mass: {results['mgas'][-1]:.2e} Msun")
    print(f"Final metallicity: {results['metallicity'][-1]:.4f}")
    
    return results


def example_multiple_models():
    """Compare evolution with different yield models."""
    
    print("\nExample 2: Comparing Yield Models")
    print("-" * 60)
    
    models = ['kobayashi', 'lc18', 'karakas']
    results_all = {}
    
    for model_name in models:
        print(f"  Running {model_name}...")
        
        sam = galaxy_sam.GalaxySAM(
            yield_model=model_name,
            metallicity=0.02,
            imf_type='chabrier',
            nbint=500,
        )
        
        results = sam.evolve()
        results_all[model_name] = results
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Galaxy SAM Evolution - Model Comparison')
    
    time = results_all[models[0]]['time']
    z_sun = constants.ZSUN_ASPLUND
    
    # Stellar mass
    ax = axes[0, 0]
    for model in models:
        ax.semilogy(time, results_all[model]['mstar'], label=model, linewidth=2)
    ax.set_ylabel('Stellar Mass (Msun)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gas mass
    ax = axes[0, 1]
    for model in models:
        ax.semilogy(time, results_all[model]['mgas'], label=model, linewidth=2)
    ax.set_ylabel('Gas Mass (Msun)')
    ax.grid(True, alpha=0.3)
    
    # Metallicity
    ax = axes[1, 0]
    for model in models:
        z = results_all[model]['metallicity']
        logz = np.log10(np.clip(z, 1e-5, 1.0) / z_sun)
        ax.plot(time, logz, label=model, linewidth=2)
    ax.set_ylabel('log(Z/Zsun)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # SFR
    ax = axes[1, 1]
    for model in models:
        sfr = results_all[model]['sfr']
        ax.semilogy(time, np.clip(sfr, 1e-10, np.max(sfr)), label=model, linewidth=2)
    ax.set_ylabel('SFR (Msun/yr)')
    ax.set_xlabel('Time (Gyr)')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('example_model_comparison.png', dpi=150, bbox_inches='tight')
    print("  Plot saved: example_model_comparison.png")
    
    return results_all


def example_parameter_study():
    """Study effect of different parameters."""
    
    print("\nExample 3: Parameter Study - Infall Timescale Effect")
    print("-" * 60)
    
    tscales = [3.0, 7.0, 15.0]  # Gyr
    results_all = {}
    
    for tscale in tscales:
        print(f"  Running with tscale_infall = {tscale} Gyr...")
        
        sam = galaxy_sam.GalaxySAM(
            yield_model='kobayashi',
            metallicity=0.02,
            tscale_infall=tscale,
            nbint=500,
        )
        
        results = sam.evolve()
        results_all[f'{tscale:.1f}'] = results
    
    # Plot parameter study
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = results_all[list(results_all.keys())[0]]['time']
    z_sun = constants.ZSUN_ASPLUND
    
    for label, results in results_all.items():
        z = results['metallicity']
        logz = np.log10(np.clip(z, 1e-5, 1.0) / z_sun)
        ax.plot(time, logz, label=f'τ = {label} Gyr', linewidth=2)
    
    ax.set_xlabel('Time (Gyr)', fontsize=12)
    ax.set_ylabel('log(Z/Zsun)', fontsize=12)
    ax.set_title('Effect of Infall Timescale on Metallicity Evolution')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_parameter_study.png', dpi=150, bbox_inches='tight')
    print("  Plot saved: example_parameter_study.png")
    
    return results_all


def example_multi_metallicity():
    """Track evolution across metallicity range."""
    
    print("\nExample 4: Multi-Metallicity Evolution")
    print("-" * 60)
    
    metallicities = [0.002, 0.01, 0.02, 0.05]  # Msun values
    results_all = {}
    
    for z_init in metallicities:
        z_sun = constants.ZSUN_ASPLUND
        logz_init = np.log10(z_init / z_sun)
        print(f"  Running with Z = {z_init:.4f} (log(Z/Zsun) = {logz_init:.2f})...")
        
        sam = galaxy_sam.GalaxySAM(
            yield_model='kobayashi',
            metallicity=z_init,
            nbint=500,
        )
        
        results = sam.evolve()
        results_all[f'{logz_init:.2f}'] = results
    
    # Plot multi-metallicity evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = results_all[list(results_all.keys())[0]]['time']
    z_sun = constants.ZSUN_ASPLUND
    
    for label, results in results_all.items():
        z = results['metallicity']
        logz = np.log10(np.clip(z, 1e-5, 1.0) / z_sun)
        ax.plot(time, logz, label=f'Initial log(Z/Zsun) = {label}', linewidth=2)
    
    ax.set_xlabel('Time (Gyr)', fontsize=12)
    ax.set_ylabel('log(Z/Zsun)', fontsize=12)
    ax.set_title('Metallicity Evolution from Different Initial Conditions')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Solar')
    
    plt.tight_layout()
    plt.savefig('example_multi_metallicity.png', dpi=150, bbox_inches='tight')
    print("  Plot saved: example_multi_metallicity.png")
    
    return results_all


def main():
    """Run all examples."""
    
    print("=" * 60)
    print("Galaxy SAM Evolution - Examples")
    print("=" * 60)
    
    # Run examples
    results1 = example_basic_evolution()
    results2 = example_multiple_models()
    results3 = example_parameter_study()
    results4 = example_multi_metallicity()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
