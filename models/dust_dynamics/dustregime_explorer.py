import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def create_updated_dust_regimes_plot():
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define the boundaries
    stokes_min, stokes_max = 1e-6, 1e2
    eps_min, eps_max = 1e-4, 0.1

    ax.set_xlim(stokes_min, stokes_max)
    ax.set_ylim(eps_min, eps_max)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Background Regions
    # 1. Perfectly Coupled (Passive Scalar) - ONLY low St AND low eps
    ax.fill_between([stokes_min, 1e-3], eps_min, 0.01, color='#e0f2fe', alpha=1.0)
    
    # 2. TVA / Mono-Fluid Regime - L-Shape (Mass-loaded OR Drifting)
    ax.fill_between([stokes_min, 1e-3], 0.01, eps_max, color='#dcfce7', alpha=1.0)
    ax.fill_between([1e-3, 0.1], eps_min, eps_max, color='#dcfce7', alpha=1.0)

    # 3. Decoupled (Inertial)
    ax.fill_between([0.1, stokes_max], eps_min, eps_max, color='#fee2e2', alpha=1.0)

    # Horizontal Line for Back-Reaction
    ax.axhline(0.01, color='#64748b', linestyle='--', linewidth=2)

    # Add Region Labels (Adjusted for new Y-axis limits)
    ax.text(3e-5, 1e-3, 'Perfectly Coupled\n(Passive Scalar)', ha='center', va='center', 
            color='#0284c7', fontweight='bold', fontsize=11)
    ax.text(3e-5, 0.03, 'Mass-Loaded Mixture\n(TVA / Mono-fluid)', ha='center', va='center', 
            color='#16a34a', fontweight='bold', fontsize=11)
    ax.text(1e-2, 1e-3, 'TVA Drift Regime\n(Diffusion)', ha='center', va='center', 
            color='#16a34a', fontweight='bold', fontsize=11)
    ax.text(3, 1e-3, 'Decoupled Regime\n(Inertial)', ha='center', va='center', 
            color='#dc2626', fontweight='bold', fontsize=11)

    # Add a border line to separate the passive scalar from the mass-loaded regime
    ax.plot([stokes_min, 1e-3], [0.01, 0.01], color='#16a34a', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Stokes Number ($St = t_{stop} / t_{dyn}$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dust-to-Gas Ratio ($\\epsilon = \\rho_d / \\rho_g$)', fontsize=13, fontweight='bold')
    ax.set_title('Astrophysical Dust Regimes & ISM Grain Scattering', fontsize=15, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    plt.savefig('dust_regimes_clean.pdf', dpi=300, format='pdf')

    # ---------------------------------------------------------
    # CALCULATE AND PLOT PHYSICAL DATA POINTS
    # ---------------------------------------------------------
    
    # Physics Constants
    k_B = 1.38e-16        # erg/K
    m_p = 1.67e-24        # g
    Myr_to_s = 3.154e13   # s
    rho_s = 2.2           # g/cm^3 (internal dust density)

    # ISM Phases Assumptions: {n (cm^-3), T (K), mu, t_dyn (Myr), eps}
    phases = {
        'HIM': {'n': 0.005, 'T': 1e6, 'mu': 0.6, 'tdyn': 1.0, 'eps': 5e-4, 'color': '#f59e0b'},
        'WIM': {'n': 0.5, 'T': 1e4, 'mu': 0.6, 'tdyn': 3.0, 'eps': 5e-3, 'color': '#ef4444'},
        'WNM': {'n': 0.5, 'T': 8000, 'mu': 1.3, 'tdyn': 10.0, 'eps': 1e-2, 'color': '#8b5cf6'},
        'CNM': {'n': 50, 'T': 80, 'mu': 1.3, 'tdyn': 3.0, 'eps': 1e-2, 'color': '#3b82f6'},
        'Dark Cloud': {'n': 1e4, 'T': 15, 'mu': 2.33, 'tdyn': 2.0, 'eps': 2e-2, 'color': '#10b981'},
        'Galactic Outflow': {'n': 0.01, 'T': 1e6, 'mu': 0.6, 'tdyn': 10.0, 'eps': 5e-3, 'color': '#c026d3'},
        'SN Shock': {'n': 10.0, 'T': 1e7, 'mu': 0.6, 'tdyn': 0.05, 'eps': 1e-3, 'color': '#be123c'},
        'HII Region': {'n': 1000.0, 'T': 1e4, 'mu': 0.6, 'tdyn': 2.0, 'eps': 5e-3, 'color': '#06b6d4'},
        'PDR': {'n': 1e4, 'T': 500, 'mu': 1.3, 'tdyn': 1.0, 'eps': 1e-2, 'color': '#84cc16'},
        'AGN Torus': {'n': 1e6, 'T': 1000, 'mu': 2.33, 'tdyn': 0.05, 'eps': 5e-2, 'color': '#1e3a8a'}
    }

    # Grain Sizes: {radius in cm, marker shape}
    grains = {
        'PAH (10 Å)': {'a': 1e-7, 'marker': 'o'},
        'Small (100 Å)': {'a': 1e-6, 'marker': 's'},
        'Large (0.1 µm)': {'a': 1e-5, 'marker': '^'},
        'V. Large (1 µm)': {'a': 1e-4, 'marker': 'D'}
    }

    np.random.seed(42) # For reproducible jitter

    for p_name, p_data in phases.items():
        # Calculate gas phase density and sound speed
        rho_g = p_data['n'] * p_data['mu'] * m_p
        c_s = np.sqrt(k_B * p_data['T'] / (p_data['mu'] * m_p))
        t_dyn_s = p_data['tdyn'] * Myr_to_s
        
        for g_name, g_data in grains.items():
            # Calculate Epstein drag stopping time and Stokes Number
            t_stop = (rho_s * g_data['a']) / (rho_g * c_s)
            stokes = t_stop / t_dyn_s
            
            # Apply slight vertical jitter to eps to prevent markers overlapping entirely
            jitter = np.random.uniform(0.85, 1.15)
            eps_plot = p_data['eps'] * jitter

            ax.scatter(stokes, eps_plot, color=p_data['color'], marker=g_data['marker'], 
                       s=80, edgecolor='black', zorder=5, alpha=0.9)

    # ---------------------------------------------------------
    # LEGENDS AND FORMATTING
    # ---------------------------------------------------------

    # Custom Legend for Phases (Colors)
    phase_handles = [mlines.Line2D([], [], color=p['color'], marker='o', linestyle='None',
                                  markersize=8, markeredgecolor='black', label=name) 
                     for name, p in phases.items()]
    
    # Custom Legend for Grain Sizes (Markers)
    grain_handles = [mlines.Line2D([], [], color='gray', marker=g['marker'], linestyle='None',
                                  markersize=8, markeredgecolor='black', label=name) 
                     for name, g in grains.items()]

    legend1 = ax.legend(handles=phase_handles, title='Astrophysical Phase', loc='upper right', 
                        fontsize=8, bbox_to_anchor=(0.98, 0.98), ncol=2)
    ax.add_artist(legend1)
    ax.legend(handles=grain_handles, title='Grain Size', loc='upper right', 
              fontsize=9, bbox_to_anchor=(0.98, 0.60))
    
    plt.tight_layout()
    plt.savefig('dust_regimes_scatter.pdf', dpi=300, format='pdf')

if __name__ == "__main__":
    create_updated_dust_regimes_plot()