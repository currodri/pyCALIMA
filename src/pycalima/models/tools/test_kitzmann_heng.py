
import os
import numpy as np
import matplotlib.pyplot as plt
from mie_theory import MieTheory
from pathlib import Path

def run_kitzmann_heng_test():
    mie = MieTheory()
    supp_path = Path('optical_props/stx3141_supp')
    
    # Get all .dat files
    dat_files = sorted(list(supp_path.glob('*.dat')))
    
    radius_um = 0.1
    # Define a wavelength grid that extends into X-rays (0.001 um = 1.24 keV)
    wav_grid = np.logspace(-3, 3, 300) # 0.001 to 1000 microns
    
    results = {}
    
    print(f"Processing {len(dat_files)} compositions from Kitzmann & Heng 2018...")
    
    for fpath in dat_files:
        species = fpath.stem
        try:
            mie.load_kitzmann_heng(str(fpath), species)
            
            qabs_list = []
            qsca_list = []
            g_list = []
            
            # Wavelength range for this species
            w_min_tab = mie.dielectric_data[species]['wavelengths'].min()
            w_max_tab = mie.dielectric_data[species]['wavelengths'].max()
            
            # For plotting, we use the full wav_grid but cap at the max tabulated wavelength
            # and allow extension into the UV/X-ray below w_min_tab
            w_plot = wav_grid[wav_grid <= w_max_tab]
            
            for w in w_plot:
                # extend_xrays=True will use Henke for w < w_min_tab
                qa, qs, g = mie.compute_grain_properties(radius_um, w, species, use_fast_path=True, extend_xrays=True)
                qabs_list.append(qa)
                qsca_list.append(qs)
                g_list.append(g)
            
            results[species] = {
                'wav': w_plot,
                'qabs': np.array(qabs_list),
                'qsca': np.array(qsca_list),
                'g': np.array(g_list)
            }
        except Exception as e:
            print(f"Error processing {species}: {e}")

    # Plotting
    num_species = len(results)
    ncols = 4
    nrows = (num_species + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4*nrows), sharex=True)
    axes_flat = axes.flatten()
    
    legend_info = None
    for i, (species, data) in enumerate(results.items()):
        ax = axes_flat[i]
        
        # Plot Qabs and Qsca on log scale
        l_abs = ax.loglog(data['wav'], data['qabs'], label='$Q_{abs}$', color='firebrick', linewidth=1.5)
        l_sca = ax.loglog(data['wav'], data['qsca'], label='$Q_{sca}$', color='navy', linestyle='--', linewidth=1.5)
        
        # Plot g on the same axis (if positive) or secondary?
        ax_g = ax.twinx()
        l_g = ax_g.plot(data['wav'], data['g'], label='$g$', color='forestgreen', linestyle=':', linewidth=2.0)
        ax_g.set_ylim(-0.1, 1.1)
        if i % ncols == ncols - 1:
            ax_g.set_ylabel('$g$')
        
        ax.set_title(species, fontsize=14, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
        if i // ncols == nrows - 1:
            ax.set_xlabel(r'Wavelength [$\mu m$]')
        if i % ncols == 0:
            ax.set_ylabel('Efficiency $Q$')
        
        if i == 0:
            # Capture handles for the first subplot
            legend_info = (l_abs + l_sca + l_g, ['$Q_{abs}$', '$Q_{sca}$', '$g$'])

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # Combined legend with larger font
    if legend_info:
        fig.legend(legend_info[0], legend_info[1], loc='upper center', 
                   bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=32, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)
    
    out_path = 'results/kitzmann_heng_efficiencies.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    run_kitzmann_heng_test()
