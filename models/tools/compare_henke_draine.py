
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mie_theory import MieTheory
from henke_extension import HenkeExtension

def compare_draine_henke():
    mie = MieTheory()
    henke = HenkeExtension()
    
    # Paths to Draine's data
    gra_pa_path = 'optical_props/draine_lee_1984/callindex.out_CpaD03_0.01'
    gra_pe_path = 'optical_props/draine_lee_1984/callindex.out_CpeD03_0.01'
    sil_path = 'optical_props/draine_lee_1984/eps_suvSil'
    
    mie.load_dielectric_constants(gra_pa_path, 'gra_pa')
    mie.load_dielectric_constants(gra_pe_path, 'gra_pe')
    mie.load_dielectric_constants(sil_path, 'sil')
    
    # Compositions and densities for Draine's models
    models = {
        'Graphite (pa)': {
            'draine_label': 'gra_pa',
            'comp': {'C': 1},
            'rho': 2.26  # Typical Draine graphite density
        },
        'Graphite (pe)': {
            'draine_label': 'gra_pe',
            'comp': {'C': 1},
            'rho': 2.26
        },
        'Silicate': {
            'draine_label': 'sil',
            'comp': {'Mg': 1.1, 'Fe': 0.9, 'Si': 1.0, 'O': 4.0}, # Astrodust typical
            'rho': 3.50  # Draine astronomical silicate density
        }
    }
    
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)), sharex=True)
    
    for i, (name, meta) in enumerate(models.items()):
        # Get Draine's data
        wav = mie.dielectric_data[meta['draine_label']]['wavelengths']
        n_dr = mie.dielectric_data[meta['draine_label']]['n_interp'](wav)
        k_dr = mie.dielectric_data[meta['draine_label']]['k_interp'](wav)
        
        # Filter for FUV/X-ray regime (e.g., 0.0001 to 0.03 um, which corresponds to ~40 eV to 12 keV)
        mask = (wav >= 0.0001) & (wav <= 0.03)
        wav_x = wav[mask]
        n_dr_x = n_dr[mask]
        k_dr_x = k_dr[mask]
        
        # Compute Henke properties
        m_henke = henke.compute_refractive_index(meta['comp'], meta['rho'], wav_x)
        n_hk = np.real(m_henke)
        k_hk = -np.imag(m_henke)
        
        # Calculate statistics
        diff_n = np.abs((n_hk - 1) - (n_dr_x - 1)) / np.abs(n_dr_x - 1) * 100
        diff_k = np.abs(k_hk - k_dr_x) / np.abs(k_dr_x) * 100
        
        print(f"--- {name} ---")
        print(f"n-1: Median Diff = {np.median(diff_n):.2f}%, Max Diff = {np.max(diff_n):.2f}%")
        print(f"k:   Median Diff = {np.median(diff_k):.2f}%, Max Diff = {np.max(diff_k):.2f}%")
        print("")
        
        # Plot n-1
        ax = axes[i, 0]
        ax.loglog(wav_x, np.abs(n_dr_x - 1), label='Draine 2003', linestyle='-', color='black', lw=2)
        ax.loglog(wav_x, np.abs(n_hk - 1), label='Henke Extension', linestyle='--', color='red', lw=2)
        ax.set_ylabel('|n - 1|')
        ax.set_title(f"{name} (|n - 1|)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        # Plot k
        ax = axes[i, 1]
        ax.loglog(wav_x, k_dr_x, label='Draine 2003', linestyle='-', color='black', lw=2)
        ax.loglog(wav_x, k_hk, label='Henke Extension', linestyle='--', color='red', lw=2)
        ax.set_ylabel('k')
        ax.set_title(f"{name} (k)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        if i == len(models) - 1:
            axes[i, 0].set_xlabel('Wavelength (um)')
            axes[i, 1].set_xlabel('Wavelength (um)')

    plt.tight_layout()
    out_path = 'results/henke_vs_draine_comparison.png'
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    compare_draine_henke()
