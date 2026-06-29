import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Ensure this script's directory and the repository root are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../..'))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from mie_theory import MieTheory
from henke_extension import HenkeExtension

def load_lab_data(filepath, is_mg=True):
    """
    Load lab optical constant data from Daniele Rogantini files.
    
    Columns for Mg: energy[ev], n-1, k
    Columns for Si: energy[ev], n, k
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Laboratory data file not found at {filepath}")
    
    df = pd.read_csv(filepath, sep=r'\s+', header=0)
    if is_mg:
        df.columns = ['energy_ev', 'n_minus_1', 'k']
        df['abs_n_minus_1'] = np.abs(df['n_minus_1'])
    else:
        df.columns = ['energy_ev', 'n', 'k']
        df['n_minus_1'] = df['n'] - 1.0
        df['abs_n_minus_1'] = np.abs(df['n_minus_1'])
    return df

def load_draine_raw(filepath):
    """
    Load raw Draine 2003 silicate data from eps_suvSil.
    Returns DataFrame with columns: energy_ev, w_um, n_minus_1, k, abs_n_minus_1
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Draine file not found at {filepath}")
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data_start = 0
    for i, line in enumerate(lines):
        if 'wave' in line.lower() or 'w(micron)' in line.lower():
            data_start = i + 1
            break
        parts = line.split()
        if not parts:
            continue
        try:
            float(parts[0])
            if len(parts) >= 5 and '=' not in line:
                data_start = i
                break
        except ValueError:
            continue
            
    table_data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 5:
            try:
                vals = [float(p) for p in parts[:5]]
                table_data.append(vals)
            except ValueError:
                continue
                
    df = pd.DataFrame(table_data, columns=['w_um', 'eps1_minus_1', 'eps2', 'n_minus_1', 'k'])
    C_EV_UM = 1.23984193
    df['energy_ev'] = C_EV_UM / df['w_um']
    df['abs_n_minus_1'] = np.abs(df['n_minus_1'])
    
    return df.sort_values('energy_ev')

def compare_constants():
    # 1. Initialize models
    henke = HenkeExtension(os.path.join(repo_root, 'external_data/henke/f1f2_Henke.dat'))
    
    sil_path = os.path.join(repo_root, 'optical_props/draine_lee_1984/callindex.out_silD03')
    df_dr_all = load_draine_raw(sil_path)
    
    # Paths to Rogantini lab data
    mg_lab_path = os.path.join(repo_root, 'optical_props/fromDanieleRogantini/olivine100_lab_optical_constants_mg.txt')
    si_lab_path = os.path.join(repo_root, 'optical_props/fromDanieleRogantini/olivine100_lab_optical_constants_si.txt')
    
    # 2. Load lab data
    print("Loading Daniele Rogantini lab tables...")
    df_mg = load_lab_data(mg_lab_path, is_mg=True)
    df_si = load_lab_data(si_lab_path, is_mg=False)
    
    # 3. Model parameters
    # MgFeSiO4 composition and density
    comp_mgfesi = {'Mg': 1.0, 'Fe': 1.0, 'Si': 1.0, 'O': 4.0}
    rho_mgfesi = 3.71
    
    C_EV_UM = 1.23984193  # Energy (eV) * Wavelength (um)
    
    # --- Mg Edge: energies in [1100, 1550] eV ---
    energies_mg = df_mg['energy_ev'].values
    wavs_mg = C_EV_UM / energies_mg
    
    # Our method for MgFeSiO4 (Henke Extension)
    m_our_mg = henke.compute_refractive_index(comp_mgfesi, rho_mgfesi, wavs_mg)
    abs_n_minus_1_our_mg = np.abs(np.real(m_our_mg) - 1.0)
    k_our_mg = -np.imag(m_our_mg)
    
    # Draine 2003 raw data points in this range
    df_dr_mg = df_dr_all[(df_dr_all['energy_ev'] >= 1100.0) & (df_dr_all['energy_ev'] <= 1550.0)]
    
    # --- Si Edge: energies in [1500, 2300] eV ---
    energies_si = df_si['energy_ev'].values
    wavs_si = C_EV_UM / energies_si
    
    # Our method for MgFeSiO4 (Henke Extension)
    m_our_si = henke.compute_refractive_index(comp_mgfesi, rho_mgfesi, wavs_si)
    abs_n_minus_1_our_si = np.abs(np.real(m_our_si) - 1.0)
    k_our_si = -np.imag(m_our_si)
    
    # Draine 2003 raw data points in this range
    df_dr_si = df_dr_all[(df_dr_all['energy_ev'] >= 1500.0) & (df_dr_all['energy_ev'] <= 2300.0)]
    
    # 4. Calculate statistics
    print("\n==================== Magnesium (Mg) Edge Statistics ====================")
    diff_our_mg_n = 100 * np.abs(abs_n_minus_1_our_mg - df_mg['abs_n_minus_1'].values) / df_mg['abs_n_minus_1'].values
    diff_our_mg_k = 100 * np.abs(k_our_mg - df_mg['k'].values) / df_mg['k'].values
    print(f"Our Method (MgFeSiO4 Henke) vs Lab:")
    print(f"  abs(n-1) Median Diff: {np.median(diff_our_mg_n):.2f}%")
    print(f"  k Median Diff:        {np.median(diff_our_mg_k):.2f}%")
    
    if len(df_dr_mg) > 0:
        lab_n_interp = interp1d(df_mg['energy_ev'].values, df_mg['abs_n_minus_1'].values, fill_value='extrapolate')
        lab_k_interp = interp1d(df_mg['energy_ev'].values, df_mg['k'].values, fill_value='extrapolate')
        
        lab_n_at_dr = lab_n_interp(df_dr_mg['energy_ev'].values)
        lab_k_at_dr = lab_k_interp(df_dr_mg['energy_ev'].values)
        
        diff_dr_mg_n = 100 * np.abs(df_dr_mg['abs_n_minus_1'].values - lab_n_at_dr) / lab_n_at_dr
        diff_dr_mg_k = 100 * np.abs(df_dr_mg['k'].values - lab_k_at_dr) / lab_k_at_dr
        print(f"Draine 2003 Tabulated vs Lab (evaluated on the {len(df_dr_mg)} overlapping raw points):")
        print(f"  abs(n-1) Median Diff: {np.median(diff_dr_mg_n):.2f}%")
        print(f"  k Median Diff:        {np.median(diff_dr_mg_k):.2f}%")
    else:
        print("Draine 2003 Tabulated vs Lab: No overlapping tabulated points in the Mg range.")
        
    print("\n==================== Silicon (Si) Edge Statistics ====================")
    diff_our_si_n = 100 * np.abs(abs_n_minus_1_our_si - df_si['abs_n_minus_1'].values) / df_si['abs_n_minus_1'].values
    diff_our_si_k = 100 * np.abs(k_our_si - df_si['k'].values) / df_si['k'].values
    print(f"Our Method (MgFeSiO4 Henke) vs Lab:")
    print(f"  abs(n-1) Median Diff: {np.median(diff_our_si_n):.2f}%")
    print(f"  k Median Diff:        {np.median(diff_our_si_k):.2f}%")
    
    if len(df_dr_si) > 0:
        lab_n_interp = interp1d(df_si['energy_ev'].values, df_si['abs_n_minus_1'].values, fill_value='extrapolate')
        lab_k_interp = interp1d(df_si['energy_ev'].values, df_si['k'].values, fill_value='extrapolate')
        
        lab_n_at_dr = lab_n_interp(df_dr_si['energy_ev'].values)
        lab_k_at_dr = lab_k_interp(df_dr_si['energy_ev'].values)
        
        diff_dr_si_n = 100 * np.abs(df_dr_si['abs_n_minus_1'].values - lab_n_at_dr) / lab_n_at_dr
        diff_dr_si_k = 100 * np.abs(df_dr_si['k'].values - lab_k_at_dr) / lab_k_at_dr
        print(f"Draine 2003 Tabulated vs Lab:")
        print(f"  abs(n-1) Median Diff: {np.median(diff_dr_si_n):.2f}%")
        print(f"  k Median Diff:        {np.median(diff_dr_si_k):.2f}%")
    else:
        print("Draine 2003 Tabulated vs Lab: No overlapping tabulated points in the Si range (maximum tabulated energy is 1239.8 eV).")
        
    # 5. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Styles and colors
    lab_style = {'color': 'k', 'label': 'Rogantini Lab Data', 'lw': 2.5, 'alpha': 0.8}
    our_style = {'color': 'r', 'label': 'Our Method (MgFeSiO4 Henke)', 'lw': 2.5, 'ls': '--'}
    dr_tab_style = {'color': 'b', 'label': 'Draine 2003 Tabulated (Raw Points)', 'ls': ':', 'lw': 2.5}
    
    # --- Mg Edge: abs(n-1) ---
    ax = axes[0, 0]
    ax.plot(energies_mg, df_mg['abs_n_minus_1'], **lab_style)
    ax.plot(energies_mg, abs_n_minus_1_our_mg, **our_style)
    if len(df_dr_mg) > 0:
        ax.plot(df_dr_mg['energy_ev'], df_dr_mg['abs_n_minus_1'], **dr_tab_style)
    else:
        ax.plot([], [], **dr_tab_style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$|n - 1|$')
    ax.set_title(r'Mg K-edge: $|n - 1|$', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # --- Mg Edge: k ---
    ax = axes[0, 1]
    ax.plot(energies_mg, df_mg['k'], **lab_style)
    ax.plot(energies_mg, k_our_mg, **our_style)
    if len(df_dr_mg) > 0:
        ax.plot(df_dr_mg['energy_ev'], df_dr_mg['k'], **dr_tab_style)
    else:
        ax.plot([], [], **dr_tab_style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$k$')
    ax.set_title(r'Mg K-edge: Imaginary Part $k$', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # --- Si Edge: abs(n-1) ---
    ax = axes[1, 0]
    ax.plot(energies_si, df_si['abs_n_minus_1'], **lab_style)
    ax.plot(energies_si, abs_n_minus_1_our_si, **our_style)
    if len(df_dr_si) > 0:
        ax.plot(df_dr_si['energy_ev'], df_dr_si['abs_n_minus_1'], **dr_tab_style)
    else:
        ax.plot([], [], **dr_tab_style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel(r'$|n - 1|$')
    ax.set_title(r'Si K-edge: $|n - 1|$', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # --- Si Edge: k ---
    ax = axes[1, 1]
    ax.plot(energies_si, df_si['k'], **lab_style)
    ax.plot(energies_si, k_our_si, **our_style)
    if len(df_dr_si) > 0:
        ax.plot(df_dr_si['energy_ev'], df_dr_si['k'], **dr_tab_style)
    else:
        ax.plot([], [], **dr_tab_style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel(r'$k$')
    ax.set_title(r'Si K-edge: Imaginary Part $k$', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_dir = os.path.join(repo_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    out_path = os.path.join(results_dir, 'rogantini_comparison.png')
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    compare_constants()
