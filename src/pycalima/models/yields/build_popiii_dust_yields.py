#!/usr/bin/env python
"""
Build binned dust yields for Pop III Supernovae (CCSNe and PISNe) based on 
Nozawa et al. (2003) dust yields and size distributions.

This script parses a user configuration JSON file (defining the dust size bins),
determines the fraction of the grain size distribution that lies in each bin 
for each composition and SN channel, and constructs the binned yield tables.

By default, the script tests this for graphite ('C' in Nozawa) and silicate 
('Mg2SiO4' in Nozawa) using example_ic.json, saves the tables in 
model_data/nozawa_dust_yields/, and makes a diagnostic plot.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Resolve the repository root and add to path
repo_root = Path(__file__).parents[2].resolve()

from pycalima.galaxysam.yield_models import load_nozawa2003_dust_yields, load_nozawa2003_dust_dist

# Standard mapping of user composition labels to Nozawa et al. (2003) species
COMPOSITION_MAP = {
    'graphite': 'C',
    'silicate': 'Mg2SiO4',
    'c': 'C',
    'sil': 'Mg2SiO4',
    'mg2sio4': 'Mg2SiO4',
    'si': 'Si',
    'silicon': 'Si',
    'fes': 'FeS',
    'sio2': 'SiO2',
    'fe': 'Fe',
    'iron': 'Fe',
    'mgo': 'MgO',
    'mgsio3': 'MgSiO3',
    'al2o3': 'Al2O3',
    'fe3o4': 'Fe3O4'
}

# Mapping of yield channels to distribution channels in the Nozawa files
DIST_CHANNEL_MAP = {
    'CCSNe (unmixed)': 'CCSNe (unmixed)',
    'PISNe (unmixed)': 'PISNe (unmixed) P170',
    'CCSNe (mixed)': 'CCSNe (mixed) C25',
    'PISNe (mixed)': 'PISNe (mixed) P200'
}

def load_dust_bins(config_path):
    """
    Parse the user's initial conditions JSON file to extract dust bins.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with config_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)
        
    dust_bins = []
    for bd in cfg.get('dust_bins', []):
        bin_id = bd['id']
        composition = bd.get('composition', 'graphite')
        
        # Determine size limits in micron and cm
        asize_um = float(bd.get('grain_size_micron', 0.1))
        asize_cm = asize_um * 1.0e-4
        
        amin_um = float(bd.get('amin_micron', asize_um * 0.5))
        amax_um = float(bd.get('amax_micron', asize_um * 2.0))
        amin_cm = amin_um * 1.0e-4
        amax_cm = amax_um * 1.0e-4
        
        dust_bins.append({
            'id': bin_id,
            'composition': composition,
            'asize_micron': asize_um,
            'amin_micron': amin_um,
            'amax_micron': amax_um,
            'amin_cm': amin_cm,
            'amax_cm': amax_cm
        })
    return dust_bins

def compute_bin_fraction(df_dist, dist_channel, nozawa_comp, amin_cm, amax_cm, integration_mode='mass'):
    """
    Integrate the size distribution dN/da to find the fraction in [amin_cm, amax_cm].
    
    Uses substitution u = log10(a) to perform robust numerical integration in log-space.
    """
    # Filter distribution
    df_filt = df_dist[(df_dist['channel'] == dist_channel) & (df_dist['composition'] == nozawa_comp)]
    if df_filt.empty:
        return 0.0
        
    df_filt = df_filt.sort_values('grain_size')
    a_pts = df_filt['grain_size'].to_numpy()
    dnda_pts = df_filt['dN_da'].to_numpy()
    
    # Avoid log(0)
    dnda_pts = np.maximum(dnda_pts, 1e-30)
    
    log_a = np.log10(a_pts)
    log_dnda = np.log10(dnda_pts)
    
    # Linear interpolation in log-log space.
    # Outside the range, we assume dN/da is 0, i.e., log10(dN/da) is -99.0
    interp_func = interp1d(log_a, log_dnda, kind='linear', bounds_error=False, fill_value=-99.0)
    
    # Distribution limits
    u_dist_min = log_a.min()
    u_dist_max = log_a.max()
    
    # Bin limits
    u_bin_min = np.log10(amin_cm)
    u_bin_max = np.log10(amax_cm)
    
    # Overlap range
    u_start = max(u_dist_min, u_bin_min)
    u_end = min(u_dist_max, u_bin_max)
    
    if u_start >= u_end:
        return 0.0
        
    # Integrands (substitution: a = 10^u, da = ln(10)*10^u du, ln(10) cancels out)
    if integration_mode == 'mass':
        # Mass integral \int a^3 dN/da da = \int 10^(4u + f(u)) du
        integrand = lambda u: 10.0**(4.0 * u + interp_func(u))
    elif integration_mode == 'number':
        # Number integral \int dN/da da = \int 10^(u + f(u)) du
        integrand = lambda u: 10.0**(u + interp_func(u))
    else:
        raise ValueError(f"Unknown integration_mode: {integration_mode}")
        
    num_val, _ = quad(integrand, u_start, u_end, limit=100)
    denom_val, _ = quad(integrand, u_dist_min, u_dist_max, limit=100)
    
    if denom_val <= 0.0:
        return 0.0
        
    return num_val / denom_val

def get_yield_at_nominal_masses(df_yields, channel, nozawa_comp, nominal_masses):
    """
    Interpolate Nozawa total yields to clean nominal progenitor mass grids.
    """
    df_filtered = df_yields[(df_yields['channel'] == channel) & (df_yields['composition'] == nozawa_comp)]
    if df_filtered.empty:
        return np.zeros_like(nominal_masses)
        
    df_sorted = df_filtered.sort_values('progenitor_mass')
    masses = df_sorted['progenitor_mass'].to_numpy()
    yields = df_sorted['dust_mass'].to_numpy()
    
    if len(masses) == 1:
        return np.full_like(nominal_masses, yields[0])
        
    f = interp1d(masses, yields, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(nominal_masses)

def build_and_save_tables(config_path, output_dir, integration_mode='mass'):
    """
    Main function to compute fractions, scale yields, save files, and create plots.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dust bins from: {config_path}")
    dust_bins = load_dust_bins(config_path)
    print(f"Found {len(dust_bins)} dust bins in configuration:")
    for db in dust_bins:
        print(f"  - {db['id']}: {db['composition']} ({db['amin_micron']} to {db['amax_micron']} um)")
        
    # Load raw yields and size distributions
    df_yields = load_nozawa2003_dust_yields()
    df_dist = load_nozawa2003_dust_dist()
    
    # Define nominal mass grids
    nominal_grids = {
        'CCSNe': np.array([13.0, 20.0, 25.0, 30.0]),
        'PISNe': np.array([170.0, 200.0])
    }
    
    # Pre-calculate fractions for each channel and each dust bin
    fractions = {} # (channel, bin_id) -> fraction
    print(f"\nComputing size distribution fractions (mode: {integration_mode}):")
    
    for channel in DIST_CHANNEL_MAP.keys():
        dist_channel = DIST_CHANNEL_MAP[channel]
        print(f"  Channel: {channel}")
        
        # Group sum verification
        comp_sums = {}
        
        for db in dust_bins:
            bin_id = db['id']
            user_comp = db['composition']
            nozawa_comp = COMPOSITION_MAP.get(user_comp.lower())
            
            if not nozawa_comp:
                print(f"    Warning: No matching Nozawa composition for '{user_comp}' (bin {bin_id})")
                fractions[(channel, bin_id)] = 0.0
                continue
                
            frac = compute_bin_fraction(df_dist, dist_channel, nozawa_comp, db['amin_cm'], db['amax_cm'], integration_mode)
            fractions[(channel, bin_id)] = frac
            print(f"    - Bin {bin_id} ({nozawa_comp}): fraction = {frac:.4f}")
            
            comp_sums[nozawa_comp] = comp_sums.get(nozawa_comp, 0.0) + frac
            
        for comp, total_frac in comp_sums.items():
            print(f"    => Total fraction captured for {comp}: {total_frac:.4f}")
            
    # Compute binned yields and save tables
    results = {}
    
    print("\nConstructing and saving tables:")
    for channel in DIST_CHANNEL_MAP.keys():
        model_type = 'CCSNe' if 'CCSNe' in channel else 'PISNe'
        nominal_masses = nominal_grids[model_type]
        
        ch_bin_yields = {}
        ch_total_yields = {}
        
        # Create output file header
        header_lines = [
            f"# Pop III SN dust yields from Nozawa et al. (2003)",
            f"# SN Channel: {channel}",
            f"# Size distribution integration: {integration_mode}-weighted",
            f"# Bins defined in: {config_path.name}",
            f"#"
        ]
        
        # Calculate yield for each bin and gather total yields
        for db in dust_bins:
            bin_id = db['id']
            user_comp = db['composition']
            nozawa_comp = COMPOSITION_MAP.get(user_comp.lower())
            
            # Fetch total yield interpolated to nominal masses
            if nozawa_comp:
                total_y = get_yield_at_nominal_masses(df_yields, channel, nozawa_comp, nominal_masses)
                ch_total_yields[nozawa_comp] = total_y
                frac = fractions[(channel, bin_id)]
                bin_y = total_y * frac
            else:
                bin_y = np.zeros_like(nominal_masses)
                
            ch_bin_yields[bin_id] = bin_y
            
            header_lines.append(
                f"#   {bin_id}: {user_comp} ({db['amin_micron']:.3f}-{db['amax_micron']:.3f} um), "
                f"fraction={fractions.get((channel, bin_id), 0.0):.4f}"
            )
            
        header_lines.append("#")
        
        # Build the table columns
        cols = ['progenitor_mass'] + [db['id'] for db in dust_bins]
        df_out = pd.DataFrame(columns=cols)
        df_out['progenitor_mass'] = nominal_masses
        for db in dust_bins:
            df_out[db['id']] = ch_bin_yields[db['id']]
            
        # Format filename
        clean_channel = channel.replace(' ', '_').replace('(', '').replace(')', '').lower()
        filename = f"nozawa_popiii_{clean_channel}_yields.txt"
        file_path = output_dir / filename
        
        # Write to file
        with file_path.open('w', encoding='utf-8') as f:
            f.write('\n'.join(header_lines) + '\n')
            # Write data format: aligned columns
            df_out.to_string(f, index=False, justify='left', formatters={
                'progenitor_mass': lambda x: f"{x:12.3f}",
                **{db['id']: (lambda x: f"{x:15.8e}") for db in dust_bins}
            })
            
        print(f"  Saved: {file_path}")
        
        results[channel] = {
            'nominal_masses': nominal_masses,
            'bin_yields': ch_bin_yields,
            'total_yields': ch_total_yields
        }
        
    # Generate diagnostic plot
    plot_path = output_dir / "nozawa_popiii_yields.png"
    plot_popiii_yields(results, dust_bins, plot_path)
    
    return results

def plot_popiii_yields(results, dust_bins, output_path):
    """
    Create a clean, publication-quality 2x2 panel plot of the binned dust yields.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)
    
    channels = [
        ('CCSNe (unmixed)', axes[0, 0]),
        ('CCSNe (mixed)', axes[0, 1]),
        ('PISNe (unmixed)', axes[1, 0]),
        ('PISNe (mixed)', axes[1, 1])
    ]
    
    # Nice tailormade color palette for graphite and silicate bins
    comp_colors = {
        'graphite': ['#e74c3c', '#d35400', '#f39c12', '#c0392b'], # Reds/Oranges
        'silicate': ['#3498db', '#1f618d', '#1abc9c', '#117a65']  # Blues/Teals
    }
    
    color_counters = {'graphite': 0, 'silicate': 0}
    bin_colors = {}
    for db in dust_bins:
        comp = db['composition'].lower()
        if comp in comp_colors:
            idx = color_counters[comp] % len(comp_colors[comp])
            bin_colors[db['id']] = comp_colors[comp][idx]
            color_counters[comp] += 1
        else:
            bin_colors[db['id']] = '#7f8c8d'
            
    for channel_name, ax in channels:
        ch_data = results[channel_name]
        masses = ch_data['nominal_masses']
        
        # Plot total Nozawa yields as background reference (dashed black/grey)
        for comp, total_y in ch_data['total_yields'].items():
            if np.any(total_y > 0):
                # Standard labeling for plot
                comp_lbl = 'Silicate (Mg2SiO4)' if comp == 'Mg2SiO4' else 'Graphite (C)'
                ax.plot(masses, total_y, linestyle='--', color='#2c3e50', alpha=0.4, 
                        linewidth=1.5, label=f"Total Nozawa {comp_lbl}", zorder=1)
        
        # Plot each bin's yield
        for db in dust_bins:
            bin_id = db['id']
            bin_y = ch_data['bin_yields'][bin_id]
            
            # Only plot if there is positive yield in the channel
            if np.any(bin_y > 0):
                lbl = f"{bin_id} ({db['composition']}, {db['amin_micron']:.3f}-{db['amax_micron']:.3f} $\\mu$m)"
                ax.plot(masses, bin_y, marker='o', markersize=6, color=bin_colors[bin_id], 
                        linewidth=2.0, label=lbl, zorder=3)
                ax.scatter(masses, bin_y, color=bin_colors[bin_id], s=40, zorder=4)
                
        ax.set_title(channel_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(r'Progenitor Mass ($M_\odot$)', fontsize=10)
        ax.set_ylabel(r'Dust Yield ($M_\odot$)', fontsize=10)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.set_yscale('log')
        
        # Custom limits and ticks per SN type
        if 'CCSNe' in channel_name:
            ax.set_xlim(12.0, 31.0)
            ax.set_xticks([13.0, 20.0, 25.0, 30.0])
            ax.set_ylim(1e-5, 2.0)
        else:
            ax.set_xlim(165.0, 205.0)
            ax.set_xticks([170.0, 200.0])
            ax.set_ylim(1e-3, 50.0)
            
        ax.legend(loc='best', fontsize=8.5, framealpha=0.9)
        
    fig.suptitle('Pop III SN Binned Dust Yields (Nozawa et al. 2003)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.91, hspace=0.25, wspace=0.22)
    
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved diagnostic yields plot: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Pop III binned dust yields from Nozawa et al. 2003.")
    parser.add_argument('--config', type=str, default=str(repo_root / 'solvers' / 'configs' / 'example_ic.json'),
                        help="Path to initial conditions JSON config file")
    parser.add_argument('--output', type=str, default=str(repo_root / 'model_data' / 'nozawa_dust_yields'),
                        help="Directory to save generated tables and plots")
    parser.add_argument('--mode', type=str, default='mass', choices=['mass', 'number'],
                        help="Integration weighting mode: 'mass'-weighted (default) or 'number'-weighted")
                        
    args = parser.parse_args()
    
    build_and_save_tables(args.config, args.output, args.mode)
