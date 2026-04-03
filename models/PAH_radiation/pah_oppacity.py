"""
PAH OPTICAL PROPERTIES COMPUTATION

Functions for computing and exporting optical properties of Polycyclic Aromatic Hydrocarbons (PAHs).
Includes reading PAH efficiencies from Draine-Lee tables and exporting cross-sections for different
PAH size bins defined in the grain configuration.

By: Curro Rodriguez (currodri@gmail.com)
"""

import os
import sys
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})

from models.grain_size_config import get_bins, get_lognormal_parameters, build_lognormal_distribution, get_optical_props_path
from models.dust_model import LogNormal_Distribution

PATH_OPTICS = str(get_optical_props_path())


def _save_optical_quicklook_plot(plot_path, wavelengths_cm, C_abs, C_sca, title):
    """Save a quick-look log-log plot of absorption, scattering and extinction."""
    wavelengths_micron = np.asarray(wavelengths_cm) * 1e4
    C_abs = np.asarray(C_abs)
    C_sca = np.asarray(C_sca)
    C_ext = C_abs + C_sca

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.loglog(wavelengths_micron, C_abs, label='C_abs', linewidth=2)
    ax.loglog(wavelengths_micron, C_sca, label='C_sca', linewidth=2)
    ax.loglog(wavelengths_micron, C_ext, label='C_ext', linewidth=2)
    ax.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$C$ [cm$^2$]')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _parse_pah_data_line(line, expected_cols=5):
    """Parse one PAH table data line robustly.

    Handles malformed rows where the last two scientific-notation values are
    concatenated without whitespace (e.g. ``2.048E-12-1.07E-08``).
    """
    clean = line.strip().replace('D', 'E').replace('d', 'e')
    if not clean:
        return None

    # Extract scientific-notation numbers even when adjacent without spaces.
    vals = re.findall(r'[+-]?\d+(?:\.\d+)?E[+-]?\d+', clean)
    if len(vals) >= expected_cols:
        return np.array([float(v) for v in vals[:expected_cols]], dtype=float)

    # Fallback for non-scientific plain floats, if present.
    arr = np.fromstring(clean, dtype=float, sep=' ')
    if arr.size >= expected_cols:
        return arr[:expected_cols]

    raise ValueError(f"Could not parse PAH row into {expected_cols} columns: {line.rstrip()}")


def pah_efficiencies(filename, verbose=False):
    """
    Read and parse PAH efficiency tables from Draine-Lee format files.
    
    This function reads optical efficiency tables (Q_ext, Q_abs, Q_sca, g) for PAHs
    at different grain sizes from a standardized formatted file.
    
    Parameters
    ----------
    filename : str
        Path to PAH efficiency file (e.g., 'li_draine_2001/PAHion_30')
    verbose : bool
        If True, print diagnostic information during parsing.
        
    Returns
    -------
    tuple
        (nwav, data, columns, dust_type) where:
        - nwav: number of wavelength points
        - data: dict mapping size strings to (nwav, 5) arrays [w, Q_ext, Q_abs, Q_sca, g]
        - columns: list of column names
        - dust_type: string identifier from file header
    """
    columns = ['w(micron)', 'Q_ext', 'Q_abs', 'Q_sca', 'g=<cos>']
    data = {}
    
    with open(filename) as f:
        # Read header (9 lines for PAH files)
        for i in range(0, 9):
            hd = f.readline()
            if i == 0:
                dust_type = hd
            elif i == 7:
                info = list(filter(None, hd.split(' ')))
                nrad = int(info[0])
                amin = float(info[1])
                amax = float(info[2])
            elif i == 8:
                info = list(filter(None, hd.split(' ')))
                nwav = int(info[0])
                wmin = float(info[1])
                wmax = float(info[2])
        
        if verbose:
            print(dust_type, nrad, nwav)
        
        # Read efficiency blocks for each size
        while True:
            f.readline()  # Blank line
            myarray = np.zeros((nwav, 5))
            a = str(f.readline().split(' ')[0])
            if a == '':
                if verbose:
                    print('End of file')
                break
            f.readline()  # Column names line
            
            # Read efficiency data
            for i in range(0, nwav):
                line = f.readline()
                dig = _parse_pah_data_line(line, expected_cols=5)
                myarray[i, :] = dig
            data[a] = myarray
    
    return nwav, data, columns, dust_type


def interpolate_pah_cross_sections_2d(pah_type, grain_size, target_wavelengths=None,
                                      efficiency=False, data_table=None):
    """
    Interpolate PAH cross sections in both size and wavelength dimensions.
    
    Parameters
    ----------
    pah_type : str
        PAH type: 'iPAH' (ionized) or 'nPAH'/'PAH' (neutral)
    grain_size : float
        Target grain size in microns
    target_wavelengths : array-like, optional
        Wavelengths in microns to interpolate to. If None, uses native wavelengths.
    efficiency : bool
        If True, return Q values (dimensionless); otherwise return C (cm^2)
    data_table : tuple, optional
        Pre-loaded (nwav, data, columns, name) tuple to avoid re-reading files
        
    Returns
    -------
    tuple
        (grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp) with units in cm
    """
    # Read table if not provided
    if data_table is None:
        if pah_type == 'iPAH':
            filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30')
            nwav, data, columns, name = pah_efficiencies(filename)
        elif pah_type == 'nPAH' or pah_type == 'PAH':
            filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
            nwav, data, columns, name = pah_efficiencies(filename)
        else:
            raise ValueError('PAH type not recognised: ', pah_type)
    else:
        nwav, data, columns, name = data_table
    
    # Build arrays of sizes and native wavelengths robustly
    keys = list(data.keys())
    sizes_raw = np.array([float(k) for k in keys])
    
    # Use first table to get native wavelength grid and detect ordering
    first_arr = data[keys[0]]
    wcol = columns.index('w(micron)')
    native_wav = first_arr[:, wcol].copy()
    
    # If wavelength axis is decreasing, flip it
    flip_wav = False
    if native_wav[0] > native_wav[-1]:
        flip_wav = True
        native_wav = native_wav[::-1]
    
    # Sort sizes ascending
    order = np.argsort(sizes_raw)
    native_sizes = sizes_raw[order]
    sorted_keys = [keys[i] for i in order]
    
    nwav_native = native_wav.size
    
    # Determine target wavelengths (in microns)
    if target_wavelengths is None:
        target_wav = native_wav.copy()
    else:
        target_wav = np.array(target_wavelengths, dtype=float)
    
    # For each native size, interpolate Q vs wavelength
    nsizes = native_sizes.size
    ntarget_wav = target_wav.size
    Q_abs_table = np.zeros((nsizes, ntarget_wav))
    Q_sca_table = np.zeros((nsizes, ntarget_wav))
    Q_ext_table = np.zeros((nsizes, ntarget_wav))
    g_table = np.zeros((nsizes, ntarget_wav))
    
    for i, key in enumerate(sorted_keys):
        arr = data[key]
        if flip_wav:
            arr = arr[::-1, :]
        
        # Get native Q arrays
        qext_native = arr[:, columns.index('Q_ext')] if 'Q_ext' in columns else None
        qabs_native = arr[:, columns.index('Q_abs')]
        qsca_native = arr[:, columns.index('Q_sca')] if 'Q_sca' in columns else np.zeros_like(qabs_native)
        g_native = arr[:, columns.index('g=<cos>')] if 'g=<cos>' in columns else np.zeros_like(qabs_native)
        
        # Interpolate in log-log
        log_native_wav = np.log10(native_wav)
        for j, tw in enumerate(target_wav):
            if qext_native is not None and (qext_native > 0).all():
                Q_ext_table[i, j] = 10.0 ** np.interp(np.log10(tw), log_native_wav, np.log10(qext_native))
            if (qabs_native > 0).all():
                Q_abs_table[i, j] = 10.0 ** np.interp(np.log10(tw), log_native_wav, np.log10(qabs_native))
            else:
                Q_abs_table[i, j] = np.interp(tw, native_wav, qabs_native)
            if (qsca_native > 0).all():
                Q_sca_table[i, j] = 10.0 ** np.interp(np.log10(tw), log_native_wav, np.log10(qsca_native))
            else:
                Q_sca_table[i, j] = np.interp(tw, native_wav, qsca_native)
            g_table[i, j] = np.interp(tw, native_wav, g_native)
    
    # Interpolate over size to the desired grain_size
    log_native_a = np.log10(native_sizes)
    log_target_a = np.log10(grain_size)
    
    Q_abs_target = np.zeros(ntarget_wav)
    Q_sca_target = np.zeros(ntarget_wav)
    Q_ext_target = np.zeros(ntarget_wav)
    g_target = np.zeros(ntarget_wav)
    
    for j in range(ntarget_wav):
        qext_vs_a = Q_ext_table[:, j] if qext_native is not None else None
        qabs_vs_a = Q_abs_table[:, j]
        qsca_vs_a = Q_sca_table[:, j]
        
        if qext_vs_a is not None and (qext_vs_a > 0).all():
            Q_ext_target[j] = 10.0 ** np.interp(log_target_a, log_native_a, np.log10(qext_vs_a))
        if (qabs_vs_a > 0).all():
            Q_abs_target[j] = 10.0 ** np.interp(log_target_a, log_native_a, np.log10(qabs_vs_a))
        else:
            Q_abs_target[j] = np.interp(grain_size, native_sizes, qabs_vs_a)
        if (qsca_vs_a > 0).all():
            Q_sca_target[j] = 10.0 ** np.interp(log_target_a, log_native_a, np.log10(qsca_vs_a))
        else:
            Q_sca_target[j] = np.interp(grain_size, native_sizes, qsca_vs_a)
        g_target[j] = np.interp(grain_size, native_sizes, g_table[:, j])
    
    # Compute Q_rp and convert to cross sections if requested
    Q_rp = Q_abs_target + (1.0 - g_target) * Q_sca_target
    
    # Geometric area (micron^2) then convert to cm^2
    area_cm2 = np.pi * (grain_size ** 2) * 1e-8
    
    wavelengths_cm = target_wav * 1e-4
    grain_size_cm = grain_size * 1e-4
    
    if efficiency:
        C_sca = Q_sca_target
        C_abs = Q_abs_target
        C_rp = Q_rp
    else:
        C_sca = Q_sca_target * area_cm2
        C_abs = Q_abs_target * area_cm2
        C_rp = Q_rp * area_cm2
    
    return grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp


def export_pah_optical_properties(output_dir='model_data/optical_properties', config_path=None):
    """
    Batch export optical properties for all PAH bins defined in grain configuration.
    
    This function:
    1. Reads all PAH bins from grain_size_distribution.json (or specified config)
    2. For each PAH bin, computes and saves its optical properties
    3. Outputs cross-section tables to model_data/optical_properties/
    
    Parameters
    ----------
    output_dir : str
        Output directory for optical property files
    config_path : str, optional
        Path to JSON configuration file. If provided, temporarily sets the config.
    """
    # Set config path if provided
    if config_path:
        from models.grain_size_config import set_config_path
        set_config_path(config_path)
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load PAH bins from configuration
    pah_bins = get_bins(is_pah=True)
    
    if not pah_bins:
        print("No PAH bins found in grain configuration.")
        return
    
    # Define PAH types mapping
    pah_type_map = {
        'graphite': 'nPAH',  # Neutral PAH for graphite
    }
    
    print(f"Exporting optical properties for {len(pah_bins)} PAH bins...")
    
    # Process each PAH bin
    for bin_info in pah_bins:
        bin_id = bin_info['id']
        composition = bin_info['composition']
        bin_rank = bin_info['bin_rank']
        
        # Get grain size parameters for this bin
        lognormal_params = get_lognormal_parameters(bin_id)
        if not lognormal_params:
            print(f"Warning: Could not find parameters for bin {bin_id}")
            continue
        
        a0 = lognormal_params.get('a0')
        if a0 is None:
            print(f"Warning: No a0 parameter for bin {bin_id}")
            continue
        
        grain_size_micron = a0
        pah_type = pah_type_map.get(composition, 'nPAH')
        
        # Compute optical properties
        try:
            grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp = \
                interpolate_pah_cross_sections_2d(
                    pah_type, grain_size_micron,
                    target_wavelengths=None, efficiency=False
                )
        except Exception as e:
            print(f"Error computing optical properties for PAH bin {bin_id}: {e}")
            continue
        
        # Create output filename based on bin metadata
        output_filename = f"{composition}_pah_bin_{bin_rank}_a{a0:.4g}micron.txt"
        output_path = os.path.join(output_dir, output_filename)
        plot_filename = f"{composition}_pah_bin_{bin_rank}_a{a0:.4g}micron_quicklook.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        # Write to file
        try:
            with open(output_path, 'w') as f:
                f.write(f"# PAH optical properties\n")
                f.write(f"# Bin ID: {bin_id}\n")
                f.write(f"# Composition: {composition}\n")
                f.write(f"# PAH Type: {pah_type}\n")
                f.write(f"# Grain size a0: {a0} micron\n")
                f.write(f"# \n")
                f.write(f"# Columns: wavelength[Angstrom] C_abs[cm^2] C_sca[cm^2] C_rp[cm^2]\n")
                
                for j in range(len(wavelengths_cm)):
                    f.write(f"{wavelengths_cm[j]:14.6e} ")
                    f.write(f"{C_abs[j]:14.6e} ")
                    f.write(f"{C_sca[j]:14.6e} ")
                    f.write(f"{C_rp[j]:14.6e}\n")

            _save_optical_quicklook_plot(
                plot_path,
                wavelengths_cm,
                C_abs,
                C_sca,
                title=f"PAH {composition} bin {bin_rank}, a0={a0:.4g} micron"
            )
            
            print(f"  ✓ Exported {output_filename}")
            print(f"  ✓ Exported {plot_filename}")
        
        except Exception as e:
            print(f"  ✗ Error writing {output_filename}: {e}")
    
    print(f"PAH optical properties exported to {output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export optical properties for all PAH bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    args = parser.parse_args()
    
    export_pah_optical_properties(config_path=args.config)
