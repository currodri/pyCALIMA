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
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
})

from models.grain_size_config import get_bins, get_lognormal_parameters, build_lognormal_distribution, get_optical_props_path
from models.dust_model import LogNormal_Distribution
from models.dust_radiation.dust_oppacity import compute_isrf_averaged_cross_sections

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
        # Compute neutral and ionised optical properties on the same wavelength grid
        try:
            grain_size_cm, wavelengths_cm_neu, C_sca_neu, C_abs_neu, C_rp_neu = \
                interpolate_pah_cross_sections_2d(
                    'nPAH', grain_size_micron,
                    target_wavelengths=None, efficiency=False
                )
            _, wavelengths_cm_ion, C_sca_ion, C_abs_ion, C_rp_ion = \
                interpolate_pah_cross_sections_2d(
                    'iPAH', grain_size_micron,
                    target_wavelengths=None, efficiency=False
                )

            if wavelengths_cm_neu.shape != wavelengths_cm_ion.shape or \
               not np.allclose(wavelengths_cm_neu, wavelengths_cm_ion):
                raise RuntimeError('Neutral and ionised PAH wavelength grids do not match.')

            wavelengths_cm = wavelengths_cm_neu
        except Exception as e:
            print(f"Error computing optical properties for PAH bin {bin_id}: {e}")
            continue
        
        # Use a unified prefix for downstream tooling consistency.
        file_stem = f"averaged_cross_section_{bin_id}"
        output_filename = f"{file_stem}.txt"
        output_path = os.path.join(output_dir, output_filename)
        plot_filename = f"{file_stem}_quicklook.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        # Write to file
        try:
            isrf_avg_neu = compute_isrf_averaged_cross_sections(
                wavelengths_cm=wavelengths_cm,
                C_abs=C_abs_neu,
                C_sca=C_sca_neu,
                C_rp=C_rp_neu,
            )
            isrf_avg_ion = compute_isrf_averaged_cross_sections(
                wavelengths_cm=wavelengths_cm,
                C_abs=C_abs_ion,
                C_sca=C_sca_ion,
                C_rp=C_rp_ion,
            )

            with open(output_path, 'w') as f:
                f.write(f"# PAH optical properties\n")
                f.write(f"# Bin ID: {bin_id}\n")
                f.write(f"# Composition: {composition}\n")
                f.write(f"# PAH blocks: neutral then ionised\n")
                f.write(f"# Grain size a0: {a0} micron\n")
                f.write(f"# NWAV\n")
                f.write(f"{len(wavelengths_cm):d}\n")
                f.write(f"# ISRF-average: Mathis83, energy range [0.1, 13.6] eV\n")
                f.write(f"# ISRF_AVG_CROSS_SECTIONS_NEUTRAL_CM2: C_abs_ISRF C_sca_ISRF C_rp_ISRF\n")
                f.write(f"{isrf_avg_neu['C_abs_isrf']: .12E} {isrf_avg_neu['C_sca_isrf']: .12E} {isrf_avg_neu['C_rp_isrf']: .12E}\n")
                f.write(f"# ISRF_AVG_CROSS_SECTIONS_IONISED_CM2: C_abs_ISRF C_sca_ISRF C_rp_ISRF\n")
                f.write(f"{isrf_avg_ion['C_abs_isrf']: .12E} {isrf_avg_ion['C_sca_isrf']: .12E} {isrf_avg_ion['C_rp_isrf']: .12E}\n")
                f.write(f"# \n")
                f.write("# Columns: lambda_neutral[Angstrom] C_abs_neutral[cm^2] C_sca_neutral[cm^2] C_rp_neutral[cm^2] | lambda_ionised[Angstrom] C_abs_ionised[cm^2] C_sca_ionised[cm^2] C_rp_ionised[cm^2]\n")
                
                for j in range(len(wavelengths_cm)):
                    f.write(f"{wavelengths_cm[j]:14.6e} ")
                    f.write(f"{C_abs_neu[j]:14.6e} ")
                    f.write(f"{C_sca_neu[j]:14.6e} ")
                    f.write(f"{C_rp_neu[j]:14.6e} ")
                    f.write("| ")
                    f.write(f"{wavelengths_cm[j]:14.6e} ")
                    f.write(f"{C_abs_ion[j]:14.6e} ")
                    f.write(f"{C_sca_ion[j]:14.6e} ")
                    f.write(f"{C_rp_ion[j]:14.6e}\n")

            _save_optical_quicklook_plot(
                plot_path,
                wavelengths_cm,
                C_abs_neu,
                C_sca_neu,
                title=f"PAH {composition} bin {bin_rank}, a0={a0:.4g} micron"
            )
            
            print(f"  ✓ Exported {output_filename}")
            print(f"  ✓ Exported {plot_filename}")
        
        except Exception as e:
            print(f"  ✗ Error writing {output_filename}: {e}")
    
    print(f"PAH optical properties exported to {output_dir}/")


def test_pah_ionised_neutral_ratio(grain_size_micron=5e-4, Emin_eV=None, Emax_eV=None,
                                   output_path=None, show=False,
                                   optical_model='Draine', Nc=None):
    """Simple check plot: ionised/neutral PAH cross-section ratios at fixed size.

    Parameters
    ----------
    grain_size_micron : float
        Grain size in microns.
    Emin_eV : float or None
        Minimum photon energy in eV. If provided with Emax_eV, the corresponding
        wavelength interval is shaded in the plot.
    Emax_eV : float or None
        Maximum photon energy in eV. If provided with Emin_eV, the corresponding
        wavelength interval is shaded in the plot.
    output_path : str or None
        Output figure path.
    show : bool
        If True, show figure interactively.
    optical_model : str
        Optical dataset to use: 'Draine' (default) or 'Malloci'.
    Nc : int or None
        Number of carbon atoms for Malloci mode. If None, inferred from
        grain_size_micron using Nc ~= 468 * (a[nm])^3.
    """
    optical_model = str(optical_model).strip().lower()

    def _safe_ratio(num, den):
        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.0)

    curves = []
    inferred_Nc = None
    if optical_model == 'draine':
        _, wav_cm_n, C_sca_n, C_abs_n, C_rp_n = interpolate_pah_cross_sections_2d(
            'nPAH', grain_size_micron, target_wavelengths=None, efficiency=False
        )
        _, wav_cm_i, C_sca_i, C_abs_i, C_rp_i = interpolate_pah_cross_sections_2d(
            'iPAH', grain_size_micron, target_wavelengths=None, efficiency=False
        )

        if wav_cm_n.shape != wav_cm_i.shape or not np.allclose(wav_cm_n, wav_cm_i):
            raise ValueError('Neutral and ionised PAH wavelength grids do not match.')

        wav_micron = wav_cm_n * 1e4
        curves = [
            (r'$C_{\rm abs}^{\rm ion}/C_{\rm abs}^{\rm neu}$', _safe_ratio(C_abs_i, C_abs_n)),
            (r'$C_{\rm sca}^{\rm ion}/C_{\rm sca}^{\rm neu}$', _safe_ratio(C_sca_i, C_sca_n)),
        ]
    elif optical_model == 'malloci':
        from models.PAH_charge.PAH_photoelectric_heating import absorption_cross_section_Berne

        if Nc is None:
            a_nm = float(grain_size_micron) * 1e3
            inferred_Nc = max(1, int(round(468.0 * (a_nm ** 3))))
        else:
            inferred_Nc = int(Nc)

        E_a, E_n, E_c, E_dc, C_a, C_n, C_c, C_dc = absorption_cross_section_Berne(inferred_Nc)
        if len(E_n) == 0:
            raise RuntimeError('Malloci cross-section table is empty for the selected Nc.')

        hc_eV_micron = 1.23984193
        E_n = np.asarray(E_n, dtype=float)
        Cn_raw = np.asarray(C_n, dtype=float)
        Ca_raw = np.asarray(C_a, dtype=float)
        Cc_raw = np.asarray(C_c, dtype=float)
        Cdc_raw = np.asarray(C_dc, dtype=float)

        valid = np.isfinite(E_n) & (E_n > 0.0) & np.isfinite(Cn_raw)
        if not np.any(valid):
            raise RuntimeError('Malloci data does not contain valid E>0 samples.')

        wav_micron = hc_eV_micron / E_n[valid]
        order = np.argsort(wav_micron)
        wav_micron = wav_micron[order]

        Cn = Cn_raw[valid][order]
        Ca = Ca_raw[valid][order]
        Cc = Cc_raw[valid][order]
        Cdc = Cdc_raw[valid][order]

        curves = [
            (r'$C_{\rm abs}^{\rm anion}/C_{\rm abs}^{\rm neu}$', _safe_ratio(Ca, Cn)),
            (r'$C_{\rm abs}^{\rm cation}/C_{\rm abs}^{\rm neu}$', _safe_ratio(Cc, Cn)),
            (r'$C_{\rm abs}^{\rm dication}/C_{\rm abs}^{\rm neu}$', _safe_ratio(Cdc, Cn)),
        ]
    else:
        raise ValueError("optical_model must be 'Draine' or 'Malloci'.")

    hc_eV_micron = 1.23984193  # eV * micron
    band_label = None
    lam_min = lam_max = None
    if Emin_eV is not None or Emax_eV is not None:
        if Emin_eV is None or Emax_eV is None:
            raise ValueError('Provide both Emin_eV and Emax_eV, or neither.')
        Emin_eV = float(Emin_eV)
        Emax_eV = float(Emax_eV)
        if Emin_eV <= 0.0 or Emax_eV <= 0.0:
            raise ValueError('Emin_eV and Emax_eV must be > 0.')
        if Emax_eV < Emin_eV:
            Emin_eV, Emax_eV = Emax_eV, Emin_eV
        lam_max = hc_eV_micron / Emin_eV
        lam_min = hc_eV_micron / Emax_eV
        band_label = f'E=[{Emin_eV:.3g}, {Emax_eV:.3g}] eV'

    if output_path is None:
        out_dir = os.path.join('model_data', 'optical_properties')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(
            out_dir,
            f'pah_ratio_{optical_model}_a{grain_size_micron:.4g}micron.png'
        )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for label, yvals in curves:
        ax.plot(wav_micron, yvals, label=label, linewidth=2)
    if lam_min is not None and lam_max is not None:
        ax.axvspan(lam_min, lam_max, color='gray', alpha=0.15, label=band_label)
        ax.axvline(lam_min, color='gray', linestyle=':', linewidth=1.0, alpha=0.8)
        ax.axvline(lam_max, color='gray', linestyle=':', linewidth=1.0, alpha=0.8)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax.set_ylabel('Cross-section ratio to neutral')
    title = f'PAH cross-section ratios at a={grain_size_micron:.4g} micron ({optical_model.capitalize()})'
    if inferred_Nc is not None:
        title += f', Nc={inferred_Nc}'
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved ratio test plot: {output_path}")
    return {
        'output_path': output_path,
        'grain_size_micron': float(grain_size_micron),
        'optical_model': optical_model,
        'Nc': None if inferred_Nc is None else int(inferred_Nc),
        'Emin_eV': None if Emin_eV is None else float(Emin_eV),
        'Emax_eV': None if Emax_eV is None else float(Emax_eV),
        'lambda_min_micron': None if lam_min is None else float(lam_min),
        'lambda_max_micron': None if lam_max is None else float(lam_max),
        'wavelength_min_micron': float(np.min(wav_micron)),
        'wavelength_max_micron': float(np.max(wav_micron)),
    }

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
    parser.add_argument(
        '--test-ion-neutral-ratio',
        action='store_true',
        help='Run a test plot of ionised/neutral PAH cross-section ratios.'
    )
    parser.add_argument(
        '--ratio-size',
        type=float,
        default=5e-4,
        help='PAH grain size in microns for --test-ion-neutral-ratio.'
    )
    parser.add_argument(
        '--ratio-output',
        type=str,
        default=None,
        help='Optional output filename for --test-ion-neutral-ratio.'
    )
    parser.add_argument(
        '--Emin',
        type=float,
        default=None,
        help='Minimum photon energy in eV for wavelength-band overlay.'
    )
    parser.add_argument(
        '--Emax',
        type=float,
        default=None,
        help='Maximum photon energy in eV for wavelength-band overlay.'
    )
    parser.add_argument(
        '--ratio-optical-model',
        type=str,
        default='Draine',
        choices=['Draine', 'Malloci', 'draine', 'malloci'],
        help='Optical model for ratio test: Draine or Malloci.'
    )
    parser.add_argument(
        '--ratio-Nc',
        type=int,
        default=None,
        help='Nc for Malloci ratio test. If omitted, inferred from --ratio-size.'
    )
    args = parser.parse_args()

    if args.test_ion_neutral_ratio:
        test_pah_ionised_neutral_ratio(
            grain_size_micron=args.ratio_size,
            Emin_eV=args.Emin,
            Emax_eV=args.Emax,
            output_path=args.ratio_output,
            show=False,
            optical_model=args.ratio_optical_model,
            Nc=args.ratio_Nc,
        )
    else:
        export_pah_optical_properties(config_path=args.config)
