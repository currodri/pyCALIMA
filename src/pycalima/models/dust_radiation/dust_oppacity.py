"""
DUST EFFICIENCIES TABLES

This set of tools have been constructed such that the public
tables from B. Draine and co. can be read, visualised
and reorganised in look-up tables for RAMSES Dust-RTZ

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import some libraries
import os
import concurrent.futures
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
})
import re
from pathlib import Path
from pycalima.models.dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,\
                        LogNormal_Distribution,PowerLaw_ExpCutoff_Distribution, \
                        Classical_LogNormal_Distribution
from pycalima.models.grain_size_config import get_optical_props_path, get_lognormal_parameters, load_grain_size_config
from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

PATH_OPTICS = str(get_optical_props_path())
_REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_TABLES = str(_REPO_ROOT / 'model_data' / 'optical_properties')
PATH_MODEL_OPTICAL_OUTPUT = _REPO_ROOT / 'model_data' / 'optical_properties'
PATH_EXTERNAL_DATA = _REPO_ROOT / 'external_data'
# Note: PAH-specific functions are now in models.PAH_radiation.pah_oppacity
# Functions


def _table_output_path(path):
    if os.path.isabs(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    out_path = os.path.join(PATH_TABLES, path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def read_dielectric_file(filename):
    """
    Read a dielectric data file in either Draine 2003 or astronomical silicate format.

    Returns a metadata dictionary with the parsed table stored under ``table``.
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    is_draine2003 = lines[0].startswith("ICOMP=")

    metadata = {
        'icomp': None,
        'temperature_K': None,
    }

    if is_draine2003:
        metadata['icomp'] = lines[0].split(":", 1)[-1].strip()
        metadata['radius_micron'] = float(lines[1].split('=')[0].strip())
        metadata['temperature_K'] = float(lines[2].split('=')[0].strip())
        metadata['n_wavelengths'] = int(lines[3].split('=')[0].strip())
        col_names = ['wavelength_um', 'eps1_minus_1', 'eps2', 'Re_n_minus_1', 'Im_n']
        data_start = 5
    else:
        header_index = None
        for i, line in enumerate(lines):
            if '=' in line and 'radius' in line:
                metadata['radius_micron'] = float(line.split('=')[0].strip())
            elif '=' in line and 'wavelengths' in line:
                metadata['n_wavelengths'] = int(line.split('=')[0].strip())
            elif line.lower().startswith('wave') or 'wave(' in line:
                header_index = i
                break

        if header_index is None:
            raise ValueError(f'Could not locate dielectric table header in {filename}')

        col_names = ['wavelength_um', 'eps1_minus_1', 'eps2', 'Re_n_minus_1', 'Im_n']
        data_start = header_index + 1

    table_data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) == 5:
            table_data.append(list(map(float, parts)))

    metadata['table'] = pd.DataFrame(table_data, columns=col_names)
    return metadata


def save_imn_file(metadata, outfile):
    """
    Save wavelength (Angstrom) and Im_n from dielectric data in the same table
    directory used for exported optical properties.
    """
    df = metadata['table'].copy()

    if 'wavelength_um' not in df.columns or 'Im_n' not in df.columns:
        raise KeyError("Input table must contain 'wavelength_um' and 'Im_n' columns")

    df['wavelength_A'] = df['wavelength_um'].astype(float) * 1e4
    df_sorted = df.sort_values('wavelength_A', ascending=True)
    data = df_sorted[['wavelength_A', 'Im_n']].to_numpy(dtype=float)

    outpath = _table_output_path(outfile)
    with open(outpath, 'w') as f:
        f.write(f"{len(data):8d}\n")
        np.savetxt(f, data, fmt="%.12e %.12e")


def export_dielectric_tables_for_bin(bin_id, composition, output_dir=None):
    from pycalima.models.grain_size_config import get_bins, get_lognormal_parameters
    """
    Export the dielectric Im_n tables for a dust bin.

    The output filenames follow the same bin-stem convention as the cross-section
    export, using ``Im_n_<bin_id>``, ``Im_n_<bin_id>_pe`` and ``Im_n_<bin_id>_pa``.
    """
    composition_key = str(composition).lower()
    if output_dir is None:
        output_dir = PATH_MODEL_OPTICAL_OUTPUT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load bin info for header
    all_bins = get_bins()
    bin_info = next((b for b in all_bins if b['id'] == bin_id), {})
    bin_rank = bin_info.get('bin_rank', 'N/A')
    lognormal_params = get_lognormal_parameters(bin_id)
    a0 = lognormal_params.get('a0', 'N/A') if lognormal_params else 'N/A'

    source_dir = Path(PATH_OPTICS) / 'draine_lee_1984'
    if composition_key == 'graphite':
        sources = [
            (source_dir / 'callindex.out_CpeD03_0.10', f'Im_n_{bin_id}_pe'),
            (source_dir / 'callindex.out_CpaD03_0.10', f'Im_n_{bin_id}_pa'),
        ]
    elif composition_key == 'silicate':
        sources = [
            (source_dir / 'eps_suvSil', f'Im_n_{bin_id}'),
        ]
    else:
        raise ValueError(f"Unsupported composition '{composition}'.")

    saved_paths = []
    for src_path, out_stem in sources:
        metadata = read_dielectric_file(str(src_path))
        # Write directly to output_dir without routing through _table_output_path
        # (which prepends PATH_TABLES and would double the directory on paths
        # that already include model_data/optical_properties).
        df = metadata['table'].copy()
        df['wavelength_A'] = df['wavelength_um'].astype(float) * 1e4
        df_sorted = df.sort_values('wavelength_A', ascending=True)
        
        # Extract native wavelength and Im_n arrays
        wav_A = df_sorted['wavelength_A'].to_numpy(dtype=float)
        Im_n = df_sorted['Im_n'].to_numpy(dtype=float)
        
        # Ensure we avoid log10 of zero or negative values
        Im_n = np.maximum(Im_n, 1e-30)
        
        log10_wav_A = np.log10(wav_A)
        log10_Im_n = np.log10(Im_n)
        
        # Create a perfectly uniform log-spaced grid for wavelengths (in Angstroms)
        # preserving the native range and number of points.
        target_log10_wav_A = np.linspace(log10_wav_A.min(), log10_wav_A.max(), len(log10_wav_A))
        
        # Interpolate log10(Im_n) vs log10(wavelength) to the uniform grid.
        target_log10_Im_n = np.interp(target_log10_wav_A, log10_wav_A, log10_Im_n)
        
        # We export ACTUAL values (not logs) to be consistent with the header
        # and to match the Fortran code's expected input (which takes log10 itself).
        data_arr = np.column_stack([10**target_log10_wav_A, 10**target_log10_Im_n])
        
        from pycalima.models.grain_size_config import get_header_lines
        headers = get_header_lines(
            title="Dust dielectric properties (Im_n)",
            script_name="models/dust_radiation/dust_oppacity.py",
            bin_info=f"Bin ID: {bin_id}, Composition: {composition}, Bin rank: {bin_rank}, Grain size a0: {a0} micron",
            val_desc="Columns: lambda[Angstrom] Im_n"
        )

        dest = output_dir / out_stem
        with open(dest, 'w') as fout:
            for line in headers:
                fout.write(f"{line}\n")
            fout.write(f"# NWAV\n")
            fout.write(f"{len(data_arr):8d}\n")
            np.savetxt(fout, data_arr, fmt="%20.12e %20.12e")
        saved_paths.append(str(dest))

    return saved_paths


def compute_isrf_averaged_absorption_efficiency_all_sizes(E_min=0.1, E_max=13.6,
                                                          nE=2000,
                                                          save=True,
                                                          outfile='isrf_averaged_qabs_mathis83.csv',
                                                          print_integrated_uE=False):
    """Compute Mathis83-ISRF averaged Q_abs for all graphite and silicate sizes.

    The average is
        <Q_abs> = Integral[Q_abs(E) * u_E(E) dE] / Integral[u_E(E) dE)
    where ``u_E`` is given by ``Mathis83_radiation_field``.

    Parameters
    ----------
    E_min, E_max : float
        Integration limits in eV.
    nE : int
        Number of points in the integration energy grid.
    save : bool
        If True, save the output table to disk.
    outfile : str
        Output file path. Relative paths are written under ``PATH_TABLES``.
    print_integrated_uE : bool
        If True, print the integrated ISRF energy density over the selected
        energy range in erg cm^-3.

    Returns
    -------
    pandas.DataFrame
        Columns: grain_type, size_micron, size_angstrom, qabs_isrf_avg.
    """
    e_min = float(max(E_min, 1e-4))
    e_max = float(min(E_max, 13.6))
    if e_max <= e_min:
        raise ValueError(f'Invalid energy range: E_min={E_min}, E_max={E_max}')

    E_grid = np.logspace(np.log10(e_min), np.log10(e_max), int(nE))
    uE = np.array([Mathis83_radiation_field(float(E)) for E in E_grid], dtype=float)
    uE = np.where(np.isfinite(uE), uE, 0.0)
    denom = np.trapezoid(uE, E_grid)
    if denom <= 0.0:
        raise RuntimeError('Mathis83_radiation_field produced zero integrated energy density.')
    if print_integrated_uE:
        print(f"Integrated ISRF energy density [{e_min:.3g}, {e_max:.3g}] eV: {denom:.6e} erg cm^-3")
        hard_min = max(6.0, e_min)
        hard_max = min(13.6, e_max)
        if hard_max > hard_min:
            hard_mask = (E_grid >= hard_min) & (E_grid <= hard_max)
            if np.count_nonzero(hard_mask) >= 2:
                uE_hard = np.trapezoid(uE[hard_mask], E_grid[hard_mask])
            else:
                uE_hard = 0.0
            print(f"Integrated ISRF energy density [{hard_min:.3g}, {hard_max:.3g}] eV: {uE_hard:.6e} erg cm^-3")
        else:
            print("Integrated ISRF energy density [6, 13.6] eV: 0.000000e+00 erg cm^-3 (outside selected range)")

    conv_eum = 1.239841984  # E[eV] * lambda[um]
    records = []
    table_map = {
        'graphite': os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81'),
        'silicate': os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81'),
    }

    for grain_type, table_file in table_map.items():
        _, data, columns, _ = dust_efficiencies(table_file)
        wcol = columns.index('w(micron)')
        qcol = columns.index('Q_abs')

        for size_key in sorted(data.keys(), key=float):
            arr = np.asarray(data[size_key], dtype=float)
            wav_um = arr[:, wcol]
            q_abs = arr[:, qcol]

            E_tab = conv_eum / np.maximum(wav_um, 1e-30)
            order = np.argsort(E_tab)
            q_on_grid = np.interp(E_grid, E_tab[order], q_abs[order], left=0.0, right=0.0)

            q_avg = np.trapezoid(q_on_grid * uE, E_grid) / denom
            size_micron = float(size_key)
            records.append({
                'grain_type': grain_type,
                'size_micron': size_micron,
                'size_angstrom': size_micron * 1e4,
                'qabs_isrf_avg': float(q_avg),
            })

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(['grain_type', 'size_micron']).reset_index(drop=True)

    if save:
        if os.path.isabs(outfile):
            out = outfile
        else:
            os.makedirs(PATH_TABLES, exist_ok=True)
            out = os.path.join(PATH_TABLES, outfile)
        df.to_csv(out, index=False)
        print('Saved ISRF-averaged Q_abs table to', out)

    return df


def compute_isrf_averaged_cross_sections(wavelengths_cm, C_abs, C_sca, C_rp,
                                         E_min=0.1, E_max=13.6):
    """Compute Mathis83-ISRF averaged cross sections from spectral arrays.

    Parameters
    ----------
    wavelengths_cm : array-like
        Wavelength grid in cm.
    C_abs, C_sca, C_rp : array-like
        Cross-sections in cm^2 sampled on the same wavelength grid.
    E_min, E_max : float
        Energy integration limits in eV.

    Returns
    -------
    dict
        {'C_abs_isrf', 'C_sca_isrf', 'C_rp_isrf'} in cm^2.
    """
    wav_cm = np.asarray(wavelengths_cm, dtype=float)
    c_abs = np.asarray(C_abs, dtype=float)
    c_sca = np.asarray(C_sca, dtype=float)
    c_rp = np.asarray(C_rp, dtype=float)

    if not (wav_cm.size == c_abs.size == c_sca.size == c_rp.size):
        raise ValueError('wavelengths_cm, C_abs, C_sca and C_rp must have the same length.')

    e_min = float(max(E_min, 1e-6))
    e_max = float(min(E_max, 13.6))
    if e_max <= e_min:
        raise ValueError(f'Invalid energy range: E_min={E_min}, E_max={E_max}')

    energy_eV = 1.239841984e-4 / np.maximum(wav_cm, 1e-300)
    valid = np.isfinite(energy_eV) & np.isfinite(c_abs) & np.isfinite(c_sca) & np.isfinite(c_rp)
    valid &= (energy_eV >= e_min) & (energy_eV <= e_max)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError('Insufficient valid samples to compute ISRF-averaged cross sections.')

    e = energy_eV[valid]
    c_abs_e = c_abs[valid]
    c_sca_e = c_sca[valid]
    c_rp_e = c_rp[valid]

    order = np.argsort(e)
    e = e[order]
    c_abs_e = c_abs_e[order]
    c_sca_e = c_sca_e[order]
    c_rp_e = c_rp_e[order]

    u_e = np.array([Mathis83_radiation_field(float(x)) for x in e], dtype=float)
    u_e = np.where(np.isfinite(u_e), u_e, 0.0)
    denom = np.trapezoid(u_e, e)
    if denom <= 0.0:
        raise RuntimeError('Mathis83_radiation_field produced zero integrated energy density.')

    c_abs_isrf = np.trapezoid(c_abs_e * u_e, e) / denom
    c_sca_isrf = np.trapezoid(c_sca_e * u_e, e) / denom
    c_rp_isrf = np.trapezoid(c_rp_e * u_e, e) / denom
    return {
        'C_abs_isrf': float(c_abs_isrf),
        'C_sca_isrf': float(c_sca_isrf),
        'C_rp_isrf': float(c_rp_isrf),
    }


def plot_isrf_averaged_qabs_vs_size(E_min=0.1, E_max=13.6, nE=2000,
                                    savefile='qabs_vs_size_mathis83_isrf.pdf',
                                    also_save_table=True,
                                    tablefile='isrf_averaged_qabs_mathis83.csv'):
    """Plot ISRF-averaged Q_abs versus grain size for graphite and silicate.

    The figure is saved into ``model_data/optical_properties`` by default.
    """
    df = compute_isrf_averaged_absorption_efficiency_all_sizes(
        E_min=E_min,
        E_max=E_max,
        nE=nE,
        save=also_save_table,
        outfile=tablefile,
        print_integrated_uE=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8), dpi=300, facecolor='w', edgecolor='k')
    ax.set_xlabel(r'$a$ [$\AA$]', fontsize=14)
    ax.set_ylabel(r'$\langle Q_{\rm abs} \rangle_{\rm ISRF}$', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both', axis='both', direction='in', labelsize=12)
    ax.minorticks_on()

    styles = {
        'graphite': {'color': 'steelblue', 'linestyle': '-'},
        'silicate': {'color': 'sandybrown', 'linestyle': '-'},
    }
    for grain_type in ['graphite', 'silicate']:
        sub = df[df['grain_type'] == grain_type]
        if len(sub) == 0:
            continue
        ax.plot(sub['size_angstrom'].to_numpy(dtype=float),
                sub['qabs_isrf_avg'].to_numpy(dtype=float),
                label=grain_type,
                linewidth=2.0,
                **styles.get(grain_type, {}))

    # Overplot analytic Draine-like approximations as dashed curves:
    # silicate: qabs = 0.18 * (a / 0.1 micron)^0.6
    # graphite: qabs = 0.8  * (a / 0.1 micron)^0.85
    aA_all = df['size_angstrom'].to_numpy(dtype=float)
    aA_grid = np.logspace(np.log10(np.nanmin(aA_all)), np.log10(np.nanmax(aA_all)), 300)
    a_micron_grid = aA_grid * 1e-4

    sil_mask = (a_micron_grid >= 0.01) & (a_micron_grid <= 1.0)
    gra_mask = (a_micron_grid >= 0.005) & (a_micron_grid <= 0.15)

    qabs_sil_approx = 0.18 * (a_micron_grid[sil_mask] / 0.1) ** 0.6
    qabs_gra_approx = 0.8 * (a_micron_grid[gra_mask] / 0.1) ** 0.85

    ax.plot(aA_grid[gra_mask], qabs_gra_approx,
            linestyle='--', linewidth=2.0,
            color=styles['graphite']['color'],
            label='graphite approx')
    ax.plot(aA_grid[sil_mask], qabs_sil_approx,
            linestyle='--', linewidth=2.0,
            color=styles['silicate']['color'],
            label='silicate approx')

    ax.legend(loc='best', frameon=False, fontsize=12)
    fig.subplots_adjust(top=0.97, bottom=0.14, left=0.15, right=0.98)

    out_dir = PATH_MODEL_OPTICAL_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / savefile
    fig.savefig(out_path, format='pdf', dpi=300)
    plt.close(fig)
    print('Saved Qabs-vs-size plot to', str(out_path))
    return str(out_path)

def dust_efficiencies(filename,print_info=False):
    """
    This function allows for the construction of a clean and
    nice dataset.
    """
    columns = ['w(micron)','Q_abs', 'Q_sca', 'g=<cos>']
    data = {}

    with open(filename) as f:
        # Begin by reading the header
        for i in range(0,5):
            hd = f.readline()
            if i == 1:
                dust_type = hd
            elif i==3:
                info = list(filter(None, hd.split(' ')))
                nrad = int(info[0])
                amin = float(info[1])
                amax = float(info[2])
            elif i==4:
                info = list(filter(None, hd.split(' ')))
                nwav = int(info[0])
                wmin = float(info[1])
                wmax = float(info[2])
        if print_info: print(dust_type,nrad,nwav)
        
        while True:
            f.readline() # Blank line
            myarray = np.zeros((nwav,4))
            a = str(f.readline().split(' ')[0])
            if a == '':
                if print_info:  print('End of file')
                break
            f.readline() # Column names
            for i in range(0, nwav):
                line = f.readline()
                myarray[i,:] = np.fromstring(line, dtype=float, sep=' ')
            data[a] = myarray

    return nwav,data,columns,dust_type

def plot_efficiencies(filename,dust_type='grains',
                      do_average=True,
                      output_average=True):

    fig, axes = plt.subplots(3,1, figsize=(6,9),dpi=300,facecolor='w',edgecolor='k',sharey=True, sharex=True)

    if dust_type == 'grains':
        nwav,data,columns,name = dust_efficiencies(filename)
    else:
        # Import PAH reader for non-grain dust types
        from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
        nwav,data,columns,name = pah_efficiencies(filename)
    
    if 'PAH' in name:
        dist = [LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0]),
                LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])]
        ndist = 2
        linestyles = ['-.','-']
        name = ['smallPAHs','largePAHs']
    elif 'Graphite' in name:
        dist = [LogNormal_Distribution(basic_a0[2],basic_amin[2],basic_amax[2],basic_sigma[2],basic_s[2]),
                LogNormal_Distribution(basic_a0[3],basic_amin[3],basic_amax[3],basic_sigma[3],basic_s[3])]
        ndist = 2
        linestyles = ['-.','-']
        name = ['smallC','largeC']
    elif 'silicate' in name:
        dist = [LogNormal_Distribution(basic_a0[5],basic_amin[5],basic_amax[5],basic_sigma[5],basic_s[5]),
                LogNormal_Distribution(basic_a0[6],basic_amin[6],basic_amax[6],basic_sigma[6],basic_s[6])]
        ndist = 2
        linestyles = ['-.','-']
        name = ['smallSil','largeSil']
    for a in data:
        print('a = ',a)
        Q_sca = data[a][:,columns.index('Q_sca')]
        Q_abs = data[a][:,columns.index('Q_abs')]
        g     = data[a][:,columns.index('g=<cos>')]
        w     = data[a][:,columns.index('w(micron)')]
        Q_rp  = Q_abs + (1-g)*Q_sca
        
        # if float(a) == 1e-1:
        #     axes[0].plot(w,Q_abs,alpha=0.3,linewidth=0.5,color='k',linestyle=':')
        #     axes[1].plot(w,Q_sca,alpha=0.3,linewidth=0.5,color='r',linestyle=':')
        #     axes[2].plot(w,Q_rp,alpha=0.3,linewidth=0.5,color='b',linestyle=':')
        # elif float(a) == 5.012E-03:
        #     axes[0].plot(w,Q_abs,alpha=0.3,linewidth=0.5,color='k',linestyle='--')
        #     axes[1].plot(w,Q_sca,alpha=0.3,linewidth=0.5,color='r',linestyle='--')
        #     axes[2].plot(w,Q_rp,alpha=0.3,linewidth=0.5,color='b',linestyle='--')
        # else:
        
        axes[0].plot(w,Q_abs*np.pi*float(a)**2.* 1e-8,alpha=0.3,linewidth=0.5,color='k')
        axes[1].plot(w,Q_sca*np.pi*float(a)**2.* 1e-8,alpha=0.3,linewidth=0.5,color='r')
        axes[2].plot(w,Q_rp*np.pi*float(a)**2.* 1e-8,alpha=0.3,linewidth=0.5,color='b')
    if do_average:
        for i in range(0,ndist):
            nwav = len(w)
            Q_sca_eff = np.zeros(nwav)
            Q_abs_eff = np.zeros(nwav)
            Q_rp_eff  = np.zeros(nwav)
            nrad = len(data.keys())
            akeys= list(data.keys())
            for j in range(0, nwav):
                sizes = np.zeros(nrad)
                Q_sca = np.zeros(nrad)
                Q_abs = np.zeros(nrad)
                Q_rp  = np.zeros(nrad)
                for k in range(0,nrad):
                    tmpdt = data[akeys[k]]
                    sizes[k] = float(akeys[k])
                    Q_sca[k] = tmpdt[j,columns.index('Q_sca')]
                    Q_abs[k] = tmpdt[j,columns.index('Q_abs')]
                    g        = tmpdt[j,columns.index('g=<cos>')]
                    w        = tmpdt[:,columns.index('w(micron)')]
                    Q_rp[k]  = Q_abs[k] + (1-g)*Q_sca[k]
                Q_sca_eff[j] = dist[i].averaged_over_number(Q_sca*np.pi*sizes**2.,sizes)
                Q_abs_eff[j] = dist[i].averaged_over_number(Q_abs*np.pi*sizes**2.,sizes)
                Q_rp_eff[j]  = dist[i].averaged_over_number(Q_rp*np.pi*sizes**2.,sizes)
            axes[0].plot(w,Q_abs_eff* 1e-8,linewidth=2,color='k',linestyle=linestyles[i],label=name[i])
            axes[1].plot(w,Q_sca_eff* 1e-8,linewidth=2,color='r',linestyle=linestyles[i])
            axes[2].plot(w,Q_rp_eff* 1e-8,linewidth=2,color='b',linestyle=linestyles[i])
            if output_average:
                # Convert wavelength from micron to angstrom 
                w = w[::-1] * 1e4
                # Convert cross section from micron^2 to cm^2
                Q_abs_eff = Q_abs_eff[::-1] * 1e-8
                Q_sca_eff = Q_sca_eff[::-1] * 1e-8 
                Q_rp_eff = Q_rp_eff[::-1] * 1e-8
                if not os.path.exists(PATH_TABLES):
                    os.makedirs(PATH_TABLES)
                f = open(os.path.join(PATH_TABLES, 'averaged_cross_section_%.4f_micron_%s'%(dist[i].a0,filename.split('/')[-1])), 'w', encoding="utf-8")
                f.write("{:8d}".format(nwav)+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(w[j])+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(Q_abs_eff[j])+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(Q_sca_eff[j])+'\n')
                for j in range(0,nwav):
                    f.write("{:14.6e}".format(Q_rp_eff[j])+'\n')
                f.close()

    # Load the zubko et al. 2004 cross-sections for comparison
    data_zubko = np.loadtxt('zubko_2004_bare_gr_s.dat')
    axes[0].plot(data_zubko[:,0],data_zubko[:,1],'k--',label='Zubko et al. 2004')

    # Load the CLOUDY cross-sections for comparison
    data_cloudy = np.loadtxt(PATH_EXTERNAL_DATA / 'grains_CLOUDY.dat')
    axes[0].plot(data_cloudy[:,0],data_cloudy[:,1]*1e8,'r--',label='CLOUDY')


    for i in range(0,3):
        ax = axes[i]
        ax.tick_params(labelsize=14)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([1e-3,1e3])
        # ax.set_ylim([1e-10,1e-3])
    axes[0].set_ylabel(r'$C_{\rm abs}$ [cm$^2$]', fontsize=16)
    axes[1].set_ylabel(r'$C_{\rm sca}$ [cm$^2$]', fontsize=16)
    axes[2].set_ylabel(r'$C_{\rm rp}$ [cm$^2$]', fontsize=16)
    axes[2].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=16)
    axes[0].legend(loc='best',fontsize=14,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.06,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig('cross_section_%s.pdf'%filename.split('/')[-1], format='pdf', dpi=300)

def plot_sil_comp():

    fig, axes = plt.subplots(3,1, figsize=(6,9),dpi=300,facecolor='w',edgecolor='k',sharey=True, sharex=True)

    nwav,data,columns,name = dust_efficiencies(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Sil_21'))
    nwav_suv,data_suv,columns_suv,name_suv = dust_efficiencies(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_21'))

    nsizes = len(data.keys())
    Q_abs = np.zeros((nwav,nsizes,2))
    Q_sca = np.zeros((nwav,nsizes,2))
    g     = np.zeros((nwav,nsizes,2))
    w     = data[list(data.keys())[0]][:,columns.index('w(micron)')]
    for i,a in enumerate(data):
        Q_sca[:,i,0] = data[a][:,columns.index('Q_sca')]
        Q_abs[:,i,0] = data[a][:,columns.index('Q_abs')]
        g[:,i,0]     = data[a][:,columns.index('g=<cos>')]

    for i,a in enumerate(data_suv):
        Q_sca[:,i,1] = data_suv[a][:,columns.index('Q_sca')]
        Q_abs[:,i,1] = data_suv[a][:,columns.index('Q_abs')]
        g[:,i,1]     = data_suv[a][:,columns.index('g=<cos>')]

    Q_rp = Q_abs + (1-g)*Q_sca
        
    for i,a in enumerate(data):
        axes[0].plot(w,Q_abs[:,i,0]/Q_abs[:,i,1],linewidth=0.5,color='k')
        axes[1].plot(w,Q_sca[:,i,0]/Q_sca[:,i,1],linewidth=0.5,color='r')
        axes[2].plot(w,Q_rp[:,i,0]/Q_rp[:,i,1],linewidth=0.5,color='b')

    for i in range(0,3):
        ax = axes[i]
        ax.tick_params(labelsize=14)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.set_xscale('log')
        ax.set_xlim([1e-3,1e3])
        # ax.set_ylim([1e-10,1e-3])
    axes[0].set_ylabel(r'$Q_{\rm abs}/Q_{\rm abs,suv}$', fontsize=16)
    axes[1].set_ylabel(r'$Q_{\rm sca}/Q_{\rm sca,suv}$', fontsize=16)
    axes[2].set_ylabel(r'$Q_{\rm rp}/Q_{\rm rp,suv}$', fontsize=16)
    axes[2].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=16)
    fig.subplots_adjust(top=0.99,bottom=0.06,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig('compare_silicate_cs.pdf', format='pdf', dpi=300)

def interpolate_cross_sections_2d(dust_type, grain_size, target_wavelengths=None,
                                  efficiency=False, data_table=None, use_li_draine=None):
    """
    Interpolate cross sections in both size and wavelength.

    Parameters
    - dust_type: same as interpolate_cross_sections (silicate, graphite, iPAH, nPAH, PAH)
    - grain_size: target grain size in microns
    - target_wavelengths: array-like of wavelengths in microns to interpolate to.
        If None, uses the native wavelengths from the table.
    - efficiency: if True, return Q values (dimensionless); otherwise return C (cm^2)
    - data_table: optional (nwav, data, columns, name) tuple to avoid re-reading files
    - use_li_draine: optional boolean override for Li & Draine (2001) carbonaceous blend

    Returns (grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp)
    Similar units/shape as interpolate_cross_sections.
    """
    from pycalima.models.dust_radiation.dust_emission import USE_LI_DRAINE_2001_CARBONACEOUS
    if use_li_draine is None:
        use_li_draine = USE_LI_DRAINE_2001_CARBONACEOUS

    if use_li_draine and dust_type == 'graphite':
        # Get classical graphite first
        g_size_cm, wav_cm, C_sca_gra, C_abs_gra, C_rp_gra = interpolate_cross_sections_2d(
            'graphite', grain_size, target_wavelengths=target_wavelengths, efficiency=efficiency, data_table=data_table, use_li_draine=False
        )
        # Get neutral PAH
        _, wav_pah, C_sca_pah, C_abs_pah, C_rp_pah = interpolate_cross_sections_2d(
            'PAH', grain_size, target_wavelengths=target_wavelengths, efficiency=efficiency, use_li_draine=False
        )
        
        eps_PAH = 0.99 * min(1.0, (0.005 / max(grain_size, 1e-30)) ** 3)
        
        if len(wav_cm) == len(wav_pah) and np.allclose(wav_cm, wav_pah):
            C_sca_pah_interp = C_sca_pah
            C_abs_pah_interp = C_abs_pah
            C_rp_pah_interp = C_rp_pah
        else:
            sort_idx = np.argsort(wav_pah)
            C_sca_pah_interp = np.interp(wav_cm, wav_pah[sort_idx], C_sca_pah[sort_idx])
            C_abs_pah_interp = np.interp(wav_cm, wav_pah[sort_idx], C_abs_pah[sort_idx])
            C_rp_pah_interp = np.interp(wav_cm, wav_pah[sort_idx], C_rp_pah[sort_idx])
            
        C_sca_carb = eps_PAH * C_sca_pah_interp + (1.0 - eps_PAH) * C_sca_gra
        C_abs_carb = eps_PAH * C_abs_pah_interp + (1.0 - eps_PAH) * C_abs_gra
        C_rp_carb = eps_PAH * C_rp_pah_interp + (1.0 - eps_PAH) * C_rp_gra
        
        return g_size_cm, wav_cm, C_sca_carb, C_abs_carb, C_rp_carb

    # Read table if not provided
    if data_table is None:
        if dust_type == 'silicate':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81')
            nwav, data, columns, name = dust_efficiencies(filename)
        elif dust_type == 'graphite':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81')
            nwav, data, columns, name = dust_efficiencies(filename)
        elif dust_type == 'iPAH' or dust_type == 'nPAH' or dust_type == 'PAH':
            # Import PAH-specific function
            from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies, interpolate_pah_cross_sections_2d
            # Use PAH-specific interpolator instead
            return interpolate_pah_cross_sections_2d(dust_type, grain_size, target_wavelengths, efficiency, data_table)
        else:
            raise ValueError('Dust type not recognised: ', dust_type)
    else:
        nwav, data, columns, name = data_table

    # Build arrays of sizes and native wavelengths robustly from the data dict
    keys = list(data.keys())
    sizes_raw = np.array([float(k) for k in keys])

    # use the first table to get native wavelength grid and detect ordering
    first_arr = data[keys[0]]
    wcol = columns.index('w(micron)')
    native_wav = first_arr[:, wcol].copy()
    # If the wavelength axis is decreasing, we'll flip it when reading arrays
    flip_wav = False
    if native_wav[0] > native_wav[-1]:
        flip_wav = True
        native_wav = native_wav[::-1]

    # Sort sizes ascending and remember original keys order
    order = np.argsort(sizes_raw)
    native_sizes = sizes_raw[order]
    sorted_keys = [keys[i] for i in order]

    nwav_native = native_wav.size

    # Determine target wavelengths (in microns)
    if target_wavelengths is None:
        target_wav = native_wav.copy()
    else:
        target_wav = np.array(target_wavelengths, dtype=float)

    # For each native size, interpolate Q vs wavelength to the target wavelengths
    nsizes = native_sizes.size
    ntarget_wav = target_wav.size
    Q_abs_table = np.zeros((nsizes, ntarget_wav))
    Q_sca_table = np.zeros((nsizes, ntarget_wav))
    g_table = np.zeros((nsizes, ntarget_wav))

    for i, key in enumerate(sorted_keys):
        arr = data[key]
        if flip_wav:
            arr = arr[::-1, :]
        # get native Q arrays
        qabs_native = arr[:, columns.index('Q_abs')]
        qsca_native = arr[:, columns.index('Q_sca')] if 'Q_sca' in columns else np.zeros_like(qabs_native)
        g_native = arr[:, columns.index('g=<cos>')] if 'g=<cos>' in columns else np.zeros_like(qabs_native)

        # Interpolate in log-log for Q (avoid negative or zero) where appropriate
        # For small values, fall back to linear interp of Q
        # Use log10(native_wav) which is increasing after potential flip
        log_native_wav = np.log10(native_wav)
        for j, tw in enumerate(target_wav):
            if (qabs_native > 0).all():
                Q_abs_table[i, j] = 10.0 ** np.interp(np.log10(tw), log_native_wav, np.log10(qabs_native))
            else:
                Q_abs_table[i, j] = np.interp(tw, native_wav, qabs_native)
            if (qsca_native > 0).all():
                Q_sca_table[i, j] = 10.0 ** np.interp(np.log10(tw), log_native_wav, np.log10(qsca_native))
            else:
                Q_sca_table[i, j] = np.interp(tw, native_wav, qsca_native)
            g_table[i, j] = np.interp(tw, native_wav, g_native)

    # Now interpolate over size to the desired grain_size
    # do interpolation in log-log for Q vs a
    log_native_a = np.log10(native_sizes)
    log_target_a = np.log10(grain_size)

    Q_abs_target = np.zeros(ntarget_wav)
    Q_sca_target = np.zeros(ntarget_wav)
    g_target = np.zeros(ntarget_wav)
    for j in range(ntarget_wav):
        qabs_vs_a = Q_abs_table[:, j]
        qsca_vs_a = Q_sca_table[:, j]
        # avoid zeros for log interpolation
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
    # geometric area (micron^2) then convert to cm^2
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
        # ensure units in cm^2
        # (area_cm2 already in cm^2, Q dimensionless)

    return grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp


def compute_cross_sections_mie(composition, grain_size_micron, target_wavelengths=None,
                               efficiency=False):
    """
    Compute efficiencies or cross sections for a single grain size directly using Mie theory.

    Parameters
    - composition: 'silicate' or 'graphite'
    - grain_size_micron: grain size in microns
    - target_wavelengths: array-like of wavelengths in microns. If None, uses a default grid.
    - efficiency: if True, return Q values (dimensionless); otherwise return C (cm^2)

    Returns (grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp)
    """
    
    from pycalima.models.tools.mie_theory import MieTheory
    mie = MieTheory()

    # Load dielectrics
    mie.load_dielectric_constants(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'eps_suvSil'), 'suvSil')
    mie.load_dielectric_constants(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'callindex.out_CpaD03_0.01'), 'graphite_pa')
    mie.load_dielectric_constants(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'callindex.out_CpeD03_0.01'), 'graphite_pe')

    if target_wavelengths is None:
        target_wav = np.logspace(-3, 3, 241)
    else:
        target_wav = np.array(target_wavelengths, dtype=float)

    if composition == 'silicate':
        species_info = 'suvSil'
    elif composition == 'graphite':
        species_info = {'parallel': 'graphite_pa', 'perpendicular': 'graphite_pe'}
    else:
        raise ValueError(f"Unknown composition: {composition}")

    qabs = np.zeros(len(target_wav))
    qsca = np.zeros(len(target_wav))
    g = np.zeros(len(target_wav))

    for w_idx, w_um in enumerate(target_wav):
        qa, qs, gg = mie.compute_grain_properties(grain_size_micron, w_um, species_info, extend_xrays=True)
        qabs[w_idx] = qa
        qsca[w_idx] = qs
        g[w_idx] = gg

    qrp = qabs + (1.0 - g) * qsca

    wavelengths_cm = target_wav * 1e-4
    grain_size_cm = grain_size_micron * 1e-4

    if efficiency:
        return grain_size_cm, wavelengths_cm, qsca, qabs, qrp

    # Compute cross sections: area = pi * a^2 (in cm^2)
    area_cm2 = np.pi * (grain_size_micron * 1e-4) ** 2
    C_abs = qabs * area_cm2
    C_sca = qsca * area_cm2
    C_rp = qrp * area_cm2

    return grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp


def plot_cs_sne(rho_gas,D_smallPAHs,D_largePAHs,D_smallC,D_largeC,D_smallSil,D_largeSil,export=False):

    # 1. Set up the figure
    fig, axes = plt.subplots(2,2, figsize=(10,6),dpi=300,facecolor='w',edgecolor='k',sharey=False,sharex=False)
    axes[0,0].set_ylabel(r'$a^4n(a)$', fontsize=16)
    axes[0,1].set_ylabel(r'$C_{\rm abs}$, $C_{\rm sca}$ [cm$^2$] \& $g$', fontsize=16)
    axes[1,0].set_ylabel(r'$a^4n(a)$', fontsize=16)
    axes[1,1].set_ylabel(r'$C_{\rm abs}$, $C_{\rm sca}$ [cm$^2$] \& $g$', fontsize=16)
    axes[1,0].set_xlabel(r'$a$ [$\mu$m]', fontsize=16)
    axes[1,1].set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=16)

    # 2. Setup the different size distributions
    # Gao et al. 2020 for the empirically derived extinction curve of the supernova SN2012cu
    # (https://www.sciencedirect.com/science/article/pii/S0032063318300321?via%3Dihub)
    Gao_2020_sil = PowerLaw_ExpCutoff_Distribution(5e-3,5,0.04,0.5,3.3)
    Gao_2020_gra = PowerLaw_ExpCutoff_Distribution(5e-3,5,0.03,0.5,2.2)

    # Asano et al. (2013) log-normal distribution parameters (originally used
    # for AGB production) it is also used for SNe ejecta in Hirashita & Aoyama (2019)
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.2555H/abstract)
    Asano_2013_sil = LogNormal_Distribution(0.1,5e-3,5.,0.47,3.3)
    Asano_2013_gra = LogNormal_Distribution(0.1,5e-3,5.,0.47,2.2)

    # Nozawa et al. (2007) power-law distributions for Mg2SiO4 and C grains
    # after the effect of sputtering and shattering in Pop III ejecta
    # (https://ui.adsabs.harvard.edu/abs/2007ApJ...666..955N/abstract)
    # NOTE: They do not provide the numerical values for this, so I have
    # obtain them by copying their table and fitting the power-law function
    Nozawa_2007_sil = PowerLaw_ExpCutoff_Distribution(1.6e-3,1.0,5.23e-02,1.25,3.3)
    Nozawa_2007_gra = PowerLaw_ExpCutoff_Distribution(1.6e-3,1.0,2.14e-02,1.15,2.2)

    # Marassi et al. (2019) log-normal distribution using the Limongi & Chieffi (2018) SNe
    # yields for the ejecta of massive stars
    # (https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.3109M/abstract)
    Marassi_2019_sil = Classical_LogNormal_Distribution(0.025,1e-3,1,0.1,2.2)
    Marassi_2019_gra = Classical_LogNormal_Distribution(0.075,1e-3,1,0.1,2.2)

    # RAMSES Dust: Using the resulting grain size distribution from the G8 simulation
    # with initial 0.003 Zsun and DTMinit=1d-3 and 18 pc resolution (output 10)
    # fCs  =     0.005     0.010
    # fCl  =     0.464     0.990
    # fSils=     0.010     0.018
    # fSill=     0.522     0.982
    # fs   =     0.014
    # fl   =     0.986
    # fC   =     0.468
    # fSil =     0.532
    ramses_silLarge = LogNormal_Distribution(1e-1,5e-3,1.0,0.75,3.3)
    ramses_silSmall = LogNormal_Distribution(5e-3,5e-4,0.1,0.75,3.3)
    ramses_graLarge = LogNormal_Distribution(1e-1,5e-3,1.0,0.75,2.2)
    ramses_graSmall = LogNormal_Distribution(5e-3,5e-4,0.1,0.75,2.2)
    

    # 3. Plot the size distribution on the first axes
    a = np.logspace(np.log10(5e-3),np.log10(0.8),100)
    axes[0,0].plot(a,a**4*Gao_2020_sil.n_density(rho_gas*D_largeSil,a),
                 label='Gao et al. 2020',color='#8CBA80',linestyle='-')
    axes[1,0].plot(a,a**4*Gao_2020_gra.n_density(rho_gas*D_largeC,a),
                    color='#8CBA80',linestyle='-')
    axes[0,0].plot(a,a**4*Asano_2013_sil.n_density(rho_gas*D_largeSil,a),
                 label='Asano et al. 2013',color='#658E9C',linestyle='-')
    axes[1,0].plot(a,a**4*Asano_2013_gra.n_density(rho_gas*D_largeC,a),
                    color='#658E9C',linestyle='-')
    axes[0,0].plot(a,a**4*Nozawa_2007_sil.n_density(rho_gas*D_largeSil,a),
                    label='Nozawa et al. 2007',color='#F5A65B',linestyle='-')
    axes[1,0].plot(a,a**4*Nozawa_2007_gra.n_density(rho_gas*D_largeC,a),
                    color='#F5A65B',linestyle='-')
    axes[0,0].plot(a,a**4*Marassi_2019_sil.n_density(rho_gas*D_largeSil,a),
                    label='Marassi et al. 2019',color='#F28C8C',linestyle='-')
    axes[1,0].plot(a,a**4*Marassi_2019_gra.n_density(rho_gas*D_largeC,a),
                    color='#F28C8C',linestyle='-')
    axes[0,0].plot(a,a**4*ramses_silLarge.n_density(0.986*rho_gas*D_largeSil,a)+a**4*ramses_silSmall.n_density(0.014*rho_gas*D_largeSil,a),
                    label='RAMSES',color='k',linestyle='-')
    axes[1,0].plot(a,a**4*ramses_graLarge.n_density(0.986*rho_gas*D_largeC,a)+a**4*ramses_graSmall.n_density(0.014*rho_gas*D_largeSil,a),
                    color='k',linestyle='-')

    axes[0,0].set_yscale('log')
    axes[0,0].set_xscale('log')
    axes[0,0].legend(loc='best',fontsize=10,frameon=False)
    axes[0,0].set_ylim([4e-30,3e-27])
    axes[0,0].tick_params(labelsize=14)
    axes[0,0].xaxis.set_ticks_position('both')
    axes[0,0].yaxis.set_ticks_position('both')
    axes[0,0].minorticks_on()
    axes[0,0].tick_params(which='both',axis="both",direction="in")

    axes[0,0].plot(a,5e-28*a**(0.5),':',color='gray',linewidth=2)
    axes[0,0].text(0.5, 0.52, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes[0,0].transAxes,fontsize=14,rotation=17)

    axes[1,0].set_yscale('log')
    axes[1,0].set_xscale('log')
    axes[1,0].set_ylim([4e-30,3e-27])
    axes[1,0].tick_params(labelsize=14)
    axes[1,0].xaxis.set_ticks_position('both')
    axes[1,0].yaxis.set_ticks_position('both')
    axes[1,0].minorticks_on()
    axes[1,0].tick_params(which='both',axis="both",direction="in")

    axes[1,0].plot(a,5e-28*a**(0.5),':',color='gray',linewidth=2)
    axes[1,0].text(0.13, 0.37, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=axes[1,0].transAxes,fontsize=14,rotation=17)


    # 4. Compute and plot the number-averaged cross-section
    nwav_Gra,data_Gra,columns_Gra,name_Gra = dust_efficiencies(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81'))
    nwav_Sil,data_Sil,columns_Sil,name_Sil = dust_efficiencies(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81'))

    nrad = len(data_Sil.keys())
    C_sca_Asano_2013_sil = np.zeros(nwav_Sil)
    C_abs_Asano_2013_sil = np.zeros(nwav_Sil)
    g_Asano_2013_sil = np.zeros(nwav_Sil)
    C_sca_Asano_2013_gra = np.zeros(nwav_Gra)
    C_abs_Asano_2013_gra = np.zeros(nwav_Gra)
    g_Asano_2013_gra = np.zeros(nwav_Gra)

    C_sca_Gao_2020_sil = np.zeros(nwav_Sil)
    C_abs_Gao_2020_sil = np.zeros(nwav_Sil)
    g_Gao_2020_sil = np.zeros(nwav_Sil)
    C_sca_Gao_2020_gra = np.zeros(nwav_Gra)
    C_abs_Gao_2020_gra = np.zeros(nwav_Gra)
    g_Gao_2020_gra = np.zeros(nwav_Gra)

    C_sca_Nozawa_2007_sil = np.zeros(nwav_Sil)
    C_abs_Nozawa_2007_sil = np.zeros(nwav_Sil)
    g_Nozawa_2007_sil = np.zeros(nwav_Sil)
    C_sca_Nozawa_2007_gra = np.zeros(nwav_Gra)
    C_abs_Nozawa_2007_gra = np.zeros(nwav_Gra)
    g_Nozawa_2007_gra = np.zeros(nwav_Gra)

    C_sca_Marassi_2019_sil = np.zeros(nwav_Sil)
    C_abs_Marassi_2019_sil = np.zeros(nwav_Sil)
    g_Marassi_2019_sil = np.zeros(nwav_Sil)
    C_sca_Marassi_2019_gra = np.zeros(nwav_Gra)
    C_abs_Marassi_2019_gra = np.zeros(nwav_Gra)
    g_Marassi_2019_gra = np.zeros(nwav_Gra)

    C_sca_ramses_silLarge = np.zeros(nwav_Sil)
    C_abs_ramses_silLarge = np.zeros(nwav_Sil)
    g_ramses_silLarge = np.zeros(nwav_Sil)
    C_sca_ramses_graLarge = np.zeros(nwav_Gra)
    C_abs_ramses_graLarge = np.zeros(nwav_Gra)
    g_ramses_graLarge = np.zeros(nwav_Gra)


    nrad = len(data_Sil.keys())
    akeys= list(data_Sil.keys())
    for j in range(0,nwav_Sil):
        sizes_Sil = np.zeros(nrad)
        Q_sca_Sil = np.zeros(nrad)
        Q_abs_Sil = np.zeros(nrad)
        g_Sil = np.zeros(nrad)
        w_Sil = np.zeros(nrad)
        sizes_Gra = np.zeros(nrad)
        Q_sca_Gra = np.zeros(nrad)
        Q_abs_Gra = np.zeros(nrad)
        g_Gra = np.zeros(nrad)
        w_Gra = np.zeros(nrad)
        for k in range(0,nrad):
            tmpdt = data_Sil[akeys[k]]
            sizes_Sil[k] = float(akeys[k])
            Q_sca_Sil[k] = tmpdt[j,columns_Sil.index('Q_sca')]
            Q_abs_Sil[k] = tmpdt[j,columns_Sil.index('Q_abs')]
            g_Sil[k]     = tmpdt[j,columns_Sil.index('g=<cos>')]
            w_Sil        = tmpdt[:,columns_Sil.index('w(micron)')]
            tmpdt = data_Gra[akeys[k]]
            sizes_Gra[k] = float(akeys[k])
            Q_sca_Gra[k] = tmpdt[j,columns_Gra.index('Q_sca')]
            Q_abs_Gra[k] = tmpdt[j,columns_Gra.index('Q_abs')]
            g_Gra[k]     = tmpdt[j,columns_Gra.index('g=<cos>')]
            w_Gra        = tmpdt[:,columns_Gra.index('w(micron)')]
        C_sca_Asano_2013_sil[j] = Asano_2013_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Asano_2013_sil[j] = Asano_2013_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Asano_2013_sil[j]     = Asano_2013_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Asano_2013_gra[j] = Asano_2013_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Asano_2013_gra[j] = Asano_2013_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Asano_2013_gra[j]     = Asano_2013_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_Gao_2020_sil[j] = Gao_2020_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Gao_2020_sil[j] = Gao_2020_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Gao_2020_sil[j]     = Gao_2020_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Gao_2020_gra[j] = Gao_2020_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Gao_2020_gra[j] = Gao_2020_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Gao_2020_gra[j]     = Gao_2020_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_Nozawa_2007_sil[j] = Nozawa_2007_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Nozawa_2007_sil[j] = Nozawa_2007_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Nozawa_2007_sil[j]     = Nozawa_2007_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Nozawa_2007_gra[j] = Nozawa_2007_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Nozawa_2007_gra[j] = Nozawa_2007_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Nozawa_2007_gra[j]     = Nozawa_2007_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_Marassi_2019_sil[j] = Marassi_2019_sil.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_Marassi_2019_sil[j] = Marassi_2019_sil.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_Marassi_2019_sil[j]     = Marassi_2019_sil.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_Marassi_2019_gra[j] = Marassi_2019_gra.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_Marassi_2019_gra[j] = Marassi_2019_gra.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_Marassi_2019_gra[j]     = Marassi_2019_gra.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

        C_sca_ramses_silLarge[j] = ramses_silLarge.averaged_over_number(Q_sca_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_abs_ramses_silLarge[j] = ramses_silLarge.averaged_over_number(Q_abs_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        g_ramses_silLarge[j]     = ramses_silLarge.averaged_over_number(g_Sil*np.pi*sizes_Sil**2.,sizes_Sil)
        C_sca_ramses_graLarge[j] = ramses_graLarge.averaged_over_number(Q_sca_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        C_abs_ramses_graLarge[j] = ramses_graLarge.averaged_over_number(Q_abs_Gra*np.pi*sizes_Gra**2.,sizes_Gra)
        g_ramses_graLarge[j]     = ramses_graLarge.averaged_over_number(g_Gra*np.pi*sizes_Gra**2.,sizes_Gra)

    axes[0,1].plot(w_Sil,C_abs_Asano_2013_sil * 1e-8,linewidth=2,color='#658E9C',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Asano_2013_sil * 1e-8,linewidth=2,color='#658E9C',linestyle='--')
    axes[0,1].plot(w_Sil,g_Asano_2013_sil * 1e-8,linewidth=2,color='#658E9C',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_Gao_2020_sil * 1e-8,linewidth=2,color='#8CBA80',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Gao_2020_sil * 1e-8,linewidth=2,color='#8CBA80',linestyle='--')
    axes[0,1].plot(w_Sil,g_Gao_2020_sil * 1e-8,linewidth=2,color='#8CBA80',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_Nozawa_2007_sil * 1e-8,linewidth=2,color='#F5A65B',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Nozawa_2007_sil * 1e-8,linewidth=2,color='#F5A65B',linestyle='--')
    axes[0,1].plot(w_Sil,g_Nozawa_2007_sil * 1e-8,linewidth=2,color='#F5A65B',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_Marassi_2019_sil * 1e-8,linewidth=2,color='#F28C8C',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_Marassi_2019_sil * 1e-8,linewidth=2,color='#F28C8C',linestyle='--')
    axes[0,1].plot(w_Sil,g_Marassi_2019_sil * 1e-8,linewidth=2,color='#F28C8C',linestyle=':')
    axes[0,1].plot(w_Sil,C_abs_ramses_silLarge * 1e-8,linewidth=2,color='k',linestyle='-')
    axes[0,1].plot(w_Sil,C_sca_ramses_silLarge * 1e-8,linewidth=2,color='k',linestyle='--')
    axes[0,1].plot(w_Sil,g_ramses_silLarge * 1e-8,linewidth=2,color='k',linestyle=':')


    axes[1,1].plot(w_Gra,C_abs_Asano_2013_gra * 1e-8,linewidth=2,color='#658E9C',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Asano_2013_gra * 1e-8,linewidth=2,color='#658E9C',linestyle='--')
    axes[1,1].plot(w_Gra,g_Asano_2013_gra * 1e-8,linewidth=2,color='#658E9C',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_Gao_2020_gra * 1e-8,linewidth=2,color='#8CBA80',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Gao_2020_gra * 1e-8,linewidth=2,color='#8CBA80',linestyle='--')
    axes[1,1].plot(w_Gra,g_Gao_2020_gra * 1e-8,linewidth=2,color='#8CBA80',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_Nozawa_2007_gra * 1e-8,linewidth=2,color='#F5A65B',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Nozawa_2007_gra * 1e-8,linewidth=2,color='#F5A65B',linestyle='--')
    axes[1,1].plot(w_Gra,g_Nozawa_2007_gra * 1e-8,linewidth=2,color='#F5A65B',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_Marassi_2019_gra * 1e-8,linewidth=2,color='#F28C8C',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_Marassi_2019_gra * 1e-8,linewidth=2,color='#F28C8C',linestyle='--')
    axes[1,1].plot(w_Gra,g_Marassi_2019_gra * 1e-8,linewidth=2,color='#F28C8C',linestyle=':')
    axes[1,1].plot(w_Gra,C_abs_ramses_graLarge * 1e-8,linewidth=2,color='k',linestyle='-')
    axes[1,1].plot(w_Gra,C_sca_ramses_graLarge * 1e-8,linewidth=2,color='k',linestyle='--')
    axes[1,1].plot(w_Gra,g_ramses_graLarge * 1e-8,linewidth=2,color='k',linestyle=':')

    # 5. If the export flag is True, we save these number-averaged cross-sections to individual files
    # indicating well the names as well as the properties of the underlying distribution assumed in the
    # header of the file
    if export:
        folder = './cross_section_sne/'
        # Convert wavelength from micron to angstrom 
        w_Sil = w_Sil[::-1] * 1e4
        w_Gra = w_Gra[::-1] * 1e4
        # Convert cross section from micron^2 to cm^2
        C_abs_Asano_2013_sil = C_abs_Asano_2013_sil[::-1] * 1e-8
        C_sca_Asano_2013_sil = C_sca_Asano_2013_sil[::-1] * 1e-8
        g_Asano_2013_sil = g_Asano_2013_sil[::-1]
        C_abs_Asano_2013_gra = C_abs_Asano_2013_gra[::-1] * 1e-8
        C_sca_Asano_2013_gra = C_sca_Asano_2013_gra[::-1] * 1e-8
        g_Asano_2013_gra = g_Asano_2013_gra[::-1]

        C_abs_Gao_2020_sil = C_abs_Gao_2020_sil[::-1] * 1e-8
        C_sca_Gao_2020_sil = C_sca_Gao_2020_sil[::-1] * 1e-8
        g_Gao_2020_sil = g_Gao_2020_sil[::-1]
        C_abs_Gao_2020_gra = C_abs_Gao_2020_gra[::-1] * 1e-8
        C_sca_Gao_2020_gra = C_sca_Gao_2020_gra[::-1] * 1e-8
        g_Gao_2020_gra = g_Gao_2020_gra[::-1]

        C_abs_Nozawa_2007_sil = C_abs_Nozawa_2007_sil[::-1] * 1e-8
        C_sca_Nozawa_2007_sil = C_sca_Nozawa_2007_sil[::-1] * 1e-8
        g_Nozawa_2007_sil = g_Nozawa_2007_sil[::-1]
        C_abs_Nozawa_2007_gra = C_abs_Nozawa_2007_gra[::-1] * 1e-8
        C_sca_Nozawa_2007_gra = C_sca_Nozawa_2007_gra[::-1] * 1e-8
        g_Nozawa_2007_gra = g_Nozawa_2007_gra[::-1]

        C_abs_Marassi_2019_sil = C_abs_Marassi_2019_sil[::-1] * 1e-8
        C_sca_Marassi_2019_sil = C_sca_Marassi_2019_sil[::-1] * 1e-8
        g_Marassi_2019_sil = g_Marassi_2019_sil[::-1]
        C_abs_Marassi_2019_gra = C_abs_Marassi_2019_gra[::-1] * 1e-8
        C_sca_Marassi_2019_gra = C_sca_Marassi_2019_gra[::-1] * 1e-8
        g_Marassi_2019_gra = g_Marassi_2019_gra[::-1]

        C_abs_ramses_silLarge = C_abs_ramses_silLarge[::-1] * 1e-8
        C_sca_ramses_silLarge = C_sca_ramses_silLarge[::-1] * 1e-8
        g_ramses_silLarge = g_ramses_silLarge[::-1]
        C_abs_ramses_graLarge = C_abs_ramses_graLarge[::-1] * 1e-8
        C_sca_ramses_graLarge = C_sca_ramses_graLarge[::-1] * 1e-8
        g_ramses_graLarge = g_ramses_graLarge[::-1]
        
        # Export the data to a file
        with open(folder+'/cross_section_sne_Asano_2013_sil.dat','w') as f:
            f.write('# Asano et al. 2013 silicates\n')
            f.write('# Modified Log-normal distribution (Hirashita 2015) for AGB production\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=5 [micron], alpha=0.47, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Asano_2013_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Asano_2013_sil[j])+" "+
                        "{:14.6e}".format(g_Asano_2013_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Asano_2013_gra.dat','w') as f:
            f.write('# Asano et al. 2013 graphite\n')
            f.write('# Modified Log-normal distribution (Hirashita 2015) for AGB production\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=5 [micron], alpha=0.47, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Asano_2013_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Asano_2013_gra[j])+" "+
                        "{:14.6e}".format(g_Asano_2013_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Gao_2020_sil.dat','w') as f:
            f.write('# Gao et al. 2020 silicates\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.04 [micron], amin=0.005 [micron], amax=5 [micron], alpha=0.5, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Gao_2020_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Gao_2020_sil[j])+" "+
                        "{:14.6e}".format(g_Gao_2020_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Gao_2020_gra.dat','w') as f:
            f.write('# Gao et al. 2020 graphite\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.03 [micron], amin=0.005 [micron], amax=5 [micron], alpha=0.5, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Gao_2020_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Gao_2020_gra[j])+" "+
                        "{:14.6e}".format(g_Gao_2020_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Nozawa_2007_sil.dat','w') as f:
            f.write('# Nozawa et al. 2007 silicates\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.0523 [micron], amin=0.0016 [micron], amax=1 [micron], alpha=1.25, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Nozawa_2007_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Nozawa_2007_sil[j])+" "+
                        "{:14.6e}".format(g_Nozawa_2007_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Nozawa_2007_gra.dat','w') as f:
            f.write('# Nozawa et al. 2007 graphite\n')
            f.write('# Power-law with exponential cutoff distribution\n')
            f.write('# with acut=0.0214 [micron], amin=0.0016 [micron], amax=1 [micron], alpha=1.15, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Nozawa_2007_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Nozawa_2007_gra[j])+" "+
                        "{:14.6e}".format(g_Nozawa_2007_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Marassi_2019_sil.dat','w') as f:
            f.write('# Marassi et al. 2019 silicates\n')
            f.write('# Log-normal distribution\n')
            f.write('# with a0=0.025 [micron], amin=0.001 [micron], amax=1 [micron], alpha=0.1, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_Marassi_2019_sil[j])+" "+
                        "{:14.6e}".format(C_sca_Marassi_2019_sil[j])+" "+
                        "{:14.6e}".format(g_Marassi_2019_sil[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_Marassi_2019_gra.dat','w') as f:
            f.write('# Marassi et al. 2019 graphite\n')
            f.write('# Log-normal distribution\n')
            f.write('# with a0=0.075 [micron], amin=0.001 [micron], amax=1 [micron], alpha=0.1, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_Marassi_2019_gra[j])+" "+
                        "{:14.6e}".format(C_sca_Marassi_2019_gra[j])+" "+
                        "{:14.6e}".format(g_Marassi_2019_gra[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_ramses_silLarge.dat','w') as f:
            f.write('# RAMSES silicates\n')
            f.write('# Modified log-normal distribution (Hirashita 2015)\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=1 [micron], alpha=0.75, s=3.3 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Sil):
                f.write("{:14.6e}".format(w_Sil[j])+" "+
                        "{:14.6e}".format(C_abs_ramses_silLarge[j])+" "+
                        "{:14.6e}".format(C_sca_ramses_silLarge[j])+" "+
                        "{:14.6e}".format(g_ramses_silLarge[j])+'\n')
            f.close()
        with open(folder+'/cross_section_sne_ramses_graLarge.dat','w') as f:
            f.write('# RAMSES graphite\n')
            f.write('# Modified log-normal distribution (Hirashita 2015)\n')
            f.write('# with a0=0.1 [micron], amin=5e-3 [micron], amax=1 [micron], alpha=0.75, s=2.2 [g/cm3]\n')
            f.write('# w [Angstrom] C_abs [cm^2] C_sca [cm^2] g\n')
            for j in range(0,nwav_Gra):
                f.write("{:14.6e}".format(w_Gra[j])+" "+
                        "{:14.6e}".format(C_abs_ramses_graLarge[j])+" "+
                        "{:14.6e}".format(C_sca_ramses_graLarge[j])+" "+
                        "{:14.6e}".format(g_ramses_graLarge[j])+'\n')
            f.close()

    axes[0,1].set_yscale('log')
    axes[0,1].set_xscale('log')
    axes[0,1].tick_params(labelsize=14)
    axes[0,1].xaxis.set_ticks_position('both')
    axes[0,1].yaxis.set_ticks_position('both')
    axes[0,1].minorticks_on()
    axes[0,1].tick_params(which='both',axis="both",direction="in")
    axes[0,1].yaxis.set_label_position("right")
    axes[0,1].yaxis.tick_right()
    axes[0,1].set_ylim([4e-17,3e-9])

    axes[1,1].set_yscale('log')
    axes[1,1].set_xscale('log')
    axes[1,1].tick_params(labelsize=14)
    axes[1,1].xaxis.set_ticks_position('both')
    axes[1,1].yaxis.set_ticks_position('both')
    axes[1,1].minorticks_on()
    axes[1,1].tick_params(which='both',axis="both",direction="in")
    axes[1,1].yaxis.set_label_position("right")
    axes[1,1].yaxis.tick_right()
    axes[1,1].set_ylim([4e-17,3e-9])

    dummy_lines = [axes[1,1].plot([],[],color='k',linestyle='-',label=r'$C_{\rm abs}$')[0],
                   axes[1,1].plot([],[],color='k',linestyle='--',label=r'$C_{\rm sca}$')[0],
                   axes[1,1].plot([],[],color='k',linestyle=':',label=r'$g\times 10^{-13}$')[0]]
    first_legend = axes[1,1].legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=14)
    axes[1,1].add_artist(first_legend)


    # 5. Add text indicating that the top row is for silicates while the bottom row is for graphite
    axes[0,1].text(0.75, 0.91, 'Silicates', verticalalignment='top', horizontalalignment='left',
                   transform=axes[0,1].transAxes,fontsize=16,fontdict={'weight': 'bold'})
    axes[1,1].text(0.75, 0.91, 'Graphite', verticalalignment='top', horizontalalignment='left',
                   transform=axes[1,1].transAxes,fontsize=16,fontdict={'weight': 'bold'})

    # 6. Adjust figure and save
    fig.subplots_adjust(top=0.99,bottom=0.09,left=0.08,right=0.92,hspace=0,wspace=0)
    fig.savefig(folder+'/cross_section_sne.pdf', format='pdf', dpi=300)


def compute_extinction_curve(dust_types, dists, mass_fractions,
                             mdust_per_H=None, convert_to_A_per_NH=True,
                             nsize_per_bin=10, verbose=False,
                             optical_dir=None, pah_state='neutral'):
    """
    Compute a composite extinction curve from either:
    1) precomputed DustBin/PAHBin tables, or
    2) the legacy size-distribution integration path.

    Parameters
    - data_list : list of dict
        Each element is a `data` dict as returned by `dust_efficiencies` or
        `pah_efficiencies`. Keys are size strings and values are arrays with
        wavelength and Q columns.
    - columns_list : list of list
        Matching list of `columns` lists (the column names returned by the
        reader functions) for each data dict. If a single `columns` is
        supplied, it will be reused for all components.
    - dists : list
        List of distribution objects (instances of LogNormal_Distribution,
        PowerLaw_ExpCutoff_Distribution, etc.) describing the grain size
        distribution for each component. The distributions must accept sizes
        in microns (the same units as the data keys).
        For precomputed bin usage this argument is accepted but ignored.
    - mass_fractions : list or array
        Mass fraction of the total dust mass assigned to each component.
        These should sum to 1.0 (the function will normalize if they don't).
    - mdust_per_H : float, optional
        If provided (g of dust per H nucleus), the function also returns
        A_lambda / N_H in magnitudes per H by using
            A/N_H = 1.086 * kappa_lambda * mdust_per_H
    - convert_to_A_per_NH : bool
        If True and mdust_per_H is provided, compute and return A_lambda/N_H.
    - size_unit_micron : bool
        If True (default) the size keys are interpreted as microns as used
        throughout this codebase.
    - verbose : bool
        Print progress/info if True.
    - optical_dir : str or Path or None
        Directory containing averaged_cross_section_<BinID>.txt files.
        Only used when `dust_types` are bin IDs.
    - pah_state : str
        PAH block to read from PAH precomputed tables: 'neutral' or 'ionised'.

    Returns
    A dict with keys:
    - 'wavelength' : 1D array of wavelengths [micron]
    - 'kappa' : 1D array [cm^2 / g_dust]
    - 'components' : list of per-component kappa arrays (same units)
    - 'A_per_NH' : 1D array of A_lambda/N_H [mag per H] if mdust_per_H provided else None

    Notes
        - If `dust_types` are bin IDs (for example 'PAHbin_01', 'DustBin_03'),
            the function reads precomputed tables from model_data/optical_properties.
        - Otherwise it falls back to the legacy integration over size distributions.

    Example usage
    -------------
    nwav,data,columns,name = dust_efficiencies(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81'))
    k = compute_extinction_curve([data], [columns], [dist], [1.0], mdust_per_H=1e-26)
    """
    # normalize inputs to lists
    if optical_dir is None:
        optical_dir = PATH_MODEL_OPTICAL_OUTPUT

    if isinstance(dust_types, str):
        dust_types = [dust_types]

    if not isinstance(dists, (list, tuple)):
        dists = [dists]

    mass_fractions = np.array(mass_fractions, dtype=float)
    if mass_fractions.size != len(dust_types):
        raise ValueError('mass_fractions length must match number of components')
    # normalize mass fractions
    if mass_fractions.sum() <= 0:
        raise ValueError('mass_fractions must sum to a positive value')
    mass_fractions = mass_fractions / mass_fractions.sum()

    # Preferred path: use precomputed optical-property tables for DustBin/PAHBin IDs.
    # This avoids recomputing size-integrated cross sections.
    is_precomputed_bins = all(
        isinstance(comp, str) and ('DustBin_' in comp or 'PAHBin_' in comp)
        for comp in dust_types
    )

    if is_precomputed_bins:
        component_tables = []
        wavelength_sets = []

        for comp in dust_types:
            wav_i, cabs_i, csca_i, _ = _read_precomputed_cross_section_table(
                comp,
                optical_dir=optical_dir,
                pah_state=pah_state,
            )
            order_i = np.argsort(wav_i)
            wav_i = wav_i[order_i]
            cext_i = (cabs_i + csca_i)[order_i]
            component_tables.append((comp, wav_i, cext_i))
            wavelength_sets.append(wav_i)

        wav = np.unique(np.concatenate(wavelength_sets))
        kappas_comp = np.zeros((len(dust_types), len(wav)))
        for i, (_, wav_i, cext_i) in enumerate(component_tables):
            kappas_comp[i, :] = np.interp(wav, wav_i, cext_i, left=0.0, right=0.0)

        kappa_total = np.tensordot(mass_fractions, kappas_comp, axes=(0, 0))

        A_per_NH = None
        A_per_component = None
        if mdust_per_H is not None and convert_to_A_per_NH:
            A_per_NH = 1.086 * kappa_total * float(mdust_per_H)
            A_per_component = (
                1.086
                * kappas_comp
                * float(mdust_per_H)
                * mass_fractions[:, np.newaxis]
            )

        return {
            'wavelength': wav,
            'kappa': kappa_total,
            'components': kappas_comp,
            'A_per_component': A_per_component,
            'A_per_NH': A_per_NH,
        }

    req_wav_micron = np.logspace(-1.5,1,100)  # 0.1 micron to 10 micron
    kappas_comp = np.zeros((len(dists), len(req_wav_micron)))

    # loop over grain components
    for icomp, (dist,material) in enumerate(zip(dists, dust_types)):
        kappa_dist = np.zeros((nsize_per_bin, len(req_wav_micron)))
        size_bins = np.logspace(np.log10(dist.amin), np.log10(dist.amax), nsize_per_bin)
        # load the optical files
        if material == 'silicate':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81')
            nwav, data, columns, name = dust_efficiencies(filename)
        elif material == 'graphite':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81')
            nwav, data, columns, name = dust_efficiencies(filename)
        elif material == 'iPAH' or material == 'nPAH' or material == 'PAH':
            # Import PAH-specific reader
            from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
            if material == 'iPAH':
                filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30')
            else:
                filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
            nwav, data, columns, name = pah_efficiencies(filename)
        else:
            raise ValueError('Dust type not recognised: ', material)
        data_table = nwav, data, columns, name

        # get number distribution normalized to 1.0 units of dust mass
        n_for_unit_mass = dist.n_density(1.0, size_bins)  # sizes in cm
        
        # loop over the grain sizes in the bin
        for isize, a in enumerate(size_bins):
            a_cm, wav_cm, C_sca, C_abs, C_rp = interpolate_cross_sections_2d(
                material, a*1e4, req_wav_micron, data_table=data_table
            )
            C_ext = C_abs + C_sca  # cm^2
            kappa_dist[isize, :] = C_ext * n_for_unit_mass[isize]  # cm^2/g
        
        kappas_comp[icomp,:] = np.trapezoid(kappa_dist, size_bins, axis=0)  # cm^2/g
        
        
    # combine components by mass fractions (mass fraction refers to fraction of dust mass)
    kappa_total = np.tensordot(mass_fractions, kappas_comp, axes=(0, 0))

    A_per_NH = None
    if mdust_per_H is not None and convert_to_A_per_NH:
        # A/N_H = 1.086 * kappa_lambda * mdust_per_H
        A_per_NH = 1.086 * kappa_total * float(mdust_per_H)
        A_per_component = 1.086 * kappas_comp * float(mdust_per_H) * mass_fractions[:, np.newaxis]
    else:
        A_per_component = None

    return {
        'wavelength': req_wav_micron,
        'kappa': kappa_total,
        'components': kappas_comp,
        'A_per_component': A_per_component,
        'A_per_NH': A_per_NH
    }


def getCrosssection_BARE_GR_S_DUST(lambda_angstrom):
    """
    Harley's fit to the effective absorption cross section of dust
    for the Zubko et al. (2004) BARE-GR-S model.

    Parameters
    ----------
    lambda_angstrom : float or array-like
        Wavelength in Angstroms.

    Returns
    -------
    Cabs : float or ndarray
        Absorption cross section in cm^2 per H.
    """

    # Polynomial coefficients (degree 10)
    fit_vals = np.array([
        -1.59319023e+01, -1.60473171e+00,  6.20612550e-01,
         6.42859480e-01, -4.08743189e-01, -1.59224607e-01,
         7.37953364e-02,  1.60696953e-02, -5.96977205e-03,
        -5.57671237e-04,  1.80437634e-04
    ])

    # Convert wavelength from Angstroms to microns
    lambda_microns = np.asarray(lambda_angstrom) * 1e-4

    # Compute polynomial in log10(lambda_microns)
    loglam = np.log10(lambda_microns)
    logC = np.zeros_like(loglam, dtype=float)

    for i, lam in enumerate(loglam):
        sum_val = 0.0
        for j, coeff in enumerate(fit_vals):
            sum_val += coeff * (lam ** j)
        logC[i] = sum_val

    # Convert from log10(Cext) to Cext
    Cabs = 10.0 ** logC

    return Cabs


def _read_precomputed_cross_section_table(bin_id, optical_dir=None, pah_state='neutral'):
    """Read one precomputed DustBin/PAHBin optical table.

    Returns
    -------
    tuple of ndarray
        wavelength_micron, C_abs, C_sca, C_rp
    """
    if optical_dir is None:
        optical_dir = PATH_MODEL_OPTICAL_OUTPUT

    file_path = Path(optical_dir) / f'averaged_cross_section_{bin_id}.txt'
    if not file_path.exists():
        raise FileNotFoundError(f'Optical-property file not found for {bin_id}: {file_path}')

    pah_state_token = str(pah_state).strip().lower()
    use_ionised = pah_state_token in ('ionised', 'ionized')

    rows = []
    in_table = False
    with open(file_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if line.lower().startswith('# columns:'):
                    in_table = True
                continue
            if not in_table:
                continue

            if '|' in line:
                left_block, right_block = line.split('|', 1)
                tokens = right_block.split() if use_ionised else left_block.split()
            else:
                tokens = line.split()

            if len(tokens) < 4:
                continue
            rows.append([float(value) for value in tokens[:4]])

    if len(rows) == 0:
        raise ValueError(f'Could not parse optical-property rows for {bin_id}: {file_path}')

    data = np.asarray(rows, dtype=float)
    wavelength_micron = data[:, 0] * 1e-4
    return wavelength_micron, data[:, 1], data[:, 2], data[:, 3]


def _resolve_bin_materials(component_bins, pah_state='neutral'):
    """Resolve each bin ID to the material token used by interpolators."""
    cfg = load_grain_size_config()
    meta_by_id = {entry['id']: entry for entry in cfg['bins']}

    pah_state_token = str(pah_state).strip().lower()
    pah_material = 'iPAH' if pah_state_token in ('ionised', 'ionized') else 'nPAH'

    materials = []
    for bin_id in component_bins:
        if bin_id not in meta_by_id:
            raise KeyError(f'Bin {bin_id} not found in grain-size configuration')
        meta = meta_by_id[bin_id]
        if bool(meta['is_pah']):
            materials.append(pah_material)
        else:
            materials.append(meta['composition'])
    return materials


def _build_q_table_2d(optical_table, target_wav_micron):
    """Build a 2D Q table at (native_sizes, target_wavelengths) from an optical data tuple.

    This is called ONCE per material so that the per-size inner loop in
    ``_integrate_q_table_2d`` can be replaced by a single vectorised pass.

    Parameters
    ----------
    optical_table : tuple
        ``(nwav, data, columns, name)`` as returned by ``dust_efficiencies`` or
        ``pah_efficiencies``.
    target_wav_micron : ndarray
        Target wavelength grid in microns.

    Returns
    -------
    native_sizes : ndarray, shape (N,)
        Native grain sizes in microns, sorted ascending.
    Q_abs_2d, Q_sca_2d, g_2d : ndarray, shape (N, nwav_target)
    """
    _, data, columns, _ = optical_table
    target_wav = np.asarray(target_wav_micron, dtype=float)
    nwav_target = len(target_wav)

    keys = list(data.keys())
    sizes_raw = np.array([float(k) for k in keys])
    order = np.argsort(sizes_raw)
    native_sizes = sizes_raw[order]
    sorted_keys = [keys[i] for i in order]

    wcol = columns.index('w(micron)')
    qabs_col = columns.index('Q_abs')
    qsca_col = columns.index('Q_sca') if 'Q_sca' in columns else None
    g_col    = columns.index('g=<cos>') if 'g=<cos>' in columns else None

    first_arr = data[sorted_keys[0]]
    native_wav = first_arr[:, wcol].copy()
    flip_wav = native_wav[0] > native_wav[-1]
    if flip_wav:
        native_wav = native_wav[::-1]
    log_native_wav = np.log10(native_wav)
    log_target_wav = np.log10(target_wav)

    nsizes_native = len(native_sizes)
    Q_abs_2d = np.zeros((nsizes_native, nwav_target))
    Q_sca_2d = np.zeros((nsizes_native, nwav_target))
    g_2d     = np.zeros((nsizes_native, nwav_target))

    for i, key in enumerate(sorted_keys):
        arr = data[key]
        if flip_wav:
            arr = arr[::-1, :]
        qa = arr[:, qabs_col]
        qs = arr[:, qsca_col] if qsca_col is not None else np.zeros_like(qa)
        g  = arr[:, g_col]    if g_col    is not None else np.zeros_like(qa)
        # log-log interpolation with safe floor
        Q_abs_2d[i] = 10.0 ** np.interp(log_target_wav, log_native_wav,
                                         np.log10(np.maximum(qa, 1e-100)))
        Q_sca_2d[i] = 10.0 ** np.interp(log_target_wav, log_native_wav,
                                         np.log10(np.maximum(qs, 1e-100)))
        g_2d[i]     = np.interp(target_wav, native_wav, g)

    return native_sizes, Q_abs_2d, Q_sca_2d, g_2d


def _integrate_q_table_2d(native_sizes, Q_abs_2d, Q_sca_2d, g_2d,
                           target_sizes_micron, weights):
    """Vectorised integration of C(a,lambda)*weights(a) over grain sizes.

    Replaces the ``nsize`` calls to ``interpolate_cross_sections_2d`` with a
    single pass of numpy operations.  The size interpolation loop is over
    ``nwav`` (typically 100) rather than ``nsize`` (100-200), and each
    iteration is a vectorised ``np.interp`` over all target sizes at once.

    Parameters
    ----------
    native_sizes : ndarray, shape (N,)
        From ``_build_q_table_2d``, in microns, sorted ascending.
    Q_abs_2d, Q_sca_2d, g_2d : ndarray, shape (N, nwav)
        From ``_build_q_table_2d``.
    target_sizes_micron : ndarray, shape (M,)
        Integration quadrature grid in microns.
    weights : ndarray, shape (M,)
        Values of the weighting function at each quadrature point (e.g.
        dn/da in 1/(micron·H) or n(a) in cm^-3/micron).

    Returns
    -------
    cabs, csca, crp : ndarray, shape (nwav,)
        Integrated cross sections (units follow from ``weights``).
    """
    nwav = Q_abs_2d.shape[1]
    log_native_a = np.log10(native_sizes)
    log_target_a = np.log10(target_sizes_micron)

    Q_abs_t = np.zeros((len(target_sizes_micron), nwav))
    Q_sca_t = np.zeros_like(Q_abs_t)
    g_t     = np.zeros_like(Q_abs_t)

    for j in range(nwav):
        Q_abs_t[:, j] = 10.0 ** np.interp(log_target_a, log_native_a,
                                            np.log10(np.maximum(Q_abs_2d[:, j], 1e-100)))
        Q_sca_t[:, j] = 10.0 ** np.interp(log_target_a, log_native_a,
                                            np.log10(np.maximum(Q_sca_2d[:, j], 1e-100)))
        g_t[:, j]     = np.interp(target_sizes_micron, native_sizes, g_2d[:, j])

    area_cm2 = np.pi * (target_sizes_micron * 1e-4) ** 2  # (M,)
    C_abs = Q_abs_t * area_cm2[:, np.newaxis]
    C_sca = Q_sca_t * area_cm2[:, np.newaxis]
    C_rp  = (Q_abs_t + (1.0 - g_t) * Q_sca_t) * area_cm2[:, np.newaxis]

    w = weights[:, np.newaxis]
    cabs = np.trapezoid(w * C_abs, target_sizes_micron, axis=0)
    csca = np.trapezoid(w * C_sca, target_sizes_micron, axis=0)
    crp  = np.trapezoid(w * C_rp,  target_sizes_micron, axis=0)
    return cabs, csca, crp


def _compute_component_cross_sections_legacy(component_bins, target_wavelengths,
                                             nsize_per_bin=30,
                                             pah_state='neutral', verbose=False,
                                             distribution_class=None):
    """Compute bin cross sections by integrating raw tables over JSON size distributions.

    Returns per-component cross sections normalized per gram of dust (cm^2 / g_dust),
    matching the normalization used in ``compute_extinction_curve``.
    """
    materials = _resolve_bin_materials(component_bins, pah_state=pah_state)
    target_wavelengths = np.asarray(target_wavelengths, dtype=float)

    ncomp = len(component_bins)
    nwav = len(target_wavelengths)
    cabs_comps = np.zeros((ncomp, nwav))
    csca_comps = np.zeros((ncomp, nwav))
    crp_comps = np.zeros((ncomp, nwav))

    # Build optical tables once per unique material (avoids repeated file I/O)
    optical_cache = {}
    q_table_cache = {}
    for material in set(materials):
        if material == 'silicate':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81')
            optical_cache[material] = dust_efficiencies(filename)
        elif material == 'graphite':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81')
            optical_cache[material] = dust_efficiencies(filename)
        elif material == 'iPAH':
            from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
            filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30')
            optical_cache[material] = pah_efficiencies(filename)
        elif material == 'nPAH':
            from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
            filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
            optical_cache[material] = pah_efficiencies(filename)
        else:
            raise ValueError(f'Unsupported material for legacy interpolation: {material}')
        # Pre-build the 2D Q table at the target wavelength grid
        q_table_cache[material] = _build_q_table_2d(optical_cache[material], target_wavelengths)

    _dist_cls = distribution_class if distribution_class is not None else LogNormal_Distribution

    def _process_bin(args):
        i, bin_id, material = args
        p = get_lognormal_parameters(bin_id)
        dist = _dist_cls(
            p['a0'] * 1e-4,
            p['amin'] * 1e-4,
            p['amax'] * 1e-4,
            p['sigma'],
            p['s'],
        )
        # size_bins in cm; convert to microns for _integrate_q_table_2d
        size_bins_cm = np.logspace(np.log10(dist.amin), np.log10(dist.amax), int(nsize_per_bin))
        size_bins_um = size_bins_cm * 1e4
        n_for_unit_mass = dist.n_density(1.0, size_bins_cm)
        # dn/da is per cm; convert to per micron so the trapezoid integral is consistent
        weights_per_um = n_for_unit_mass / 1e4

        native_sizes, Q_abs_2d, Q_sca_2d, g_2d = q_table_cache[material]
        cabs, csca, crp = _integrate_q_table_2d(
            native_sizes, Q_abs_2d, Q_sca_2d, g_2d,
            size_bins_um, weights_per_um,
        )
        if verbose:
            print(f'[plot_extinction_from_massfractions] legacy component done: {bin_id} ({material})')
        return i, cabs, csca, crp

    args_list = [(i, bin_id, mat)
                 for i, (bin_id, mat) in enumerate(zip(component_bins, materials))]

    with concurrent.futures.ThreadPoolExecutor() as pool:
        for i, cabs, csca, crp in pool.map(_process_bin, args_list):
            cabs_comps[i, :] = cabs
            csca_comps[i, :] = csca
            crp_comps[i, :] = crp

    return cabs_comps, csca_comps, crp_comps


def compute_component_cross_sections_mie(component_bins, target_wavelengths,
                                         nsize_per_bin=30,
                                         pah_state='neutral', verbose=False,
                                         distribution_class=None,
                                         distribution_class_map=None):
    """Compute bin cross sections by integrating Mie theory calculations over JSON size distributions.

    Returns per-component cross sections normalized per gram of dust (cm^2 / g_dust).

    distribution_class_map : dict {bin_id: class}, optional
        Per-bin override of distribution_class. Takes precedence over distribution_class
        for any bin_id key present in the map.
    """
    
    from pycalima.models.tools.mie_theory import MieTheory

    mie = MieTheory()

    # Load dielectric constants
    mie.load_dielectric_constants(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'eps_suvSil'), 'suvSil')
    mie.load_dielectric_constants(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'callindex.out_CpaD03_0.01'), 'graphite_pa')
    mie.load_dielectric_constants(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'callindex.out_CpeD03_0.01'), 'graphite_pe')

    materials = _resolve_bin_materials(component_bins, pah_state=pah_state)
    target_wavelengths = np.asarray(target_wavelengths, dtype=float)

    ncomp = len(component_bins)
    nwav = len(target_wavelengths)
    cabs_comps = np.zeros((ncomp, nwav))
    csca_comps = np.zeros((ncomp, nwav))
    crp_comps = np.zeros((ncomp, nwav))

    # For PAH bins, we fall back to the legacy tables
    optical_cache = {}
    q_table_cache = {}
    for material in set(materials):
        if material in ('iPAH', 'nPAH'):
            if material == 'iPAH':
                from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
                filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30')
                optical_cache[material] = pah_efficiencies(filename)
            elif material == 'nPAH':
                from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
                filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
                optical_cache[material] = pah_efficiencies(filename)
            q_table_cache[material] = _build_q_table_2d(optical_cache[material], target_wavelengths)

    _dist_cls_default = distribution_class if distribution_class is not None else LogNormal_Distribution

    def _process_bin(args):
        i, bin_id, material = args
        _dist_cls = (distribution_class_map or {}).get(bin_id, _dist_cls_default)
        p = get_lognormal_parameters(bin_id)
        dist = _dist_cls(
            p['a0'] * 1e-4,
            p['amin'] * 1e-4,
            p['amax'] * 1e-4,
            p['sigma'],
            p['s'],
        )
        size_bins_cm = np.logspace(np.log10(dist.amin), np.log10(dist.amax), int(nsize_per_bin))
        size_bins_um = size_bins_cm * 1e4
        n_for_unit_mass = dist.n_density(1.0, size_bins_cm)
        weights_per_um = n_for_unit_mass / 1e4

        if material in ('iPAH', 'nPAH'):
            native_sizes, Q_abs_2d, Q_sca_2d, g_2d = q_table_cache[material]
            cabs, csca, crp = _integrate_q_table_2d(
                native_sizes, Q_abs_2d, Q_sca_2d, g_2d,
                size_bins_um, weights_per_um,
            )
        else:
            if material == 'silicate':
                species_info = 'suvSil'
            elif material == 'graphite':
                species_info = {'parallel': 'graphite_pa', 'perpendicular': 'graphite_pe'}
            else:
                raise ValueError(f"Unknown material: {material}")

            Q_abs_t = np.zeros((len(size_bins_um), nwav))
            Q_sca_t = np.zeros((len(size_bins_um), nwav))
            g_t     = np.zeros((len(size_bins_um), nwav))

            for s_idx, a_um in enumerate(size_bins_um):
                for w_idx, w_um in enumerate(target_wavelengths):
                    qa, qs, g = mie.compute_grain_properties(a_um, w_um, species_info, extend_xrays=True)
                    Q_abs_t[s_idx, w_idx] = qa
                    Q_sca_t[s_idx, w_idx] = qs
                    g_t[s_idx, w_idx]     = g

            area_cm2 = np.pi * (size_bins_um * 1e-4) ** 2
            C_abs = Q_abs_t * area_cm2[:, np.newaxis]
            C_sca = Q_sca_t * area_cm2[:, np.newaxis]
            C_rp  = (Q_abs_t + (1.0 - g_t) * Q_sca_t) * area_cm2[:, np.newaxis]

            w = weights_per_um[:, np.newaxis]
            cabs = np.trapezoid(w * C_abs, size_bins_um, axis=0)
            csca = np.trapezoid(w * C_sca, size_bins_um, axis=0)
            crp  = np.trapezoid(w * C_rp,  size_bins_um, axis=0)

        if verbose:
            print(f'Mie component done: {bin_id} ({material})')
        return i, cabs, csca, crp

    args_list = [(i, bin_id, mat)
                 for i, (bin_id, mat) in enumerate(zip(component_bins, materials))]

    with concurrent.futures.ThreadPoolExecutor() as pool:
        for i, cabs, csca, crp in pool.map(_process_bin, args_list):
            cabs_comps[i, :] = cabs
            csca_comps[i, :] = csca
            crp_comps[i, :] = crp

    return cabs_comps, csca_comps, crp_comps


# ---------------------------------------------------------------------------
# Zubko et al. (2004) BARE-GR-S grain size distribution functions
# ---------------------------------------------------------------------------

def _zubko_dnda_parametric(a_micron, A, c0, b0, a1, b1, m1, a2, b2, m2, a3, b3, m3, a4, b4, m4):
    """Zubko (2004) parameterized grain size distribution.

    Parameters
    ----------
    a_micron : float or ndarray
        Grain size in microns.

    Returns
    -------
    float or ndarray
        dn/da in 1/(micron * H atom).
    """
    a = np.asarray(a_micron, dtype=float)
    logg = (
        c0
        + b0 * np.log10(a)
        - b1 * np.abs(np.log10(a / a1)) ** m1
        - (b2 * np.abs(np.log10(a / a2)) ** m2 if b2 != 0.0 else 0.0)
        - b3 * np.abs(a - a3) ** m3
        - (b4 * np.abs(a - a4) ** m4 if b4 != 0.0 else 0.0)
    )
    return A * 10.0 ** logg


def _zubko_dnda_graphite(a_micron):
    """Zubko (2004) graphite dn/da.  a in microns, returns 1/(micron * H atom)."""
    return _zubko_dnda_parametric(
        a_micron,
        A=1.905816e-7, c0=-9.86,      b0=-5.02082,
        a1=0.415861,   b1=5.81215e-3, m1=4.63229,
        a2=1.0,        b2=0.0,        m2=0.0,
        a3=0.160344,   b3=1125.02,    m3=3.69897,
        a4=0.160501,   b4=1126.02,    m4=3.69967,
    )


def _zubko_dnda_pah(a_micron):
    """Zubko (2004) PAH dn/da.  a in microns, returns 1/(micron * H atom)."""
    return _zubko_dnda_parametric(
        a_micron,
        A=2.227433e-7,  c0=-8.02895,  b0=-3.45764,
        a1=1.0,         b1=1183.96,   m1=-8.20551,
        a2=1.0,         b2=0.0,       m2=0.0,
        a3=-5.29496e-3, b3=1.0e24,    m3=12.0146,
        a4=1.0,         b4=0.0,       m4=0.0,
    )


def _zubko_dnda_silicate(a_micron):
    """Zubko (2004) silicate dn/da.  a in microns, returns 1/(micron * H atom)."""
    return _zubko_dnda_parametric(
        a_micron,
        A=1.471288e-7,  c0=-8.47091,   b0=-3.68708,
        a1=7.64943e-3,  b1=2.37316e-5, m1=22.5489,
        a2=1.0,         b2=0.0,        m2=0.0,
        a3=0.480229,    b3=2961.28,    m3=12.1717,
        a4=1.0,         b4=0.0,        m4=0.0,
    )


def compute_zubko2004_bare_gr_s_cross_sections(
    wavelengths_micron=None,
    nsize=100,
    pah_neutral_fraction=0.5,
    verbose=False,
):
    """
    Compute size-integrated absorption/scattering cross sections per H atom
    for the Zubko et al. (2004) BARE-GR-S dust model.

    Model parameters
    ----------------
    mdust_per_H  = 1.44e-26 g/H
    Mass fractions: PAH 4.57%, graphite 29.47%, silicate 65.96%.
    PAH optical properties: Li & Draine 2001 (li_draine_2001), half neutral/half ionised.
    Graphite optical properties: Draine & Lee 1984 (Gra_81), rho = 2.24 g/cm^3.
    Silicate optical properties: Draine & Lee 1984 suvSil (suvSil_81), rho = 3.5 g/cm^3.

    Size distribution functions from Zubko et al. (2004) as implemented in SKIRT:
      Graphite : 0.00035 – 0.33 micron
      PAH      : 0.00035 – 0.005 micron
      Silicate : 0.00035 – 0.37 micron

    The dn/da functions are absolutely normalised per H atom so no additional
    mass-fraction weighting is required.

    Parameters
    ----------
    wavelengths_micron : array-like or None
        Target wavelength grid in microns.  Defaults to logspace(-1.5, 1, 100).
    nsize : int
        Number of quadrature points in grain size per component.
    pah_neutral_fraction : float
        Fraction of the PAH population that is neutral (default 0.5).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        wavelength       - wavelength grid [micron]
        C_abs_total      - total C_abs per H [cm^2/H]
        C_sca_total      - total C_sca per H [cm^2/H]
        C_rp_total       - total C_rp per H [cm^2/H]
        C_ext_total      - C_abs + C_sca per H [cm^2/H]
        C_abs_graphite   - graphite contribution [cm^2/H]
        C_sca_graphite
        C_abs_silicate   - silicate contribution [cm^2/H]
        C_sca_silicate
        C_abs_pah        - combined PAH contribution [cm^2/H]
        C_sca_pah
        C_abs_pah_neutral
        C_abs_pah_ionised
    """
    if wavelengths_micron is None:
        wav = np.logspace(-1.5, 1.0, 100)
    else:
        wav = np.asarray(wavelengths_micron, dtype=float)

    nwav = len(wav)

    from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
    gra_table     = dust_efficiencies(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81'))
    sil_table     = dust_efficiencies(os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81'))
    pah_neu_table = pah_efficiencies(os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30'))
    pah_ion_table = pah_efficiencies(os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30'))

    # Pre-build 2D Q tables once per material (avoids rebuilding inside the size loop)
    q_gra     = _build_q_table_2d(gra_table,     wav)
    q_sil     = _build_q_table_2d(sil_table,     wav)
    q_pah_neu = _build_q_table_2d(pah_neu_table, wav)
    q_pah_ion = _build_q_table_2d(pah_ion_table, wav)

    def _integrate_component(q_tables, dnda_func, amin_um, amax_um, label):
        """Vectorised size integration using pre-built Q tables."""
        a_grid   = np.logspace(np.log10(amin_um), np.log10(amax_um), int(nsize))
        dnda_val = dnda_func(a_grid)  # 1/(micron * H)
        native_sizes, Q_abs_2d, Q_sca_2d, g_2d = q_tables
        cabs, csca, crp = _integrate_q_table_2d(
            native_sizes, Q_abs_2d, Q_sca_2d, g_2d, a_grid, dnda_val
        )
        if verbose:
            print(f'[compute_zubko2004_bare_gr_s] done integrating {label}.')
        return cabs, csca, crp

    # Define the four independent tasks
    tasks = [
        (q_gra,     _zubko_dnda_graphite, 0.00035, 0.33,  'graphite'),
        (q_sil,     _zubko_dnda_silicate, 0.00035, 0.37,  'silicate'),
        (q_pah_neu, _zubko_dnda_pah,      0.00035, 0.005, 'PAH neutral'),
        (q_pah_ion, _zubko_dnda_pah,      0.00035, 0.005, 'PAH ionised'),
    ]

    if verbose:
        print('[compute_zubko2004_bare_gr_s] integrating 4 components in parallel...')

    # ThreadPoolExecutor is safe here: numpy releases the GIL for heavy computations
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_integrate_component, *t) for t in tasks]
        results = [f.result() for f in futures]

    (cabs_gra, csca_gra, crp_gra),         \
    (cabs_sil, csca_sil, crp_sil),         \
    (cabs_pah_neu, csca_pah_neu, crp_pah_neu), \
    (cabs_pah_ion, csca_pah_ion, crp_pah_ion) = results

    f_neu = float(np.clip(pah_neutral_fraction, 0.0, 1.0))
    f_ion = 1.0 - f_neu
    cabs_pah = f_neu * cabs_pah_neu + f_ion * cabs_pah_ion
    csca_pah = f_neu * csca_pah_neu + f_ion * csca_pah_ion
    crp_pah  = f_neu * crp_pah_neu  + f_ion * crp_pah_ion

    cabs_total = cabs_gra + cabs_sil + cabs_pah
    csca_total = csca_gra + csca_sil + csca_pah
    crp_total  = crp_gra  + crp_sil  + crp_pah
    cext_total = cabs_total + csca_total

    # The optical tables (Gra_81, suvSil_81, PAH) cover wavelengths up to
    # 1000 µm.  Beyond that, _build_q_table_2d silently clamps Q to the last
    # tabulated value (np.interp flat-fill), which gives unphysical flat cross
    # sections instead of the correct ~1/λ Rayleigh fall-off.  Mask those
    # wavelengths with NaN so callers cannot silently use wrong values.
    _WAV_TABLE_MAX_UM = 1000.0
    _out_of_range = wav > _WAV_TABLE_MAX_UM
    if _out_of_range.any():
        if verbose:
            print(f'[compute_zubko2004_bare_gr_s] WARNING: {_out_of_range.sum()} '
                  f'wavelength(s) exceed the optical table maximum '
                  f'({_WAV_TABLE_MAX_UM} µm) — setting those cross sections to NaN.')
        for _arr in (cabs_total, csca_total, crp_total, cext_total,
                     cabs_gra, csca_gra, cabs_sil, csca_sil,
                     cabs_pah, csca_pah, cabs_pah_neu, cabs_pah_ion):
            _arr[_out_of_range] = np.nan

    if verbose:
        print('[compute_zubko2004_bare_gr_s] done.')

    return {
        'wavelength':        wav,
        'C_abs_total':       cabs_total,
        'C_sca_total':       csca_total,
        'C_rp_total':        crp_total,
        'C_ext_total':       cext_total,
        'C_abs_graphite':    cabs_gra,
        'C_sca_graphite':    csca_gra,
        'C_abs_silicate':    cabs_sil,
        'C_sca_silicate':    csca_sil,
        'C_abs_pah':         cabs_pah,
        'C_sca_pah':         csca_pah,
        'C_abs_pah_neutral': cabs_pah_neu,
        'C_abs_pah_ionised': cabs_pah_ion,
    }


def export_zubko2004_cross_sections(
    outfile='zubko2004_bare_gr_s_cross_sections.dat',
    wavelengths_micron=None,
    nsize=200,
    pah_neutral_fraction=0.5,
    verbose=True,
    plot_comparison=True,
    out_png='zubko2004_bare_gr_s_comparison.png',
):
    """Compute and export the Zubko et al. (2004) BARE-GR-S dust cross sections
    to a Fortran-readable ASCII file, and optionally produce a comparison plot
    against the published reference tables (``zubko_2004_bare_gr_s.dat``).

    The three columns written are:

    - Wavelength            [Angstrom]
    - C_abs                 [cm^2 / H atom]  -- absorption cross section
    - C_sca                 [cm^2 / H atom]  -- scattering cross section
    - (1-g)*C_sca           [cm^2 / H atom]  -- radiation-pressure scattering term
                                                 derived as C_rp - C_abs, where
                                                 C_rp = integral[dn/da*(C_abs(a)+(1-g(a))*C_sca(a))da]
                                                 accounts for grain-size dependence of g(a)

    The quantity (1-g)*C_sca is derived as  C_rp - C_abs, where C_rp is the
    size-integrated radiation-pressure cross section
        C_rp = integral[ dn/da * (C_abs(a) + (1-g(a))*C_sca(a)) da ]
    computed directly during the grain-size integration, which correctly
    accounts for the grain-size dependence of the asymmetry parameter g.

    The header lines all start with '!' (the Fortran line-comment character)
    so that a Fortran 90 driver can skip them with a simple loop::

        integer :: i
        open(10, file='zubko2004_bare_gr_s_cross_sections.dat', status='old')
        do i = 1, N_HEADER_LINES     ! see NHEADER in the file header itself
            read(10, *)
        end do
        do i = 1, N_WAV
            read(10, *) wav_A(i), cabs(i), csca_rp(i)
        end do

    Parameters
    ----------
    outfile : str or Path
        Output file path.  Relative paths are written relative to the current
        working directory.
    wavelengths_micron : array-like or None
        Wavelength grid in microns.  If ``None``, a logarithmic grid of 400
        points from 0.03 to 1000 microns is used.
    nsize : int
        Number of grain-size quadrature points for the size integration.
        Default 200.
    pah_neutral_fraction : float
        Fraction of PAH grains that are neutral (rest are ionised).
        Default 0.5 (equal mix, as in Zubko et al. 2004).
    verbose : bool
        Print progress and the output file path.
    plot_comparison : bool
        If ``True`` (default), produce a two-panel comparison plot of the
        recomputed vs reference C_abs and C_sca curves.
    out_png : str or Path
        Path for the comparison plot PNG.  Ignored when
        ``plot_comparison=False``.

    Returns
    -------
    dict
        path        - absolute path to the written ASCII file (str)
        wavelength  - wavelength grid [micron]
        C_abs       - recomputed C_abs per H [cm^2/H]
        C_sca       - recomputed C_sca per H [cm^2/H]
        C_sca_rp    - recomputed (1-g)*C_sca per H [cm^2/H]
        C_abs_ref   - reference C_abs per H [cm^2/H], interpolated to output grid
        C_sca_ref   - reference C_sca per H [cm^2/H], interpolated to output grid
        C_sca_rp_ref- reference (1-g)*C_sca per H [cm^2/H], interpolated to output grid
        plot_path   - absolute path to the comparison plot, or None
    """
    import datetime

    if wavelengths_micron is None:
        wavelengths_micron = np.logspace(np.log10(1e-3), np.log10(1e3), 1000)
    wav_micron = np.asarray(wavelengths_micron, dtype=float)

    if verbose:
        print('[export_zubko2004_cross_sections] computing cross sections '
              f'({len(wav_micron)} wavelengths, nsize={nsize}) ...')

    result = compute_zubko2004_bare_gr_s_cross_sections(
        wavelengths_micron=wav_micron,
        nsize=nsize,
        pah_neutral_fraction=pah_neutral_fraction,
        verbose=verbose,
    )

    wav_angstrom  = wav_micron * 1e4          # micron -> Angstrom
    cabs          = result['C_abs_total']     # cm^2/H
    csca          = result['C_sca_total']     # cm^2/H
    csca_rp       = result['C_rp_total'] - result['C_abs_total']  # (1-g)*C_sca  cm^2/H
    nwav = len(wav_angstrom)

    outpath = Path(outfile).expanduser().resolve()
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the header.  Count lines so the Fortran driver can skip them.
    # ------------------------------------------------------------------
    header_lines = [
        '! ============================================================',
        '! Zubko et al. (2004) BARE-GR-S dust model — recomputed cross sections',
        '! ============================================================',
        '!',
        '! Reference: Zubko, Dwek & Arendt (2004), ApJS, 152, 211',
        '!   DOI: 10.1086/382351',
        '!',
        '! Model: BARE-GR-S (bare graphite + silicate + PAH, no ice mantles)',
        '!',
        '! Grain components and optical data sources',
        '!   Graphite   : Draine & Lee (1984) tables  — Gra_81',
        '!                grain density  rho = 2.24 g cm^-3',
        '!                size range    [0.00035, 0.33]  micron',
        '!   Silicate   : Draine & Lee (1984) tables  — suvSil_81',
        '!                grain density  rho = 3.5  g cm^-3',
        '!                size range    [0.00035, 0.37]  micron',
        '!   PAH        : Li & Draine (2001) tables',
        f'!                neutral fraction  f_neu = {pah_neutral_fraction:.3f}',
        '!                size range    [0.00035, 0.005] micron',
        '!',
        '! Integrated dust-to-gas mass ratio',
        '!   m_dust / H = 1.44e-26 g H^-1',
        '!',
        '! Column description',
        '!   Col 1 — wavelength          [Angstrom]',
        '!   Col 2 — C_abs               [cm^2 H^-1]   absorption cross section',
        '!   Col 3 — C_sca               [cm^2 H^-1]   scattering cross section',
        '!   Col 4 — (1-g)*C_sca         [cm^2 H^-1]   radiation-pressure scattering term',
        '!             derived as C_rp - C_abs, where C_rp = integral[dn/da*(C_abs(a)+(1-g(a))*C_sca(a))da]',
        '!             accounts for the grain-size dependence of the asymmetry parameter g(a)',
        '!',
        f'! Computation parameters: nsize = {nsize}',
        f'! Generated: {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC',
        f'! N_WAV = {nwav}',
        '!',
        '! To read in Fortran 90:',
        '!   integer, parameter :: NHEADER = <see NHEADER value below>',
        '!   integer, parameter :: NWAV    = <see N_WAV above>',
        '!   real(8) :: wav_A(NWAV), cabs(NWAV), csca(NWAV), csca_rp(NWAV)',
        '!   do i = 1, NHEADER ; read(unit, *) ; end do',
        '!   do i = 1, NWAV',
        '!     read(unit, *) wav_A(i), cabs(i), csca(i), csca_rp(i)',
        '!   end do',
        '!',
    ]
    # Add the NHEADER count as the very last header line (+1 for the NHEADER line itself)
    nheader = len(header_lines) + 1
    header_lines.append(f'! NHEADER = {nheader}')
    # Column label row (also a comment so Fortran skips it)
    header_lines.append(
        f'! {"wav_angstrom":>22s}  {"C_abs_cm2_per_H":>22s}  {"C_sca_cm2_per_H":>22s}  {"(1-g)Csca_cm2_per_H":>22s}'
    )

    # Only write rows where cabs, csca, and csca_rp are all finite (NaN rows mean
    # the wavelength is beyond the optical table coverage and values are invalid).
    _valid = np.isfinite(cabs) & np.isfinite(csca) & np.isfinite(csca_rp)
    nwav_valid = int(_valid.sum())
    # Update N_WAV in the header to reflect the actual number of rows written.
    header_lines = [ln.replace(f'! N_WAV = {nwav}', f'! N_WAV = {nwav_valid}')
                    for ln in header_lines]

    with open(outpath, 'w') as fout:
        for line in header_lines:
            fout.write(line + '\n')
        for i in range(nwav):
            if _valid[i]:
                fout.write(f'{wav_angstrom[i]:26.8e}  {cabs[i]:26.8e}  {csca[i]:26.8e}  {csca_rp[i]:26.8e}\n')

    if verbose:
        print(f'[export_zubko2004_cross_sections] wrote {nwav_valid} rows '
              f'(of {nwav} requested) to {outpath}')

    # ------------------------------------------------------------------
    # Load the Zubko et al. (2004) reference tables for comparison.
    # File columns: lam[um]  Cabs[cm^2]  Csca[cm^2]  Tau  a  g
    # Convert from cm^2 (per dust column) to cm^2/H using the factor
    # that normalises per H atom (N_H / cm^2 proportionality constant).
    # ------------------------------------------------------------------
    _NH_FACTOR = 1784268.76   # cm^2/H normalisation for Zubko 2004 BARE-GR-S
    _ref_path = PATH_EXTERNAL_DATA / 'zubko_2004_bare_gr_s.dat'
    _ref_data = np.loadtxt(str(_ref_path), comments='#')
    _ref_wav_um  = _ref_data[:, 0]                    # micron
    _ref_cabs_H  = _ref_data[:, 1] / _NH_FACTOR       # cm^2/H
    _ref_csca_H  = _ref_data[:, 2] / _NH_FACTOR       # cm^2/H
    _ref_g           = _ref_data[:, 5]                              # asymmetry parameter g
    _ref_csca_rp_H   = (1.0 - _ref_g) * _ref_csca_H               # (1-g)*C_sca  cm^2/H
    # Interpolate reference onto our output wavelength grid (log-space safe)
    _ref_ord = np.argsort(_ref_wav_um)
    cabs_ref     = np.interp(wav_micron, _ref_wav_um[_ref_ord], _ref_cabs_H[_ref_ord],
                             left=np.nan, right=np.nan)
    csca_ref     = np.interp(wav_micron, _ref_wav_um[_ref_ord], _ref_csca_H[_ref_ord],
                             left=np.nan, right=np.nan)
    csca_rp_ref  = np.interp(wav_micron, _ref_wav_um[_ref_ord], _ref_csca_rp_H[_ref_ord],
                             left=np.nan, right=np.nan)

    # ------------------------------------------------------------------
    # Comparison plot — three panels: C_abs, C_sca, C_rp
    # ------------------------------------------------------------------
    plot_path = None
    if plot_comparison:
        fig, (ax_abs, ax_sca, ax_rp) = plt.subplots(3, 1, figsize=(8, 10), dpi=150,
                                                      sharex=True)
        ax_abs.loglog(wav_micron, cabs, color='steelblue', lw=2,
                      label='Recomputed (this work)')
        ax_abs.loglog(_ref_wav_um, _ref_cabs_H, 'k--', lw=1.5,
                      label='Zubko et al. (2004) reference')
        ax_abs.set_ylabel(r'$C_{\rm abs}$ [cm$^2$ H$^{-1}$]', fontsize=12)
        ax_abs.legend(fontsize=11, frameon=False)
        ax_abs.grid(alpha=0.2, which='both')
        ax_abs.set_title('Zubko et al. (2004) BARE-GR-S — cross-section comparison',
                         fontsize=12)

        ax_sca.loglog(wav_micron, csca, color='darkorange', lw=2,
                      label=r'Recomputed $C_{\rm sca}$')
        ax_sca.loglog(_ref_wav_um, _ref_csca_H, 'k--', lw=1.5,
                      label=r'$C_{\rm sca}$ reference (Zubko 2004)')
        ax_sca.set_ylabel(r'$C_{\rm sca}$ [cm$^2$ H$^{-1}$]', fontsize=12)
        ax_sca.legend(fontsize=11, frameon=False)
        ax_sca.grid(alpha=0.2, which='both')

        ax_rp.loglog(wav_micron, csca_rp, color='mediumseagreen', lw=2,
                     label=r'Recomputed $(1-g)\,C_{\rm sca}$')
        ax_rp.loglog(_ref_wav_um, _ref_csca_rp_H, 'k--', lw=1.5,
                     label=r'$(1-g)\,C_{\rm sca}$ reference (Zubko 2004)')
        ax_rp.set_xlabel(r'Wavelength [$\mu$m]', fontsize=12)
        ax_rp.set_ylabel(r'$(1-g)\,C_{\rm sca}$ [cm$^2$ H$^{-1}$]', fontsize=12)
        ax_rp.legend(fontsize=11, frameon=False)
        ax_rp.grid(alpha=0.2, which='both')

        fig.tight_layout()
        plot_outpath = Path(out_png).expanduser().resolve()
        plot_outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_outpath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        plot_path = str(plot_outpath)
        if verbose:
            print(f'[export_zubko2004_cross_sections] comparison plot saved to {plot_path}')

    return {
        'path':      str(outpath),
        'wavelength': wav_micron,
        'C_abs':         cabs,
        'C_sca':         csca,
        'C_sca_rp':      csca_rp,
        'C_abs_ref':     cabs_ref,
        'C_sca_ref':     csca_ref,
        'C_sca_rp_ref':  csca_rp_ref,
        'plot_path':     plot_path,
    }


def plot_extinction_from_massfractions(dust_bins, dust_mass_fractions,
                                      pah_bins=None, pah_mass_fractions=None,
                                      out_png='test_extinction_curve.png',
                                      pah_state='neutral', verbose=False,
                                      optical_dir=None, mdust_per_H=1e-26,
                                      cabs_method='precomputed',
                                      nsize_per_bin=30,
                                      distribution_class=None,
                                      distribution_class_map=None):
    """
    Read precomputed DustBin/PAHBin optical tables, combine them with the
    supplied mass fractions, and plot the resulting extinction curve
    normalized to the V-band value (lambda_V = 0.55 micron).

    Parameters
    - dust_bins : list[str] or str
        Dust-bin IDs from the JSON configuration, for example
        ['DustBin_01', 'DustBin_02', 'DustBin_03', 'DustBin_04'].
    - dust_mass_fractions : array-like or dict
        Mass fractions associated with `dust_bins`. If a dict is supplied,
        its keys must be the dust-bin IDs.
    - pah_bins : list[str] or str or None
        Optional PAH-bin IDs from the JSON configuration, for example
        ['PAHbin_01', 'PAHbin_02'].
    - pah_mass_fractions : array-like or dict or None
        Mass fractions associated with `pah_bins`. If omitted, PAH bins are not
        included.
    - out_png : str or None
        If provided, save the plot to this path.
    - pah_state : str
        Which PAH block to read from the precomputed PAH tables: 'neutral'
        or 'ionised'.
    - verbose : bool
        Print info during processing.
    - optical_dir : str or Path or None
        Directory containing averaged_cross_section_<BinID>.txt files.
    - mdust_per_H : float
        Dust mass per H nucleus (g/H) for converting to A_lambda/N_H if desired.
    - cabs_method : str
        Method used to compute Cabs/Csca/Crp for the whole dust mixture.
        Accepted values:
        - 'precomputed': read averaged_cross_section_<BinID>.txt tables (default)
        - 'legacy': integrate raw optical tables over JSON log-normal distributions
          (old method)
        - 'mie': compute efficiencies directly using Mie theory and integrate (new method)
    - nsize_per_bin : int
        Number of size samples for the 'legacy' integration method.
    

    Returns
    -------
    dict
        Dictionary with wavelength grid, per-component curves, total cross
        sections, and the normalized extinction curve.
    """
    if optical_dir is None:
        optical_dir = PATH_MODEL_OPTICAL_OUTPUT

    if isinstance(dust_bins, str):
        dust_bins = [dust_bins]
    if pah_bins is None:
        pah_bins = []
    elif isinstance(pah_bins, str):
        pah_bins = [pah_bins]

    dust_bins = list(dust_bins)
    pah_bins = list(pah_bins)

    if isinstance(dust_mass_fractions, dict):
        dust_mf = np.array([dust_mass_fractions[bin_id] for bin_id in dust_bins], dtype=float)
    else:
        dust_mf = np.asarray(dust_mass_fractions, dtype=float)

    if len(dust_bins) != dust_mf.size:
        raise ValueError('dust_bins and dust_mass_fractions must have the same length')

    if len(pah_bins) == 0:
        pah_mf = np.array([], dtype=float)
    elif isinstance(pah_mass_fractions, dict):
        pah_mf = np.array([pah_mass_fractions[bin_id] for bin_id in pah_bins], dtype=float)
    else:
        if pah_mass_fractions is None:
            raise ValueError('pah_mass_fractions must be provided when pah_bins are supplied')
        pah_mf = np.asarray(pah_mass_fractions, dtype=float)

    if len(pah_bins) != pah_mf.size:
        raise ValueError('pah_bins and pah_mass_fractions must have the same length')

    component_bins = pah_bins + dust_bins
    component_mf = np.concatenate((pah_mf, dust_mf))
    if component_mf.size == 0:
        raise ValueError('At least one DustBin or PAHBin must be provided')
    if np.sum(component_mf) <= 0.0:
        raise ValueError('Mass fractions must sum to a positive value')

    component_mf = component_mf / np.sum(component_mf)

    if verbose:
        print('[plot_extinction_from_massfractions] bins:', component_bins)
        print('[plot_extinction_from_massfractions] normalized mass fractions:', component_mf)

    ncomp = len(component_bins)
    cabs_method_token = str(cabs_method).strip().lower()
    if cabs_method_token in ('precomputed', 'pre-computed', 'table', 'tables'):
        component_tables = []
        wavelength_sets = []
        for bin_id in component_bins:
            wav_i, cabs_i, csca_i, crp_i = _read_precomputed_cross_section_table(
                bin_id,
                optical_dir=optical_dir,
                pah_state=pah_state,
            )
            order_i = np.argsort(wav_i)
            wav_i = wav_i[order_i]
            cabs_i = cabs_i[order_i]
            csca_i = csca_i[order_i]
            crp_i = crp_i[order_i]
            component_tables.append((bin_id, wav_i, cabs_i, csca_i, crp_i))
            wavelength_sets.append(wav_i)

        wav = np.unique(np.concatenate(wavelength_sets))
        cabs_comps = np.zeros((ncomp, len(wav)))
        csca_comps = np.zeros((ncomp, len(wav)))
        crp_comps = np.zeros((ncomp, len(wav)))

        for i, (_, wav_i, cabs_i, csca_i, crp_i) in enumerate(component_tables):
            cabs_comps[i, :] = np.interp(wav, wav_i, cabs_i, left=0.0, right=0.0)
            csca_comps[i, :] = np.interp(wav, wav_i, csca_i, left=0.0, right=0.0)
            crp_comps[i, :] = np.interp(wav, wav_i, crp_i, left=0.0, right=0.0)
    elif cabs_method_token in ('mie', 'miedust'):
        wav = np.logspace(-1.5, 1.0, 100)
        cabs_comps, csca_comps, crp_comps = compute_component_cross_sections_mie(
            component_bins,
            target_wavelengths=wav,
            nsize_per_bin=nsize_per_bin,
            pah_state=pah_state,
            verbose=verbose,
            distribution_class=distribution_class,
            distribution_class_map=distribution_class_map,
        )
    elif cabs_method_token in ('legacy', 'old', 'raw', 'integration'):
        wav = np.logspace(-1.5, 1.0, 100)
        cabs_comps, csca_comps, crp_comps = _compute_component_cross_sections_legacy(
            component_bins,
            target_wavelengths=wav,
            nsize_per_bin=nsize_per_bin,
            pah_state=pah_state,
            verbose=verbose,
            distribution_class=distribution_class,
        )
    else:
        raise ValueError(
            "cabs_method must be one of 'precomputed', 'legacy' or 'mie' "
            f"(got {cabs_method})"
        )

    cext_comps = cabs_comps + csca_comps
    cabs_total = np.tensordot(component_mf, cabs_comps, axes=(0, 0))
    csca_total = np.tensordot(component_mf, csca_comps, axes=(0, 0))
    crp_total = np.tensordot(component_mf, crp_comps, axes=(0, 0))
    cext_total = cabs_total + csca_total

    total_y = 1.086 * cext_total
    comps_y = 1.086 * cext_comps

    # Group bin indices by composition for per-composition curves
    _bin_comp = {b['id']: b['composition'] for b in load_grain_size_config()['bins']}
    _comp_idx = {'pah': [], 'graphite': [], 'silicate': []}
    for _i, _bid in enumerate(component_bins):
        if _bid in pah_bins:
            _comp_idx['pah'].append(_i)
        else:
            _cp = _bin_comp.get(_bid, 'graphite')
            _comp_idx[_cp].append(_i)
    cext_by_comp = {
        _c: sum(component_mf[_i] * cext_comps[_i] for _i in _idxs) if _idxs
            else np.zeros(len(wav))
        for _c, _idxs in _comp_idx.items()
    }

    # find V band (0.55 micron) index for normalization
    lambda_V = 0.55
    idx_V = np.argmin(np.abs(wav - lambda_V))
    yV = total_y[idx_V]
    if not np.isfinite(yV) or yV == 0:
        finite = np.where(np.isfinite(total_y))[0]
        if finite.size == 0:
            raise RuntimeError('No finite values in extinction result to normalize')
        idx_V = finite[0]
        yV = total_y[idx_V]

    # normalize total and per-component curves by the total value at V
    y_norm = total_y / yV
    comp_norm = comps_y / yV

    # Compute Zubko et al. (2004) BARE-GR-S cross sections on the same wavelength grid
    zubko_result = compute_zubko2004_bare_gr_s_cross_sections(
        wavelengths_micron=wav, nsize=200, verbose=verbose
    )
    zubko_cabs_H = zubko_result['C_abs_total']  # cm^2/H
    zubko_cext_H = zubko_result['C_ext_total']  # cm^2/H
    zubko_total_y = 1.086 * zubko_cext_H
    zubko_idx_V = np.argmin(np.abs(wav - lambda_V))
    zubko_yV = zubko_total_y[zubko_idx_V]
    if np.isfinite(zubko_yV) and zubko_yV > 0:
        zubko_y_norm = zubko_total_y / zubko_yV
    else:
        zubko_y_norm = np.full_like(zubko_total_y, np.nan)

    # --- Top panel: grain size distributions (a^4 n(a)) scaled by mass fractions ---
    params = [get_lognormal_parameters(bin_id) for bin_id in component_bins]
    amin_cm = min(p['amin'] for p in params) * 1e-4
    amax_cm = max(p['amax'] for p in params) * 1e-4
    a_cm = np.logspace(np.log10(amin_cm), np.log10(amax_cm), 200)
    a_micron = a_cm * 1e4
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(7, 9), dpi=220,
                                         gridspec_kw={'height_ratios': [1, 1, 1.2]})
    colour = sns.color_palette('tab10', n_colors=max(ncomp, 3))

    # plot each component's size distribution scaled by its mass fraction
    _dist_cls_default = distribution_class if distribution_class is not None else LogNormal_Distribution
    distributions = []
    for _bid, p in zip(component_bins, params):
        _dist_cls_bin = (distribution_class_map or {}).get(_bid, _dist_cls_default)
        distributions.append(
            _dist_cls_bin(
                p['a0'] * 1e-4,
                p['amin'] * 1e-4,
                p['amax'] * 1e-4,
                p['sigma'],
                p['s'],
            )
        )

    # One solid line per composition (PAH / graphite / silicate); no per-bin lines, no total
    _comp_plot = [
        ('pah',      'navy',      'PAH'),
        ('graphite', 'darkgreen', 'Graphite'),
        ('silicate', 'darkred',   'Silicate'),
    ]
    for _c, _col, _lbl in _comp_plot:
        _idxs = _comp_idx[_c]
        if not _idxs:
            continue
        comp_dist = np.zeros_like(a_cm)
        for _i in _idxs:
            try:
                comp_dist += a_cm**4 * distributions[_i].n_density(component_mf[_i] * mdust_per_H, a_cm)
            except Exception:
                pass
        ax_top.plot(a_micron, comp_dist, color=_col, lw=2, ls='-', label=_lbl)

    # Zubko et al. (2004) BARE-GR-S grain size distributions for comparison
    # a^4 * dn/da converted to cm^3/H: (a_um*1e-4)^4 * (dnda_per_um * 1e4)
    _zubko_specs = [
        ('Zubko PAH',      _zubko_dnda_pah,      0.00035, 0.005, 'navy'),
        ('Zubko graphite', _zubko_dnda_graphite,  0.00035, 0.33,  'darkgreen'),
        ('Zubko silicate', _zubko_dnda_silicate,  0.00035, 0.37,  'darkred'),
    ]
    for _zlabel, _zfunc, _zamin, _zamax, _zcol in _zubko_specs:
        _az = np.logspace(np.log10(_zamin), np.log10(_zamax), 300)
        _az_cm = _az * 1e-4
        _y_z = _az_cm**4 * _zfunc(_az) * 1e4  # cm^3/H
        ax_top.plot(_az, _y_z, '--', lw=1.5, color=_zcol, label=_zlabel)

    ax_top.plot(a_micron,3e-27*a_micron**(.5),':',color='gray',linewidth=2)
    ax_top.text(0.2, 0.6, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax_top.transAxes,fontsize=12,rotation=16)

    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel(r'$a^4 n(a)$ (scaled by mass fraction)', fontsize=12)
    ax_top.set_xlabel(r'$a$ [$\mu$m]', fontsize=12)
    ax_top.set_ylim([5e-30, 1e-27])
    ax_top.set_xlim([amin_cm * 1e4, amax_cm * 1e4])
    ax_top.tick_params(labelsize=10)
    ax_top.grid(alpha=0.2, which='both')
    ax_top.legend(fontsize=10, loc='best', frameon=False, ncol=2)


    # --- Middle panel: mass-fraction weighted Cext per H ---
    x = 1.0 / wav
    order = np.argsort(x)
    _comp_styles = {'pah': ('navy', '-', 'PAH'), 'graphite': ('darkgreen', '-', 'Graphite'),
                    'silicate': ('darkred', '-', 'Silicate')}
    for _c, (_col, _ls, _lbl) in _comp_styles.items():
        _curve = cext_by_comp[_c] * mdust_per_H
        if np.any(_curve > 0):
            ax_mid.plot(x[order], _curve[order], color=_col, ls=_ls, lw=1.5, label=_lbl)
    ax_mid.plot(x[order], cext_total[order] * mdust_per_H, color='k', lw=2, label='Total')

    # Load the CLOUDY cross-sections for comparison
    data_cloudy = np.loadtxt(PATH_EXTERNAL_DATA / 'grains_CLOUDY.dat')
    _cloudy_order = np.argsort(data_cloudy[:, 0])
    _cloudy_yV = 1.086 * np.interp(lambda_V, data_cloudy[_cloudy_order, 0], data_cloudy[_cloudy_order, 1])
    ax_mid.plot(1/data_cloudy[:,0],data_cloudy[:,1],'r--',label='CLOUDY')
    ax_bot.plot(1/data_cloudy[:,0], 1.086*data_cloudy[:,1]/_cloudy_yV, 'r--')

    # Plot Harley's values
    harley_eV = np.array([0.1,  1.0, 8.245, 12.343, 14.371, 18.710, 29.321, 58.615])
    harley_wav_micron = 1.23984 / harley_eV
    harley_Cabs = np.array([5.190E-17, 7.611E-16, 2.140E-15, 2.830E-15, 2.955E-15, 2.929E-15, 2.442E-15, 1.303E-15])/1784268.76
    _harley_order = np.argsort(harley_wav_micron)
    _harley_yV = 1.086 * np.interp(lambda_V, harley_wav_micron[_harley_order], harley_Cabs[_harley_order])
    ax_mid.plot(1/harley_wav_micron, harley_Cabs, 'go', label='Harley', markersize=6)
    ax_bot.plot(1/harley_wav_micron, 1.086*harley_Cabs/_harley_yV, 'go', markersize=6)

    # Plot Zubko BARE-GR-S fit
    zb_wav_micron = np.logspace(-1.5,1,100)
    zb_wav_angstrom = zb_wav_micron * 1e4
    zb_Cabs = getCrosssection_BARE_GR_S_DUST(zb_wav_angstrom)/1784268.76
    _zb_yV = 1.086 * np.interp(lambda_V, zb_wav_micron, zb_Cabs)
    ax_mid.plot(1/zb_wav_micron, zb_Cabs, 'm--', label='Zubko et al. (2024) BARE-GR-S (RAMSES)', linewidth=2)
    ax_bot.plot(1/zb_wav_micron, 1.086*zb_Cabs/_zb_yV, 'm--', linewidth=2)

    # Plot the data from Zubko et al. (2004) BARE-GR-S model
    zubko_data = np.loadtxt(PATH_EXTERNAL_DATA / 'zubko_BAREGRS_extinction.csv', delimiter=',')
    _zubko_paper_wav = 1.0 / zubko_data[:, 0]   # 1/micron -> micron
    _zubko_paper_cabs = zubko_data[:, 1] * 1e-21
    _zubko_paper_order = np.argsort(_zubko_paper_wav)
    _zubko_paper_yV = 1.086 * np.interp(lambda_V, _zubko_paper_wav[_zubko_paper_order],
                                         _zubko_paper_cabs[_zubko_paper_order])
    ax_mid.plot(zubko_data[:,0], _zubko_paper_cabs, 'm-.', label='Zubko et al. (2004) BARE-GR-S (Paper)', linewidth=2)
    ax_bot.plot(zubko_data[:,0], 1.086*_zubko_paper_cabs/_zubko_paper_yV, 'm-.', linewidth=2)

    # Plot the recomputed Zubko et al. (2004) BARE-GR-S cross sections
    _zub_x = 1.0 / wav
    _zub_order = np.argsort(_zub_x)
    ax_mid.plot(_zub_x[_zub_order], zubko_cext_H[_zub_order], color='purple', lw=2,
                linestyle='-', label='Zubko et al. (2004) BARE-GR-S (recomputed)')
    ax_bot.plot(_zub_x[_zub_order], zubko_y_norm[_zub_order], color='purple', lw=2,
                linestyle='-', label='Zubko et al. (2004) BARE-GR-S (recomputed)')

    ax_mid.set_xlabel(r'$\lambda^{-1} [\mu {\rm m}^{-1}]$', fontsize=12)
    ylabel = r'$C_{\rm ext} [{\rm cm}^2 / {\rm H}]$'
    ax_mid.set_ylabel(ylabel, fontsize=12)
    ax_mid.tick_params(labelsize=10)
    ax_mid.grid(alpha=0.25, which='both')
    ax_mid.set_title('Extinction cross-section', fontsize=12)
    ax_mid.set_yscale('log')
    ax_mid.legend(fontsize=12, loc='best', ncol=1,frameon=False)
    ax_mid.set_xlim([0,16])
    ax_mid.set_ylim([2e-23,5e-21])

    # --- Bottom panel: extinction curve normalized at V ---
    x = 1.0 / wav
    order = np.argsort(x)
    for _c, (_col, _ls, _lbl) in _comp_styles.items():
        _curve = cext_by_comp[_c]
        if np.any(_curve > 0):
            ax_bot.plot(x[order], (_curve / yV)[order], color=_col, ls=_ls, lw=1.5, label=_lbl)
    ax_bot.plot(x[order], y_norm[order], color='k', lw=2, label='Total')

    ax_bot.set_xlabel(r'$\lambda^{-1} [\mu {\rm m}^{-1}]$', fontsize=12)
    ylabel = r'$A_\lambda / A_V$'
    ax_bot.set_ylabel(ylabel, fontsize=12)
    ax_bot.tick_params(labelsize=10)
    ax_bot.grid(alpha=0.25)
    ax_bot.set_title('Extinction curve normalized at V (%.2f $\\mu$m)' % lambda_V, fontsize=12)
    ax_bot.legend(fontsize=12, loc='best', ncol=3,frameon=False)
    ax_bot.set_xlim([0,10])

    # Data for MW (Pei 1992)
    mw_wav_inv = np.array([0.21,0.29,0.45,0.61,0.80,1.11,
                           1.43,1.82,2.27,2.50,2.91,3.65,
                           4.0,4.17,4.35,4.57,4.76,5.0,5.26,
                           5.56,5.88,6.25,6.71,7.18,7.60,
                           8.0,8.5,9.0,9.5,10.])
    mw_Alambda_over_AB = np.array([-3.02,-2.91,-2.76,-2.58,-2.23,-1.60,-0.78,
                                   0.0,1.0,1.3,1.8,3.10,4.19,4.90,
                                   5.77,6.57,6.23,5.52,4.90,4.65,4.60,4.73,
                                   4.99,5.36,5.91,6.55,7.45,8.45,
                                   9.80,11.30])
    mw_RV = 3.08
    mw_A_lambda_over_AV = mw_Alambda_over_AB / mw_RV + 1.0
    ax_bot.scatter(mw_wav_inv, mw_A_lambda_over_AV, color='grey', label='MW (Pei 1992)', s=20, alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    if verbose:
        print(f'[plot_extinction_from_massfractions] saved plot to {out_png}')

    plt.close(fig)

    return {
        'wavelength': wav,
        'bin_ids': component_bins,
        'cabs_method': cabs_method_token,
        'mass_fractions': component_mf,
        'C_abs_total': cabs_total,
        'C_sca_total': csca_total,
        'C_rp_total': crp_total,
        'C_ext_total': cext_total,
        'C_abs_components': cabs_comps,
        'C_sca_components': csca_comps,
        'C_rp_components': crp_comps,
        'A_total': total_y,
        'A_components': comps_y,
        'A_over_AV': y_norm,
        'A_over_AV_components': comp_norm,
    }


def fit_massfractions_to_zubko2004(
    dust_bins,
    dust_mass_fractions_init,
    pah_bins=None,
    pah_mass_fractions_init=None,
    out_png='fit_zubko_extinction.png',
    pah_state='neutral',
    verbose=True,
    optical_dir=None,
    mdust_per_H=1e-26,
    cabs_method='precomputed',
    nsize_per_bin=30,
    nsize_zubko=200,
    wav_fit_range=(0.1, 10.0),
    composition_penalty_weight=10.0,
    zubko_mf_ref=None,
    distribution_class=None,
):
    """Fit bin mass fractions to best reproduce the Zubko et al. (2004) BARE-GR-S
    extinction cross section.

    The per-component cross-section tables are computed once, then a chi-squared
    minimisation in log-space finds the mass-fraction vector that minimises the
    difference between the weighted sum of per-component C_ext curves and the
    Zubko reference.

    Strategy
    --------
    The problem is parameterised with ``ncomp - 1`` free ratios so that the
    sum-to-one constraint is satisfied by construction (no equality constraint
    needed).  A global search with ``scipy.optimize.differential_evolution`` is
    run first; its best solution is then polished automatically by the built-in
    ``polish=True`` option (L-BFGS-B).  This avoids the gradient / linesearch
    failures that local methods (SLSQP, etc.) suffer when started far from the
    optimum or near a boundary.

    After convergence the best-fit fractions are printed and
    ``plot_extinction_from_massfractions`` is called to produce a comparison
    figure.

    Parameters
    ----------
    dust_bins : list[str]
        Dust-bin IDs.
    dust_mass_fractions_init : array-like
        Initial guess for dust mass fractions (will be renormalized).
    pah_bins : list[str] or None
        PAH-bin IDs.
    pah_mass_fractions_init : array-like or None
        Initial guess for PAH mass fractions.
    out_png : str
        Output filename for the best-fit comparison plot.
    pah_state : str
        'neutral' or 'ionised'.
    verbose : bool
        Print progress and results.
    optical_dir : str or Path or None
        Directory with precomputed cross-section tables.
    mdust_per_H : float
        Dust mass per H nucleus [g/H] for the absolute cross-section scale.
    cabs_method : str
        'precomputed', 'legacy', or 'mie'.
    nsize_per_bin : int
        Number of size quadrature points for legacy method.
    nsize_zubko : int
        Number of size quadrature points for the Zubko reference integral.
    wav_fit_range : tuple(float, float)
        Wavelength range in microns over which the chi-squared is evaluated.
    composition_penalty_weight : float
        Weight applied to the composition constraint penalty.  The spectral
        chi-squared is normalised by the number of fit wavelength points, so
        it represents the *mean* squared log-residual per point (O(1) for a
        reasonable fit).  The penalty uses the same scale::

            objective = chi2_spec / nwav + w * [((mf_PAH  - ref_PAH)  / ref_PAH)²
                                                + ((mf_GRA  - ref_GRA)  / ref_GRA)²
                                                + ((mf_SIL  - ref_SIL)  / ref_SIL)²]

        A value of ``w = 10`` means that deviating by 100 % in any single
        composition type costs as much as 10 wavelength points with a
        log-residual of 1 (i.e. a factor-of-10 flux error each).
        Set to ``0`` to disable the constraint entirely.  Default is ``10.0``.
    zubko_mf_ref : dict or None
        Override the Zubko BARE-GR-S reference composition fractions.  Keys
        are ``'pah'``, ``'graphite'``, ``'silicate'``; values are mass
        fractions (they do **not** need to sum to 1 — they are used only as
        penalty targets).  Any missing key falls back to the default values
        ``{'pah': 0.0457, 'graphite': 0.2947, 'silicate': 0.6596}``.

    Returns
    -------
    dict
        best_mass_fractions : ndarray  -- best-fit fractions (normalized, sum=1)
        bin_ids             : list[str]
        chi2                : float    -- final chi-squared value (spectral only)
        chi2_composition    : float    -- unweighted composition penalty
        converged           : bool
        plot_result         : dict     -- return value of plot_extinction_from_massfractions
    """
    from scipy.optimize import differential_evolution

    if optical_dir is None:
        optical_dir = PATH_MODEL_OPTICAL_OUTPUT

    # ------------------------------------------------------------------ #
    # 1. Assemble bin lists and initial fractions                         #
    # ------------------------------------------------------------------ #
    if isinstance(dust_bins, str):
        dust_bins = [dust_bins]
    if pah_bins is None:
        pah_bins = []
    elif isinstance(pah_bins, str):
        pah_bins = [pah_bins]
    dust_bins = list(dust_bins)
    pah_bins  = list(pah_bins)

    dust_mf0 = np.asarray(dust_mass_fractions_init, dtype=float)
    pah_mf0  = np.asarray(pah_mass_fractions_init, dtype=float) \
               if pah_mass_fractions_init is not None else np.array([], dtype=float)

    component_bins = pah_bins + dust_bins
    mf0 = np.concatenate((pah_mf0, dust_mf0))
    mf0 = mf0 / mf0.sum()          # normalize initial guess
    ncomp = len(component_bins)

    if verbose:
        print('[fit_massfractions_to_zubko2004] bins:', component_bins)
        print('[fit_massfractions_to_zubko2004] initial mass fractions:', mf0)

    # ------------------------------------------------------------------ #
    # 1b. Classify each component bin by grain type                       #
    #                                                                      #
    # PAH bins are passed explicitly; dust bins are classified by reading  #
    # their 'composition' field ('graphite' or 'silicate') from the JSON   #
    # grain size configuration.                                            #
    # ------------------------------------------------------------------ #
    _zubko_mf_ref_default = {'pah': 0.0457, 'graphite': 0.2947, 'silicate': 0.6596}
    if zubko_mf_ref is None:
        zubko_mf_ref = _zubko_mf_ref_default
    else:
        zubko_mf_ref = {**_zubko_mf_ref_default, **zubko_mf_ref}

    # Read composition directly from the JSON grain-size configuration so we
    # get the actual 'graphite'/'silicate' field rather than relying on the
    # lognormal-parameter dict (which does not carry composition metadata).
    _cfg = load_grain_size_config()
    _bin_meta = {b['id']: b for b in _cfg['bins']}

    comp_types = []
    for i, bin_id in enumerate(component_bins):
        if i < len(pah_bins):
            comp_types.append('pah')
        else:
            meta = _bin_meta.get(bin_id)
            if meta is None:
                raise KeyError(f"Bin '{bin_id}' not found in grain_size_distribution.json")
            comp_types.append(str(meta['composition']).lower())

    pah_mask = np.array([t == 'pah'      for t in comp_types])
    gra_mask = np.array([t == 'graphite' for t in comp_types])
    sil_mask = np.array([t == 'silicate' for t in comp_types])

    ref_pah = float(zubko_mf_ref.get('pah',      0.0))
    ref_gra = float(zubko_mf_ref.get('graphite', 0.0))
    ref_sil = float(zubko_mf_ref.get('silicate', 0.0))

    if verbose:
        print('[fit_massfractions_to_zubko2004] grain types:', dict(zip(component_bins, comp_types)))
        print(f'[fit_massfractions_to_zubko2004] Zubko composition targets  —  '
              f'PAH: {ref_pah:.4f}  graphite: {ref_gra:.4f}  silicate: {ref_sil:.4f}')
        print(f'[fit_massfractions_to_zubko2004] composition_penalty_weight = {composition_penalty_weight}')

    # ------------------------------------------------------------------ #
    # 2. Pre-compute per-component C_ext tables (done only once)         #
    # ------------------------------------------------------------------ #
    cabs_method_token = str(cabs_method).strip().lower()
    if cabs_method_token in ('precomputed', 'pre-computed', 'table', 'tables'):
        component_tables = []
        wavelength_sets  = []
        for bin_id in component_bins:
            wav_i, cabs_i, csca_i, crp_i = _read_precomputed_cross_section_table(
                bin_id, optical_dir=optical_dir, pah_state=pah_state
            )
            ord_i = np.argsort(wav_i)
            component_tables.append((wav_i[ord_i], (cabs_i + csca_i)[ord_i]))
            wavelength_sets.append(wav_i[ord_i])

        wav = np.unique(np.concatenate(wavelength_sets))
        cext_comps = np.zeros((ncomp, len(wav)))
        for i, (wav_i, cext_i) in enumerate(component_tables):
            cext_comps[i] = np.interp(wav, wav_i, cext_i, left=0.0, right=0.0)

    elif cabs_method_token in ('mie', 'miedust'):
        wav = np.logspace(-1.5, 1.0, 100)
        cabs_c, csca_c, _ = compute_component_cross_sections_mie(
            component_bins,
            target_wavelengths=wav,
            nsize_per_bin=nsize_per_bin,
            pah_state=pah_state,
            verbose=verbose,
            distribution_class=distribution_class,
        )
        cext_comps = cabs_c + csca_c
    elif cabs_method_token in ('legacy', 'old', 'raw', 'integration'):
        wav = np.logspace(-1.5, 1.0, 100)
        cabs_c, csca_c, _ = _compute_component_cross_sections_legacy(
            component_bins,
            target_wavelengths=wav,
            nsize_per_bin=nsize_per_bin,
            pah_state=pah_state,
            verbose=verbose,
            distribution_class=distribution_class,
        )
        cext_comps = cabs_c + csca_c
    else:
        raise ValueError(f"cabs_method must be 'precomputed', 'legacy', or 'mie' (got {cabs_method})")

    # ------------------------------------------------------------------ #
    # 3. Pre-compute Zubko reference C_ext (done only once)              #
    # ------------------------------------------------------------------ #
    if verbose:
        print('[fit_massfractions_to_zubko2004] computing Zubko reference ...')
    zubko_result = compute_zubko2004_bare_gr_s_cross_sections(
        wavelengths_micron=wav, nsize=nsize_zubko, verbose=False
    )
    zubko_cext = zubko_result['C_ext_total']   # cm^2/H

    # Restrict fit to the requested wavelength range
    wav_mask = (wav >= wav_fit_range[0]) & (wav <= wav_fit_range[1])
    if wav_mask.sum() < 2:
        raise ValueError(
            f'wav_fit_range {wav_fit_range} leaves fewer than 2 wavelength points; '
            'widen the range or check the wavelength grid.'
        )
    zubko_fit = zubko_cext[wav_mask]          # cm^2/H
    cext_fit  = cext_comps[:, wav_mask]       # (ncomp, nwav_fit) in cm^2/g_dust

    log_zubko = np.log10(np.maximum(zubko_fit, 1e-100))
    nwav_fit = int(wav_mask.sum())

    # ------------------------------------------------------------------ #
    # 4. Chi-squared objective in log-space                               #
    #                                                                      #
    # Parameterise with ncomp-1 free "shares" s_i ∈ [0, 1].              #
    # The last fraction is derived as max(0, 1 - sum(s)), so the simplex  #
    # constraint is enforced by construction.  The vector is renormalized  #
    # after clipping for numerical safety.                                 #
    #                                                                      #
    # Scaling: the spectral term is normalised by nwav_fit so it gives    #
    # the mean squared log-residual per wavelength point (O(1) for a      #
    # reasonable fit).  The composition penalty terms are also O(1) at    #
    # 100 % relative error, so composition_penalty_weight=10 means the    #
    # constraint is worth ~10 wavelength points.                          #
    # ------------------------------------------------------------------ #
    def _mf_from_shares(s):
        s = np.asarray(s, dtype=float)
        mf = np.empty(ncomp)
        mf[:-1] = np.clip(s, 0.0, 1.0)
        mf[-1]  = max(0.0, 1.0 - mf[:-1].sum())
        total = mf.sum()
        return mf / total if total > 0 else mf

    def _chi2(s):
        mf = _mf_from_shares(s)
        cext_model = np.dot(mf, cext_fit) * mdust_per_H
        cext_model = np.maximum(cext_model, 1e-100)
        residuals = np.log10(cext_model) - log_zubko
        # Normalise by number of wavelength points so the spectral and
        # composition terms are on the same O(1) scale.
        chi2_spec_norm = float(np.sum(residuals ** 2)) / nwav_fit

        if composition_penalty_weight > 0.0:
            penalty = 0.0
            if ref_pah > 0.0 and pah_mask.any():
                penalty += ((mf[pah_mask].sum() - ref_pah) / ref_pah) ** 2
            if ref_gra > 0.0 and gra_mask.any():
                penalty += ((mf[gra_mask].sum() - ref_gra) / ref_gra) ** 2
            if ref_sil > 0.0 and sil_mask.any():
                penalty += ((mf[sil_mask].sum() - ref_sil) / ref_sil) ** 2
            return chi2_spec_norm + composition_penalty_weight * penalty

        return chi2_spec_norm

    # Search bounds: each share in [0, 1]; the last fraction is implicit
    bounds_de = [(0.0, 1.0)] * (ncomp - 1)

    # Seed the initial population with the user-supplied guess
    init_pop = np.tile(mf0[:-1], (max(15 * (ncomp - 1), 20), 1))
    rng = np.random.default_rng(42)
    init_pop += rng.uniform(-0.1, 0.1, init_pop.shape)
    init_pop = np.clip(init_pop, 0.0, 1.0)

    if verbose:
        print('[fit_massfractions_to_zubko2004] running differential_evolution '
              f'(ncomp={ncomp}, nwav_fit={wav_mask.sum()}) ...')

    de_result = differential_evolution(
        _chi2,
        bounds_de,
        init=init_pop,
        seed=42,
        maxiter=2000,
        tol=1e-14,
        popsize=15,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,          # L-BFGS-B polishing step after DE
        workers=1,
        disp=verbose,
    )

    mf_best = _mf_from_shares(de_result.x)
    chi2_final = de_result.fun

    # Decompose the final objective into its spectral and composition parts
    _cext_best = np.dot(mf_best, cext_fit) * mdust_per_H
    _cext_best = np.maximum(_cext_best, 1e-100)
    chi2_spectral = float(np.sum((np.log10(_cext_best) - log_zubko) ** 2))
    chi2_comp_pen = 0.0
    if ref_pah > 0.0 and pah_mask.any():
        chi2_comp_pen += ((mf_best[pah_mask].sum() - ref_pah) / ref_pah) ** 2
    if ref_gra > 0.0 and gra_mask.any():
        chi2_comp_pen += ((mf_best[gra_mask].sum() - ref_gra) / ref_gra) ** 2
    if ref_sil > 0.0 and sil_mask.any():
        chi2_comp_pen += ((mf_best[sil_mask].sum() - ref_sil) / ref_sil) ** 2

    # ------------------------------------------------------------------ #
    # 5. Print results                                                    #
    # ------------------------------------------------------------------ #
    print('\n' + '='*60)
    print('fit_massfractions_to_zubko2004  —  RESULTS')
    print('='*60)
    print(f'  Converged          : {de_result.success}  ({de_result.message})')
    print(f'  Final chi² (total) : {chi2_final:.6g}')
    print(f'    spectral part    : {chi2_spectral:.6g}')
    print(f'    composition pen. : {chi2_comp_pen:.6g}  (weight={composition_penalty_weight})')
    print(f'  Wavelength fit range: {wav_fit_range[0]:.3g} – {wav_fit_range[1]:.3g} µm '
          f'({wav_mask.sum()} points)')
    print()
    print(f'  {"Bin ID":<20}  {"Type":<10}  {"Init MF":>10}  {"Best-fit MF":>12}')
    print(f'  {"-"*20}  {"-"*10}  {"-"*10}  {"-"*12}')
    for i, bin_id in enumerate(component_bins):
        print(f'  {bin_id:<20}  {comp_types[i]:<10}  {mf0[i]:>10.4f}  {mf_best[i]:>12.6f}')
    print()
    print(f'  {"Type":<12}  {"Best-fit total":>16}  {"Zubko ref":>12}  {"Rel. error":>12}')
    print(f'  {"-"*12}  {"-"*16}  {"-"*12}  {"-"*12}')
    for _type, _mask, _ref in [('PAH',      pah_mask, ref_pah),
                                ('graphite', gra_mask, ref_gra),
                                ('silicate', sil_mask, ref_sil)]:
        if _mask.any() and _ref > 0.0:
            _tot = mf_best[_mask].sum()
            _rel = (_tot - _ref) / _ref
            print(f'  {_type:<12}  {_tot:>16.6f}  {_ref:>12.4f}  {_rel:>+12.2%}')
    print('='*60 + '\n')

    # ------------------------------------------------------------------ #
    # 6. Plot using the best-fit fractions                                #
    # ------------------------------------------------------------------ #
    n_pah        = len(pah_bins)
    mf_best_pah  = mf_best[:n_pah]
    mf_best_dust = mf_best[n_pah:]

    plot_result = plot_extinction_from_massfractions(
        dust_bins=dust_bins,
        dust_mass_fractions=mf_best_dust,
        pah_bins=pah_bins if pah_bins else None,
        pah_mass_fractions=mf_best_pah if n_pah > 0 else None,
        out_png=out_png,
        pah_state=pah_state,
        verbose=verbose,
        optical_dir=optical_dir,
        mdust_per_H=mdust_per_H,
        cabs_method=cabs_method,
        nsize_per_bin=nsize_per_bin,
        distribution_class=distribution_class,
    )

    return {
        'best_mass_fractions': mf_best,
        'bin_ids':             component_bins,
        'chi2':                chi2_spectral,
        'chi2_composition':    chi2_comp_pen,
        'converged':           de_result.success,
        'plot_result':         plot_result,
    }


def compute_bb_averaged_cross_sections(bin_edges_ev, temperature_k, config_path=None, optical_dir=None, pah_state='neutral', nE=1000):
    """
    Compute the photon number weighted average cross section for different radiation bins
    assuming a BlackBody spectrum, for each dust/PAH bin.

    Parameters
    ----------
    bin_edges_ev : array-like of float
        Energy edges of different radiation bins in eV (length N+1 for N bins).
    temperature_k : float
        Temperature of the BlackBody radiation field in Kelvin.
    config_path : str or Path, optional
        Path to the grain-size configuration JSON file. If None, uses the active config.
    optical_dir : str or Path, optional
        Directory containing the precomputed cross-section files (e.g. `averaged_cross_section_<BinID>.txt`).
        If None, defaults to `PATH_MODEL_OPTICAL_OUTPUT`.
    pah_state : str, optional
        The state of PAH, either 'neutral' or 'ionised'. Default is 'neutral'.
    nE : int, optional
        Number of integration points within each radiation bin. Default is 1000.

    Returns
    -------
    dict
        A dictionary containing:
        - 'C_abs': 2D numpy array of shape (n_dust_pah_bins, n_radiation_bins)
        - 'C_sca': 2D numpy array of shape (n_dust_pah_bins, n_radiation_bins)
        - 'C_rp':  2D numpy array of shape (n_dust_pah_bins, n_radiation_bins)
        - 'dust_pah_bins': list of str, the dust/PAH bin IDs
        - 'radiation_bins': list of tuples (E_low, E_high) representing the radiation bins
    """
    # 1. Load grain-size configuration to get the list of dust and PAH bins
    cfg = load_grain_size_config(config_path=config_path)
    bin_ids = [b['id'] for b in cfg['bins']]
    n_bins = len(bin_ids)

    # 2. Parse radiation bin edges
    edges = np.asarray(bin_edges_ev, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin_edges_ev must be a 1D array-like with at least 2 elements.")
    
    n_rad_bins = len(edges) - 1
    rad_bins = [(float(edges[j]), float(edges[j+1])) for j in range(n_rad_bins)]

    # 3. Initialize output arrays
    C_abs_avg = np.zeros((n_bins, n_rad_bins))
    C_sca_avg = np.zeros((n_bins, n_rad_bins))
    C_rp_avg  = np.zeros((n_bins, n_rad_bins))

    # Constants
    conv_eum = 1.239841984  # E[eV] * lambda[um]
    kB = 8.617333262145e-5  # Boltzmann constant in eV/K

    # 4. Load cross sections for each bin
    # We do this once to avoid repeated file reads
    bin_cross_sections = {}
    for bin_id in bin_ids:
        wav_um, cabs, csca, crp = _read_precomputed_cross_section_table(
            bin_id, optical_dir=optical_dir, pah_state=pah_state
        )
        # Convert wavelength to energy in eV
        E_table = conv_eum / np.maximum(wav_um, 1e-30)
        # Sort in ascending order of energy for np.interp
        order = np.argsort(E_table)
        bin_cross_sections[bin_id] = {
            'E': E_table[order],
            'C_abs': cabs[order],
            'C_sca': csca[order],
            'C_rp': crp[order]
        }

    # 5. Compute the photon-weighted average for each dust/PAH bin and each radiation bin
    for j, (E_low, E_high) in enumerate(rad_bins):
        if E_low >= E_high:
            raise ValueError(f"Invalid radiation bin edges: {E_low} to {E_high}. Edges must be strictly increasing.")

        # Create integration grid
        E_grid = np.linspace(E_low, E_high, num=nE)

        # Compute BlackBody photon spectrum weights: n_gamma(E) dE \propto E^2 / (exp(E / kBT) - 1)
        if temperature_k <= 0.0:
            # At T=0, blackbody photon density is 0.
            weights = np.zeros_like(E_grid)
        else:
            kBT = kB * temperature_k
            x = E_grid / kBT
            x0 = x[0]
            if x0 > 700.0:
                # All x in the bin are > 700. Use scaled weights to avoid overflow/underflow.
                weights = E_grid**2 * np.exp(-(x - x0))
            else:
                # Standard calculation using expm1 for stability.
                safe_x = np.minimum(x, 700.0)
                weights = np.where(x > 700.0, E_grid**2 * np.exp(-x), E_grid**2 / np.expm1(safe_x))

        denom = np.trapezoid(weights, E_grid)

        # Loop over dust/PAH bins
        for i, bin_id in enumerate(bin_ids):
            data = bin_cross_sections[bin_id]
            # Interpolate cross sections onto the integration grid
            cabs_grid = np.interp(E_grid, data['E'], data['C_abs'])
            csca_grid = np.interp(E_grid, data['E'], data['C_sca'])
            crp_grid  = np.interp(E_grid, data['E'], data['C_rp'])

            if denom > 0.0:
                C_abs_avg[i, j] = np.trapezoid(cabs_grid * weights, E_grid) / denom
                C_sca_avg[i, j] = np.trapezoid(csca_grid * weights, E_grid) / denom
                C_rp_avg[i, j]  = np.trapezoid(crp_grid * weights, E_grid) / denom
            else:
                # If denom is 0, default to unweighted (flat) average over the bin
                C_abs_avg[i, j] = np.trapezoid(cabs_grid, E_grid) / (E_high - E_low)
                C_sca_avg[i, j] = np.trapezoid(csca_grid, E_grid) / (E_high - E_low)
                C_rp_avg[i, j]  = np.trapezoid(crp_grid, E_grid) / (E_high - E_low)

    return {
        'C_abs': C_abs_avg,
        'C_sca': C_sca_avg,
        'C_rp': C_rp_avg,
        'dust_pah_bins': bin_ids,
        'radiation_bins': rad_bins
    }

