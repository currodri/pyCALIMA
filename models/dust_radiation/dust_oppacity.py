"""
DUST EFFICIENCIES TABLES

This set of tools have been constructed such that the public
tables from B. Draine and co. can be read, visualised
and reorganised in look-up tables for RAMSES Dust-RTZ

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import some libraries
import os
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
from models.dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,\
                        LogNormal_Distribution,PowerLaw_ExpCutoff_Distribution, \
                        Classical_LogNormal_Distribution
from models.grain_size_config import get_optical_props_path, get_lognormal_parameters, load_grain_size_config
from models.tools.radiation_fields import Mathis83_radiation_field

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
        save_imn_file(metadata, str(output_dir / out_stem))
        saved_paths.append(str(output_dir / out_stem))

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
        from models.PAH_radiation.pah_oppacity import pah_efficiencies
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
                                  efficiency=False, data_table=None):
    """
    Interpolate cross sections in both size and wavelength.

    Parameters
    - dust_type: same as interpolate_cross_sections (silicate, graphite, iPAH, nPAH, PAH)
    - grain_size: target grain size in microns
    - target_wavelengths: array-like of wavelengths in microns to interpolate to.
        If None, uses the native wavelengths from the table.
    - efficiency: if True, return Q values (dimensionless); otherwise return C (cm^2)
    - data_table: optional (nwav, data, columns, name) tuple to avoid re-reading files

    Returns (grain_size_cm, wavelengths_cm, C_sca, C_abs, C_rp)
    Similar units/shape as interpolate_cross_sections.
    """
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
            from models.PAH_radiation.pah_oppacity import pah_efficiencies, interpolate_pah_cross_sections_2d
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
    1) precomputed DustBin/PAHbin tables, or
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

    # Preferred path: use precomputed optical-property tables for DustBin/PAHbin IDs.
    # This avoids recomputing size-integrated cross sections.
    is_precomputed_bins = all(
        isinstance(comp, str) and ('DustBin_' in comp or 'PAHbin_' in comp)
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
            from models.PAH_radiation.pah_oppacity import pah_efficiencies
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
    """Read one precomputed DustBin/PAHbin optical table.

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


def _compute_component_cross_sections_legacy(component_bins, target_wavelengths,
                                             nsize_per_bin=30,
                                             pah_state='neutral', verbose=False):
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

    optical_cache = {}

    for i, (bin_id, material) in enumerate(zip(component_bins, materials)):
        if material not in optical_cache:
            if material == 'silicate':
                filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81')
                optical_cache[material] = dust_efficiencies(filename)
            elif material == 'graphite':
                filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81')
                optical_cache[material] = dust_efficiencies(filename)
            elif material == 'iPAH':
                from models.PAH_radiation.pah_oppacity import pah_efficiencies
                filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30')
                optical_cache[material] = pah_efficiencies(filename)
            elif material == 'nPAH':
                from models.PAH_radiation.pah_oppacity import pah_efficiencies
                filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
                optical_cache[material] = pah_efficiencies(filename)
            else:
                raise ValueError(f'Unsupported material for legacy interpolation: {material}')

        p = get_lognormal_parameters(bin_id)
        dist = LogNormal_Distribution(
            p['a0'] * 1e-4,
            p['amin'] * 1e-4,
            p['amax'] * 1e-4,
            p['sigma'],
            p['s'],
        )

        size_bins = np.logspace(np.log10(dist.amin), np.log10(dist.amax), int(nsize_per_bin))
        n_for_unit_mass = dist.n_density(1.0, size_bins)

        cabs_dist = np.zeros((len(size_bins), nwav))
        csca_dist = np.zeros((len(size_bins), nwav))
        crp_dist = np.zeros((len(size_bins), nwav))

        for isize, a_cm in enumerate(size_bins):
            _, _, csca_i, cabs_i, crp_i = interpolate_cross_sections_2d(
                material,
                a_cm * 1e4,
                target_wavelengths=target_wavelengths,
                data_table=optical_cache[material],
            )
            cabs_dist[isize, :] = cabs_i * n_for_unit_mass[isize]
            csca_dist[isize, :] = csca_i * n_for_unit_mass[isize]
            crp_dist[isize, :] = crp_i * n_for_unit_mass[isize]

        cabs_comps[i, :] = np.trapezoid(cabs_dist, size_bins, axis=0)
        csca_comps[i, :] = np.trapezoid(csca_dist, size_bins, axis=0)
        crp_comps[i, :] = np.trapezoid(crp_dist, size_bins, axis=0)

        if verbose:
            print(f'[plot_extinction_from_massfractions] legacy component done: {bin_id} ({material})')

    return cabs_comps, csca_comps, crp_comps

def plot_extinction_from_massfractions(dust_bins, dust_mass_fractions,
                                      pah_bins=None, pah_mass_fractions=None,
                                      out_png='test_extinction_curve.png',
                                      pah_state='neutral', verbose=False,
                                      optical_dir=None, mdust_per_H=1e-26,
                                      cabs_method='precomputed',
                                      nsize_per_bin=30):
    """
    Read precomputed DustBin/PAHbin optical tables, combine them with the
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
        raise ValueError('At least one DustBin or PAHbin must be provided')
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
    elif cabs_method_token in ('legacy', 'old', 'raw', 'integration'):
        wav = np.logspace(-1.5, 1.0, 100)
        cabs_comps, csca_comps, crp_comps = _compute_component_cross_sections_legacy(
            component_bins,
            target_wavelengths=wav,
            nsize_per_bin=nsize_per_bin,
            pah_state=pah_state,
            verbose=verbose,
        )
    else:
        raise ValueError(
            "cabs_method must be one of 'precomputed' or 'legacy' "
            f"(got {cabs_method})"
        )

    cext_comps = cabs_comps + csca_comps
    cabs_total = np.tensordot(component_mf, cabs_comps, axes=(0, 0))
    csca_total = np.tensordot(component_mf, csca_comps, axes=(0, 0))
    crp_total = np.tensordot(component_mf, crp_comps, axes=(0, 0))
    cext_total = cabs_total + csca_total

    total_y = 1.086 * cext_total
    comps_y = 1.086 * cext_comps

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
    distributions = []
    for p in params:
        distributions.append(
            LogNormal_Distribution(
                p['a0'] * 1e-4,
                p['amin'] * 1e-4,
                p['amax'] * 1e-4,
                p['sigma'],
                p['s'],
            )
        )

    for i, bin_id in enumerate(component_bins):
        try:
            n_vs_a = distributions[i].n_density(1.0, a_cm)
        except Exception:
            n_vs_a = np.zeros_like(a_cm)
        ydist = a_cm**4 * n_vs_a * component_mf[i] * mdust_per_H
        ax_top.plot(a_micron, ydist, color=colour[i], lw=2, label=bin_id)

    # combined distribution
    total_dist = np.zeros_like(a_micron)
    for i in range(ncomp):
        total_dist += a_cm**4 * distributions[i].n_density(component_mf[i]*mdust_per_H, a_cm)
    ax_top.plot(a_micron, total_dist, color='k', lw=2)

    ax_top.plot(a_micron,3e-27*a_micron**(.5),':',color='gray',linewidth=2)
    ax_top.text(0.2, 0.6, r'MRN ($n\propto a^{-3.5}$)',
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax_top.transAxes,fontsize=12,rotation=16)

    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel(r'$a^4 n(a)$ (scaled by mass fraction)', fontsize=12)
    ax_top.set_xlabel(r'$a$ [$\mu$m]', fontsize=12)
    ax_top.set_ylim([5e-30,1e-27])
    ax_top.set_xlim([amin_cm * 1e4, amax_cm * 1e4])
    ax_top.tick_params(labelsize=10)
    ax_top.grid(alpha=0.2, which='both')
    ax_top.legend(fontsize=10, loc='best', frameon=False, ncol=2)


    # --- Middle panel: mass-fraction weighted Cabs ---
    x = 1.0 / wav
    order = np.argsort(x)
    ax_mid.plot(x[order], cabs_total[order], color='k', lw=2)

    # plot per-component normalized contributions (if present)
    for i, bin_id in enumerate(component_bins):
        comp_curve = component_mf[i] * cabs_comps[i, :]
        if np.any(np.isfinite(comp_curve)):
            ax_mid.plot(x[order], comp_curve[order], lw=2, color=colour[i])

    # Load the CLOUDY cross-sections for comparison
    data_cloudy = np.loadtxt(PATH_EXTERNAL_DATA / 'grains_CLOUDY.dat')
    ax_mid.plot(1/data_cloudy[:,0],data_cloudy[:,1],'r--',label='CLOUDY')
    ax_bot.plot(1/data_cloudy[:,0],data_cloudy[:,1]/yV*1.086,'r--')

    # Plot Harley's values
    harley_eV = np.array([0.1,  1.0, 8.245, 12.343, 14.371, 18.710, 29.321, 58.615])
    harley_wav_micron = 1.23984 / harley_eV
    harley_Cabs = np.array([5.190E-17, 7.611E-16, 2.140E-15, 2.830E-15, 2.955E-15, 2.929E-15, 2.442E-15, 1.303E-15])/1784268.76
    ax_mid.plot(1/harley_wav_micron, harley_Cabs, 'go', label='Harley', markersize=6)
    ax_bot.plot(1/harley_wav_micron, 1.086*harley_Cabs/yV, 'go', markersize=6)

    # Plot Zubko BARE-GR-S fit
    zb_wav_micron = np.logspace(-1.5,1,100)
    zb_wav_angstrom = zb_wav_micron * 1e4
    zb_Cabs = getCrosssection_BARE_GR_S_DUST(zb_wav_angstrom)/1784268.76
    ax_mid.plot(1/zb_wav_micron, zb_Cabs, 'm--', label='Zubko et al. (2024) BARE-GR-S (RAMSES)', linewidth=2)
    ax_bot.plot(1/zb_wav_micron, 1.086*zb_Cabs/yV, 'm--', linewidth=2)

    # Plot the data from Zubko et al. (2004) BARE-GR-S model
    zubko_data = np.loadtxt(PATH_EXTERNAL_DATA / 'zubko_BAREGRS_extinction.csv', delimiter=',')
    ax_mid.plot(zubko_data[:,0], zubko_data[:,1]*1e-21, 'm-.', label='Zubko et al. (2024) BARE-GR-S (Paper)', linewidth=2)
    ax_bot.plot(zubko_data[:,0], 1.086*zubko_data[:,1]*1e-21/yV, 'm-.', linewidth=2)

    ax_mid.set_xlabel(r'$\lambda^{-1} [\mu {\rm m}^{-1}]$', fontsize=12)
    ylabel = r'$C_{\rm abs} [{\rm cm}^2]$'
    ax_mid.set_ylabel(ylabel, fontsize=12)
    ax_mid.tick_params(labelsize=10)
    ax_mid.grid(alpha=0.25, which='both')
    ax_mid.set_title('Absorption cross-section', fontsize=12)
    ax_mid.set_yscale('log')
    ax_mid.legend(fontsize=12, loc='best', ncol=1,frameon=False)
    ax_mid.set_xlim([0,16])
    ax_mid.set_ylim([3e-25,5e-21])

    # --- Bottom panel: extinction curve normalized at V ---
    x = 1.0 / wav
    order = np.argsort(x)
    ax_bot.plot(x[order], y_norm[order], color='k', lw=2, label='Total')

    # plot per-component normalized contributions (if present)
    for i, bin_id in enumerate(component_bins):
        comp_curve = comp_norm[i, :]
        if np.any(np.isfinite(comp_curve)):
            ax_bot.plot(x[order], comp_curve[order], label=bin_id, lw=2, color=colour[i])

    ax_bot.set_xlabel(r'$\lambda^{-1} [\mu {\rm m}^{-1}]$', fontsize=12)
    ylabel = r'$A_\lambda / A_V$'
    ax_bot.set_ylabel(ylabel, fontsize=12)
    ax_bot.tick_params(labelsize=10)
    ax_bot.grid(alpha=0.25)
    ax_bot.set_title('Extinction curve normalized at V (%.2f $\\mu$m)' % lambda_V, fontsize=12)
    ax_bot.legend(fontsize=12, loc='best', ncol=3,frameon=False)
    ax_bot.set_xlim([0,16])

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
