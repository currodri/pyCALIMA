"""
COULOMB ENHANCEMENT FACTOR

The functions below are used for the computation and plotting
of the Coulomb enhancement factor for ion-grain collisions.

"""
# LIBRARIES
import os
import numpy as np
import pandas as pd


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_EXTERNAL_DATA_DIR = os.path.join(_REPO_ROOT, 'external_data')


def _external_data_path(filename):
    return os.path.join(_EXTERNAL_DATA_DIR, filename)

# FUNCTIONS
def cmp_D_WD99(charge_dist,x,Zi,T,a):
    # This is based on Eq. 6-7 in Weingartner & Draine (1999) which allows
    # the computation of the Coulomb enhancement factor from the charge
    # distribution
    # (https://iopscience.iop.org/article/10.1086/307197)

    e = 4.8032047e-10 # statC
    kB = 1.380649e-16   # erg/K
    D = 0.0
    if Zi != 0:
        for i in range(0, len(charge_dist)):
            Zg = x[i]
            if Zg*Zi>0:
                B = np.exp(-Zg*Zi*e**2/(kB*T*a))
            elif Zg*Zi<0:
                B = 1.0 - Zg*Zi*e**2/(kB*T*a)
            elif Zg==0:
                B = 1.0 + np.sqrt(np.pi*Zi**2*e**2/(2.0*kB*T*a))
            D = D + charge_dist[i] * B
    else:
        D = 1.0
    D = max(D,1e-10)
    return D

def _compute_D_phi_for_size(args):
    """Worker helper: compute both D and average potential for a single size/env in one call.

    Returns tuple (D, phi) where either may be NaN on error.
    """
    try:
        Gtot, ne_val, T_val, material, a_micron, a_cm, Zi = args
        from models.dust_charge.dust_charging import equilibrium_charge_for_grain
        Zs, P, rates, Zmean_eq, Zsigma_eq = equilibrium_charge_for_grain(
            Gtot, ne_val, T_val, material, a_cm,
            radiation_model='Mathis', rad_field=None, yield_params=None, debug=False)
        if Zs is None or P is None or len(Zs) == 0 or len(P) == 0:
            return (float('nan'), float('nan'))
        # D
        D = cmp_D_WD99(P, Zs, Zi, T_val, a_cm)
        # mean Z (use returned Zmean_eq if available)
        if Zmean_eq is None:
            meanZ = float(np.sum(np.asarray(Zs) * np.asarray(P)))
        else:
            meanZ = float(Zmean_eq)
        # convert radius from cm to meters
        a_m = a_cm * 1e-2
        e_SI = 1.602176634e-19
        epsilon0_SI = 8.854187817e-12
        if a_m <= 0:
            phi = float('nan')
        else:
            phi = meanZ * e_SI / (4.0 * np.pi * epsilon0_SI * a_m)
        return (float(D), float(phi))
    except Exception:
        return (float('nan'), float('nan'))
    
def plot_coulomb_enhancement(Gtot,Zi,nsizes=10):
    import concurrent.futures
    import os
    from tqdm import tqdm
    from models.dust_charge.dust_charging import equilibrium_charge_for_grain
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        })
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7,5), dpi=300, facecolor='w', edgecolor='k')
    
    # Coulomb enhancement factor from Weingartner & Draine (1999)
    # This is given for graphitic and silicate grains in 
    # CNM: nH=30 Hcc, T=100K, xe=0.0015
    # WNM: nH=0.4 Hcc, T=6000K, xe=0.1
    # WIM: nH=0.1 Hcc, T=8000K, xe=0.99
    DCNM_gra = pd.read_csv(_external_data_path('weingartner_draine_1999_coulomb_enhancement_CNM_gra.csv'),header=1,names=['CNM,gra_x','CNM,gra_y'])
    DCNM_sil = pd.read_csv(_external_data_path('weingartner_draine_1999_coulomb_enhancement_CNM_sil.csv'),header=1,names=['CNM,sil_x','CNM,sil_y'])
    DWNM_gra = pd.read_csv(_external_data_path('weingartner_draine_1999_coulomb_enhancement_WNM_gra.csv'),header=1,names=['WNM,gra_x','WNM,gra_y'])
    DWNM_sil = pd.read_csv(_external_data_path('weingartner_draine_1999_coulomb_enhancement_WNM_sil.csv'),header=1,names=['WNM,sil_x','WNM,sil_y'])
    DWIM_gra = pd.read_csv(_external_data_path('weingartner_draine_1999_coulomb_enhancement_WIM_gra.csv'),header=1,names=['WIM,gra_x','WIM,gra_y'])
    DWIM_sil = pd.read_csv(_external_data_path('weingartner_draine_1999_coulomb_enhancement_WIM_sil.csv'),header=1,names=['WIM,sil_x','WIM,sil_y'])

    
    # Define colors for phases and line styles for materials/source
    phase_colors = {'CNM': 'cornflowerblue', 'WNM': 'goldenrod', 'WIM': 'firebrick'}
    linestyle_map = {
        'silicate_mine': '-',
        'graphite_mine': '--',
        'silicate_WD99': '-.',
        'graphite_WD99': ':'
    }
    
    # Plot WD99 data
    WD99_data = [
        (DCNM_gra, 'CNM', 'graphite'),
        (DCNM_sil, 'CNM', 'silicate'),
        (DWNM_gra, 'WNM', 'graphite'),
        (DWNM_sil, 'WNM', 'silicate'),
        (DWIM_gra, 'WIM', 'graphite'),
        (DWIM_sil, 'WIM', 'silicate')
    ]
    
    for data_df, phase, material in WD99_data:
        x = np.asarray(data_df.iloc[:, 0])
        y = np.asarray(data_df.iloc[:, 1])
        linestyle = linestyle_map[f'{material}_WD99']
        color = phase_colors[phase]
        ax.plot(x*1e4, y, linestyle=linestyle, color=color, lw=2, alpha=0.6)

    asizes_micron = np.logspace(-3,1,nsizes) # in micron
    asizes_cm = asizes_micron * 1e-4
    materials = ['graphite','silicate']
    nmaterials = len(materials)

    # Separate figure for average electrostatic potential (Volts)
    fig_phi, ax_phi = plt.subplots(1, 1, sharex=True, figsize=(7,5), dpi=300, facecolor='w', edgecolor='k')
    ax_phi.set_xscale('log')
    ax_phi.set_xlabel(r'$a$ [$\mu$m]', fontsize=16)
    ax_phi.set_ylabel(r'Average potential $\langle U \rangle$ (V)', fontsize=16)
    ax_phi.tick_params(labelsize=14)
    ax_phi.xaxis.set_ticks_position('both')
    ax_phi.yaxis.set_ticks_position('both')
    ax_phi.minorticks_on()
    ax_phi.tick_params(which='both',axis="both",direction="in")
    ax_phi.set_ylim([-2,2.5])
    ax_phi.set_xlim([7e-4,0.25])

    # plot the results from Draine_potential_graphite.csv and Draine_potential_silicate.csv\
    phases_Draine = ['CNM','WNM']
    for i, phase in enumerate(phases_Draine):
        for mat_idx, mat in enumerate(materials):
            if mat == 'graphite':
                data = np.loadtxt(_external_data_path(f'Draine_potential_graphite_{phase}.csv'), delimiter=',', skiprows=1)
                linestyle = linestyle_map['graphite_WD99']
            elif mat == 'silicate':
                data = np.loadtxt(_external_data_path(f'Draine_potential_silicate_{phase}.csv'), delimiter=',', skiprows=1)
                linestyle = linestyle_map['silicate_WD99']
            sizes_draine = data[:, 0]*1e-4  # in cm
            potentials_draine = data[:, 1]  # in eV
            ax_phi.plot(sizes_draine, potentials_draine, 
                        linestyle=linestyle, 
                        color=phase_colors[phase], lw=2, alpha=0.6)

    max_workers = min(nsizes, os.cpu_count() or 1)
    print(f'Using max_workers={max_workers} for parallel computation')

    # helper to run a mapping (with optional tqdm)
    def _map_with_tqdm(executor, func, tasks, desc=None):
        try:
            from tqdm import tqdm as _tqdm
        except Exception:
            _tqdm = None
        it = executor.map(func, tasks)
        if _tqdm is not None:
            return list(_tqdm(it, total=len(tasks), desc=desc))
        else:
            return list(it)

    # Generic environment loop (env_name, T_val, ne_val, linestyle)
    environments = [
        ('CNM', 100, 0.03, '-'),
        ('WNM', 6000, 0.03, '--'),
        ('WIM', 8000, 0.099, ':'),
    ]


    for env_name, T_env, ne_env, env_ls in environments:
        for i in range(0, nmaterials):
            # build tasks for this material and environment
            tasks = []
            for j in range(nsizes):
                Gtot_local = Gtot
                T_val = T_env
                ne_val = ne_env
                a_micron = asizes_micron[j]
                a_cm = asizes_cm[j]
                tasks.append((Gtot_local, ne_val, T_val, materials[i], a_micron, a_cm, Zi))

            # attempt parallel computation for D and phi in a single solver call; if it fails, fall back to sequential
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = _map_with_tqdm(executor, _compute_D_phi_for_size, tasks, desc=f'{env_name} {materials[i]} D+phi')
                # results is a list of (D, phi) tuples
                D_list, phi_list = zip(*results) if len(results) else ([], [])
                D_arr = np.asarray(D_list, dtype=float)
                phi_arr = np.asarray(phi_list, dtype=float)
            except Exception:
                # sequential fallback for both D and phi
                D_list = []
                phi_list = []
                for t in tasks:
                    try:
                        dval, phival = _compute_D_phi_for_size(t)
                        D_list.append(dval)
                        phi_list.append(phival)
                    except Exception:
                        D_list.append(float('nan'))
                        phi_list.append(float('nan'))
                D_arr = np.asarray(D_list, dtype=float)
                phi_arr = np.asarray(phi_list, dtype=float)

            # plot D on left axis
            color = phase_colors[env_name]
            linestyle = linestyle_map[f'{materials[i]}_mine']
            ax.plot(asizes_micron, D_arr, color=color, linestyle=linestyle, lw=2)
            
            # plot phi on separate figure
            ax_phi.plot(asizes_micron, phi_arr, color=color, linestyle=linestyle, lw=2)
    
    # Legend for phases (colors)
    dummy_lines = [
        ax.plot([],[], color=phase_colors['CNM'], linestyle='-', label='CNM', lw=2)[0],
        ax.plot([],[], color=phase_colors['WNM'], linestyle='-', label='WNM', lw=2)[0],
        ax.plot([],[], color=phase_colors['WIM'], linestyle='-', label='WIM', lw=2)[0]
    ]
    first_legend = ax.legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=16)
    ax.add_artist(first_legend)
    
    # Legend for materials and source (linestyles)
    dummy_lines = [
        ax.plot([],[], color='k', linestyle='-', label='Silicate (mine)', lw=2)[0],
        ax.plot([],[], color='k', linestyle='--', label='Graphite (mine)', lw=2)[0],
        ax.plot([],[], color='k', linestyle='-.', label='Silicate (WD99)', lw=2)[0],
        ax.plot([],[], color='k', linestyle=':', label='Graphite (WD99)', lw=2)[0]
    ]
    second_legends = ax.legend(handles=dummy_lines, loc='upper right', frameon=False, fontsize=16)
    ax.add_artist(second_legends)


    ax.set_ylabel(r'Coulomb enhancement $F_{\rm C}(a)$', fontsize=18)
    ax.set_xlabel(r'$a$ [$\mu$m]',fontsize=18)
    ax.set_ylim([1e-3,300])
    ax.set_xlim([7e-4,0.25])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=16)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # ax.axvline(x=0.1, color='cornflowerblue', linestyle='-',alpha=0.6)
    # ax.axvline(x=0.01, color='steelblue', linestyle='--',alpha=0.6)
    # ax.axvline(x=0.1, color='sandybrown', linestyle='-',alpha=0.6)
    # ax.axvline(x=0.005, color='saddlebrown', linestyle='--',alpha=0.6)
    
    fig.subplots_adjust(top=0.99,bottom=0.11,left=0.11,right=0.99)
    fig.savefig('dust_coulomb_enhancement.pdf',format='pdf',dpi=300)
    plt.close(fig)


    # Legend for phases (colors)
    dummy_lines = [
        ax_phi.plot([],[], color=phase_colors['CNM'], linestyle='-', label='CNM')[0],
        ax_phi.plot([],[], color=phase_colors['WNM'], linestyle='-', label='WNM')[0],
        ax_phi.plot([],[], color=phase_colors['WIM'], linestyle='-', label='WIM')[0]
    ]
    first_legend = ax_phi.legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=14)
    ax_phi.add_artist(first_legend)
    
    # Legend for materials and source (linestyles)
    dummy_lines = [
        ax_phi.plot([],[], color='k', linestyle='-', label='Silicate (mine)')[0],
        ax_phi.plot([],[], color='k', linestyle='--', label='Graphite (mine)')[0],
        ax_phi.plot([],[], color='k', linestyle='-.', label='Silicate (WD99)')[0],
        ax_phi.plot([],[], color='k', linestyle=':', label='Graphite (WD99)')[0]
    ]
    second_legends = ax_phi.legend(handles=dummy_lines, loc='upper right', frameon=False, fontsize=12)
    ax_phi.add_artist(second_legends)

    # finalize and save the average potential figure
    fig_phi.subplots_adjust(top=0.99,bottom=0.11,left=0.11,right=0.99)
    fig_phi.savefig('dust_avg_potential.pdf', format='pdf', dpi=300)
    plt.close(fig_phi)