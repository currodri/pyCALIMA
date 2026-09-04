"""
COULOMB ENHANCEMENT FACTOR

The functions below are used for the computation and plotting
of the Coulomb enhancement factor for ion-grain collisions.

"""
# LIBRARIES
from pycalima.models import grain_size_config
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from pycalima import _paths


_EXTERNAL_DATA_DIR = str(_paths.get_external_data_path())

e = 4.8032047e-10 # statC
kB = 1.380649e-16   # erg/K
inv_sqrt2 = 0.7071067811865475
inv_sqrt2pi = 0.3989422804014327

def _external_data_path(filename):
    return os.path.join(_EXTERNAL_DATA_DIR, filename)

# FUNCTIONS
def cmp_D_WD99(charge_dist,x,Zi,T,a):
    # This is based on Eq. 6-7 in Weingartner & Draine (1999) which allows
    # the computation of the Coulomb enhancement factor from the charge
    # distribution
    # (https://iopscience.iop.org/article/10.1086/307197)
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

def compute_D_analytical(mu, sigma, Zi, T, a):
    """
    Computes the Coulomb enhancement factor D analytically 
    assuming a Gaussian charge distribution.
    
    Parameters:
    mu    : Mean grain charge <Zg>
    sigma : Standard deviation of grain charge Zg
    Zi    : Charge of the incident particle (e.g., -1 for electrons)
    T     : Temperature in Kelvin
    a     : Grain radius in cm
    """
    
    # Calculate the dimensionless plasma parameter alpha
    # Based on the term alpha = Zi * e^2 / (a * kB * T)
    alpha = (Zi * e**2) / (a * kB * T)
    
    # PDF and CDF of the Gaussian
    # Using scipy.stats.norm for numerical stability
    phi = norm.cdf
    pdf = norm.pdf
    
    # Attractive term (Contribution for Zg such that Zg*Zi < 0)
    # Integral of (1 - alpha * Zg) * f(Zg)
    # For Zi > 0, this is the region Zg < 0. For Zi < 0, it is Zg > 0.
    if Zi > 0:
        # Attractive: Zg < 0
        term_attr = (1 - alpha * mu) * phi(-mu / sigma) + alpha * sigma * pdf(-mu / sigma)
        # Repulsive: Zg > 0
        term_rep = np.exp(-alpha * mu + 0.5 * (alpha**2) * (sigma**2)) * (1 - phi((alpha * sigma**2 - mu) / sigma))
    else:
        # If Zi < 0 (electrons), the logic for attractiveness/repulsion flips
        # Repulsive: Zg < 0
        term_rep = np.exp(-alpha * mu + 0.5 * (alpha**2) * (sigma**2)) * phi((-mu + alpha * sigma**2) / sigma)
        # Attractive: Zg > 0
        term_attr = (1 - alpha * mu) * phi(mu / sigma) + alpha * sigma * pdf(mu / sigma)

    D = term_attr + term_rep
    return max(D, 1e-10)

def compute_D_moment_expansion(mu, sigma, Zi, T, a):
    # alpha = Zi * e^2 / (a * kB * T)
    alpha = (Zi * e**2) / (a * kB * T)
    
    # D(mu) evaluation
    if mu * Zi > 0:
        D_mu = np.exp(-mu * alpha)
        # Second derivative of exp(-alpha * Z) is alpha^2 * exp(-alpha * Z)
        D_double_prime = (alpha**2) * np.exp(-mu * alpha)
    elif mu * Zi < 0:
        D_mu = 1.0 - mu * alpha
        # Second derivative of (1 - alpha * Z) is 0
        D_double_prime = 0.0
    else:
        # Handling the transition point Z=0 (neutral)
        # For simplicity, we use the value at 0
        D_mu = 1.0 + np.sqrt(np.pi * Zi**2 * e**2 / (2.0 * kB * T * a))
        D_double_prime = 0.0 # Curvature is negligible at the cusp
        
    # Apply expansion
    D_approx = D_mu + 0.5 * D_double_prime * (sigma**2)
    
    return max(D_approx, 1e-10)

def compute_D_split_expansion(mu, sigma, Zi, T, a):
    e, kB = 4.803e-10, 1.381e-16
    alpha = (Zi * e**2) / (a * kB * T)
    
    # Probabilities of regimes (using the Gaussian assumption for P(Z))
    p_rep = 1.0 - norm.cdf(0, mu, sigma) if Zi > 0 else norm.cdf(0, mu, sigma)
    p_attr = 1.0 - p_rep
    
    # 1. Linear expansion for attraction (Exact for linear functions)
    # E[1 - alpha*Z] = 1 - alpha*mu
    D_attr = (1.0 - alpha * mu) * p_attr
    
    # 2. Second-order expansion for repulsion
    # E[exp(-alpha*Z)] approx exp(-alpha*mu) * (1 + 0.5 * alpha^2 * sigma^2)
    D_rep = np.exp(-alpha * mu) * (1.0 + 0.5 * (alpha**2) * (sigma**2)) * p_rep
    
    return max(D_attr + D_rep, 1e-10)

def compute_D_skewed_expansion(charge_dist, x, mu, sigma, Zi, T, a):
    e, kB = 4.803e-10, 1.381e-16
    alpha = (Zi * e**2) / (a * kB * T)
    
    # Calculate Skewness (gamma) from your actual discrete distribution
    gamma = np.sum(((x - mu) / sigma)**3 * charge_dist)
    
    # Base Gaussian-based exponential approximation
    D_base = np.exp(-alpha * mu + 0.5 * (alpha**2) * (sigma**2))
    
    # Skewness correction factor
    skew_correction = (1.0 + (alpha**3) * (sigma**3) * gamma / 6.0)
    
    return D_base * skew_correction

def compute_D_RAMSES(Z_avg, Zsigma, Zi, T, a):
    """
    Computes the Coulomb enhancement factor D by generating the discrete
    Gaussian charge distribution and summing the contribution of each state.
    """
    
    # 2. Generate the discretized distribution (Matching your Fortran logic)
    Zmin = int(round(Z_avg - 3 * Zsigma))
    Zmax = int(round(Z_avg + 3 * Zsigma))
    n_charge = Zmax - Zmin + 1
        
    Zdust = np.arange(Zmin, Zmax + 1, dtype=float)
    factor_pdf = -0.5 * (1.0 / Zsigma)**2
    fcharge = np.exp(factor_pdf * (Zdust - Z_avg)**2)
    fcharge /= np.sum(fcharge) # Normalization
    
    # 3. Compute the Coulomb enhancement factor D for this distribution
    # This matches the WD99 loop logic
    D = 0.0
    if Zi != 0:
        # Pre-compute alpha constant
        alpha = (Zi * e**2) / (a * kB * T)
        
        for i in range(len(Zdust)):
            Zg = Zdust[i]
            # WD99 logic for Coulomb factor
            if Zg * Zi > 0:
                B = np.exp(-Zg * alpha)
            elif Zg * Zi < 0:
                B = 1.0 - Zg * alpha
            else: # Zg == 0
                # Using the WD99 neutral grain limit
                B = 1.0 + np.sqrt(np.pi * Zi**2 * e**2 / (2.0 * kB * T * a))
            
            D += fcharge[i] * B
    else:
        D = 1.0
        
    return max(D, 1e-10)

def compute_D_hybrid(mu, sigma, Zi, T, a, n_charge_threshold=20,debug=False):
    """
    Hybrid solver: Uses Moment Expansion for high-charge regimes (stable)
    and Discrete Discretization for low-charge/jagged regimes (accurate).
    """
    alpha = (Zi * e**2) / (a * kB * T)
    print('alpha: ',alpha)
    # 1. Determine charge range to estimate n_charge
    Zmin = int(round(mu - 3 * sigma))
    Zmax = int(round(mu + 3 * sigma))
    n_charge = Zmax - Zmin + 1
    
    # 2. Hybrid Decision
    # Threshold condition: 
    # Use expansion if charges are high (mu > 3*sigma) AND grid is large enough
    if (abs(mu / sigma) > 3.0) and (n_charge > n_charge_threshold):
        # Moment Expansion (Second-order)
        # If mu * Zi < 0: Grain is Attractive
        if mu * Zi > 0: 
            # Repulsive Regime: Second-order expansion of exp(-alpha * Z)
            # Expands around mu: exp(-alpha*mu) * (1 + 0.5 * alpha^2 * sigma^2)
            D_approx = np.exp(-alpha * mu) * (1.0 + 0.5 * (alpha**2) * (sigma**2))
        else:
            # Attractive Regime: Linear expansion of (1 - alpha * Z)
            # Expansion around mu is exact (second derivative is 0)
            D_approx = 1.0 - alpha * mu
        if debug: print(f'Using moment expansion: mu/sigma = {mu/sigma:.2f}, n_charge = {n_charge}')
        return max(D_approx, 1e-10)
    else:
        if debug: print(f'Using discrete discretization: mu/sigma = {mu/sigma:.2f}, n_charge = {n_charge}')
        # Fallback to Discrete Discretization (The "Safe" Path)
        return compute_total_coulomb_factor(mu, sigma, Zi, T, a)

def compute_total_coulomb_factor(Z_avg, Zsigma, Zi, T, a, Zdust_max_size=200):
    # This is the discrete implementation developed in the previous step
    Zmin = int(round(Z_avg - 3 * Zsigma))
    Zmax = int(round(Z_avg + 3 * Zsigma))
    n_charge = Zmax - Zmin + 1
    if n_charge > Zdust_max_size:
        n_charge = Zdust_max_size
        Zmin = int(round(Z_avg - n_charge / 2.0))
    Zdust = np.arange(Zmin, Zmin + n_charge)
    fcharge = np.exp(-0.5 * ((Zdust - Z_avg) / Zsigma)**2)
    fcharge /= np.sum(fcharge)
    
    alpha = (Zi * e**2) / (a * kB * T)
    D = 0.0
    for i in range(len(Zdust)):
        Zg = Zdust[i]
        if Zg * Zi > 0: B = np.exp(-Zg * alpha)
        elif Zg * Zi < 0: B = 1.0 - Zg * alpha
        else: B = 1.0 + np.sqrt(np.pi * Zi**2 * e**2 / (2.0 * kB * T * a))
        D += fcharge[i] * B
    return max(D, 1e-10)

def compute_D_lognormal_hybrid(mu, sigma, Zi, T, a, Z_min=4, Z_max=84):
    """
    Computes the Coulomb enhancement factor D using the Log-Normal 
    property for repulsion and Linear expectation for attraction.
    """
    alpha = (Zi * e**2) / (a * kB * T)
    
    # 1. Normalizing factor: The probability mass within the physical grid
    # This accounts for the truncation of the distribution at [Z_min, Z_max]
    norm_mass = norm.cdf(Z_max, mu, sigma) - norm.cdf(Z_min, mu, sigma)
    
    # 2. Attractive part (Linear expectation: E[1 - alpha*Z])
    # Integral of (1 - alpha*Z) * f(Z) = 1 - alpha * E[Z]
    # For Zi > 0, attractive is Z < 0. For Zi < 0, attractive is Z > 0.
    # Note: Using the truncated mean property here would be ideal, 
    # but the simple mean is often sufficient if the grid is well-centered.
    term_attr = (1.0 - alpha * mu)
    
    # 3. Repulsive part (Log-Normal expectation: E[exp(-alpha*Z)])
    # E[exp(-alpha*Z)] = exp(-alpha*mu + 0.5 * alpha^2 * sigma^2)
    term_rep = np.exp(-alpha * mu + 0.5 * (alpha**2) * (sigma**2))
    
    # 4. Weighted combination based on probability of being in repulsive/attractive regime
    # P(repulsive) vs P(attractive)
    if Zi > 0:
        # Repulsive if Z > 0, Attractive if Z < 0
        p_rep = 1.0 - norm.cdf(0, mu, sigma)
        p_attr = norm.cdf(0, mu, sigma)
    else:
        p_rep = norm.cdf(0, mu, sigma)
        p_attr = 1.0 - norm.cdf(0, mu, sigma)
        
    D = (term_attr * p_attr + term_rep * p_rep) / norm_mass
    
    return max(D, 1e-10)

def compute_D_saddle_point(mu, sigma, Zi, T, a):
    e, kB = 4.803e-10, 1.381e-16
    alpha = (Zi * e**2) / (a * kB * T)
    
    # The integrand in log-space: ln(D(Z)) + ln(P(Z))
    # P(Z) is Gaussian: ln(P(Z)) = -0.5 * ((Z-mu)/sigma)**2 + const
    # D(Z) = exp(-alpha * Z) (assuming repulsive for Z > 0)
    def log_integrand(Z):
        # We focus on the repulsive part where D(Z) = exp(-alpha * Z)
        return -alpha * Z - 0.5 * ((Z - mu) / sigma)**2
    
    # Find the peak (Z_star) of the combined function
    # The derivative of the log-integrand is -alpha - (Z - mu)/sigma^2
    # Setting to 0: Z_star = mu - alpha * sigma^2
    Z_star = mu - alpha * (sigma**2)
    
    # Calculate D at Z_star and adjust by the local variance
    # This is essentially the Laplace approximation
    D_star = np.exp(-alpha * Z_star) * np.exp(-0.5 * ((Z_star - mu)/sigma)**2)
    
    # Scale by the width of the Gaussian (sqrt(2*pi)*sigma)
    D_final = D_star * np.sqrt(2 * np.pi) * sigma
    
    return max(D_final, 1e-10)

def _compute_D_phi_for_size(args):
    """Worker helper: compute both D and average potential for a single size/env in one call.

    Returns tuple (D, phi) where either may be NaN on error.
    """
    try:
        Gtot, ne_val, T_val, material, a_micron, a_cm, Zi = args
        from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
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

def _compute_error_analytic_for_size(args):
    """Worker helper: compute the error of the analytic approximation for a single size/env in one call.

    Returns tuple (error_D) where error_D may be NaN on error.
    """
    import matplotlib.pyplot as plt
    try:
        Gtot, ne_val, T_val, material, a_micron, a_cm, Zi = args
        from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
        Zs, P, rates, Zmean_eq, Zsigma_eq = equilibrium_charge_for_grain(
            Gtot, ne_val, T_val, material, a_cm,
            radiation_model='Mathis', rad_field=None, yield_params=None, debug=False)
        if Zs is None or P is None or len(Zs) == 0 or len(P) == 0:
            return (float('nan'))
        D_discrete = cmp_D_WD99(P, Zs, Zi, T_val, a_cm)
        D_analytic = compute_D_hybrid(Zmean_eq,Zsigma_eq,Zi,T_val,a_cm)
        error_D = (D_discrete - D_analytic) / D_discrete
        return (float(error_D))
    except Exception:
        return (float('nan'))

def plot_coulomb_enhancement(Gtot,Zi,nsizes=10):
    import concurrent.futures
    import os
    from tqdm import tqdm
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
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

def plot_analytic_coulomb(Gtot,Zi,nsizes=10):
    import concurrent.futures
    import os
    from tqdm import tqdm
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
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
    
    # Define colors for phases and line styles for materials/source
    phase_colors = {'CNM': 'cornflowerblue', 'WNM': 'goldenrod', 'WIM': 'firebrick'}
    linestyle_map = {
        'silicate_mine': '-',
        'graphite_mine': '--',
    }

    asizes_micron = np.logspace(-3,1,nsizes) # in micron
    asizes_cm = asizes_micron * 1e-4
    materials = ['graphite','silicate']
    nmaterials = len(materials)

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
                    results = _map_with_tqdm(executor, _compute_error_analytic_for_size, tasks, desc=f'{env_name} {materials[i]} error(D)')
                # results is a list of (D, phi) tuples
                D_list = [r for r in results]
                D_arr = np.asarray(D_list, dtype=float)
            except Exception:
                # sequential fallback for both D and phi
                D_list = []
                for t in tasks:
                    try:
                        dval = _compute_error_analytic_for_size(t)
                        D_list.append(dval)
                    except Exception:
                        D_list.append(float('nan'))
                D_arr = np.asarray(D_list, dtype=float)

            # plot D on left axis
            color = phase_colors[env_name]
            linestyle = linestyle_map[f'{materials[i]}_mine']
            ax.plot(asizes_micron, D_arr, color=color, linestyle=linestyle, lw=2)
    
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
        ax.plot([],[], color='k', linestyle='--', label='Graphite (mine)', lw=2)[0]
    ]
    second_legends = ax.legend(handles=dummy_lines, loc='upper right', frameon=False, fontsize=16)
    ax.add_artist(second_legends)


    ax.set_ylabel(r'Error in Coulomb enhancement', fontsize=18)
    ax.set_xlabel(r'$a$ [$\mu$m]',fontsize=18)
    ax.set_ylim([-2,2])
    ax.set_xlim([7e-4,10])
    ax.set_xscale('log')
    ax.tick_params(labelsize=16)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    fig.tight_layout()
    fig.savefig('coulomb_analytic_comparison.pdf',format='pdf',dpi=300)
    plt.close(fig)

def plot_single_size_dist(Gtot,Tgas,ne,Zi,asize_micron,material='graphite'):
    import concurrent.futures
    import os
    from tqdm import tqdm
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
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

    # set environment parameters
    ne_val = ne
    T_val = Tgas
    a_cm = asize_micron * 1e-4

    Zs, P, rates, Zmean_eq, Zsigma_eq = equilibrium_charge_for_grain(
            Gtot, ne_val, T_val, material, a_cm,
            radiation_model='Mathis', rad_field=None, yield_params=None, debug=False)

    ax.step(Zs,P,color='k',lw=2,linestyle='-')
    Zs_gaussian = np.linspace(Zmean_eq - 3*Zsigma_eq, Zmean_eq + 3*Zsigma_eq, 100)
    P_gaussian = 1/(Zsigma_eq*np.sqrt(2*np.pi)) * np.exp(-(Zs_gaussian - Zmean_eq)**2/(2*Zsigma_eq**2))
    ax.plot(Zs_gaussian,P_gaussian,color='r',lw=2,linestyle='--')

    # Compute the Coulomb factors and the diagnostics
    D_discrete = cmp_D_WD99(P, Zs, Zi, T_val, a_cm)
    D_analytic = compute_D_analytical(Zmean_eq, Zsigma_eq, Zi, T_val, a_cm)
    D_exp = compute_D_moment_expansion(Zmean_eq, Zsigma_eq, Zi, T_val, a_cm)
    D_lognorm = compute_D_lognormal_hybrid(Zmean_eq, Zsigma_eq, Zi, T_val, a_cm)
    D_saddle = compute_D_saddle_point(Zmean_eq, Zsigma_eq, Zi, T_val, a_cm)
    D_split = compute_D_split_expansion(Zmean_eq, Zsigma_eq, Zi, T_val, a_cm)
    D_skewed = compute_D_skewed_expansion(P, Zs, Zmean_eq, Zsigma_eq, Zi, T_val, a_cm)
    D_ramses = compute_D_RAMSES(Zmean_eq, Zsigma_eq, Zi, T_val, a_cm)
    D_hybrid = compute_D_hybrid(Zmean_eq, Zsigma_eq, Zi, T_val, a_cm,debug=True)
    print(f'{"-" * 65}')
    print(f'TRUE DISTRIBUTION INFORMATION: a = {a_cm * 1e4:.4f} um | T = {T_val} K')
    print(f'{"-" * 65}')
    print(f'Mean (mu)     : {Zmean_eq:+8.3f} | Width (sigma): {Zsigma_eq:8.3f}')
    print(f'Physical Grid : Z in [{Zs[0]}, {Zs[-1]}]')
    print(f'D_discrete (true distribution)  : {D_discrete:8.3e}')
    print(f'D_analytic (Gaussian) : {D_analytic:8.3e}')
    print(f'Error(Gaussian): {(D_discrete-D_analytic)/D_discrete}')
    print(f'D_exp (moment expansion) : {D_exp:8.3e}')
    print(f'Error(Moment Ex): {(D_discrete-D_exp)/D_discrete}')
    print(f'D_lognorm (hybrid) : {D_lognorm:8.3e}')
    print(f'Error(Lognorm): {(D_discrete-D_lognorm)/D_discrete}')
    print(f'D_saddle (saddle point) : {D_saddle:8.3e}')
    print(f'Error(Saddle): {(D_discrete-D_saddle)/D_discrete}')
    print(f'D_split (split expansion) : {D_split:8.3e}')
    print(f'Error(Split): {(D_discrete-D_split)/D_discrete}')
    print(f'D_skewed (skewed expansion) : {D_skewed:8.3e}')
    print(f'Error(Skewed): {(D_discrete-D_skewed)/D_discrete}')
    print(f'D_ramses (RAMSES) : {D_ramses:8.3e}')
    print(f'Error(RAMSES): {(D_discrete-D_ramses)/D_discrete}')
    print(f'D_hybrid (hybrid) : {D_hybrid:8.3e}')
    print(f'Error(Hybrid): {(D_discrete-D_hybrid)/D_discrete}')
    print(f'{"-" * 65}\n')
    
    fig.tight_layout()
    fig.savefig('test_distribution.pdf',format='pdf',dpi=300)
    plt.close(fig)