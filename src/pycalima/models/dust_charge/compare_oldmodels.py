"""
COMPARE GRAIN CHARGE DISTRIBUTIONS

The tools here provided allow a sanity check comparison of the
equlibrium grain charge distribution from the modelling in CALIMA
vs. the fitting formulae from Ibanez-Mejias et al. 2019 (IM19)
and the original Weingartner & Draine 2001 (WD01) results.
"""
# LIBRARIES
import numpy as np


# FUNCTIONS
def charge_equilibrium_from_rates(k_up, k_down, Zmin, Zmax):
    """
    Compute steady-state charge distribution f(Z) for integer charges Zmin..Zmax
    given upward (Z->Z+1) and downward (Z->Z-1) transition rates.

    Parameters
    ----------
    k_up : callable
        Function taking integer Z and returning rate [s^-1] for transition Z->Z+1
    k_down : callable
        Function taking integer Z and returning rate [s^-1] for transition Z->Z-1
    Zmin, Zmax : int
        Inclusive bounds for integer charges to consider.

    Returns
    -------
    f : ndarray
        Normalized probability array for charges Zmin..Zmax
    Z : ndarray
        Array of integer charges from Zmin to Zmax

    Notes
    -----
    Solves detailed balance in steady state:
        f(Z) * k_up(Z) = f(Z+1) * k_down(Z+1)
    which can be recursively solved up to an overall normalization.
    """
    Z = np.arange(Zmin, Zmax+1, dtype=int)
    n = len(Z)
    f = np.zeros(n, dtype=float)

    # Choose a reference Z0 (we'll pick Zmin) and set f(Zmin)=1 before normalizing
    f[0] = 1.0
    # Upwards recursion: build f(Z+1) from f(Z)
    for i in range(0, n-1):
        Zi = Z[i]
        kip = k_up(Zi)
        kdown_next = k_down(Zi+1)
        # avoid division by zero; if both zero, set next prob to zero
        if kdown_next > 0.0:
            f[i+1] = f[i] * (kip / kdown_next)
        else:
            f[i+1] = 0.0

    # If some lower charges might be populated by downward transitions from reference,
    # we could also recurse downward. Here reference is Zmin so nothing below.

    # Normalize
    s = np.sum(f)
    if s <= 0.0:
        raise RuntimeError('All probabilities zero in charge_equilibrium_from_rates')
    f /= s
    return f, Z


def grain_charge_equilibrium_WD01(grain_type, a_cm, radiation_field, C_abs, Im, ne, nH, T, Zmin=None, Zmax=None,
                                 ion_charge=1, method='WD01_simple'):
    """
    Compute equilibrium grain charge distribution following Weingartner & Draine (2001)
    using photoemission (photoelectric) and collisional charging (electron/ion capture).

    This is a practical implementation that requires the caller to provide the
    radiation_field and absorption cross section arrays. It computes:
      k_pe(Z): photoemission rate [s^-1] (Z -> Z+1)
      k_e(Z): electron capture rate [s^-1] (Z -> Z-1)
      k_ion(Z): ion capture rate [s^-1] (Z -> Z+1)

    The net upward rate used is k_up(Z) = k_pe(Z) + k_ion(Z) and the downward
    rate is k_down(Z) = k_e(Z).

    Parameters
    ----------
    grain_type : {'silicate','graphite'}
        Grain composition
    a_cm : float
        Grain radius in cm
    radiation_field : ndarray (N,3)
        Array with columns [E_eV, I_lambda?, flux_photon?]. The function expects
        column 0 = photon energy in eV and column 2 = energy density or flux-like
        quantity where integrand = flux/E * ... (compatible with routines in
        `dust_photoelectric_heating.py`).
    C_abs : ndarray (N,)
        Absorption cross section for each photon energy (cm^2)
    ne : float
        Electron density [cm^-3]
    nH : float
        Hydrogen density [cm^-3]
    T : float
        Gas temperature [K]
    Zmin, Zmax : int, optional
        Charge bounds to consider. If None, automatically set to +/- 10 around the
        mean estimated charge from the Ibanez-Mejias fit.
    ion_charge : int
        Ion charge (usually +1)

    Returns
    -------
    f, Z : ndarray
        Normalized charge distribution and corresponding integer charges

    Notes
    -----
    This implementation uses simplified formulae for collisional capture rates
    following DS87/WD01 (see `DS87_J_function` and `DS87_lambda_function`) and
    computes photoemission rate by integrating the photoelectric yield times photon
    flux divided by photon energy times cross section. The photoelectric yield model
    is not reimplemented here; instead this function expects `C_abs` and
    `radiation_field` consistent with `dust_photoelectric_heating.compute_photoelectric_heating_rate`.
    """
    # Lazy import to avoid circular import at module load
    from pycalima.models.dust_charge.dust_photoelectric_heating import (
        escape_fraction_attempting_electrons,
        photoelectric_yield_graphite,
        photoelectric_yield_silicate,
        min_energy_ejection,
        ionisation_potential_valence,
        min_photon_energy,
        graphite_work_function,
        silicate_work_function,
        DS87_J_function,
    )
    # use cgs Boltzmann constant to avoid unyt dependency
    kB = 1.380649e-16  # erg/K

    # Determine a representative grain radius in nm used by some helper functions
    a_nm = a_cm * 1e7  # cm -> nm

    # Estimate mean charge if Zmin/Zmax not provided
    if Zmin is None or Zmax is None:
        try:
            from pycalima.models.dust_charge.IM19_charging import grain_mean_charge
            Zmean = int(round(grain_mean_charge(1.0, T, ne, 'silicate' if grain_type=='silicate' else 'graphite', f'{int(a_nm)}A')))
        except Exception:
            Zmean = 0
        Zmin = -max(10, abs(Zmean) + 10) if Zmin is None else Zmin
        Zmax = max(10, abs(Zmean) + 10) if Zmax is None else Zmax

    # Photon energies and flux-like column index expectation
    E = radiation_field[:,0]  # eV
    flux_term = radiation_field[:,2]

    # Helper: photoemission rate k_pe(Z)
    def k_pe(Z, yield_func=None):
        # compute IPV and minimum photon energy for ejection
        if grain_type == 'graphite':
            IPV = ionisation_potential_valence(graphite_work_function, Z, a_nm)
        else:
            IPV = ionisation_potential_valence(silicate_work_function, Z, a_nm)

        Emin_ej = min_photon_energy(IPV, Z, a_nm)
        # Integrate yield * (flux / E) * C_abs over E > Emin_ej
        mask = E >= Emin_ej
        if not np.any(mask):
            return 0.0

        # If a full radiation & Im & C_abs are available, call compute_photoemission_rate
        if radiation_field is not None and C_abs is not None:
            try:
                from pycalima.models.dust_charge.dust_photoelectric_heating import compute_photoemission_rate
                args = (Z, a_nm, radiation_field, grain_type, Im, C_abs)
                return compute_photoemission_rate(args)
            except Exception:
                pass

        # fallback crude default: small constant yield for photons above threshold
        yields_masked = np.full(np.count_nonzero(mask), 0.1)
        integrand = yields_masked * (flux_term[mask] / E[mask]) * C_abs[mask]
        # result in s^-1 (photons * yield * cross-section integrated)
        return float(np.trapezoid(integrand, E[mask]))

    # Helper: collisional capture rates
    e_c = 4.8032047e-10  # statC
    m_e = 9.1093837015e-28  # g

    def k_e(Z):
        # electron thermal velocity * cross section * coulomb focusing
        v_th = np.sqrt(8. * kB * T / (np.pi * m_e))
        sigma_geom = np.pi * a_cm**2
        # Coulomb factor J from DS87
        # a_cm is provided in cm here; DS87_J_function expects a in meters -> convert
        J = DS87_J_function(Z, -1., a_cm / 100.0, T)
        # electron density ne assumed provided
        return ne * v_th * sigma_geom * J

    def k_ion(Z):
        # ions (protons) capture rate; use hydrogen as dominant ion with charge +1
        m_p = 1.67262192369e-24
        v_th_ion = np.sqrt(8. * kB * T / (np.pi * m_p))
        sigma_geom = np.pi * a_cm**2
        # Coulomb factor for positive charge interacting with ion of charge +1
        # convert a_cm (cm) to meters for DS87_J_function
        J = DS87_J_function(Z, ion_charge, a_cm / 100.0, T)
        return nH * v_th_ion * sigma_geom * J

    # Define up/down rates used by solver
    def k_up(Z):
        return k_pe(Z) + k_ion(Z)

    def k_down(Z):
        return k_e(Z)

    # Compute equilibrium distribution
    f, Z = charge_equilibrium_from_rates(k_up, k_down, Zmin, Zmax)
    return f, Z


def compare_charge_dist_mathis(grain_type, grain_size_cm, ne, nH, T, plot=False):
    """
    Compare equilibrium charge distribution computed with the WD01-based
    solver against the Ibanez-Mejias (fitting) result used in `grain_charge_dist`.

    Parameters
    ----------
    grain_type : {'silicate','graphite'}
    grain_size_cm : float
        Grain radius in cm
    ne, nH : float
        Electron and hydrogen densities [cm^-3]
    T : float
        Gas temperature [K]
    plot : bool
        If True, plot the two distributions for visual comparison.

    Returns
    -------
    result : dict
        Contains keys 'Z', 'f_WD', 'f_fit' where f_WD is the distribution from
        `grain_charge_equilibrium_WD01` and f_fit is from `grain_charge_dist`.
    """
    from pycalima.models.dust_charge.dust_photoelectric_heating import read_dielectric_file
    from pycalima.models.dust_charge.IM19_charging import grain_charge_dist
    from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections
    # 1. Load Mathis ISRF (file included in repo as mathis1983.dat)
    data = np.loadtxt('mathis1983.dat')
    # file has columns: wavelength (nm), intensity (photons s-1 cm-2 nm-1)
    wav_nm = data[:,0]
    intensity_phot = data[:,1]

    # Convert to photon energy in eV: E[eV] = hc / lambda
    h = 6.6260755e-27  # erg s
    c = 2.99792458e10  # cm/s
    eV2erg = 1.602176634e-12
    wav_cm = wav_nm * 1e-7
    wav_micron = wav_nm * 1e-3
    E_erg = h * c / wav_cm
    E_eV = E_erg / eV2erg

    # Build radiation_field array compatible with other routines: [E_eV, placeholder, flux_like]
    # Convert intensity (photons s-1 cm-2 nm-1) to photons s-1 cm-2 eV-1
    # d(lambda)/dE = -hc/E^2 => intensity_E = intensity_lambda * (lambda^2/(hc))
    intensity_phot_per_eV = intensity_phot * (wav_cm**2) / (h*c)  # photons s-1 cm-2 eV-1
    # For our routines we use flux_term = intensity_phot_per_eV * E (photons * E?)
    # The photoemission integrand in existing code uses (radiation_field[:,2] / radiation_field[:,0])
    # so we set column 2 = intensity_phot_per_eV * E_eV (so flux/E -> intensity_phot_per_eV)
    flux_term = intensity_phot_per_eV * E_eV
    radiation_field = np.column_stack([E_eV, wav_nm[::-1], flux_term])

    # 2. Get absorption cross section for this grain size using dust_emission helpers
    # Get the dielectric properties and interpolate to the desired wavelengths
    if grain_type == 'graphite':
        data_perp = read_dielectric_file('draine_lee_1984/callindex.out_CpeD03_0.10')
        data_para = read_dielectric_file('draine_lee_1984/callindex.out_CpaD03_0.10')
        Imperp = np.interp(wav_micron[::-1],data_perp['table']['wavelength_um'][::-1], data_perp['table']['Im_n'][::-1])
        Impara = np.interp(wav_micron[::-1],data_para['table']['wavelength_um'][::-1], data_para['table']['Im_n'][::-1])
        Im = np.column_stack([Imperp[::-1], Impara[::-1]])  # first column perpendicular, second parallel
    elif grain_type == 'silicate':
        data_sil = read_dielectric_file('draine_lee_1984/eps_suvSil')
        Im = np.interp(wav_micron[::-1],data_sil['table']['wavelength_um'][::-1], data_sil['table']['Im_n'][::-1])[::-1]
    
    # interpolate_cross_sections returns (a0_micron, wav_cm, C_sca, C_abs, C_rp)
    grain_size_micron = grain_size_cm * 1e4
    _, wav_out_cm, _, C_abs_out, _ = interpolate_cross_sections('graphite' if grain_type=='graphite' else 'silicate', grain_size_micron)
    # interpolate_cross_sections returns wavelengths in cm; we need C_abs on the same E grid as ISRF
    # convert wav_out_cm to E_eV_out and interpolate C_abs_out to E_eV
    h = 6.6260755e-27
    c = 2.99792458e10
    eV2erg = 1.602176634e-12
    E_out_eV = (h*c) / wav_out_cm / eV2erg
    # Interpolate C_abs_out(E) onto E_eV
    C_abs = np.interp(E_eV, E_out_eV, C_abs_out)

    # 3. Compute WD01 equilibrium using our wrapper
    f_WD, Z_WD = grain_charge_equilibrium_WD01(grain_type, grain_size_cm, radiation_field, C_abs, Im, ne, nH, T)

    # 4. Compute fitting distribution from grain_charge_dist (Ibanez-Mejias fit)
    # Convert grain size to IM19 label (e.g., '50A', '100A' etc.) by nearest
    a_nm = grain_size_cm * 1e7
    # available radii in grain_charge_dist fit: 3.5,5,10,50,100,500,1000 A
    im_sizes_A = np.array([3.5,5.,10.,50.,100.,500.,1000.])
    nearest = im_sizes_A[np.argmin(np.abs(im_sizes_A - a_nm))]
    radius_label = f'{int(nearest)}A'
    f_fit, Z_fit = grain_charge_dist(1.0, T, ne, 'silicate' if grain_type=='silicate' else 'graphite', radius_label)

    # 5. Rebin/align Z arrays to a common Z range
    Zmin = min(Z_WD[0], Z_fit[0])
    Zmax = max(Z_WD[-1], Z_fit[-1])
    Z_common = np.arange(Zmin, Zmax+1)

    def regrid(Z_src, f_src, Z_common):
        f_common = np.zeros(len(Z_common))
        for i,z in enumerate(Z_common):
            if z in Z_src:
                f_common[i] = f_src[np.where(Z_src == z)[0][0]]
        # normalize
        s = f_common.sum()
        if s>0:
            f_common /= s
        return f_common

    f_WD_common = regrid(Z_WD, f_WD, Z_common)
    f_fit_common = regrid(Z_fit, f_fit, Z_common)

    # 6. Optionally plot
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(Z_common, f_WD_common, drawstyle='steps-mid', label='WD01 solver')
        ax.plot(Z_common, f_fit_common, drawstyle='steps-mid', label='Ibanez-Mejias fit')
        ax.set_xlabel('Charge Z')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.set_yscale('log')
        plt.show()

    result = {'Z': Z_common, 'f_WD': f_WD_common, 'f_fit': f_fit_common}
    return result