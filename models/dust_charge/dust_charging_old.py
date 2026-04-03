"""
DUST CHARGING IN THE INTERSTELLAR MEDIUM

The scripts here included allow the computation of equlibrium charge
distributions for dust grains in the interstellar medium.

By: Curro Rodriguez Montero (currodri@gmail.com)

"""

# IMPORT LIBRARIES
import os
import numpy as np
import gc
from functools import lru_cache
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})

# CONSTANTS
graphite_work_function = 4.4 # [eV]
silicate_work_function = 8.0 # [eV]
epsilon_0 =  8.8541878188e-21 # Vacuum permittivity [F/nm]
e = 1.602176634e-19           # Elementary charge [C]
me = 9.1093837015e-28         # Electron mass [g]
h_cgs = 6.62607015e-27         # Planck constant [erg s]
c_cgs = 2.99792458e10          # Speed of light [cm/s]
kb_cgs = 1.380649e-16          # Boltzmann constant [erg/K]
silicate_band_gap = 5.0 # [eV]
electron_escape_length = 1 # [nm]
eV2erg = 1.602176634e-12  # Conversion factor from eV to erg
_four_pi_eps0 = 4. * np.pi * epsilon_0
_e_statC = 4.8032047e-10  # statcoulomb

DS87_theta_nu = np.array([0.4203,0.5000,0.5823,0.6296,0.6621,0.6865,0.7560,0.8146])
DS87_nu = np.array([0.5,1,2,3,4,5,10,20])

# UTILITIES
def ensure_array(x):
    return np.array(x, copy=False)


def get_system_memory_bytes():
    """Return total physical memory in bytes. Try psutil, then platform fallbacks."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    import sys, subprocess
    try:
        if sys.platform.startswith('darwin'):
            out = subprocess.check_output(['sysctl', '-n', 'hw.memsize'])
            return int(out.strip())
        elif sys.platform.startswith('linux'):
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        parts = line.split()
                        # value in kB
                        return int(parts[1]) * 1024
    except Exception:
        pass
    # fallback: assume 8 GiB
    return 8 * 1024 ** 3


def get_process_rss_bytes():
    """Return current process resident set (bytes) using resource.ru_maxrss with platform convention."""
    import resource, sys
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith('darwin'):
        return int(ru)
    else:
        return int(ru) * 1024

# default small floor for safe division
_TINY = 1e-300
_INF_RATIO = 1e300


# FUNCTIONS
def _coulomb_energy_over_a(Z, a):
    # a: in nm
    # returns e^2 (Z+1) / (4 pi eps0 a) with broadcasting
    return e * (Z + 1.0) / (_four_pi_eps0 * a)

def ionisation_potential_valence_vec(W,Z,a): 
    # a: in nm
    # Eq 2 Weingartner & Draine 2001
    return W + e/_four_pi_eps0 * ((Z + 0.5) / a + (Z+2.)/a * (0.03/a))

def electron_affinity_graphite_vec(Z, a):
    """Vectorized electron affinity for graphite. a in nm, returns eV."""
    # uses graphite_work_function constant
    return graphite_work_function + (e / _four_pi_eps0) / a * ((Z - 0.5) - 0.4 / (a + 0.7))

def electron_affinity_silicate_vec(Z, a):
    """Vectorized electron affinity for silicate. a in nm, returns eV."""
    return silicate_work_function - silicate_band_gap + (e / _four_pi_eps0) * (Z - 0.5) / a

def min_energy_ejection_vec(Z, a):
    # a: in nm
    # attenuation factor (1 + (2.7/a)^0.75)
    att = 1.0 + np.power(2.7 / a, 0.75)
    Emin_neg = - (e / _four_pi_eps0) * (Z + 1.0) / (a * att)
    return np.where(Z >= 0, 0.0, Emin_neg)

def photodetachment_energy_graphite_vec(Z, a):
    # a: in nm
    Emin = min_energy_ejection_vec(Z, a)
    Eaff = electron_affinity_graphite_vec(Z+1, a)
    return Eaff + Emin

def photodetachment_energy_silicate_vec(Z, a):
    # a: in nm
    Emin = min_energy_ejection_vec(Z, a)
    Eaff = electron_affinity_silicate_vec(Z+1, a)
    return Eaff + Emin

def photodetachment_cross_section_vec(E, E_det, Z):
    # sigma: in cm^2
    # E, E_det in J; output in m^2 with WD01 prefactor (converted if needed)
    x = (E - E_det) / 3.0
    # base sigma
    sigma = 1.2e-17 * np.abs(Z) * x / np.power(1.0 + (x * x) / 3.0, 2.0)
    # threshold: zero where E < E_det
    sigma = np.where(x < 0.0, 0.0, sigma)
    return sigma

def min_photon_energy_vec(IPV, Z, a):
    # a: in nm
    Emin = min_energy_ejection_vec(Z, a)
    return np.where(Z >= -1, IPV, IPV + Emin)

def parameter_theta_vec(E, Emin_ej, Z, a):
    # a: in nm
    coul = _coulomb_energy_over_a(Z, a)
    # for Z >= 0 add + e^2(Z+1)/(4πϵ0 a)
    add_term = np.where(Z >= 0, coul, 0.0)
    return E - Emin_ej + add_term

def escape_fraction_attempting_electrons_vec(hnu, Emin_ej, Z, a):
    # a: in nm
    coul = - _coulomb_energy_over_a(Z, a)
    Elow = coul
    Ehigh = hnu - Emin_ej
    denom = np.maximum((Ehigh - Elow) ** 3.0, 1e-300)
    y2_pos = (Ehigh ** 2.0) * (Ehigh - 3.0 * Elow) / denom
    # for Z < 0, y2 = 1
    return np.where(Z >= 0, np.clip(y2_pos, 0.0, 1.0), 1.0)

def photon_attenuation_length_graphite_vec(wav, Imperp, Impar):
    l_inv = (4.0 * np.pi / wav) * ( (2.0/3.0) * Imperp + (1.0/3.0) * Impar )
    return 1.0 / np.maximum(l_inv, 1e-300)

def photon_attenuation_length_silicate_vec(wav, Im):
    return wav / np.maximum(4.0 * np.pi * Im, 1e-300)

def Watson73_y1_vec(a, la, le):
    beta = a / la
    alpha = a / le + a / la
    num = (beta / alpha) ** 2.0 * (alpha**2 - 2.0*alpha + 2.0 - 2.0 * np.exp(-alpha))
    den = np.maximum(beta**2 - 2.0*beta + 2.0 - 2.0 * np.exp(-beta), 1e-300)
    return num / den

def BT94_y0_graphite_vec(theta, W):
    x = np.maximum(theta / np.maximum(W, 1e-300), 0.0)
    num = 9e-3 * x**5
    den = 1.0 + 3.7e-2 * x**5
    return num / den

def y0_silicate_vec(theta, W):
    x = np.maximum(theta / np.maximum(W, 1e-300), 0.0)
    return 0.5 * x / (1.0 + 5.0 * x)

def photoelectric_yield_graphite_vec(W, Zs, a, le, E, wav, Imperp, Impar):
    """
    Vectorized version of photoelectric_yield_graphite for all (E, Z).
    E and wav are 1D arrays [N_E]; Zs is 1D [N_Z].
    Returns Y(E,Z) array of shape [N_E, N_Z].
    """
    E, Z = np.meshgrid(E, Zs, indexing='ij')

    IPV = ionisation_potential_valence_vec(W, Z, a)
    Emin_ej = min_photon_energy_vec(IPV, Z, a)

    mask = E >= Emin_ej

    theta = parameter_theta_vec(E, Emin_ej, Z, a)
    y0 = BT94_y0_graphite_vec(theta, W)

    la = photon_attenuation_length_graphite_vec(wav[:, None], Imperp[:, None], Impar[:, None])
    y1 = Watson73_y1_vec(a, la, le)

    y2 = escape_fraction_attempting_electrons_vec(E, Emin_ej, Z, a)

    Y = np.where(mask, y2 * np.minimum(y0 * y1, 1.0), 0.0)
    return Y

def photoelectric_yield_silicate_vec(W, Zs, a, le, E, wav, Im):
    """
    Vectorized version of photoelectric_yield_silicate.
    Returns Y(E,Z) array [N_E, N_Z].
    """
    E, Z = np.meshgrid(E, Zs, indexing='ij')

    IPV = ionisation_potential_valence_vec(W, Z, a)
    Emin_ej = min_photon_energy_vec(IPV, Z, a)
    mask = E >= Emin_ej

    theta = parameter_theta_vec(E, Emin_ej, Z, a)
    y0 = y0_silicate_vec(theta, W)

    la = photon_attenuation_length_silicate_vec(wav[:, None], Im[:, None])
    y1 = Watson73_y1_vec(a, la, le)

    y2 = escape_fraction_attempting_electrons_vec(E, Emin_ej, Z, a)

    Y = np.where(mask, y2 * np.minimum(y0 * y1, 1.0), 0.0)
    return Y

# Yield adapters returning Y(nu,Z) for R_pe integration
def yield_graphite_vectorized(nu, Zs, a, params):
    W = params['W']
    le = params['le']
    wav = params.get('wav', c_cgs / np.asarray(nu)* 1e7)  # convert to nm
    Imperp = params['Imperp']
    Impar  = params['Impar']
    E = h_cgs * np.asarray(nu) / eV2erg # convert to eV
    return photoelectric_yield_graphite_vec(W, np.asarray(Zs), a*1e9, le, E, wav, Imperp, Impar)

def yield_silicate_vectorized(nu, Zs, a, params):
    W = params['W']
    le = params['le']
    wav = params.get('wav', c_cgs / np.asarray(nu)* 1e7)  # convert to nm
    Im  = params['Im']
    E = h_cgs * np.asarray(nu) / eV2erg # convert to eV
    return photoelectric_yield_silicate_vec(W, np.asarray(Zs), a*1e9, le, E, wav, Im)

def compute_Rpe_vectorized(nu, J_nu, C_abs_nu, a, Zs, yield_func, yield_params, J_is_per_sr=False, pdt_func=None):

    Zs = np.asarray(Zs, dtype=int)
    N_nu = int(nu.size)
    N_Z = int(Zs.size)

    # determine maximum temporary bytes allowed
    max_tmp_bytes = None
    try:
        if isinstance(yield_params, dict) and 'max_tmp_bytes' in yield_params:
            max_tmp_bytes = int(yield_params['max_tmp_bytes'])
        elif isinstance(yield_params, dict) and 'max_cache_bytes' in yield_params and yield_params['max_cache_bytes'] is not None:
            # conservative fraction of cache budget
            max_tmp_bytes = max(2 * 1024 * 1024, int(yield_params['max_cache_bytes'] // 8))
    except Exception:
        max_tmp_bytes = None
    if max_tmp_bytes is None:
        max_tmp_bytes = 64 * 1024 * 1024  # 64 MiB default

    # estimate memory per column (one Z) for a single float64 column of length N_nu
    bytes_per_Zcol = N_nu * 8
    # compute chunk size that keeps temporary arrays under max_tmp_bytes; allow some overhead
    est_overhead_factor = 4  # account for Y, integrand, and intermediate temporaries
    chunk = max(1, int(max_tmp_bytes // (bytes_per_Zcol * est_overhead_factor)))

    # decide whether to include photodetachment: include by default unless explicitly disabled
    include_pd = True
    if isinstance(yield_params, dict) and ('include_photodetachment' in yield_params):
        include_pd = bool(yield_params.get('include_photodetachment'))

    # if even a single-column temporary would exceed the budget, fall back to streaming per-Z
    if bytes_per_Zcol * est_overhead_factor > max_tmp_bytes:
        if isinstance(yield_params, dict) and yield_params.get('debug'):
            print(f'[compute_Rpe_vectorized] streaming per-Z because single-column ~{bytes_per_Zcol * est_overhead_factor} bytes > max_tmp_bytes={max_tmp_bytes}')
        else:
            # non-verbose notification to stdout so users running examples see streaming mode
            try:
                print('[compute_Rpe_vectorized] INFO: using streaming per-Z memory-sparing mode')
            except Exception:
                pass
        out = np.zeros(N_Z, dtype=float)
        Ccol = C_abs_nu[:, None]
        Jcol = J_nu[:, None]
        # prepare photodetachment helpers if requested
        E_eV = h_cgs * np.asarray(nu) / eV2erg
        if include_pd and pdt_func is None:
            if isinstance(yield_params, dict) and callable(yield_params.get('photodetachment_func')):
                pdt_func = yield_params.get('photodetachment_func')
            else:
                material = 'graphite'
                if isinstance(yield_params, dict):
                    material = yield_params.get('material', 'graphite')
                pdt_func = photodetachment_energy_graphite_vec if material == 'graphite' else photodetachment_energy_silicate_vec
        for idx, Z in enumerate(Zs):
            Y = yield_func(nu, np.array([Z], dtype=int), a, yield_params)
            with np.errstate(divide='ignore', invalid='ignore'):
                integrand = (Ccol * Jcol) * Y
            if J_is_per_sr:
                integrand *= 4.0 * np.pi
            base_int = float(np.trapz(integrand[:, 0], nu))
            add_pd = 0.0
            if include_pd and pdt_func is not None and Z < 0:
                a_nm = float(a) * 1e9
                Zarr = np.array([Z])
                E_pdt_val = float(pdt_func(Zarr, a_nm)[0])
                sigma = photodetachment_cross_section_vec(E_eV, E_pdt_val, Z) * 1e-4
                integrand_pd = sigma * J_nu 
                add_pd = float(np.trapz(integrand_pd, nu))
            out[idx] = base_int + add_pd
            try:
                del Y
                del integrand
            except Exception:
                pass
            gc.collect()
        return out

    # if we can compute all at once, do it; otherwise iterate over chunks
    if N_Z <= chunk:
        Y = yield_func(nu, Zs, a, yield_params)
        Ccol = C_abs_nu[:, None]
        Jcol = J_nu[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            integrand = (Ccol * Jcol) * Y
        if J_is_per_sr:
            integrand *= 4.0 * np.pi
        Rpe_Z = np.trapz(integrand, nu, axis=0)
        # If requested, compute photodetachment heating contribution and add to Rpe_Z
        if include_pd:
            # shared conversions/helpers
            E_eV = h_cgs * np.asarray(nu) / eV2erg
            pd_add = np.zeros_like(Rpe_Z)
            # convert a (m) to nm for helpers
            a_nm = float(a) * 1e9
            # choose pdt function once
            if pdt_func is None:
                if isinstance(yield_params, dict) and callable(yield_params.get('photodetachment_func')):
                    pdt_func = yield_params.get('photodetachment_func')
                else:
                    material = 'graphite'
                    if isinstance(yield_params, dict):
                        material = yield_params.get('material', 'graphite')
                    pdt_func = photodetachment_energy_graphite_vec if material == 'graphite' else photodetachment_energy_silicate_vec
            for iz, Z in enumerate(Zs):
                Zint = Z
                if Zint >= 0:
                    pd_add[iz] = 0.0
                    continue
                E_pdt_val = float(pdt_func(np.array([Zint]), a_nm)[0])
                sigma = photodetachment_cross_section_vec(E_eV, E_pdt_val, Zint) * 1e-4
                integrand_pd = sigma * J_nu
                pd_add[iz] = float(np.trapz(integrand_pd, nu))
            Rpe_Z = Rpe_Z + pd_add
        # free large temporaries immediately
        try:
            del Y
            del integrand
            del Ccol
            del Jcol
            del pd_add
        except Exception:
            pass
        gc.collect()
        return np.asarray(Rpe_Z, dtype=float)
    else:
        out = np.zeros(N_Z, dtype=float)
        # process in chunks of size 'chunk'
        # prepare photodetachment helpers once for chunk processing
        E_eV = h_cgs * np.asarray(nu) / eV2erg
        # choose pdt function once (if requested)
        if include_pd and pdt_func is None:
            if isinstance(yield_params, dict) and callable(yield_params.get('photodetachment_func')):
                pdt_func = yield_params.get('photodetachment_func')
            else:
                material = 'graphite'
                if isinstance(yield_params, dict):
                    material = yield_params.get('material', 'graphite')
                pdt_func = photodetachment_energy_graphite_vec if material == 'graphite' else photodetachment_energy_silicate_vec
        for i in range(0, N_Z, chunk):
            ii = slice(i, min(i + chunk, N_Z))
            Zchunk = Zs[ii]
            Y = yield_func(nu, Zchunk, a, yield_params)
            Ccol = C_abs_nu[:, None]
            Jcol = J_nu[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                integrand = (Ccol * Jcol) * Y
            if J_is_per_sr:
                integrand *= 4.0 * np.pi
            out_chunk = np.trapz(integrand, nu, axis=0)
            # photodetachment contribution for this chunk (compute per-Z)
            if include_pd and pdt_func is not None:
                pd_add = np.zeros_like(out_chunk)
                a_nm = float(a) * 1e9
                for iz, Z in enumerate(Zchunk):
                    Zint = Z
                    if Zint >= 0:
                        pd_add[iz] = 0.0
                        continue
                    E_pdt_val = float(pdt_func(np.array([Zint]), a_nm)[0])
                    sigma = photodetachment_cross_section_vec(E_eV, E_pdt_val, Zint) * 1e-4
                    integrand_pd = sigma * J_nu
                    pd_add[iz] = float(np.trapz(integrand_pd, nu))
                out_chunk = out_chunk + pd_add
                try:
                    del pd_add
                except Exception:
                    pass
            out[ii] = out_chunk
            # free temporaries for this chunk to keep peak memory low
            try:
                del Y
                del integrand
                del Ccol
                del Jcol
            except Exception:
                pass
            gc.collect()
        return np.asarray(out, dtype=float)


def unit_diagnostics(nu, J_nu, C_abs_nu, a, yield_func, yield_params, E_band=(6.0,13.6)):
    """
    Print unit sanity checks for the R_pe calculation.
    nu: Hz array
    J_nu: photons m^-2 s^-1 Hz^-1
    C_abs_nu: m^2
    a: m
    """
    nu = np.asarray(nu)
    J_nu = np.asarray(J_nu)
    C_abs_nu = np.asarray(C_abs_nu)

    # Convert nu to energy in eV
    eV2J = 1.602176634e-19
    h_SI = 6.62607015e-34
    E_J = h_SI * nu
    E_eV = E_J / eV2J

    mask = (E_eV >= E_band[0]) & (E_eV <= E_band[1])
    if not np.any(mask):
        print('[unit_diagnostics] No points in E band')
        return

    # integrated photon flux in band (photons m^-2 s^-1)
    photon_flux = np.trapz(J_nu[mask], nu[mask])

    # mean absorption cross section in band
    mean_C = np.trapz(C_abs_nu[mask]*J_nu[mask], nu[mask]) / np.trapz(J_nu[mask], nu[mask])

    # effective bandwidth in Hz
    bw = nu[mask].max() - nu[mask].min()

    # sample yields at mid-energy for Z=0 and a
    mid_idx = np.argmin(np.abs(E_eV - np.mean(E_band)))
    sample_nu = np.array([nu[mid_idx]])
    Ys = yield_func(sample_nu, np.array([0]), a, yield_params)
    sample_Y = float(Ys[mid_idx, 0]) if Ys.ndim == 2 else float(Ys[0])

    est_Rpe = mean_C * photon_flux * sample_Y

    print('[unit_diagnostics] Photon flux in band (photons m^-2 s^-1): {:.3e}'.format(photon_flux))
    print('[unit_diagnostics] Mean C_abs in band (m^2): {:.3e}'.format(mean_C))
    print('[unit_diagnostics] Sample yield at E~{:.2f} eV: {:.3e}'.format(E_eV[mid_idx], sample_Y))
    print('[unit_diagnostics] Bandwidth (Hz): {:.3e}'.format(bw))
    print('[unit_diagnostics] Simple estimate R_pe ~ mean_C * photon_flux * Y = {:.3e} s^-1'.format(est_Rpe))
    return {
        'photon_flux': photon_flux,
        'mean_C': mean_C,
        'sample_Y': sample_Y,
        'bandwidth': bw,
        'est_Rpe': est_Rpe
    }

def DS87_J_function_vec(Z, q, a, T):
    # Pure-numeric implementation of the Draine & Sutin (1987) J-tilde factor.
    # Inputs:
    #  - Z : array-like of integer grain charges
    #  - q : ion/electron charge (can be scalar or length-1 array); for electrons q=-1
    #  - a : grain radius in meters (the rest of the module uses m)
    #  - T : temperature in K
    # The function returns J(nu) with nu = Z / q and tau = a_cm * kb_cgs * T / (q^2 * e_statC^2)
    Z = np.asarray(Z, dtype=float)
    q = np.asarray(q, dtype=float)

    # convert grain radius from meters to cm for cgs constants
    a_cm = np.asarray(a, dtype=float) * 100.0

    # compute nu = Z/q with numpy broadcasting
    with np.errstate(divide='ignore', invalid='ignore'):
        nu = Z / q

    # compute tau in cgs: tau = a(cm) * k_B (erg/K) * T / ( (q * e_statC)^2 )
    # q may be scalar or array; ensure broadcasting to nu shape
    denom_q2 = (q ** 2) * (_e_statC ** 2)
    tau = (a_cm * kb_cgs * T) / np.maximum(denom_q2, 1e-300)
    try:
        tau = np.broadcast_to(tau, np.shape(nu))
    except Exception:
        tau = np.asarray(tau)

    # numeric safety
    tau_safe = np.maximum(tau, 1e-300)

    # allocate output
    J = np.zeros_like(nu, dtype=float)

    # masks for nu branches
    nu_zero = (nu == 0.0)
    nu_neg = (nu < 0.0)
    nu_pos = (nu > 0.0)

    # nu == 0: J = 1 + sqrt(pi/(2 tau))
    if np.any(nu_zero):
        J[nu_zero] = 1.0 + np.sqrt(np.pi / (2.0 * tau_safe[nu_zero]))

    # nu < 0: J = (1 - nu/tau) * (1 + sqrt(2/(tau - 2 nu)))
    if np.any(nu_neg):
        tn = tau_safe[nu_neg]
        nun = nu[nu_neg]
        inner = np.maximum(tn - 2.0 * nun, 1e-300)
        J[nu_neg] = (1.0 - nun / tn) * (1.0 + np.sqrt(2.0 / inner))

    # nu > 0: theta_nu = 1/(1 + 1/sqrt(nu)); J = (1 + 1/sqrt(4 tau + 3 nu))^2 * exp(-theta_nu / tau)
    if np.any(nu_pos):
        tp = tau_safe[nu_pos]
        nup = np.maximum(nu[nu_pos], 1e-300)
        theta_nu = 1.0 / (1.0 + 1.0 / np.sqrt(nup))
        root_term = 1.0 / np.sqrt(4.0 * tp + 3.0 * nup)
        pref = (1.0 + root_term) ** 2
        expo = np.exp(-theta_nu / tp)
        J[nu_pos] = pref * expo

    # clamp non-finite values to zero
    J = np.where(np.isfinite(J), J, 0.0)
    return J

def autoionisation_potential_graphite(a):
    """Autoionisation potential for graphite (Weingartner & Draine 2001, Eq. 23)."""
    a = np.asarray(a, dtype=float)
    return 3.9 + 0.12 * a + 2.0 / a

def autoionisation_potential_silicate(a):
    """Autoionisation potential for silicate (Weingartner & Draine 2001, Eq. 23)."""
    a = np.asarray(a, dtype=float)
    return 2.5 + 0.07 * a + 8.0 / a


# --- Most negative allowed charge (Eq. 24, WD01)
def most_negative_allowed_charge_graphite(a):
    """Most negative allowed charge for graphite (Weingartner & Draine 2001, Eq. 24)."""
    a = np.asarray(a, dtype=float)
    U_ait = autoionisation_potential_graphite(a)
    return np.floor(-U_ait / 14.4 * a)

def most_negative_allowed_charge_silicate(a):
    """Most negative allowed charge for silicate (Weingartner & Draine 2001, Eq. 24)."""
    a = np.asarray(a, dtype=float)
    U_ait = autoionisation_potential_silicate(a)
    return np.floor(-U_ait / 14.4 * a)

# --- Most positive allowed charge (Eq. 22, WD01)
def most_positive_allowed_charge(a,W,hnu_max):
    """Most positive allowed charge (Weingartner & Draine 2001, Eq. 22)."""
    a = np.asarray(a, dtype=float)
    W = np.asarray(W, dtype=float)
    return np.floor(((hnu_max - W)/14.4 * a + 0.5 - 0.3 / a)/(1. + 0.3 / a) )

def electron_sticking_coefficient_graphite(Z, a):
    """
    Vectorized version of electron_sticking_coefficient_graphite.
    Z and a can be scalars or numpy arrays.
    """
    Z = np.asarray(Z, dtype=float)
    a = np.asarray(a, dtype=float)
    Nc = 468.0 * a**3
    base = 0.5 * (1.0 - np.exp(-a / electron_escape_length))
    factor = 1.0 / (1.0 + np.exp(20.0 - Nc))

    # Most negative charge (depends on 10×a)
    Zmin = most_negative_allowed_charge_graphite(a * 10.0)

    # Start with neutral case
    s = np.zeros_like(Z)

    # Z == 0
    mask0 = (Z == 0)
    # base and factor are functions of a; broadcast them to Z-shape when assigning
    if np.ndim(base) == 0 or base.shape == ():  # scalar
        s[mask0] = base * factor
    else:
        # broadcast to Z shape
        s[mask0] = (base * factor).reshape((-1,))[0] if np.size(base) == 1 else (base * factor)

    # Z < 0 and Z > Zmin
    mask_neg = (Z < 0) & (Z > Zmin)
    if np.ndim(base) == 0 or base.shape == ():
        s[mask_neg] = base * factor
    else:
        s[mask_neg] = (base * factor).reshape((-1,))[0] if np.size(base) == 1 else (base * factor)

    # Z < Zmin → already 0
    # Z > 0
    mask_pos = (Z > 0)
    if np.ndim(base) == 0 or base.shape == ():
        s[mask_pos] = base
    else:
        s[mask_pos] = base.reshape((-1,))[0] if np.size(base) == 1 else base
    return s


def electron_sticking_coefficient_silicate(Z, a):
    """
    Vectorized version of electron_sticking_coefficient_silicate.
    Z and a can be scalars or numpy arrays.
    """
    Z = np.asarray(Z, dtype=float)
    a = np.asarray(a, dtype=float)
    Nc = 468.0 * a**3
    base = 0.5 * (1.0 - np.exp(-a / electron_escape_length))
    factor = 1.0 / (1.0 + np.exp(20.0 - Nc))

    # Most negative charge (depends on 10×a)
    Zmin = most_negative_allowed_charge_silicate(a * 10.0)

    s = np.zeros_like(Z)

    mask0 = (Z == 0)
    if np.ndim(base) == 0 or base.shape == ():
        s[mask0] = base * factor
    else:
        s[mask0] = (base * factor).reshape((-1,))[0] if np.size(base) == 1 else (base * factor)

    mask_neg = (Z < 0) & (Z > Zmin)
    if np.ndim(base) == 0 or base.shape == ():
        s[mask_neg] = base * factor
    else:
        s[mask_neg] = (base * factor).reshape((-1,))[0] if np.size(base) == 1 else (base * factor)

    mask_pos = (Z > 0)
    if np.ndim(base) == 0 or base.shape == ():
        s[mask_pos] = base
    else:
        s[mask_pos] = base.reshape((-1,))[0] if np.size(base) == 1 else base

    return s

def collisional_rates_electrons_vector(a, Zs, n_e, T_e, s_e_func):
    vth = np.sqrt(8.0 * kb_cgs * T_e / (np.pi * me))
    cross = np.pi * a * a * 1e4 # convert m^2 to cm^2
    Jtilde = DS87_J_function_vec(Zs, np.array([-1.0]), a, T_e)
    s = s_e_func(Zs,a*1e9)
    J_e = s * cross * n_e * vth * Jtilde  # [s^-1]
    # if np.any(J_e <= 0.0):
    #     print('[collisional_rates_electrons_vector] Warning: some electron collisional rates <= 0.0')
    #     print('vth, cross, n_e, Jtilde, s_e:', vth, cross, n_e, Jtilde, s)
    return np.asarray(J_e, dtype=float)


def collisional_rates_ions_vector(a, Zs, ion_species):
    """
    Compute total ion collisional capture rates for a set of Zs.

    Parameters
    ----------
    a : scalar or array-like
        Grain radius in meters (same units as used elsewhere).
    Zs : array-like
        Integer charge states.
    ion_species : list of dicts
        Each dict must contain keys: 'n' (cm^-3), 'T' (K), 'm' (kg), 'z' (ion charge, positive integer)

    Returns
    -------
    J_ion_total : ndarray
        Total ion capture rate for each Z in s^-1.
    """
    Zs = np.asarray(Zs, dtype=float)
    a = np.asarray(a, dtype=float)
    cross = np.pi * a * a * 1e4  # m^2 -> cm^2

    J_total = np.zeros_like(Zs, dtype=float)
    for ion in ion_species:
        n_i = float(ion.get('n', 0.0))
        T_i = float(ion.get('T', 1.0))
        m_kg = float(ion.get('m', 1.0))
        z_i = float(ion.get('z', 1.0))

        # convert mass from kg to g for consistency with kb_cgs (erg/K) and velocities in cm/s
        m_g = m_kg * 1e3

        # thermal speed in cm/s
        vth_i = np.sqrt(8.0 * kb_cgs * T_i / (np.pi * m_g))

        # Jtilde from Draine & Sutin (1987) factor, pass ion charge as positive q
        Jtilde_i = DS87_J_function_vec(Zs, np.array([z_i]), a, T_i)

        # sticking/capture probability for ions ~ 1 (assume full sticking)
        s_i = 1.0

        J_i = s_i * cross * n_i * vth_i * Jtilde_i
        J_total = J_total + J_i

    return np.asarray(J_total, dtype=float)

# --------------------------
# Helpers to manage incremental R_pe caching while expanding window
# --------------------------
class RpeCache:
    """
    Manage R_pe evaluations for sets of integer Z values.
    Allows incremental evaluation: if new Zs are requested, only compute those not cached.
    Uses vectorized compute_Rpe_vectorized for efficiency.
    """
    def __init__(self, nu, J_nu, C_abs_nu, a, yield_func, yield_params, max_cache_bytes=2 * 1024 ** 3):
        self.nu = np.asarray(nu)
        self.J_nu = np.asarray(J_nu)
        self.C_abs_nu = np.asarray(C_abs_nu)
        self.a = a
        # if no yield_func provided, choose a reasonable default based on material
        if yield_func is None:
            mat = 'graphite'
            try:
                if yield_params is not None and isinstance(yield_params, dict):
                    mat = yield_params.get('material', 'graphite')
            except Exception:
                mat = 'graphite'
            self.yield_func = yield_graphite_vectorized if mat == 'graphite' else yield_silicate_vectorized
        else:
            self.yield_func = yield_func
        self.yield_params = yield_params
        self.cache = {}   # {Z: Rpe}
        # approximate maximum allowed memory for cached Rpe entries (bytes)
        # default ~2 GiB; each cached float64 ~8 bytes (we ignore Python dict overhead)
        if max_cache_bytes is None:
            self.max_cache_bytes = None
        else:
            self.max_cache_bytes = int(max_cache_bytes)

    def get_Rpe_for_Zs(self, Zs):
        Zs = np.asarray(Zs, dtype=int)
        missing = [Z for Z in Zs if Z not in self.cache]
        # Estimate memory footprint (approx) and limit caching if necessary.
        n_current = len(self.cache)
        n_missing = len(missing)
        # conservative accounting: assume each cached entry costs ~64 KiB to include Python object + dict overhead
        # this prevents underestimating memory use for large caches; tune if needed
        bytes_per_entry = 64 * 1024
        est_bytes_after = (n_current + n_missing) * bytes_per_entry

        # If the estimated post-cache footprint would exceed the budget, do not grow the cache.
        # Instead compute missing entries in small chunks without storing them to the persistent cache.
        if n_missing > 0 and self.max_cache_bytes is not None and est_bytes_after > self.max_cache_bytes:
            # compute missing entries in small batches to limit temporary 2D array sizes
            nu_len = int(self.nu.size)
            # estimate memory per column roughly (nu_len * 8 bytes) and allow a modest overhead
            bytes_per_Zcol = max(8 * nu_len, 1024)
            # allow temporary usage up to a fraction of max_cache_bytes
            max_tmp_bytes = max(1, int(self.max_cache_bytes // 8))
            chunk = max(1, int(max_tmp_bytes // bytes_per_Zcol))
            if chunk <= 0:
                chunk = 1

            rem_map = {}
            for i in range(0, len(missing), chunk):
                Zchunk = np.array(missing[i:i+chunk], dtype=int)
                Rvals = compute_Rpe_vectorized(self.nu, self.J_nu, self.C_abs_nu,
                                               self.a, Zchunk, self.yield_func, self.yield_params)
                for Z, R in zip(Zchunk, Rvals):
                    rem_map[Z] = float(R)

            # build return list using cached entries and rem_map for missing ones
            out = []
            for Z in Zs:
                Zint = Z
                if Zint in self.cache:
                    out.append(self.cache[Zint])
                else:
                    out.append(rem_map.get(Zint, 0.0))
            return np.array(out, dtype=float)
        # If we reach here, either missing is small or we have room to cache all
        if len(missing) > 0:
            # compute in vectorized batch
            Zm = np.array(missing, dtype=int)
            Rpe_vals = compute_Rpe_vectorized(self.nu, self.J_nu, self.C_abs_nu,
                                              self.a, Zm, self.yield_func, self.yield_params)
            for Z, R in zip(Zm, Rpe_vals):
                self.cache[Z] = float(R)
        # return array in the same order as Zs
        return np.array([self.cache[Z] for Z in Zs], dtype=float)

    def get_Rpe_single(self, Z):
        return self.get_Rpe_for_Zs([Z])[0]

# --------------------------
# Zref and window finder (optimized, reuses RpeCache)
# --------------------------
def find_Zref_and_bounds_optimized(a, n_e, T_e, ion_species, nu, J_nu, C_abs_nu,
                                   yield_func, yield_params,
                                   Z_start=0, max_search=2000,
                                   initial_halfwidth=20, tol_edge=1e-4,
                                   max_halfwidth=2000,
                                   material='graphite',
                                   max_cache_bytes=None):
    """
    Optimized finder for Zref, Zmin, Zmax.
    - ion_species: list of dicts with keys ['n','T','m','z'] for each ion type (z in + units)
    Returns (Zref, Zmin, Zmax, RpeCache)
    """
    # initialize RpeCache (respect max_cache_bytes when provided)
    if max_cache_bytes is None:
        rpc = RpeCache(nu, J_nu, C_abs_nu, a, yield_func, yield_params)
    else:
        rpc = RpeCache(nu, J_nu, C_abs_nu, a, yield_func, yield_params, max_cache_bytes=int(max_cache_bytes))

    # quick approximate Zref using collisional-only rates (cheap)
    # we compute ratio r(Z) = R_plus(R_pe_est~0 + ion capture) / R_minus(electrons)
    if yield_params is None:
        yield_params = {}
    if 's_e_func' in yield_params:
        s_e_func = yield_params['s_e_func']
    else:
        s_e_func = electron_sticking_coefficient_graphite if material == 'graphite' else electron_sticking_coefficient_silicate
    def ratio_collisional_only(Z):
        # compute electron capture
        J_e = collisional_rates_electrons_vector(np.array([a]), np.array([Z]), n_e, T_e, s_e_func)[0]
        # compute ion capture using provided ion_species (may be empty)
        J_ion_total = collisional_rates_ions_vector(np.array([a]), np.array([Z]), ion_species)[0] if ion_species else 0.0
        # R_plus ~ J_ion_total (ignoring R_pe) ; R_minus ~ J_e
        denom = J_e
        num = J_ion_total
        if denom <= 0:
            return _INF_RATIO if num > 0 else 0.0
        return num / denom

    # Compute physically allowed min/max charges and clamp search to that range.
    # most_negative_allowed_* and most_positive_allowed_charge expect the grain
    # radius in Angstroms (Å). The input `a` is in meters, so convert:
    try:
        a_angstrom = float(a) * 1e10
    except Exception:
        a_angstrom = float(np.asarray(a, dtype=float)) * 1e10

    # Determine hnu_max in eV from nu (Hz)
    try:
        nu_arr = np.asarray(nu, dtype=float)
        E_eV = h_cgs * nu_arr / eV2erg
        hnu_max = float(np.nanmax(E_eV)) if E_eV.size > 0 else 13.6
    except Exception:
        hnu_max = 13.6

    # Work function W may be provided in yield_params; fall back to material constants
    W_val = None
    try:
        if isinstance(yield_params, dict) and 'W' in yield_params:
            W_val = float(yield_params.get('W'))
    except Exception:
        W_val = None
    if W_val is None:
        W_val = graphite_work_function if material == 'graphite' else silicate_work_function

    # Compute allowed Z bounds (both functions expect Angstrom input)
    if material == 'graphite':
        Zmin_allowed = most_negative_allowed_charge_graphite(a_angstrom) + 1
    else:
        Zmin_allowed = most_negative_allowed_charge_silicate(a_angstrom) + 1
    Zmax_allowed = most_positive_allowed_charge(a_angstrom, W_val, hnu_max)
    # ensure integer bounds and sensible ordering
    try:
        Zmin_allowed = int(np.floor(Zmin_allowed))
    except Exception:
        Zmin_allowed = int(-max_halfwidth)
    try:
        Zmax_allowed = int(np.floor(Zmax_allowed))
    except Exception:
        Zmax_allowed = int(max_halfwidth)
    # Ensure sensible ordering
    if Zmin_allowed > Zmax_allowed:
        # fallback to symmetric small window around Z_start if formulas disagree
        Zmin_allowed = max(Zmin_allowed, int(Z_start - initial_halfwidth))
        Zmax_allowed = min(Zmax_allowed, int(Z_start + initial_halfwidth))
    # find coarse Zref by scanning small range using collisional-only ratio
    Z0 = int(Z_start)
    Zref = Z0
    r0 = ratio_collisional_only(Z0)
    if np.isfinite(r0) and r0 > 1.0:
        for Z in range(Z0, Z0 + max_search):
            if ratio_collisional_only(Z) < 1.0:
                Zref = max(Z0, Z - 1)
                break
    elif np.isfinite(r0) and r0 < 1.0:
        for Z in range(Z0, Z0 - max_search, -1):
            if ratio_collisional_only(Z) > 1.0:
                Zref = min(Z0, Z + 1)
                break
    # else keep Zref = Z0

    # Enforce that the coarse Zref lies within the physically allowed bounds
    if Zref < Zmin_allowed:
        Zref = Zmin_allowed
    if Zref > Zmax_allowed:
        Zref = Zmax_allowed

    # Now refine Zref by including R_pe evaluated on a narrow window around Zref:
    halfw = initial_halfwidth
    while halfw <= max_halfwidth:
        Zmin = Zref - halfw
        Zmax = Zref + halfw
        # clamp to allowed physical bounds
        if Zmin < Zmin_allowed:
            Zmin = Zmin_allowed
        if Zmax > Zmax_allowed:
            Zmax = Zmax_allowed
        # if clamping collapsed the window, ensure at least a minimal window
        if Zmin > Zmax:
            Zmin = Zmin_allowed
            Zmax = Zmax_allowed
        Zs = np.arange(Zmin, Zmax + 1, dtype=int)

        # ensure R_pe computed for this Zs (cached)
        Rpe_arr = rpc.get_Rpe_for_Zs(Zs)

        # compute collisional J_e and ion capture J_ion
        J_e_arr = collisional_rates_electrons_vector(np.array([a]), Zs, n_e, T_e, s_e_func)
        J_ion_arr = collisional_rates_ions_vector(np.array([a]), Zs, ion_species)

        R_plus_arr = J_ion_arr + Rpe_arr
        R_minus_arr = J_e_arr

        # compute ratios r(Z)=R_plus(Z)/R_minus(Z+1) across window (for Z = Zmin...Zmax-1)
        # careful at last point: we need R_minus(Zmax+1). If not cached, compute it quickly.
        r_list = []
        for idx, Z in enumerate(Zs[:-1]):
            num = R_plus_arr[idx]
            # R_minus at next Z
            denomZ = Z + 1
            if denomZ in rpc.cache:
                denom = rpc.cache.get(denomZ) if False else None  # we don't store J_e in rpc; compute directly
                denom = collisional_rates_electrons_vector(np.array([a]), np.array([denomZ]), n_e, T_e, s_e_func)[0]
            if denom <= 0:
                r_list.append(_INF_RATIO if num > 0 else 0.0)
            else:
                r_list.append(num / denom)

        r_arr = np.asarray(r_list)
        # find index where ratio crosses unity: r(Z) > 1 then r(Z_next) < 1 -> crossing near that Z
        crossing_indices = np.where(r_arr < 1.0)[0]
        if crossing_indices.size > 0:
            # pick first index where r < 1; peak near previous Z
            idx_cross = crossing_indices[0]
            Zref_candidate = Zs[max(0, idx_cross)]
            # refine choice using neighboring points: choose Z where |R_plus(Z)-R_minus(Z+1)| minimal
            # compute difference near idx_cross
            neigh = np.arange(max(0, idx_cross - 2), min(len(r_arr), idx_cross + 3))
            diffs = []
            for j in neigh:
                num = R_plus_arr[j]
                denom = collisional_rates_electrons_vector(np.array([a]), np.array([Zs[j] + 1]), n_e, T_e, s_e_func)[0]
                diff = abs(num - denom)
                diffs.append(diff)
            best_rel = neigh[np.argmin(diffs)]
            Zref_new = Zs[best_rel]
            Zref = int(Zref_new)
        else:
            # no crossing inside window: expand window around current Zref
            halfw = int(halfw * 2) if halfw < max_halfwidth else halfw + 10
            continue

        # Now with chosen Zref, test whether edges are negligible
        # Build P(Z) over this window using recursion and check P at edges
        Zs_full = np.arange(Zmin, Zmax + 1, dtype=int)
        N = len(Zs_full)
        idx_ref = np.where(Zs_full == Zref)[0]
        if idx_ref.size == 0:
            # Zref fell outside window — this can happen after clamping; re-center Zref to allowed range
            if Zref < Zmin_allowed:
                Zref = Zmin_allowed
            elif Zref > Zmax_allowed:
                Zref = Zmax_allowed
            else:
                halfw = int(halfw * 2)
            continue
        idx0 = int(idx_ref[0])

        # ensure Rpe for full window (should be cached already)
        Rpe_full = rpc.get_Rpe_for_Zs(Zs_full)
        J_e_full = collisional_rates_electrons_vector(np.array([a]), Zs_full, n_e, T_e, s_e_func)
        J_ion_full = collisional_rates_ions_vector(np.array([a]), Zs_full, ion_species)

        Rplus_full = J_ion_full + Rpe_full
        Rminus_full = J_e_full

        # recursion
        P = np.zeros_like(Rplus_full, dtype=float)
        P[idx0] = 1.0
        # upward
        for j in range(idx0, N - 1):
            denom = Rminus_full[j + 1]
            num = Rplus_full[j]
            if denom <= 0.0:
                ratio = 0.0 if num == 0 else _INF_RATIO
            else:
                ratio = num / denom
            # if ratio extremely large, cap in log-space by setting next to previous * big number
            P[j + 1] = P[j] * ratio
        # downward
        for j in range(idx0, 0, -1):
            denom = Rplus_full[j - 1]
            num = Rminus_full[j]
            if denom <= 0.0:
                ratio = 0.0 if num == 0 else _INF_RATIO
            else:
                ratio = num / denom
            P[j - 1] = P[j] * ratio

        # normalize and check edge probabilities
        total = np.sum(P)
        if total <= 0:
            halfw = int(halfw * 2)
            continue
        P /= total
        if P[0] < tol_edge and P[-1] < tol_edge:
            # final clamp before returning
            Zmin_ret = max(int(Zmin), Zmin_allowed)
            Zmax_ret = min(int(Zmax), Zmax_allowed)
            Zref_ret = int(np.clip(int(Zref), Zmin_allowed, Zmax_allowed))
            return int(Zref_ret), int(Zmin_ret), int(Zmax_ret), rpc
        else:
            # expand window and retry (but reusing cached R_pe)
            halfw = int(halfw * 2) if halfw < max_halfwidth else halfw + 50

    # if we reach here, fallback: return clamped window centered on Zref
    Zmin_fb = max(int(Zref - initial_halfwidth), Zmin_allowed)
    Zmax_fb = min(int(Zref + initial_halfwidth), Zmax_allowed)
    Zref_fb = int(np.clip(int(Zref), Zmin_allowed, Zmax_allowed))
    return int(Zref_fb), int(Zmin_fb), int(Zmax_fb), rpc


# --------------------------
# Full equilibrium solver (vectorized + optimized)
# --------------------------
def compute_equilibrium_charge_distribution_vectorized(
    a,
    n_e, T_e,
    ion_species,
    nu, J_nu, C_abs_nu,
    yield_func=None,
    yield_params=None,
    Z_start=0,
    initial_halfwidth=20,
    tol_edge=1e-4,
    debug=False,
    max_cache_bytes=None
):
    """
    Compute equilibrium P(Z) using vectorized photoemission and optimized bounds finder.

    Arguments:
      - a: grain radius (m)
      - n_e, T_e: electron density [m^-3] and temperature [K]
      - ion_species: list of dicts, each with keys:
           {'n': number density [m^-3], 'T': temperature [K], 'm': mass [kg], 'z': charge (e.g. +1)}
      - nu, J_nu: frequency grid [Hz] and photon flux per Hz [photons m^-2 s^-1 Hz^-1]
      - C_abs_nu: absorption cross-section on nu grid [m^2]
      - yield_func: vectorized yield function callable (nu, Zs, a, yield_params) -> (N_nu,N_Z)
      - yield_params: dict passed to yield_func
      - Z_start: initial search center
      - initial_halfwidth, tol_edge: bounds finder params
    Returns:
      - Zs (np.array), P (np.array), rates (dict), Zmean, Zsigma
    """
    if yield_params is None:
        yield_params = {}
    # default yield function by material if not provided
    material = yield_params.get('material', 'graphite')
    if yield_func is None:
        yield_func = yield_graphite_vectorized if material == 'graphite' else yield_silicate_vectorized
    # electron sticking function
    s_e_func = yield_params.get('s_e_func', electron_sticking_coefficient_graphite if material == 'graphite' else electron_sticking_coefficient_silicate)

    # 1) find Zref and window, get rpc cache
    Zref, Zmin, Zmax, rpc = find_Zref_and_bounds_optimized(
        a, n_e, T_e, ion_species, nu, J_nu, C_abs_nu, yield_func, yield_params,
        Z_start=Z_start, initial_halfwidth=initial_halfwidth, tol_edge=tol_edge,
        material=material
    )

    # Debug: report the computed window and the physically allowed bounds
    if debug:
        try:
            # convert a (m) -> Angstrom
            a_angstrom = float(a) * 1e10
            nu_arr = np.asarray(nu, dtype=float)
            E_eV = h_cgs * nu_arr / eV2erg
            hnu_max = float(np.nanmax(E_eV)) if E_eV.size > 0 else 13.6
            W_val = None
            if isinstance(yield_params, dict) and 'W' in yield_params:
                W_val = float(yield_params.get('W'))
            if W_val is None:
                W_val = graphite_work_function if material == 'graphite' else silicate_work_function
            if material == 'graphite':
                Zmin_allowed = most_negative_allowed_charge_graphite(a_angstrom)
            else:
                Zmin_allowed = most_negative_allowed_charge_silicate(a_angstrom)
            Zmax_allowed = most_positive_allowed_charge(a_angstrom, W_val, hnu_max)
            print(f'[debug] Zref={Zref}, Zmin={Zmin}, Zmax={Zmax}; allowed Zmin={Zmin_allowed}, Zmax={Zmax_allowed}')
        except Exception as _ex:
            print('[debug] failed to compute allowed Z bounds:', _ex)

    Zs = np.arange(Zmin, Zmax + 1, dtype=int)

    # 2) get vectorized R_pe for this final window (should be cached, but ensure)
    if debug:
        unit_diagnostics(nu, J_nu, C_abs_nu, a, yield_func, yield_params)
    Rpe_arr = rpc.get_Rpe_for_Zs(Zs)
    if debug:
        u = compute_energy_density_from_nu_J(nu, J_nu)
        wd01_val = 6.07e-14
        ratio = u / wd01_val if wd01_val > 0 else np.inf
        print(f'[energy density check] integrated u(6-13.6eV) = {u:.3e} erg/cm^3, WD01 = {wd01_val:.3e}, ratio = {ratio:.3g}')

    if debug:
        # diagnostic range around Zref
        zlo = max(Zs[0], Zref - 10)
        zhi = min(Zs[-1], Zref + 10)
        diag_Zs = np.arange(zlo, zhi + 1, dtype=int)
        Rpe_diag = rpc.get_Rpe_for_Zs(diag_Zs)
        # compute photoemission-only rates by bypassing photodetachment
        try:
            yp = dict(yield_params) if yield_params is not None else {}
            yp['include_photodetachment'] = False
            Rpe_photoem = compute_Rpe_vectorized(np.asarray(nu), np.asarray(J_nu), np.asarray(C_abs_nu), a, diag_Zs, yield_func, yp)
        except Exception:
            # fallback: mark photoemission as NaN if computation fails
            Rpe_photoem = np.full_like(Rpe_diag, np.nan)
        # photodetachment contribution is the difference
        Rpd_diag = Rpe_diag - Rpe_photoem
        J_e_diag = collisional_rates_electrons_vector(np.array([a]), diag_Zs, n_e, T_e, s_e_func)
        J_ion_diag = collisional_rates_ions_vector(np.array([a]), diag_Zs, ion_species)
        print('\n[diagnostic rates around Zref]')
        print('   Z     R_pe_tot [s^-1]  R_pe_emis [s^-1]   R_pd [s^-1]       J_e [s^-1]        J_ion [s^-1]     R_tot/(J_e+J_ion)')
        for Zv, Rtot, Rem, Rpd, Je, Ji in zip(diag_Zs, Rpe_diag, Rpe_photoem, Rpd_diag, J_e_diag, J_ion_diag):
            denom = Je + Ji
            ratio = Rtot / denom if denom > 0 else np.inf
            print(f'{Zv:4d}  {Rtot:12.4e}  {Rem:12.4e}  {Rpd:12.4e}  {Je:12.4e}  {Ji:12.4e}  {ratio:12.4e}')
        print('\n')

        # Save photoelectric yields vs energy for diagnostic Zs
        # compute energy grid in eV
        E_eV = h_cgs * np.asarray(nu) / eV2erg
        Y = yield_func(np.asarray(nu), diag_Zs, a, yield_params)
        # Y shape expected (N_nu, N_Z)
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(8, 5), dpi=150)
        max_lines = 25
        for i, Zv in enumerate(diag_Zs):
            if i >= max_lines:
                break
            y = Y[:, i] if Y.ndim == 2 else Y
            ax.plot(E_eV, y, label=f'Z={int(Zv)}')
        ax.set_xlabel('E (eV)')
        ax.set_ylabel('Yield Y(E,Z)')
        ax.set_title(f'Photoelectric yield vs E — a={a*1e6:.4f} um, Z={zlo}..{zhi}')
        ax.set_yscale('log')
        ax.set_ylim([1e-2,1.2])
        ax.grid(True, linestyle=':', alpha=0.5)
        # Add the Draine_yield_graphite or silicate curve for reference
        data = np.loadtxt(f'./Draine_yield_{material}.csv', delimiter=',')
        print(f'[debug] overplotting Draine yield for {material} from data/Draine_yield_{material}.csv')
        ax.plot(data[:, 0], data[:, 1], label='Draine yield', color='k', linestyle=':')
        Y_test = yield_func(np.asarray(nu), np.array([0]), 4e-10, yield_params)
        ax.plot(E_eV, Y_test[:, 0], label='This code Z=0', color='gray', linestyle='-')
        print(f'[debug] overplotting this code yield for Z=0',Y_test[:,0])
        print(yield_params)
        from dust_photoelectric_heating import photoelectric_yield_graphite, photoelectric_yield_silicate
        if material == 'graphite':
            wav = c_cgs / np.asarray(nu)* 1e7
            Y_ref = np.zeros_like(E_eV)
            for e in range(len(E_eV)):
                E = E_eV[e]
                Y_ref[e] = photoelectric_yield_graphite(graphite_work_function,0,0.4,
                                                        electron_escape_length,E,
                                                        wav[e],yield_params['Imperp'][e],
                                                        yield_params['Impar'][e])
            ax.plot(E_eV, Y_ref, label='Draine func Z=0', color='red', linestyle='--')
            print(f'[debug] overplotting Draine function yield for {material} Z=0',Y_ref)
        else:
            wav = c_cgs / np.asarray(nu)* 1e7
            Y_ref = np.zeros_like(E_eV)
            for e in range(len(E_eV)):
                E = E_eV[e]
                Y_ref[e] = photoelectric_yield_silicate(silicate_work_function,0,0.4,
                                                        electron_escape_length,E,
                                                        wav[e],yield_params['Im'][e])
            ax.plot(E_eV, Y_ref, label='Draine func Z=0', color='red', linestyle='--')
            print(f'[debug] overplotting Draine function yield for {material} Z=0',Y_ref)
        ax.legend(ncol=2, fontsize='small')
        outname = f'examples/yields_debug_{material}_a{a*1e6:.4f}um_Z{zlo}_{zhi}.png'
        fig.savefig(outname, dpi=200)
        _plt.close(fig)
        print(f'[debug] saved yields plot to {outname}')

    # 3) compute collisional arrays
    # electron capture
    J_e_arr = collisional_rates_electrons_vector(np.array([a]), Zs, n_e, T_e, s_e_func)
    # ion capture: compute for provided ion_species (returns zeros if list empty)
    J_ion_arr_total = collisional_rates_ions_vector(np.array([a]), Zs, ion_species) if ion_species else np.zeros_like(J_e_arr)
    # 4) assemble R_plus and R_minus (ignore ion collisional rates)
    # to reduce peak memory avoid keeping separate copy of R_pe: reuse Rpe_arr as R_plus
    R_plus = Rpe_arr
    try:
        R_plus += J_ion_arr_total
    except Exception:
        # fallback to explicit sum if in-place fails
        R_plus = Rpe_arr + J_ion_arr_total
    R_minus = J_e_arr

    # free some temporaries / intermediates and hint GC to release memory
    try:
        del J_ion_arr_total
    except Exception:
        pass
    gc.collect()

    # 5) solve recursion using Zref index
    idx_ref = int(np.where(Zs == Zref)[0][0])
    N = len(Zs)
    # Use log-space recursion to avoid overflow: compute logP up/down and then exponentiate
    logP = np.full(N, -np.inf, dtype=float)
    logP[idx_ref] = 0.0

    # upward recursion: logP[j+1] = logP[j] + log(num/denom)
    for j in range(idx_ref, N - 1):
        denom = R_minus[j + 1]
        num = R_plus[j]
        if denom <= 0.0 or num <= 0.0:
            # keep -inf (zero probability)
            log_ratio = -np.inf
        else:
            log_ratio = np.log(num) - np.log(denom)
        if np.isfinite(logP[j]) and np.isfinite(log_ratio):
            logP[j + 1] = logP[j] + log_ratio
        else:
            logP[j + 1] = -np.inf

    # downward recursion: logP[j-1] = logP[j] + log(num/denom) where roles swapped
    for j in range(idx_ref, 0, -1):
        denom = R_plus[j - 1]
        num = R_minus[j]
        if denom <= 0.0 or num <= 0.0:
            log_ratio = -np.inf
        else:
            log_ratio = np.log(num) - np.log(denom)
        if np.isfinite(logP[j]) and np.isfinite(log_ratio):
            logP[j - 1] = logP[j] + log_ratio
        else:
            logP[j - 1] = -np.inf

    # exponentiate safely: subtract max for numerical stability
    finite_mask = np.isfinite(logP)
    if not np.any(finite_mask):
        P = np.ones(N, dtype=float) / float(N)
    else:
        max_log = np.max(logP[finite_mask])
        shifted = np.where(finite_mask, np.exp(logP - max_log), 0.0)
        total = np.sum(shifted)
        if total <= 0.0 or not np.isfinite(total):
            P = np.ones(N, dtype=float) / float(N)
        else:
            P = shifted / total

    # moments
    Zmean = float(np.sum(Zs * P))
    Zvar = float(np.sum((Zs - Zmean) ** 2 * P))
    Zsigma = np.sqrt(Zvar)

    rates = {
        'R_plus': R_plus,
        'R_minus': R_minus,
        'J_e': J_e_arr,
        # note: R_pe is not separately stored to avoid duplicate large arrays; R_plus includes R_pe contribution
    }

    # ------------------------------------------------------------------
    # Compute photoelectric heating and recombination cooling for the
    # equilibrium distribution P(Z). Use local nu/J_nu/C_abs_nu arrays and
    # the helpers in `dust_photoelectric_heating` when available.
    from dust_photoelectric_heating import compute_photoelectric_heating_rate,\
                                             compute_recombination_cooling_rate,\
                                             compute_autoionisation_cooling_rate

    # Build radiation_field expected by compute_photoelectric_heating_rate
    # E_eV, lambda_nm, I_E_surface (erg / s / cm^2 / eV)
    E_eV = h_cgs * np.asarray(nu) / eV2erg
    # wavelength in cm (c_cgs defined at module level) -> nm
    lambda_cm = c_cgs / np.asarray(nu)
    lambda_nm = lambda_cm * 1e7
    # Convert J_nu (photons / s / m^2 / Hz) -> photons / s / cm^2 / Hz
    J_nu_cm = np.asarray(J_nu) * 1e-4
    # Energy flux per eV: J_nu_cm * nu * eV2erg  (erg / s / cm^2 / eV)
    I_E_surface = J_nu_cm * np.asarray(nu) * eV2erg
    radiation_field_for_heating = np.column_stack([E_eV, lambda_nm, I_E_surface])

    # absorption cross section for use in heating function: convert m^2 -> cm^2
    C_abs_for_heating = np.asarray(C_abs_nu) * 1e4

    # dielectric arrays required by compute_photoelectric_heating_rate are expected
    # in yield_params (we pass the whole object down when needed)
    Im_for_heating = None
    if isinstance(yield_params, dict):
        if 'Im' in yield_params:
            Im_for_heating = yield_params.get('Im')
        elif 'Imperp' in yield_params and 'Impar' in yield_params:
            Im_for_heating = np.column_stack([yield_params.get('Imperp'), yield_params.get('Impar')])

    # lengths: compute_photoelectric_heating_rate expects 'a' in nm for many helpers
    a_nm_local = float(a) * 1e9
    # recombination helper expects a in microns
    a_micron_local = float(a) * 1e6

    Gamma_Z = np.zeros_like(Zs, dtype=float)
    Recomb_Z = np.zeros_like(Zs, dtype=float)
    for iZ, Zv in enumerate(Zs):
        # photoelectric heating (includes photodetachment inside compute_photoelectric_heating_rate)
        args_pe = (int(Zv), a_nm_local, radiation_field_for_heating, ( 'graphite' if material=='graphite' else 'silicate'), Im_for_heating, C_abs_for_heating)
        Gamma_Z[iZ] = float(compute_photoelectric_heating_rate(args_pe))
        # recombination cooling
        args_re = (int(Zv), a_micron_local, n_e, T_e, ( 'graphite' if material=='graphite' else 'silicate'))
        Recomb_Z[iZ] = float(compute_recombination_cooling_rate(args_re))

    Gamma_total = float(np.sum(P * Gamma_Z))
    Recomb_total = float(np.sum(P * Recomb_Z))

    # autoionisation cooling (if Zmin exists)
    auto_cooling = 0.0
    if material == 'graphite':
        Zmin = most_negative_allowed_charge_graphite(a_nm_local*10)
    else:
        Zmin = most_negative_allowed_charge_silicate(a_nm_local*10)
    if Zmin in Zs:
        idx_auto = int(np.where(Zs == Zmin)[0][0])
        prob_Zmin = P[idx_auto]
        if prob_Zmin > 0.0:
            args_ai = (Zmin, prob_Zmin, a_micron_local, n_e, T_e, ( 'graphite' if material=='graphite' else 'silicate'))
            auto_cooling = float(compute_autoionisation_cooling_rate(args_ai))

    rates['Gamma_Z'] = Gamma_Z
    rates['Gamma_total'] = Gamma_total
    rates['Recomb_Z'] = Recomb_Z
    rates['Recomb_total'] = Recomb_total
    rates['Autoionisation_cooling'] = auto_cooling
    # Compute absorbed power E_abs [erg/s] using the same radiation field used
    # for the Gamma_Z computation. radiation_field_for_heating contains columns
    # [E_eV, lambda_nm, I_E_surface(erg/s/cm^2/eV)] and C_abs_for_heating is
    # in cm^2 so their product integrates to erg/s.
    try:
        E_grid = np.asarray(radiation_field_for_heating[:, 0], dtype=float)
        I_E_surf = np.asarray(radiation_field_for_heating[:, 2], dtype=float)
        mask = np.isfinite(E_grid) & np.isfinite(I_E_surf)
        if np.any(mask):
            # integrate only over finite points and positive energies
            sel = mask & (E_grid >= 0.0)
            if np.any(sel):
                E_abs_val = float(np.trapz(C_abs_for_heating[sel] * I_E_surf[sel], E_grid[sel]))
            else:
                E_abs_val = 0.0
        else:
            E_abs_val = 0.0
    except Exception:
        E_abs_val = 0.0

    rates['E_abs'] = float(E_abs_val)
    # Efficiency: fraction of absorbed power going into net gas heating
    try:
        eff_val = 0.0
        denom = float(E_abs_val)
        # include autoionisation cooling in recombination losses when evaluating net heating
        total_rec_losses = float(Recomb_total) + float(auto_cooling)
        if denom > 0.0:
            eff_val = float((Gamma_total - total_rec_losses) / denom)
        else:
            eff_val = 0.0
    except Exception:
        eff_val = 0.0
    rates['efficiency'] = float(eff_val)
    # ------------------------------------------------------------------

    return Zs, P, rates, Zmean, Zsigma


def plot_charge_distribution(Zs, P, ax=None, title=None, xlabel='Z', ylabel='P(Z)', savefile=None):
    """
    Plot the equilibrium charge distribution as a step histogram.

    Parameters
    ----------
    Zs : array-like
        Integer charge states.
    P : array-like
        Probabilities for each Z in the same order as Zs.
    ax : matplotlib.axes.Axes or None
        Axis to plot into. If None, a new figure is created and returned.
    title : str or None
        Optional title for the plot.
    xlabel, ylabel : str
        Axis labels.
    savefile : str or None
        If provided, the figure is saved to this path.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    import matplotlib.pyplot as _plt

    Zs = np.asarray(Zs)
    P = np.asarray(P)

    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 4), dpi=150)
    else:
        fig = ax.figure

    # build step-like polygon (staircase) for discrete integer bins centered on Z
    n = len(Zs)
    if n == 0:
        raise ValueError('Empty Zs array')

    # x coordinates: start at left edge of first bin, then alternate
    x = np.empty(2 * n + 2)
    y = np.empty_like(x)
    x[0] = Zs[0] - 0.5
    y[0] = 0.0
    for i, z in enumerate(Zs):
        x[2 * i + 1] = z - 0.5
        x[2 * i + 2] = z + 0.5
        y[2 * i + 1] = P[i]
        y[2 * i + 2] = P[i]
    x[-1] = Zs[-1] + 0.5
    y[-1] = 0.0

    ax.plot(x, y, drawstyle='steps-post', color='C0')
    ax.fill_between(x, y, step='post', alpha=0.3, color='C0')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.5)

    if savefile is not None:
        fig.savefig(savefile, dpi=200)

    return fig, ax


def equilibrium_charge_for_grain(G0, ne, T, grain_type, a_micron,
                                radiation_model='Mathis', rad_field=None,
                                yield_params=None, ion_species=None, Z_start=0, debug=False):
    """
    High-level wrapper that builds the radiation and optical inputs for a single
    grain and returns the equilibrium charge distribution using the
    vectorized WD01 solver.

    Parameters
    ----------
    G0 : float
        Scaling factor for the radiation field (dimensionless). The function
        will scale the field returned by `get_radiation_field` by G0.
    ne : float
        Electron density in cm^-3.
    T : float
        Electron temperature in K.
    grain_type : {'graphite','silicate'}
        Grain material/type used to pick dielectric data and yields.
    a_micron : float
        Grain radius in microns.
    radiation_model : str
        Passed to `get_radiation_field` if `rad_field` is not provided.
    rad_field : ndarray or None
        Optional precomputed radiation field. When provided it must be a
        2D array with columns [E_eV, wavelength_nm, I_E] where I_E is the
        spectral intensity in erg s^-1 cm^-2 eV^-1. If you pass the output
        of `get_radiation_field` from the project's `dust_photoelectric_heating`
        module, ensure it contains those columns (some codepaths in the repo
        use different column orders; this wrapper assumes the energy-first
        ordering).
    yield_params : dict or None
        Extra parameters forwarded to the yield function (material, s_e_func,
        etc.). If `material` not present, it'll be inferred from `grain_type`.
    Z_start : int
        Initial centre for the Zref search.

    Returns
    -------
    Zs, P, rates, Zmean, Zsigma
        Same outputs as `compute_equilibrium_charge_distribution_vectorized`.

    Notes
    -----
    This function performs simple, conservative unit conversions so that
    the internal vectorized solver receives arrays in the expected units:
    - energies (E) are in eV
    - frequencies (nu) are in Hz
    - spectral intensity J_nu is supplied as erg s^-1 m^-2 Hz^-1
    - absorption cross section C_abs is supplied in m^2

    The wrapper intentionally ignores ion capture rates (passes an empty
    ion_species list) because the current solver is configured to consider
    electrons only per your request.
    """
    # lazy imports from other modules in the repo to avoid top-level dependency issues
    from dust_photoelectric_heating import get_radiation_field, read_dielectric_file
    from dust_emission import interpolate_cross_sections

    # constants (SI)
    h_SI = 6.62607015e-34        # J s
    c_SI = 2.99792458e8          # m / s
    erg2J = 1e-7                 # 1 erg = 1e-7 J
    cm2_to_m2 = 1e4              # cm^-2 -> m^-2
    nm_to_m = 1e-9               # nm -> m

    # sensible defaults
    if yield_params is None:
        yield_params = {}
    material = yield_params.get('material', 'graphite' if grain_type.lower().startswith('gra') or grain_type.lower().startswith('car') else 'silicate')
    if debug: print(f'[debug] using material: {material} ({grain_type})')
    # 1) Radiation field: either provided or built from model
    if rad_field is None:
        rad0, _ = get_radiation_field(radiation_model)
    else:
        rad0 = np.asarray(rad_field)

    # Diagnostic: compute G0 from the provided/built radiation field and compare
    if debug:
        G0_calc, power = compute_G0_from_rad_field(rad0)
        print('[equilibrium_charge_for_grain] Computed G0 from rad_field: {:.3f} (input G0={:.3f}), FUV power density = {:.3e} erg/s/cm^3'.format(G0_calc, G0, power))

    # If rad0 has two columns (wavelength, intensity per nm per sr) follow the
    # transformation used in `dust_photoelectric_heating` to produce
    # rad_field = [E_eV, wavelength_rev_nm, I_E (erg/s/cm^2/eV)]
    if rad0.ndim == 2 and rad0.shape[1] >= 2:
        # original: wavelength (nm) and wavelength_intensity (erg cm-2 s-1 nm-1 sr-1)
        wavelength_nm = rad0[:, 0].astype(float)
        wavelength_intensity = rad0[:, 1].astype(float)

        wav_nm_rev = wavelength_nm[::-1]         # nm
        wav_int_rev = wavelength_intensity[::-1] # erg / (s cm^2 nm sr)

        # (optional) compute E_eV for diagnostics / interpolation
        hc_eVnm = 1239.84193                      # eV·nm
        E_eV = hc_eVnm / wav_nm_rev               # eV

        # 1) convert I_lambda (erg s^-1 cm^-2 nm^-1 sr^-1) -> I_lambda_SI (J s^-1 m^-2 m^-1 sr^-1)
        #    factor = erg->J * cm^-2->m^-2 * per-nm -> per-m
        I_lambda_SI_per_m = wav_int_rev * erg2J * cm2_to_m2 * (1.0 / nm_to_m)
        # note: (1/nm) -> (1/m) multiply by 1e9, so overall factor = 1e-3 * 1e9 = 1e6 -> same as wav_int_rev*1e6

        # 2) convert wavelength to meters
        lambda_m = wav_nm_rev * nm_to_m

        # 4) photon number flux per Hz per sr: n_nu = I_nu_energy / (h * nu)
        #    but nu = c / lambda, so combine: n_nu = I_nu_energy / (h * c / lambda) = I_nu_energy * lambda / (h * c)
        #    equivalently, simplifying earlier: n_nu = I_lambda_SI_per_m * lambda^3 / (h * c^2)
        n_nu_per_sr = I_lambda_SI_per_m * (lambda_m**3) / (h_SI * c_SI**2)

        # 5) integrate over hemisphere (4π sr) if original quantity was per sr
        n_nu = n_nu_per_sr * (4 * np.pi)    # photons s^-1 m^-2 Hz^-1

        # 6) apply scaling by G0 if you want to scale the field
        n_nu *= float(G0)   # if you intend to scale; earlier you used I_E *= G0

        # final arrays:
        # nu (Hz)
        nu = (c_SI / lambda_m)
        # J_nu photons: n_nu (photons / s / m^2 / Hz)
        J_nu = n_nu

        # For absorption cross section interpolation we still need E_eV and wav (micron)
        E_for_interp = E_eV
        wav_nm_for_interp = wav_nm_rev
    else:
        raise ValueError('rad_field must be an array with at least two columns (wavelength_nm, intensity)')

    # 4) absorption cross section: interpolate compute_cross_sections output onto E grid
    # compute_cross_sections returns (a0, wav_cm, C_sca_cm2, C_abs_cm2, C_rp_cm2)
    _, wav_cs, _, C_abs_cs, _ = interpolate_cross_sections(material, a_micron)
    # wav_cs is in cm; convert to energy in eV for interpolation
    optical_E = 1.2398 / (wav_cs * 1e4)  # wav_cs(cm) *1e4 -> microns
    # interpolate C_abs to E_for_interp
    C_abs_interp_cm2 = np.interp(E_for_interp, optical_E, C_abs_cs)
    # convert to m^2
    # if a_micron <= 1e-2 and material == 'graphite':
    #     C_abs_interp_cm2 = 2. * C_abs_interp_cm2
    C_abs_interp_m2 = C_abs_interp_cm2 * 1e-4

    # 5) prepare yield function parameters: pass material and dielectric if possible
    # read dielectric files and interpolate imaginary parts to wav grid used by yields
    if material == 'graphite':
        data_perp = read_dielectric_file('draine_lee_1984/callindex.out_CpeD03_0.10')
        data_par = read_dielectric_file('draine_lee_1984/callindex.out_CpaD03_0.10')
        # interpolate to wavelengths (wav_nm_for_interp -> nm)
        # compute wav in microns expected by the dielectric tables
        wav_micron = wav_nm_for_interp * 1e-3
        Im_perp = np.interp(wav_micron, data_perp['table']['wavelength_um'][::-1], data_perp['table']['Im_n'][::-1])
        Im_par = np.interp(wav_micron, data_par['table']['wavelength_um'][::-1], data_par['table']['Im_n'][::-1])
        yield_params_local = dict(yield_params)
        yield_params_local.update({'W': graphite_work_function, 'le': electron_escape_length, 'Imperp': Im_perp, 'Impar': Im_par, 'wav': wav_nm_for_interp})
    else:
        data = read_dielectric_file('draine_lee_1984/eps_suvSil')
        wav_micron = wav_nm_for_interp * 1e-3
        Im = np.interp(wav_micron, data['table']['wavelength_um'][::-1], data['table']['Im_n'][::-1])
        yield_params_local = dict(yield_params)
        yield_params_local.update({'W': silicate_work_function, 'le': electron_escape_length, 'Im': Im, 'wav': wav_nm_for_interp})

    # 6) run equilibrium solver
    if ion_species is None:
        ion_species = []
    a_m = float(a_micron) * 1e-6

    # Default cache budget is left to RpeCache default unless specified in yield_params
    max_cache_bytes = yield_params_local.get('max_cache_bytes', None)

    # Choose yield function based on material
    if material == 'graphite':
        yield_func = yield_graphite_vectorized
    else:
        yield_func = yield_silicate_vectorized

    yield_params_local['material'] = material

    Zs, P, rates, Zmean, Zsigma = compute_equilibrium_charge_distribution_vectorized(
        a_m, ne, T, ion_species, nu, J_nu, C_abs_interp_m2,
        yield_func=yield_func, yield_params=yield_params_local, Z_start=Z_start,
        debug=debug, max_cache_bytes=max_cache_bytes
    )

    return Zs, P, rates, Zmean, Zsigma


def heating_and_cooling_for_tuple(G0, ne, T, grain_type, a_micron,
                                  radiation_model='Mathis', yield_params=None,
                                  ion_species=None, Z_start=0, debug=False):
    """
    Compute photoelectric heating and recombination cooling powers for a single
    tuple of (G0, ne, T) for a given grain size and material.

    Parameters
    ----------
    G0 : float
        Radiation field scaling factor (dimensionless).
    ne : float
        Electron density in cm^-3.
    T : float
        Electron temperature in K.
    grain_type : str
        'graphite' or 'silicate'.
    a_micron : float
        Grain radius in microns.
    radiation_model : str
        Passed to the radiation-field builder if needed (default 'Mathis').
    yield_params : dict or None
        Forwarded to the equilibrium solver / yield functions.
    ion_species : list or None
        Ion species passed to the equilibrium solver (optional).
    Z_start : int
        Initial centre for the Zref search.
    debug : bool
        Verbose debug output.

    Returns
    -------
    dict
        Dictionary with keys:
          - 'Gamma_total' : total photoelectric heating (erg/s)
          - 'Recomb_total' : total recombination cooling (erg/s)
          - 'Autoionisation_cooling' : autoionisation cooling (erg/s)
          - 'E_abs' : absorbed power (erg/s)
          - 'efficiency' : heating efficiency (dimensionless)
          - 'Zmean', 'Zsigma' : charge distribution moments
          - 'Zs', 'P' : full distribution (arrays)
    """
    # reuse the high-level equilibrium wrapper which builds radiation and optics
    Zs, P, rates, Zmean, Zsigma = equilibrium_charge_for_grain(
        G0, ne, T, grain_type, a_micron,
        radiation_model=radiation_model,
        rad_field=None,
        yield_params=yield_params,
        ion_species=ion_species,
        Z_start=Z_start,
        debug=debug
    )

    out = {
        'Gamma_total': float(rates.get('Gamma_total', 0.0)),
        'Recomb_total': float(rates.get('Recomb_total', 0.0)),
        'Autoionisation_cooling': float(rates.get('Autoionisation_cooling', 0.0)),
        'E_abs': float(rates.get('E_abs', 0.0)),
        'efficiency': float(rates.get('efficiency', 0.0)),
        'Zmean': float(Zmean),
        'Zsigma': float(Zsigma),
        'Zs': Zs,
        'P': P,
    }
    return out


def compute_G0_from_rad_field(rad_field, E_min=6.0, E_max=13.6):
    """
    Integrate a radiation field between E_min and E_max (eV) and return G0.

    Parameters
    ----------
    rad_field : ndarray
        Array with at least two columns: [wavelength_nm, intensity (erg/cm^2/s/nm/sr)].
    E_min, E_max : float
        Energy integration limits in eV.

    Returns
    -------
    G0 : float
        The integrated flux between E_min and E_max in units of the Draine
        field, i.e. divided by 1.68e-6 W/m^2.
    power_W_m2 : float
        Integrated power in W/m^2 between E_min and E_max (hemisphere, 2π sr).
    """
    # constants
    h_SI = 6.62607015e-34
    c_SI = 2.99792458e8
    eV2J = 1.602176634e-19

    rad = np.asarray(rad_field)
    if rad.ndim != 2 or rad.shape[1] < 2:
        raise ValueError('rad_field must be shape (N,2+) with columns [wavelength_nm, intensity] or [E_eV, wavelength_nm, I_E]')

    # Case A: rad has 3 columns and first column covers energy range -> assume [E_eV, wav_nm, I_E (erg/s/cm^2/eV)]
    if rad.shape[1] >= 3 and np.nanmin(rad[:, 0]) > 0 and np.nanmax(rad[:, 0]) > E_min:
        E = rad[:, 0].astype(float)
        wav_nm_sorted = rad[:, 1].astype(float)
        I_E = rad[:, 2].astype(float)  # erg / s / cm^2 / eV (may be per sr already)
        # Assume I_E is already per surface (if it was per sr, earlier code often multiplies by 2π)
        # We'll integrate I_E over E and multiply by 1 if it's full-surface. If I_E was per sr, this will undercount.
        I_E_erg_per_s_cm2_eV = I_E
    else:
        # Case B: rad has columns [wavelength_nm, intensity erg/cm^2/s/nm/sr]
        wavelength_nm = rad[:, 0].astype(float)
        wavelength_intensity = rad[:, 1].astype(float)  # erg / cm^2 / s / nm / sr

        # photon energy in eV and sort by increasing energy
        E = 1.2398 / (wavelength_nm * 1e-3)
        idx = np.argsort(E)
        E = E[idx]
        I_lambda = wavelength_intensity[idx]

        # d(lambda_nm)/dE_eV = (h*c)/(eV2J * E^2) * 1e9  (nm per eV)
        dlam_dE_nm_per_eV = (h_SI * c_SI) / (eV2J * (E**2)) * 1e9

        # I_E (erg / s / cm^2 / eV / sr) = I_lambda (erg / s / cm^2 / nm / sr) * dlambda_nm/dE_eV
        I_E_erg_per_s_cm2_eV_per_sr = I_lambda * dlam_dE_nm_per_eV

        # integrate over hemisphere (2π sr)
        I_E_erg_per_s_cm2_eV = I_E_erg_per_s_cm2_eV_per_sr * (4.0 * np.pi)

    # select energy range
    mask = (E >= E_min) & (E <= E_max)
    if not np.any(mask):
        return 0.0, 0.0

    # integrate I_E over E to get erg / s / cm^2
    integral_erg_per_s_cm2 = np.trapz(I_E_erg_per_s_cm2_eV[mask], E[mask])

    # convert to W / m^2: erg -> J (1e-7); per cm^2 -> per m^2 (1e4) => factor 1e-3
    power_W_m2 = integral_erg_per_s_cm2 * 1e-3

    # G0 reference
    G0_ref = 1.68e-6
    G0 = power_W_m2 / G0_ref

    return float(G0), float(power_W_m2)


def compute_G0_from_model(radiation_model='Draine', E_min=6.0, E_max=13.6):
    """Helper that obtains the radiation field via get_radiation_field and computes G0."""
    from dust_photoelectric_heating import get_radiation_field
    rad, _ = get_radiation_field(radiation_model)
    # earlier get_radiation_field returns [wavelength_nm, intensity erg/cm2/s/nm/sr]
    return compute_G0_from_rad_field(rad, E_min=E_min, E_max=E_max)


def compute_energy_density_from_rad_field(rad_field, E_min=6.0, E_max=13.6):
    """
    Compute the energy density (erg/cm^3) integrated between E_min and E_max.

    Assumptions:
      - If rad_field has 3+ columns and the first column looks like energy (eV),
        the third column is assumed to be I_E in erg/s/cm^2/eV (integrated over solid angle).
        In that case u = (1/c) * ∫ I_E dE.
      - If rad_field has 2 columns [wavelength_nm, I_lambda (erg/s/cm^2/nm/sr)],
        it is treated as per-steradian; we convert to I_E per eV per sr, then
        u = (4π / c) * ∫ I_E_per_sr dE.
    """
    c_cgs_local = 2.99792458e10  # cm/s
    rad = np.asarray(rad_field)
    if rad.ndim != 2 or rad.shape[1] < 2:
        raise ValueError('rad_field must have at least 2 columns')

    if rad.shape[1] >= 3 and np.nanmin(rad[:, 0]) > 0 and np.nanmax(rad[:, 0]) > E_min:
        # [E_eV, wav_nm, I_E (erg/s/cm^2/eV)] per surface
        E = rad[:, 0].astype(float)
        I_E = rad[:, 2].astype(float)
        mask = (E >= E_min) & (E <= E_max)
        if not np.any(mask):
            return 0.0
        integral_erg_per_s_cm2 = np.trapz(I_E[mask], E[mask])
        # energy density u = (1/c) * energy flux (erg s^-1 cm^-2) -> erg cm^-3
        u = integral_erg_per_s_cm2 / c_cgs_local
        return float(u)
    else:
        # [wavelength_nm, I_lambda (erg/s/cm^2/nm/sr)]
        wavelength_nm = rad[:, 0].astype(float)
        wavelength_intensity = rad[:, 1].astype(float)  # erg / cm^2 / s / nm / sr
        # photon energy E_eV
        E = 1.2398 / (wavelength_nm * 1e-3)
        idx = np.argsort(E)
        E_sorted = E[idx]
        I_lambda = wavelength_intensity[idx]
        # dlam/dE (nm per eV): using lambda_nm^2 / hc_eVnm
        hc_eVnm = 1239.84193
        wav_nm_sorted = wavelength_nm[idx]
        dlam_dE_nm_per_eV = (wav_nm_sorted**2) / hc_eVnm
        I_E_per_sr = I_lambda * dlam_dE_nm_per_eV
        mask = (E_sorted >= E_min) & (E_sorted <= E_max)
        if not np.any(mask):
            return 0.0
        integral_per_sr = np.trapz(I_E_per_sr[mask], E_sorted[mask])
        # multiply by 4π to get integrated per-surface, divide by c to get energy density
        u = (4.0 * np.pi * integral_per_sr) / c_cgs_local
        return float(u)


def compute_energy_density_from_nu_J(nu, J_nu, E_min=6.0, E_max=13.6):
    """
    Compute energy density (erg/cm^3) from nu [Hz] and J_nu [photons m^-2 s^-1 Hz^-1]
    integrated between E_min and E_max (eV).
    """
    nu = np.asarray(nu)
    J_nu = np.asarray(J_nu)
    h_SI = 6.62607015e-34
    eV2J = 1.602176634e-19
    c_SI = 2.99792458e8

    E_J = h_SI * nu
    E_eV = E_J / eV2J
    mask = (E_eV >= E_min) & (E_eV <= E_max)
    if not np.any(mask):
        return 0.0

    # energy flux S = ∫ J_nu * (h nu) dnu  [J s^-1 m^-2]
    S = np.trapz(J_nu[mask] * (h_SI * nu[mask]), nu[mask])

    # energy density u_SI = S / c [J m^-3]
    u_SI = S / c_SI

    # convert to erg/cm^3: 1 J/m^3 = 10 erg/cm^3
    u_cgs = u_SI * 10.0
    return float(u_cgs)


def compute_charge_vs_gamma(
    grain_type,
    a_micron,
    gamma_values,
    combos_per_gamma=5,
    T_samples=None,
    ne_samples=None,
    temp_bin_edges=None,
    radiation_model='Mathis',
    ion_species=None,
    yield_params=None,
    seed=0,
    savefile=None,
    debug=False,
    warn_on_single_charge=False,
    outlier_factor=1e3
):
    """
    Added optional behaviour: callers may provide `max_workers` to force the number
    of parallel processes, and `per_worker_tmp_limit` (bytes) to cap the amount of
    temporary memory each worker may attempt to allocate; when a worker's requested
    tmp exceeds the cap the compute path will fall back to streaming per-Z.
    """
    """
    For a given grain, scan a list of charging-parameter values
    gamma = G0 * sqrt(T) / ne and, for each gamma, evaluate multiple
    (G0, T, ne) combinations that produce the same gamma. For each
    combination the function computes the equilibrium charge distribution
    and returns (and optionally saves) scatter plots of Z_mean and Z_width
    versus gamma.

    Parameters
    ----------
    grain_type : str
        'graphite' or 'silicate'
    a_micron : float
        grain radius in microns
    gamma_values : array-like
        Values of gamma to scan (float)
    combos_per_gamma : int
        Number of (G0,T,ne) combinations to evaluate per gamma
    T_samples : array-like or None
        If provided, candidate T values (K) to sample from; otherwise a default
        log-spaced set is used.
    ne_samples : array-like or None
        If provided, candidate electron densities (cm^-3) to sample from; otherwise
        defaults are used.
    radiation_model, ion_species, yield_params : forwarded to equilibrium wrapper
    seed : int
        RNG seed for reproducibility
    savefile : str or None
        If provided, the plot is saved to this path; otherwise a default path is used
    debug : bool
        Forwarded to equilibrium computation

    Returns
    -------
    results : list of dict
        Each entry contains keys: gamma, G0, T, ne, Zmean, Zsigma
    fig : matplotlib.figure.Figure
        The generated figure (mean and width panels)
    """
    import matplotlib.pyplot as _plt
    import seaborn as sns
    sns.set_theme(style="white")
    # preserve previous usetex setting and apply local rc updates
    prev_usetex = _plt.rcParams.get('text.usetex', False)
    _plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    import cmasher as cmr

    rng = np.random.RandomState(seed)
    gamma_values = np.asarray(gamma_values, dtype=float)

    # default sample ranges (physically reasonable)
    if T_samples is None:
        T_samples = np.logspace(np.log10(10.0), np.log10(1e7), 100)
    else:
        T_samples = np.asarray(T_samples, dtype=float)
    if ne_samples is None:
        ne_samples = np.logspace(-4, 3, 100)  # cm^-3
    else:
        ne_samples = np.asarray(ne_samples, dtype=float)

    tasks = []
    for gamma in gamma_values:
        for i in range(combos_per_gamma):
            T = float(rng.choice(T_samples))
            ne = float(rng.choice(ne_samples))
            G0 = float(gamma * ne / np.sqrt(T))
            if G0 <= 0 or not np.isfinite(G0):
                continue
            tasks.append({'gamma': float(gamma), 'G0': G0, 'T': T, 'ne': ne,
                          'grain_type': grain_type, 'a_micron': a_micron,
                          'radiation_model': radiation_model,
                          'ion_species': ion_species, 'yield_params': yield_params,
                          'debug': debug,
                          'warn_on_single_charge': bool(warn_on_single_charge),
                          'outlier_factor': float(outlier_factor)})

    results = []
    # run tasks in parallel with ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    # memory-guarded worker selection: estimate available system memory and
    # set a conservative per-worker temporary budget, reduce worker count if necessary
    import multiprocessing as _mp
    total_mem = get_system_memory_bytes()
    proc_rss = get_process_rss_bytes()
    avail_mem = max(0, total_mem - proc_rss)
    cpu_count = _mp.cpu_count()
    # conservative fraction available for workers
    budget_for_workers = int(avail_mem * 0.6)

    # Determine a conservative per-worker temporary limit.
    # If caller provided a per_worker_tmp_limit in yield_params use it; otherwise
    # choose a small default on low-RAM machines to force streaming/serial work.
    per_worker_tmp_limit = None
    if isinstance(yield_params, dict):
        per_worker_tmp_limit = yield_params.get('per_worker_tmp_limit', None)
    try:
        if per_worker_tmp_limit is None:
            # default: 32 MiB on machines with <8 GiB RAM, otherwise 128 MiB
            if total_mem < (8 * 1024 ** 3):
                per_worker_tmp_limit = 32 * 1024 * 1024
            else:
                per_worker_tmp_limit = 128 * 1024 * 1024
        else:
            per_worker_tmp_limit = int(per_worker_tmp_limit)
    except Exception:
        per_worker_tmp_limit = 32 * 1024 * 1024

    # Determine number of workers allowed by memory. Respect caller-supplied max_workers
    supplied_max_workers = None
    if isinstance(yield_params, dict):
        supplied_max_workers = yield_params.get('max_workers', None)
    if supplied_max_workers is not None:
        try:
            n_workers = max(1, int(supplied_max_workers))
        except Exception:
            n_workers = max(1, cpu_count)
    else:
        # conservatively cap number of workers so each worker could use up to
        # per_worker_tmp_limit and leave some headroom (factor 3)
        max_by_mem = max(1, int(max(1, budget_for_workers) // (per_worker_tmp_limit * 3)))
        n_workers = max(1, min(cpu_count, max_by_mem))

    # Compute a starting per_worker_tmp based on budget and chosen workers, then clamp
    per_worker_tmp = max(2 * 1024 * 1024, budget_for_workers // max(1, n_workers))
    per_worker_tmp = min(per_worker_tmp, per_worker_tmp_limit)

    # If still likely to oversubscribe memory, reduce worker count conservatively
    try:
        if per_worker_tmp * n_workers * 3 > avail_mem and n_workers > 1:
            n_workers = max(1, int(avail_mem // (per_worker_tmp * 3)))
            n_workers = max(1, min(n_workers, cpu_count))
    except Exception:
        pass

    # attach per_worker_tmp to each task's yield_params so workers know the cap
    for t in tasks:
        yp = t.get('yield_params')
        if yp is None:
            yp = {}
        else:
            yp = dict(yp)
        yp.setdefault('max_tmp_bytes', per_worker_tmp)
        t['yield_params'] = yp

    if tqdm is not None:
        pbar = tqdm(total=len(tasks), desc='gamma-scan')
    else:
        pbar = None

    if debug:
        print(f'[compute_charge_vs_gamma] total_mem={(total_mem/(1024**3)):.2f}GB proc_rss={(proc_rss/(1024**2)):.2f}MB avail={(avail_mem/(1024**2)):.2f}MB n_workers={n_workers} per_worker_tmp={(per_worker_tmp/(1024**2)):.2f}MB')

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_compute_single_combo, t): t for t in tasks}
        worker_ru_after_list = []
        worker_ru_before_list = []
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                results.append(res)
                # collect per-worker reported memory stats if present
                if isinstance(res, dict):
                    if 'worker_ru_before' in res:
                        worker_ru_before_list.append(int(res.get('worker_ru_before', 0)))
                    if 'worker_ru_after' in res:
                        worker_ru_after_list.append(int(res.get('worker_ru_after', 0)))
            completed += 1
            if pbar is not None:
                pbar.update(1)
            else:
                # lightweight manual progress
                if len(results) % 10 == 0:
                    print(f'Processed {len(results)}/{len(tasks)} tasks')
        if pbar is not None:
            pbar.close()

    # memory summary: parent process RSS and per-worker stats (if workers reported)
    parent_ru = get_process_rss_bytes()
    try:
        parent_ru_mb = parent_ru / (1024.0 ** 2)
    except Exception:
        parent_ru_mb = float(parent_ru)
    if len(worker_ru_after_list) > 0:
        # worker values are bytes (as returned by worker using same helper)
        worker_after_mb = [v / (1024.0 ** 2) for v in worker_ru_after_list]
        worker_before_mb = [v / (1024.0 ** 2) for v in worker_ru_before_list] if worker_ru_before_list else None
        avg_mb = sum(worker_after_mb) / len(worker_after_mb)
        min_mb = min(worker_after_mb)
        max_mb = max(worker_after_mb)
        print(f'[memory summary] parent RSS = {parent_ru_mb:.2f} MB; workers reported (N={len(worker_after_mb)}) avg={avg_mb:.2f} MB min={min_mb:.2f} MB max={max_mb:.2f} MB')
        if worker_before_mb is not None:
            print(f"[memory summary] workers before-run avg={sum(worker_before_mb)/len(worker_before_mb):.2f} MB")
    else:
        print(f'[memory summary] parent RSS = {parent_ru_mb:.2f} MB; no per-worker stats reported')

    if len(results) == 0:
        _plt.rcParams['text.usetex'] = prev_usetex
        raise RuntimeError('No valid combinations evaluated for given gamma_values')

    # Prepare arrays for plotting
    gam_arr = np.array([r['gamma'] for r in results])
    Zmean_arr = np.array([r['Zmean'] for r in results])
    Zsig_arr = np.array([r['Zsigma'] for r in results])
    T_arr = np.array([r['T'] for r in results])
    ne_arr = np.array([r['ne'] for r in results])

    fig, axes = _plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # encode both T and ne: color -> log10(T), marker size -> n_e
    logT = np.log10(np.maximum(T_arr, 1e-30))
    logne = np.log10(np.maximum(ne_arr, 1e-30))
    # marker sizes mapping
    size_min, size_max = 20, 200
    if np.isfinite(logne).all() and (logne.max() - logne.min()) > 1e-12:
        sizes = size_min + (logne - logne.min()) / (logne.max() - logne.min()) * (size_max - size_min)
    else:
        sizes = np.full_like(logne, (size_min + size_max) / 2.0)
    cmap = cmr.gem
    sc1 = axes[0].scatter(gam_arr, Zmean_arr, c=logT, cmap=cmap, s=sizes, alpha=0.9)
    axes[0].set_ylim([Zmean_arr.min() - 10.0, Zmean_arr.max() + 30.0])
    axes[0].set_xlim([0.5*gam_arr.min(), 2*gam_arr.max()])
    sc2 = axes[1].scatter(gam_arr, Zsig_arr, c=logT, cmap=cmap, s=sizes, alpha=0.9)
    axes[1].set_ylim([0.0, Zsig_arr.max() * 1.1])
    axes[1].set_xlim([0.5*gam_arr.min(), 2*gam_arr.max()])

    # Overplot running median of <Z> for low-temperature points (T < 100 K)
    try:
        # If the caller provided temperature bin edges, compute medians in
        # two-dimensional bins (log10(gamma) x log10(T)). Save a file with
        # rows: log10(gamma_center), log10(T_center), median_Zmean, median_Zsigma, count
        if temp_bin_edges is not None:
            t_edges = np.asarray(temp_bin_edges, dtype=float)
            if t_edges.ndim != 1 or t_edges.size < 2:
                raise ValueError('temp_bin_edges must be a 1D array of bin edges (K)')
            # select valid entries
            mask_valid = np.isfinite(gam_arr) & np.isfinite(Zmean_arr) & np.isfinite(Zsig_arr) & np.isfinite(T_arr)
            n_valid = int(np.sum(mask_valid))
            if n_valid >= 10:
                gamma_vals_all = gam_arr[mask_valid]
                uniq_gamma = np.unique(gamma_vals_all)
                n_vals = uniq_gamma.size
                # choose number of gamma windows proportional to unique gamma count
                n_windows = int(np.clip(max(10, n_vals // 3), 10, 200))
                log_min = np.log10(np.maximum(gamma_vals_all.min(), 1e-30))
                log_max = np.log10(np.maximum(gamma_vals_all.max(), 1e-30))
                gamma_centers = np.logspace(log_min, log_max, n_windows)

                # Build log-space bin edges from centers (midpoints in log-space)
                log_centers = np.log10(gamma_centers)
                if n_windows > 1:
                    dlog = np.diff(log_centers)
                    log_edges = np.empty(n_windows + 1, dtype=float)
                    log_edges[1:-1] = log_centers[:-1] + dlog / 2.0
                    log_edges[0] = log_centers[0] - dlog[0] / 2.0
                    log_edges[-1] = log_centers[-1] + dlog[-1] / 2.0
                else:
                    delta = 0.5
                    log_edges = np.array([log_centers[0] - delta, log_centers[0] + delta])

                log_gamma_all = np.log10(np.maximum(gamma_vals_all, 1e-30))
                gamma_bin_idx = np.digitize(log_gamma_all, log_edges) - 1

                # Digitize temperatures using provided edges
                T_vals_all = T_arr[mask_valid]
                T_bin_idx = np.digitize(T_vals_all, t_edges) - 1
                n_Tbins = t_edges.size - 1

                # Build 2D matrices: rows -> gamma centers, columns -> temperature bins
                T_centers = np.empty(n_Tbins, dtype=float)
                for jt in range(n_Tbins):
                    t0 = float(t_edges[jt]); t1 = float(t_edges[jt+1])
                    if t0 > 0 and t1 > 0:
                        T_centers[jt] = np.sqrt(t0 * t1)
                    else:
                        T_centers[jt] = 10.0 ** ((np.log10(max(t0, 1e-30)) + np.log10(max(t1, 1e-30))) / 2.0)

                medianZ_mat = np.full((n_windows, n_Tbins), np.nan, dtype=float)
                medianSig_mat = np.full((n_windows, n_Tbins), np.nan, dtype=float)
                counts_mat = np.zeros((n_windows, n_Tbins), dtype=int)

                Zmean_valid = Zmean_arr[mask_valid]
                Zsig_valid = Zsig_arr[mask_valid]
                for ig in range(n_windows):
                    for jt in range(n_Tbins):
                        sel = (gamma_bin_idx == ig) & (T_bin_idx == jt)
                        if np.any(sel):
                            Zsel = Zmean_valid[sel]
                            Zsigsel = Zsig_valid[sel]
                            counts_mat[ig, jt] = int(sel.sum())
                            medianZ_mat[ig, jt] = float(np.median(Zsel))
                            medianSig_mat[ig, jt] = float(np.median(Zsigsel))

                save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dust_charge_distributions')
                os.makedirs(save_dir, exist_ok=True)
                mat = 'Gra' if 'graphite' in grain_type.lower() else 'suvSil'

                # Prepare matrix files: first column is log10(gamma_center),
                # subsequent columns correspond to temperature bin centers
                gamma_col_log = np.log10(np.asarray(gamma_centers, dtype=float))
                gamma_col = np.asarray(gamma_centers, dtype=float)  # linear gamma for plotting
                out_Z = np.column_stack([gamma_col_log, medianZ_mat])
                out_sig = np.column_stack([gamma_col_log, medianSig_mat])

                # Header listing log10 temperature centers for columns (after first gamma column)
                T_header = ' '.join([f"{np.log10(t):.6e}" for t in T_centers])
                header_line = f"T_centers_log10(K):\n{T_header}\nColumns: log10(gamma_center), then median values for each T bin from left to right"

                fn_Z = os.path.join(save_dir, f"dust_charge_Z_vs_T_{a_micron:.4f}_micron_{mat}.dat")
                fn_sig = os.path.join(save_dir, f"dust_charge_sigma_vs_T_{a_micron:.4f}_micron_{mat}.dat")
                # Save files with explicit header formatting (log10 gamma and log10 T centers)
                # First line: commented description
                # Second line: n_gamma n_T
                # Third line: commented label for T centers
                # Fourth line: space-separated log10(T_centers)
                # Fifth line: commented Columns description
                with open(fn_Z, 'w') as fh:
                    fh.write('# ngamma, nT\n')
                    fh.write(f"{n_windows} {n_Tbins}\n")
                    fh.write('# T_centers_log10(K): \n')
                    fh.write(' '.join([f"{np.log10(t):.6e}" for t in T_centers]) + '\n')
                    fh.write('# gamma_centers_log10(K**0.5 cm**-3):\n')
                    fh.write(' '.join([f"{g:.6e}" for g in gamma_col_log]) + '\n')
                    fh.write('# Columns: median values for each T bin from left to right\n')
                    for i in range(n_windows):
                        row_vals = [f"{medianZ_mat[i, j]:.6e}" if np.isfinite(medianZ_mat[i, j]) else f"{np.nan:.6e}" for j in range(n_Tbins)]
                        fh.write(' '.join(row_vals) + '\n')

                with open(fn_sig, 'w') as fh:
                    fh.write('# ngamma, nT\n')
                    fh.write(f"{n_windows} {n_Tbins}\n")
                    fh.write('# T_centers_log10(K): \n')
                    fh.write(' '.join([f"{np.log10(t):.6e}" for t in T_centers]) + '\n')
                    fh.write('# gamma_centers_log10(K**0.5 cm**-3):\n')
                    fh.write(' '.join([f"{g:.6e}" for g in gamma_col_log]) + '\n')
                    fh.write('# Columns: median values for each T bin from left to right\n')
                    for i in range(n_windows):
                        row_vals = [f"{medianSig_mat[i, j]:.6e}" if np.isfinite(medianSig_mat[i, j]) else f"{np.nan:.6e}" for j in range(n_Tbins)]
                        fh.write(' '.join(row_vals) + '\n')
                # Overlay lines for each temperature bin: one line per T center
                try:
                    # prefer cmasher if available, otherwise fallback to matplotlib colormap
                    try:
                        cmap = cmr.gem
                    except Exception:
                        cmap = _plt.get_cmap('viridis')
                    colors = cmap(np.linspace(0.0, 1.0, n_Tbins)) if callable(getattr(cmap, '__call__', None)) else [None] * n_Tbins
                except Exception:
                    colors = [None] * n_Tbins

                import matplotlib.patheffects as _pe
                for jt in range(n_Tbins):
                    y = medianZ_mat[:, jt]
                    ok = np.isfinite(y)
                    if np.any(ok):
                        ln = axes[0].plot(gamma_col[ok], y[ok], color=colors[jt] if colors[jt] is not None else None,
                                          lw=1.8, label=f'T={T_centers[jt]:.2e} K', alpha=0.95)[0]
                        try:
                            ln.set_path_effects([_pe.Stroke(linewidth=4, foreground='black'), _pe.Normal()])
                        except Exception:
                            pass
                    ysig = medianSig_mat[:, jt]
                    ok2 = np.isfinite(ysig)
                    if np.any(ok2):
                        ln2 = axes[1].plot(gamma_col[ok2], ysig[ok2], color=colors[jt] if colors[jt] is not None else None,
                                           lw=1.2, ls='--', alpha=0.9)[0]
                        try:
                            ln2.set_path_effects([_pe.Stroke(linewidth=3, foreground='black'), _pe.Normal()])
                        except Exception:
                            pass

                # Add legends if the number of T-bins is reasonable
                if n_Tbins <= 12:
                    axes[0].legend(loc='best', fontsize=10)
                    axes[1].legend(loc='best', fontsize=10)

                if debug:
                    print(f'[compute_charge_vs_gamma] saved Z matrix to {fn_Z} and sigma matrix to {fn_sig} (shape={out_Z.shape})')
            else:
                if debug:
                    print('[compute_charge_vs_gamma] not enough valid points to compute binned medians')

        # If temp_bin_edges not provided, fall back to the existing low/high running-median
        if temp_bin_edges is None:
            mask_lowT = (T_arr < 50.0) & np.isfinite(gam_arr) & np.isfinite(Zmean_arr)
            n_low = int(np.sum(mask_lowT))
            if n_low >= 10:
                gamma_low = gam_arr[mask_lowT]
                Z_low = Zmean_arr[mask_lowT]
                # build an adaptive set of sliding-window centers based on number
                # of low-T gamma samples. Instead of returning an array matching the
                # original gamma shape, we return one (gamma_center, median) pair
                # per sliding window.
                gamma_vals = gamma_low
                # use unique gamma values when choosing window count because
                # multiple (T, ne) combos map to the same gamma
                uniq_gamma = np.unique(gamma_vals)
                n_vals = uniq_gamma.size
                # choose number of windows proportional to sample size but bounded
                n_windows = 50 #int(np.clip(max(10, n_vals // 3), 10, 200))
                print(f'[compute_charge_vs_gamma] n_low={n_low} n_windows={n_windows}')
                log_min = np.log10(np.maximum(gamma_vals.min(), 1e-30))
                log_max = np.log10(np.maximum(gamma_vals.max(), 1e-30))
                gamma_centers = np.logspace(log_min, log_max, n_windows)

                # Build log-space bin edges from centers (midpoints in log-space)
                log_centers = np.log10(gamma_centers)
                if n_windows > 1:
                    dlog = np.diff(log_centers)
                    log_edges = np.empty(n_windows + 1, dtype=float)
                    log_edges[1:-1] = log_centers[:-1] + dlog / 2.0
                    log_edges[0] = log_centers[0] - dlog[0] / 2.0
                    log_edges[-1] = log_centers[-1] + dlog[-1] / 2.0
                else:
                    # single window: make a generous edge around the center
                    delta = 0.5
                    log_edges = np.array([log_centers[0] - delta, log_centers[0] + delta])

                log_gamma_low = np.log10(np.maximum(gamma_vals, 1e-30))
                # digitize into bins; bins are defined by log_edges
                bin_idx = np.digitize(log_gamma_low, log_edges) - 1

                medians_centers = []
                medians_sig_centers = []
                centers_out = []
                Zsig_low = Zsig_arr[mask_lowT]
                for ibin in range(n_windows):
                    sel = (bin_idx == ibin)
                    if np.any(sel):
                        centers_out.append(gamma_centers[ibin])
                        medians_centers.append(np.median(Z_low[sel]))
                        medians_sig_centers.append(np.median(Zsig_low[sel]))

                if len(centers_out) > 0:
                    centers_out = np.asarray(centers_out, dtype=float)
                    medians_centers = np.asarray(medians_centers, dtype=float)
                    medians_sig_centers = np.asarray(medians_sig_centers, dtype=float)
                    axes[0].plot(centers_out, medians_centers, color='limegreen', lw=2.5, ls='-', zorder=50)
                    axes[0].legend(loc='best', fontsize=12, frameon=False)
                    axes[1].plot(centers_out, medians_sig_centers, color='limegreen', lw=2.5, ls='-', label=r'median $\\sigma_Z$ ($T<10^4$ K)', zorder=50)
                    axes[1].legend(loc='best', fontsize=12, frameon=False)
                if debug:
                    print(f'[compute_charge_vs_gamma] running median plotted for {n_low} low-T points; n_windows={n_windows}')
                # Now compute running median for high-temperature points (T > 1e4 K)
                mask_highT = (T_arr > 5e4) & np.isfinite(gam_arr) & np.isfinite(Zmean_arr)
                n_high = int(np.sum(mask_highT))
                if n_high >= 10:
                    gamma_high = gam_arr[mask_highT]
                    Z_high = Zmean_arr[mask_highT]
                    log_min_h = np.log10(np.maximum(gamma_high.min(), 1e-30))
                    log_max_h = np.log10(np.maximum(gamma_high.max(), 1e-30))
                    # build adaptive sliding-window centers for high-T samples
                    gamma_vals_h = gamma_high
                    uniq_gamma_h = np.unique(gamma_vals_h)
                    n_vals_h = uniq_gamma_h.size
                    n_windows_h = 50 #int(np.clip(max(10, n_vals_h // 3), 10, 200))
                    print(f'[compute_charge_vs_gamma] n_high={n_high} n_windows_h={n_windows_h}')
                    gamma_centers_h = np.logspace(log_min_h, log_max_h, n_windows_h)

                    # Build log-space bin edges from centers
                    log_centers_h = np.log10(gamma_centers_h)
                    if n_windows_h > 1:
                        dlog_h = np.diff(log_centers_h)
                        log_edges_h = np.empty(n_windows_h + 1, dtype=float)
                        log_edges_h[1:-1] = log_centers_h[:-1] + dlog_h / 2.0
                        log_edges_h[0] = log_centers_h[0] - dlog_h[0] / 2.0
                        log_edges_h[-1] = log_centers_h[-1] + dlog_h[-1] / 2.0
                    else:
                        delta = 0.5
                        log_edges_h = np.array([log_centers_h[0] - delta, log_centers_h[0] + delta])

                    log_gamma_high = np.log10(np.maximum(gamma_vals_h, 1e-30))
                    bin_idx_h = np.digitize(log_gamma_high, log_edges_h) - 1

                    centers_h = []
                    med_high = []
                    med_high_sig = []
                    Zsig_high = Zsig_arr[mask_highT]
                    for ibin in range(n_windows_h):
                        sel = (bin_idx_h == ibin)
                        if np.any(sel):
                            centers_h.append(gamma_centers_h[ibin])
                            med_high.append(np.median(Z_high[sel]))
                            med_high_sig.append(np.median(Zsig_high[sel]))

                    if len(centers_h) > 0:
                        centers_h = np.asarray(centers_h, dtype=float)
                        med_high = np.asarray(med_high, dtype=float)
                        med_high_sig = np.asarray(med_high_sig, dtype=float)
                        axes[0].plot(centers_h, med_high, color='firebrick', lw=2.5, ls='-.', zorder=55)
                        axes[0].legend(loc='best', fontsize=12, frameon=False)
                        axes[1].plot(centers_h, med_high_sig, color='firebrick', lw=2.5, ls='-.', label=r'median $\\sigma_Z$ ($T>10^4$ K)', zorder=55)
                        axes[1].legend(loc='best', fontsize=12, frameon=False)
                    # Save median arrays to disk under dust_charge_distributions/
                    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dust_charge_distributions')
                    os.makedirs(save_dir, exist_ok=True)
                    # low-T saves: combine mean and sigma in one file
                    if n_low >= 10 and 'centers_out' in locals() and len(centers_out) > 0:
                        # map material to requested short names
                        mat = 'Gra' if 'graphite' in grain_type.lower() else 'suvSil'
                        fn_low = os.path.join(save_dir, f"dust_charge_lowT_{a_micron:.4f}_micron_{mat}.dat")
                        # only keep rows where both medians are finite
                        ok_both = np.isfinite(centers_out) & np.isfinite(medians_centers) & np.isfinite(medians_sig_centers)
                        data_both = np.column_stack([np.log10(centers_out[ok_both]), medians_centers[ok_both], medians_sig_centers[ok_both]])
                        with open(fn_low, 'w') as fh:
                            fh.write(f"{data_both.shape[0]}\n")
                            if data_both.shape[0] > 0:
                                np.savetxt(fh, data_both, fmt='%.6e')
                        if debug:
                            print(f'[compute_charge_vs_gamma] saved low-T combined medians to {fn_low} (rows={data_both.shape[0]})')
                    # high-T saves
                    if n_high >= 10 and 'centers_h' in locals() and len(centers_h) > 0:
                        mat = 'Gra' if 'graphite' in grain_type.lower() else 'suvSil'
                        fn_high = os.path.join(save_dir, f"dust_charge_highT_{a_micron:.4f}_micron_{mat}.dat")
                        ok_both_h = np.isfinite(centers_h) & np.isfinite(med_high) & np.isfinite(med_high_sig)
                        data_both_h = np.column_stack([np.log10(centers_h[ok_both_h]), med_high[ok_both_h], med_high_sig[ok_both_h]])
                        with open(fn_high, 'w') as fh:
                            fh.write(f"{data_both_h.shape[0]}\n")
                            if data_both_h.shape[0] > 0:
                                np.savetxt(fh, data_both_h, fmt='%.6e')
                        if debug:
                            print(f'[compute_charge_vs_gamma] saved high-T combined medians to {fn_high} (rows={data_both_h.shape[0]})')
                    if debug:
                        print(f'[compute_charge_vs_gamma] running median plotted for {n_high} high-T points; n_windows_h={n_windows_h}')

    except Exception:
        # non-fatal; continue silently if median computation fails
        if debug:
            print('[compute_charge_vs_gamma] running median computation failed')

    # axis scaling, labels and titles
    for ax in axes:
        ax.set_xscale('log')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in",labelsize=16)

    axes[0].set_ylabel(r'$\langle Z \rangle$', fontsize=16)
    axes[0].set_title(f"Mean grain charge vs $\gamma$ for a={a_micron:.4g} $\mu$m", fontsize=18)

    axes[1].set_xlabel(r'$\gamma = G_0 \sqrt{T} / n_e$', fontsize=16)
    axes[1].set_ylabel(r'$\sigma_Z$', fontsize=16)
    axes[1].set_title('Distribution width vs $\gamma$', fontsize=18)

    # Adjust the main plot area to leave space for the colorbar
    fig.subplots_adjust(right=0.83,left=0.105,top=0.95,bottom=0.07,hspace=0.1)

    # Create an axis for the colorbar: [left, bottom, width, height]
    # Values are in figure coordinates (0–1)
    cax = fig.add_axes([0.86, 0.1, 0.025, 0.8])  # right margin outside the plots

    # Draw the colorbar
    cb = fig.colorbar(sc1, cax=cax)
    cb.set_label(r'$\log_{10}(T/[\mathrm{K}])$', labelpad=10, fontsize=16)

    # Optional: make ticks and labels a bit cleaner
    cb.ax.tick_params(labelsize=16)

    # Precompute a radius label (Angstrom) to use for overlays so it's available
    # even if one of the overlay try-blocks fails.
    try:
        angstrom = float(a_micron) * 1e4
        allowed = np.array([3.5, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
        idx_closest = np.argmin(np.abs(allowed - angstrom))
        rval = allowed[idx_closest]
        radius_label = f"{int(rval)}A" if float(rval).is_integer() else f"{rval}A"
    except Exception:
        radius_label = '10A'

    # Overlay Ibanez-Mejias (2019) fit mean and sigma vs gamma if available
    try:
        import dust_model as _dm
        # pick a radius label closest to a_micron (convert to Angstrom)
        angstrom = float(a_micron) * 1e4
        allowed = np.array([3.5, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
        idx_closest = np.argmin(np.abs(allowed - angstrom))
        rval = allowed[idx_closest]
        if float(rval).is_integer():
            radius_label = f"{int(rval)}A"
        else:
            radius_label = f"{rval}A"

        gamma_grid = np.unique(np.sort(gam_arr))
        fit_mean = []
        fit_sigma = []
        for g in gamma_grid:
            try:
                m = _dm.grain_mean_charge(1.0, 1.0, 1.0, grain_type, radius_label, gamma=float(g))
                sigma_val = _dm.grain_charge_sigma(1.0, 1.0, 1.0, grain_type, radius_label, gamma=float(g))
            except Exception:
                m = np.nan
                sigma_val = np.nan
            fit_mean.append(m)
            fit_sigma.append(sigma_val)

        fit_mean = np.array(fit_mean)
        fit_sigma = np.array(fit_sigma)
        # plot fit curves on top
        axes[0].plot(gamma_grid, fit_mean, color='darkorange', lw=2, zorder=10)
        axes[1].plot(gamma_grid, fit_sigma, color='darkorange', lw=2, label='IM19 fit', zorder=10)
        # add small legend entry
        axes[1].legend(loc='best', fontsize=12, frameon=False)
    except Exception:
        # silently continue if overlay cannot be produced
        pass

    # size legend for n_e (pick three representative points)
    try:
        ne_ticks = np.array([10 ** v for v in np.linspace(logne.min(), logne.max(), 3)])
    except Exception:
        ne_ticks = np.array([ne_arr.min(), np.median(ne_arr), ne_arr.max()])
    size_handles = []
    for val in ne_ticks:
        lv = np.log10(max(val, 1e-30))
        if (logne.max() - logne.min()) > 1e-12:
            s = size_min + (lv - logne.min()) / (logne.max() - logne.min()) * (size_max - size_min)
        else:
            s = (size_min + size_max) / 2.0
        size_handles.append(axes[0].scatter([], [], c='k', alpha=0.6, s=s, label=r'$\log_{10}(n_e/[\mathrm{cm}^{-3}])=$'+f'{lv:.2g}'))
    axes[0].legend(handles=size_handles, fontsize=12, frameon=False)

    # Overlay theoretical most-negative allowed charge as horizontal line
    try:
        # a_micron is available in outer scope
        a_angstrom_plot = float(a_micron) * 1e4
        if 'graphite' in grain_type.lower():
            Zmin_allowed_plot = int(most_negative_allowed_charge_graphite(a_angstrom_plot))
        else:
            Zmin_allowed_plot = int(most_negative_allowed_charge_silicate(a_angstrom_plot))
        axes[0].axhline(Zmin_allowed_plot, color='k', linestyle='--', lw=1.5)
        axes[0].text(3e4, 0.8*Zmin_allowed_plot, r'$Z_{\rm min}=$'+f'{Zmin_allowed_plot}',
                     fontsize=12, verticalalignment='top', color='k')
        if debug:
            print(f'[compute_charge_vs_gamma] plotted theoretical Zmin_allowed={Zmin_allowed_plot} (a={a_micron} um)')
    except Exception as _ex:
        if debug:
            print('[compute_charge_vs_gamma] failed to compute/plot Zmin_allowed:', _ex)

    if savefile is None:
        savefile = f'examples/gamma_scan_{grain_type}_a{a_micron:.4g}um.pdf'
    try:
        fig.savefig(savefile, dpi=300, format='pdf')
    except Exception as e:
        print('[compute_charge_vs_gamma] failed to save figure:', e)

    return results, fig


def _compute_single_combo(task):
    """Top-level worker wrapper for multiprocessing.

    task: dict with keys 'gamma','G0','T','ne','grain_type','a_micron','radiation_model',
          'ion_species','yield_params','debug'
    returns: dict with added 'Zmean' and 'Zsigma' or None on failure
    """
    # memory guard: check that worker has reasonable tmp budget and system has capacity
    try:
        max_tmp = None
        yp = task.get('yield_params')
        if isinstance(yp, dict):
            max_tmp = yp.get('max_tmp_bytes', None)
        # get system available memory
        total_mem = get_system_memory_bytes()
        proc_rss = get_process_rss_bytes()
        avail_mem = max(0, total_mem - proc_rss)
        # snapshot worker ru before heavy work
        worker_ru_before = get_process_rss_bytes()
        if max_tmp is not None and avail_mem < max_tmp * 2:
            return dict(task, Zmean=None, Zsigma=None, error=f'Insufficient available memory: avail={avail_mem} < required~{max_tmp*2}')

        Zs, P, rates, Zmean, Zsigma = equilibrium_charge_for_grain(
            task['G0'], task['ne'], task['T'], task['grain_type'], task['a_micron'],
            radiation_model=task.get('radiation_model', 'Mathis'),
            ion_species=task.get('ion_species', []),
            yield_params=task.get('yield_params', None),
            Z_start=0, debug=task.get('debug', False)
        )
        # snapshot worker ru after
        worker_ru_after = get_process_rss_bytes()
        out = dict(task)
        out.update({'Zmean': Zmean, 'Zsigma': Zsigma})
        # include memory snapshots so parent can summarize
        out['worker_ru_before'] = int(worker_ru_before)
        out['worker_ru_after'] = int(worker_ru_after)
        # Optional: detect pathological single-charge domination
        try:
            warn_flag = bool(task.get('warn_on_single_charge', False))
            outlier_factor = float(task.get('outlier_factor', 1e3))
            if warn_flag and P is not None and np.any(np.isfinite(P)):
                P_arr = np.asarray(P, dtype=float)
                # ignore non-finite entries for statistics
                finite_mask = np.isfinite(P_arr)
                if np.any(finite_mask):
                    Pf = P_arr[finite_mask]
                    # identify peak index within full array
                    idx_peak = int(np.nanargmax(P_arr))
                    P_peak = float(P_arr[idx_peak])
                    # Compare peak probability to its immediate neighbors (left and right)
                    Np = P_arr.size
                    left_idx = idx_peak - 1 if (idx_peak - 1) >= 0 else None
                    right_idx = idx_peak + 1 if (idx_peak + 1) < Np else None
                    P_left = float(P_arr[left_idx]) if left_idx is not None else 0.0
                    P_right = float(P_arr[right_idx]) if right_idx is not None else 0.0
                    # small floor to avoid division by zero
                    eps = 1e-300
                    max_neighbor = max(P_left, P_right, eps)
                    ratio = P_peak / max_neighbor
                    if ratio >= outlier_factor:
                        # extract rates at the peak Z index where possible
                        peak_info = {
                            'peak_Z': int(Zs[idx_peak]),
                            'P_peak': P_peak,
                            'P_left': P_left,
                            'P_right': P_right,
                            'left_Z': int(Zs[left_idx]) if left_idx is not None else None,
                            'right_Z': int(Zs[right_idx]) if right_idx is not None else None,
                            'ratio_to_neighbors': ratio
                        }
                        try:
                            Rplus = np.asarray(rates.get('R_plus')) if rates.get('R_plus') is not None else None
                            Rminus = np.asarray(rates.get('R_minus')) if rates.get('R_minus') is not None else None
                            J_e = np.asarray(rates.get('J_e')) if rates.get('J_e') is not None else None
                            Gamma_Z = np.asarray(rates.get('Gamma_Z')) if rates.get('Gamma_Z') is not None else None
                            Recomb_Z = np.asarray(rates.get('Recomb_Z')) if rates.get('Recomb_Z') is not None else None
                        except Exception:
                            Rplus = Rminus = J_e = Gamma_Z = Recomb_Z = None
                        # grab scalar values if arrays available
                        try:
                            peak_rates = {
                                'R_plus_at_peak': float(Rplus[idx_peak]) if Rplus is not None and idx_peak < Rplus.size else None,
                                'R_minus_at_peak': float(Rminus[idx_peak]) if Rminus is not None and idx_peak < Rminus.size else None,
                                'J_e_at_peak': float(J_e[idx_peak]) if J_e is not None and idx_peak < J_e.size else None,
                                'Gamma_Z_at_peak': float(Gamma_Z[idx_peak]) if Gamma_Z is not None and idx_peak < Gamma_Z.size else None,
                                'Recomb_Z_at_peak': float(Recomb_Z[idx_peak]) if Recomb_Z is not None and idx_peak < Recomb_Z.size else None,
                                'E_abs': float(rates.get('E_abs')) if rates.get('E_abs') is not None else None,
                                'efficiency': float(rates.get('efficiency')) if rates.get('efficiency') is not None else None
                            }
                        except Exception:
                            peak_rates = {}
                        out['single_charge_issue'] = True
                        out['single_charge_details'] = {**peak_info, **peak_rates}
                        # print a debug-level warning if requested
                        print('[_compute_single_combo] WARNING: single-charge domination detected for task:', task)
                        print('  peak details:', out['single_charge_details'])
        except Exception:
            # non-fatal: do not let detection break worker
            pass
        return out
    except Exception as e:
        # return dict with error info
        return dict(task, Zmean=None, Zsigma=None, error=str(e))

# -----------------------------
# Corrected model: T-dep fades with gamma
# -----------------------------
def mean_charge_model_fade(gamma, T, k, h, g, d, b, beta, gamma_T, T0=100.0):
    """
    Extended Ibanez-Mejia form with temperature dependence that fades with gamma:
      <Z> = [k (1 - exp(-gamma/h)) gamma^g + d] * [1 - b * exp(-gamma/gamma_T) * (T/T0)^beta]
    """
    gamma = np.asarray(gamma, dtype=float)
    T = np.asarray(T, dtype=float)
    base = k * (1.0 - np.exp(-gamma / h)) * np.power(gamma, g) + d
    fade = np.exp(-gamma / gamma_T)
    temp_factor = 1.0 - b * fade * np.power(T / T0, beta)
    return base * temp_factor

# -----------------------------
# Wrapper for curve_fit (flatten x)
# -----------------------------
def model_flat(X, k, h, g, d, b, beta, gamma_T):
    gamma, T = X
    return mean_charge_model_fade(gamma, T, k, h, g, d, b, beta, gamma_T, T0=100.0)

# -----------------------------
# Smart initial guess function
# -----------------------------
def guess_params(gamma, T, Z):
    gamma = np.asarray(gamma)
    T = np.asarray(T)
    Z = np.asarray(Z)
    # Prefer to compute the Ibanez-Mejía style initial guess from
    # low-temperature datapoints (T < 100 K) when available.
    try:
        mask_lowT = (T < 100.0) & np.isfinite(gamma) & np.isfinite(Z)
        if np.sum(mask_lowT) >= 3:
            gamma_sub = gamma[mask_lowT]
            Z_sub = Z[mask_lowT]
        else:
            gamma_sub = gamma
            Z_sub = Z
    except Exception:
        gamma_sub = gamma
        Z_sub = Z

    # base amplitude k ~ high-gamma plateau minus offset
    # estimate plateau by taking median of top 10% gamma points in the chosen subset
    try:
        mask_high = gamma_sub >= np.percentile(gamma_sub, 90)
        if np.any(mask_high):
            plateau = np.median(Z_sub[mask_high])
        else:
            plateau = np.median(Z_sub)
    except Exception:
        plateau = np.median(Z_sub) if Z_sub.size>0 else 0.0
    d0 = np.min(Z_sub) if Z_sub.size>0 else np.min(Z)
    k0 = max(plateau - d0, 1e-6)

    # h: turnover scale — use median gamma where signal rises (from subset)
    finite_gamma = gamma_sub[gamma_sub>0]
    h0 = np.median(finite_gamma) if finite_gamma.size>0 else 1.0

    # g: slope at high gamma; estimate by log-log fit on high-gamma region
    try:
        mask_tail = (gamma_sub > np.percentile(finite_gamma, 70)) if finite_gamma.size>0 else (gamma_sub>0)
        X = np.log(gamma_sub[mask_tail])
        Y = np.log(np.maximum(Z_sub[mask_tail] - d0, 1e-12))
        if len(X) >= 3:
            g0, lnA = np.polyfit(X, Y, 1)
        else:
            g0 = 0.3
    except Exception:
        g0 = 0.3

    # temperature-dependent params:
    # b: amplitude of T-effect; small positive number (we use +0.3 as default)
    b0 = 0.3
    # beta: how steep T enters (power-law); start ~0.5
    beta0 = 0.5
    # gamma_T: gamma scale where T-effect fades; start at ~median gamma
    gamma_T0 = np.median(finite_gamma) if finite_gamma.size>0 else 1.0

    return [k0, h0, g0, d0, b0, beta0, gamma_T0]

# -----------------------------
# Fit routine
# -----------------------------
def fit_mean_charge(gamma_data, T_data, Z_data, p0=None, bounds=None, verbose=False):
    gamma = np.asarray(gamma_data, dtype=float)
    T = np.asarray(T_data, dtype=float)
    Z = np.asarray(Z_data, dtype=float)

    if p0 is None:
        p0 = guess_params(gamma, T, Z)

    # sensible bounds to avoid unphysical regions
    if bounds is None:
        lower = [0.0, 1e-8, -2.0, -np.inf, -np.inf, -2.0, 1e-6]
        upper = [np.inf, np.inf, 5.0, np.inf, np.inf, 5.0, np.inf]
        bounds = (lower, upper)

    popt, pcov = curve_fit(model_flat, (gamma, T), Z, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    if verbose:
        names = ["k","h","g","d","b","beta","gamma_T"]
        for n,v,e in zip(names,popt,perr):
            print(f"{n:>7s} = {v:.4g} ± {e:.4g}")
    return popt, perr, pcov