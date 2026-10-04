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
from pycalima.models.grain_size_config import get_optical_props_path
from pycalima.models.dust_charge import wdb06_yields
from pycalima.models.dust_charge.shared_physics import (
    GRAPHITE_WORK_FUNCTION,
    SILICATE_WORK_FUNCTION,
    SILICATE_BAND_GAP,
    ELECTRON_ESCAPE_LENGTH_CM,
    ME_CGS,
    H_CGS,
    C_CGS,
    KB_CGS,
    EV2ERG,
    E_STATC,
    TINY,
    _coulomb_energy_over_a,
    ionisation_potential_valence_vec,
    electron_affinity_graphite_vec,
    electron_affinity_silicate_vec,
    min_energy_ejection_vec,
    photodetachment_energy_graphite_vec,
    photodetachment_energy_silicate_vec,
    photodetachment_cross_section_vec,
    min_photon_energy_vec,
    parameter_theta_vec,
    escape_fraction_attempting_electrons_vec,
    photon_attenuation_length_graphite_vec,
    photon_attenuation_length_silicate_vec,
    Watson73_y1_vec,
    BT94_y0_graphite_vec,
    y0_silicate_vec,
    DS87_J_function_vec,
    autoionisation_potential_graphite,
    autoionisation_potential_silicate,
    most_negative_allowed_charge_graphite,
    most_negative_allowed_charge_silicate,
    most_positive_allowed_charge,
    electron_sticking_coefficient_graphite,
    electron_sticking_coefficient_silicate,
    electron_sticking_coefficient_graphite_scalar,
    electron_sticking_coefficient_silicate_scalar,
    collisional_rates_electrons_scalar,
    collisional_rates_ions_scalar,
    collisional_rates_electrons_vector,
    collisional_rates_ions_vector,
)
from pycalima import _paths
from pycalima.models.grain_size_config import get_model_data_dir
from pycalima.plotting_style import use_calima_style

PATH_OPTICS = str(get_optical_props_path())
_EXTERNAL_DATA_DIR = str(_paths.get_external_data_path())


def dust_charging_output_dir(*subdirs):
    """Directory for generated dust-charging tables.

    Resolved on every call, not at import: it depends on $CALIMA_DATA and on
    the active configuration's model_name. The previous module-level
    ``os.makedirs`` ran at *import* time and raised PermissionError on a
    read-only install prefix, taking down the whole dust_charge subpackage.
    """
    return get_model_data_dir().joinpath('dust_charging_data', *subdirs)


def _dust_charging_output_path(filename, *subdirs):
    out_dir = str(dust_charging_output_dir(*subdirs))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

# CONSTANTS
graphite_work_function = GRAPHITE_WORK_FUNCTION  # [eV]
silicate_work_function = SILICATE_WORK_FUNCTION  # [eV]
me = ME_CGS  # Electron mass [g]
h_cgs = H_CGS  # Planck constant [erg s]
c_cgs = C_CGS  # Speed of light [cm/s]
kb_cgs = KB_CGS  # Boltzmann constant [erg/K]
silicate_band_gap = SILICATE_BAND_GAP  # [eV]
electron_escape_length = ELECTRON_ESCAPE_LENGTH_CM  # [cm]
eV2erg = EV2ERG  # Conversion factor from eV to erg
_e_statC = E_STATC  # statcoulomb

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
_TINY = TINY
_INF_RATIO = 1e300


# FUNCTIONS
## Low-level cgs helper physics is imported from shared_physics.

def photoelectric_yield_graphite_vec(W, Zs, a, le, E, wav, Imperp, Impar, la=None):
    """
    Vectorized version of photoelectric_yield_graphite for all (E, Z).
    E and wav are 1D arrays [N_E]; Zs is 1D [N_Z].
    Returns Y(E,Z) array of shape [N_E, N_Z].
    """
    E = np.asarray(E, dtype=float)[:, None]
    Z = np.asarray(Zs, dtype=float)[None, :]

    IPV = ionisation_potential_valence_vec(W, Z, a)
    Emin_ej = min_photon_energy_vec(IPV, Z, a)

    mask = E >= Emin_ej

    coul = _coulomb_energy_over_a(Z, a)
    theta = parameter_theta_vec(E, Emin_ej, Z, a, coulomb_over_a=coul)
    y0 = BT94_y0_graphite_vec(theta, W)

    if la is None:
        la = photon_attenuation_length_graphite_vec(wav[:, None], Imperp[:, None], Impar[:, None])
    else:
        la = np.asarray(la, dtype=float)
        if la.ndim == 1:
            la = la[:, None]
    y1 = Watson73_y1_vec(a, la, le)

    y2 = escape_fraction_attempting_electrons_vec(E, Emin_ej, Z, a, coulomb_over_a=coul)

    Y = np.where(mask, y2 * np.minimum(y0 * y1, 1.0), 0.0)
    return Y

def photoelectric_yield_silicate_vec(W, Zs, a, le, E, wav, Im, la=None):
    """
    Vectorized version of photoelectric_yield_silicate.
    Returns Y(E,Z) array [N_E, N_Z].
    """
    E = np.asarray(E, dtype=float)[:, None]
    Z = np.asarray(Zs, dtype=float)[None, :]

    IPV = ionisation_potential_valence_vec(W, Z, a)
    Emin_ej = min_photon_energy_vec(IPV, Z, a)
    mask = E >= Emin_ej

    coul = _coulomb_energy_over_a(Z, a)
    theta = parameter_theta_vec(E, Emin_ej, Z, a, coulomb_over_a=coul)
    y0 = y0_silicate_vec(theta, W)

    if la is None:
        la = photon_attenuation_length_silicate_vec(wav[:, None], Im[:, None])
    else:
        la = np.asarray(la, dtype=float)
        if la.ndim == 1:
            la = la[:, None]
    y1 = Watson73_y1_vec(a, la, le)

    y2 = escape_fraction_attempting_electrons_vec(E, Emin_ej, Z, a, coulomb_over_a=coul)

    Y = np.where(mask, y2 * np.minimum(y0 * y1, 1.0), 0.0)
    return Y

# Yield adapters returning Y(nu,Z) for R_pe integration
def yield_graphite_vectorized(nu, Zs, a, params):
    W = params['W']
    le = params['le']
    wav = params.get('wav', c_cgs / np.asarray(nu))
    E = params.get('E_eV', h_cgs * np.asarray(nu) / eV2erg)
    Imperp = params['Imperp']
    Impar  = params['Impar']
    la = params.get('la', None)
    return photoelectric_yield_graphite_vec(W, np.asarray(Zs), a, le, E, wav, Imperp, Impar, la=la)

def yield_silicate_vectorized(nu, Zs, a, params):
    W = params['W']
    le = params['le']
    wav = params.get('wav', c_cgs / np.asarray(nu))
    Im  = params['Im']
    E = params.get('E_eV', h_cgs * np.asarray(nu) / eV2erg)
    la = params.get('la', None)
    return photoelectric_yield_silicate_vec(W, np.asarray(Zs), a, le, E, wav, Im, la=la)

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

    material = 'graphite'
    if isinstance(yield_params, dict):
        material = yield_params.get('material', 'graphite')
    pd_scale = 4.0 * np.pi if J_is_per_sr else 1.0

    E_eV = None
    if isinstance(yield_params, dict) and 'E_eV' in yield_params:
        try:
            E_eV = np.asarray(yield_params['E_eV'], dtype=float)
        except Exception:
            E_eV = None
    if E_eV is None:
        E_eV = h_cgs * np.asarray(nu) / eV2erg

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
            base_int = float(np.trapezoid(integrand[:, 0], nu))
            add_pd = 0.0
            if include_pd and pdt_func is not None and Z < 0:
                E_pdt_val = float(pdt_func(Z, a))
                sigma = photodetachment_cross_section_vec(E_eV, E_pdt_val, Z)
                integrand_pd = pd_scale * sigma * J_nu
                add_pd = float(np.trapezoid(integrand_pd, nu))
            out[idx] = base_int + add_pd
            try:
                del Y
                del integrand
            except Exception:
                pass
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
        Rpe_Z = np.trapezoid(integrand, nu, axis=0)
        # If requested, compute photodetachment heating contribution and add to Rpe_Z
        if include_pd:
            pd_add = np.zeros_like(Rpe_Z)
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
                E_pdt_val = float(pdt_func(Zint, a))
                sigma = photodetachment_cross_section_vec(E_eV, E_pdt_val, Zint)
                integrand_pd = pd_scale * sigma * J_nu
                pd_add[iz] = float(np.trapezoid(integrand_pd, nu))
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
        return np.asarray(Rpe_Z, dtype=float)
    else:
        out = np.zeros(N_Z, dtype=float)
        # process in chunks of size 'chunk'
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
            out_chunk = np.trapezoid(integrand, nu, axis=0)
            # photodetachment contribution for this chunk (compute per-Z)
            if include_pd and pdt_func is not None:
                pd_add = np.zeros_like(out_chunk)
                for iz, Z in enumerate(Zchunk):
                    Zint = Z
                    if Zint >= 0:
                        pd_add[iz] = 0.0
                        continue
                    E_pdt_val = float(pdt_func(Zint, a))
                    sigma = photodetachment_cross_section_vec(E_eV, E_pdt_val, Zint)
                    integrand_pd = pd_scale * sigma * J_nu
                    pd_add[iz] = float(np.trapezoid(integrand_pd, nu))
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
        return np.asarray(out, dtype=float)


def unit_diagnostics(nu, J_nu, C_abs_nu, a, yield_func, yield_params, E_band=(6.0,13.6)):
    """
    Print unit sanity checks for the R_pe calculation.
    nu: Hz array
    J_nu: photons cm^-2 s^-1 Hz^-1
    C_abs_nu: cm^2
    a: cm
    """
    nu = np.asarray(nu)
    J_nu = np.asarray(J_nu)
    C_abs_nu = np.asarray(C_abs_nu)

    # Convert nu to energy in eV using cgs constants
    E_eV = (h_cgs * nu) / eV2erg

    mask = (E_eV >= E_band[0]) & (E_eV <= E_band[1])
    if not np.any(mask):
        print('[unit_diagnostics] No points in E band')
        return

    # integrated photon flux in band (photons cm^-2 s^-1)
    photon_flux = np.trapezoid(J_nu[mask], nu[mask])

    # mean absorption cross section in band
    mean_C = np.trapezoid(C_abs_nu[mask]*J_nu[mask], nu[mask]) / np.trapezoid(J_nu[mask], nu[mask])

    # effective bandwidth in Hz
    bw = nu[mask].max() - nu[mask].min()

    # sample yields at mid-energy for Z=0 and a
    mid_idx = np.argmin(np.abs(E_eV - np.mean(E_band)))
    sample_nu = np.array([nu[mid_idx]])
    Ys = yield_func(sample_nu, np.array([0]), a, yield_params)
    sample_Y = float(Ys[mid_idx, 0]) if Ys.ndim == 2 else float(Ys[0])

    est_Rpe = mean_C * photon_flux * sample_Y

    print('[unit_diagnostics] Photon flux in band (photons cm^-2 s^-1): {:.3e}'.format(photon_flux))
    print('[unit_diagnostics] Mean C_abs in band (cm^2): {:.3e}'.format(mean_C))
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

## DS87, charge limits, sticking, and collisional helpers are imported from
## shared_physics.

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
                Rvals = self._rates(Zchunk)
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
            Rpe_vals = self._rates(Zm)
            for Z, R in zip(Zm, Rpe_vals):
                self.cache[Z] = float(R)
        # return array in the same order as Zs
        return np.array([self.cache[Z] for Z in Zs], dtype=float)

    def get_Rpe_single(self, Z):
        return self.get_Rpe_for_Zs([Z])[0]

    def _rates(self, Zs):
        """Photoemission + photodetachment, plus any other charge-raising
        current supplied as ``yield_params['gas_secondary'](Zs)`` (WDB06:
        secondary emission by impinging gas electrons)."""
        R = compute_Rpe_vectorized(self.nu, self.J_nu, self.C_abs_nu,
                                   self.a, Zs, self.yield_func, self.yield_params)
        extra = self.yield_params.get('gas_secondary') if isinstance(self.yield_params, dict) else None
        if extra is not None:
            R = R + np.asarray(extra(Zs), dtype=float)
        return R

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
        J_e = collisional_rates_electrons_scalar(a, Z, n_e, T_e, s_e_func)
        # compute ion capture using provided ion_species (may be empty)
        J_ion_total = collisional_rates_ions_scalar(a, Z, ion_species) if ion_species else 0.0
        # R_plus ~ J_ion_total (ignoring R_pe) ; R_minus ~ J_e
        denom = J_e
        num = J_ion_total
        if denom <= 0:
            return _INF_RATIO if num > 0 else 0.0
        return num / denom

    # Compute physically allowed min/max charges and clamp search to that range.

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

    # Compute allowed Z bounds
    if material == 'graphite':
        Zmin_allowed = most_negative_allowed_charge_graphite(a)
    else:
        Zmin_allowed = most_negative_allowed_charge_silicate(a)
    Zmax_allowed = most_positive_allowed_charge(a, W_val, hnu_max)
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
    # find coarse Zref using bisection search on the full allowed range (including Rpe)
    # We want to find where ratio crosses 1.0. 
    # Ratio is monotonically decreasing with Z.
    def get_full_ratio(Z):
        # Rpe
        Rpe_val = rpc.get_Rpe_single(Z)
        # J_e
        J_e_val = collisional_rates_electrons_scalar(a, Z, n_e, T_e, s_e_func)
        # J_ion
        J_ion_val = collisional_rates_ions_scalar(a, Z, ion_species) if ion_species else 0.0
        num = J_ion_val + Rpe_val
        denom = J_e_val
        if denom <= 0:
            return _INF_RATIO if num > 0 else 0.0
        return num / denom

    low = Zmin_allowed
    high = Zmax_allowed
    while low <= high:
        mid = (low + high) // 2
        ratio = get_full_ratio(mid)
        if ratio > 1.0:
            # too low Z, increase it
            low = mid + 1
        else:
            # too high Z, decrease it
            high = mid - 1
            
    Zref = int(np.clip(low, Zmin_allowed, Zmax_allowed))

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
                denom = collisional_rates_electrons_scalar(a, denomZ, n_e, T_e, s_e_func)
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
                denom = collisional_rates_electrons_scalar(a, Zs[j] + 1, n_e, T_e, s_e_func)
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


def _wdb06_solver_params(yield_params, a, n_e, T_e, material):
    """Complete ``yield_params`` for ``yield_model='WDB06'``: the van Hoof E_min
    photodetachment threshold and, unless switched off, secondary emission by
    impinging gas electrons (``include_gas_secondaries``, WDB06 eq. 26) and the
    transmission of fast electrons through the grain (``include_electron_
    transmission``, DS79), which removes them from the sticking."""
    params = dict(yield_params)
    params.setdefault('photodetachment_func',
                      lambda Z, a_: wdb06_yields.photodetachment_energy(Z, a_, material))
    if params.get('include_gas_secondaries', True):
        params.setdefault('gas_secondary',
                          lambda Zs: wdb06_yields.gas_secondary_rate(Zs, a, n_e, T_e, material))
    if params.get('include_electron_transmission', True):
        base = params.get('s_e_func', electron_sticking_coefficient_graphite if material == 'graphite'
                          else electron_sticking_coefficient_silicate)
        params['s_e_func'] = lambda Zs, a_: base(Zs, a_) * (
            1.0 - wdb06_yields.electron_transmission(Zs, a, T_e, material)[0])
    return params


def _wdb06_heating(Zs, a, nu, J_nu, C_abs_nu, yield_params, material):
    """Photoelectric heating per grain [erg s^-1] for each Z, WDB06 eq. 27
    summed over channels, plus photodetachment (WD01 eq. 40 with eq. 3 E_min)."""
    nu = np.asarray(nu, dtype=float)
    E = yield_params.get('E_eV', h_cgs * nu / eV2erg)
    flux = np.asarray(J_nu, dtype=float) * np.asarray(C_abs_nu, dtype=float)
    YE = wdb06_yields.yield_energy_product(
        E, Zs, a, material, la_wd01=yield_params.get('la'),
        include_secondaries=yield_params.get('include_secondaries', True))
    Gamma = np.trapezoid(flux[:, None] * YE, nu, axis=0)
    if yield_params.get('include_photodetachment', True):
        for i, Z in enumerate(np.asarray(Zs)):
            if Z >= 0:
                continue
            E_pdt = float(wdb06_yields.photodetachment_energy(Z, a, material))
            sigma = photodetachment_cross_section_vec(E, E_pdt, Z)
            E_min = float(wdb06_yields.min_energy_vH04(Z, a))
            Gamma[i] += np.trapezoid(np.asarray(J_nu) * sigma * np.maximum(E - E_pdt + E_min, 0.0), nu)
    return Gamma * eV2erg


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
    max_cache_bytes=None,
    recomb_mode=None
):
    """
    Compute equilibrium P(Z) using vectorized photoemission and optimized bounds finder.

    Arguments:
      - a: grain radius (cm)
      - n_e, T_e: electron density [cm^-3] and temperature [K]
      - ion_species: list of dicts, each with keys:
           {'n': number density [cm^-3], 'T': temperature [K], 'm': mass [g], 'z': charge (e.g. +1)}
      - nu, J_nu: frequency grid [Hz] and photon flux per Hz [photons cm^-2 s^-1 Hz^-1]
      - C_abs_nu: absorption cross-section on nu grid [cm^2]
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
    wdb06 = yield_params.get('yield_model', 'WD01') == 'WDB06'
    if wdb06:
        yield_params = _wdb06_solver_params(yield_params, a, n_e, T_e, material)
        if yield_func is None:
            yield_func = wdb06_yields.yield_wdb06_vectorized
        if yield_params.get('include_ion_transmission', True):
            # fast ions that pass through still ionised give no charge (DS79)
            ion_species = wdb06_yields.with_ion_transmission(ion_species, a, material)
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
            nu_arr = np.asarray(nu, dtype=float)
            E_eV = h_cgs * nu_arr / eV2erg
            hnu_max = float(np.nanmax(E_eV)) if E_eV.size > 0 else 13.6
            W_val = None
            if isinstance(yield_params, dict) and 'W' in yield_params:
                W_val = float(yield_params.get('W'))
            if W_val is None:
                W_val = graphite_work_function if material == 'graphite' else silicate_work_function
            if material == 'graphite':
                Zmin_allowed = most_negative_allowed_charge_graphite(a)
            else:
                Zmin_allowed = most_negative_allowed_charge_silicate(a)
            Zmax_allowed = most_positive_allowed_charge(a, W_val, hnu_max)
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
        use_calima_style()
        fig, ax = _plt.subplots(figsize=(8, 5), dpi=150)
        max_lines = 25
        for i, Zv in enumerate(diag_Zs):
            if i >= max_lines:
                break
            y = Y[:, i] if Y.ndim == 2 else Y
            ax.plot(E_eV, y, label=f'Z={int(Zv)}')
        ax.set_xlabel('E (eV)')
        ax.set_ylabel('Yield Y(E,Z)')
        ax.set_title(f'Photoelectric yield vs E — a={a:.3e} cm, Z={zlo}..{zhi}')
        ax.set_yscale('log')
        ax.set_ylim([1e-2,1.2])
        ax.grid(True, linestyle=':', alpha=0.5)
        # Add the Draine_yield_graphite or silicate curve for reference
        data = np.loadtxt(os.path.join(_EXTERNAL_DATA_DIR, f'Draine_yield_{material}.csv'), delimiter=',')
        print(f'[debug] overplotting Draine yield for {material} from external_data/Draine_yield_{material}.csv')
        ax.plot(data[:, 0], data[:, 1], label='Draine yield', color='k', linestyle=':')
        Y_test = yield_func(np.asarray(nu), np.array([0]), 4e-10, yield_params)
        ax.plot(E_eV, Y_test[:, 0], label='This code Z=0', color='gray', linestyle='-')
        print(f'[debug] overplotting this code yield for Z=0',Y_test[:,0])
        print(yield_params)
        from pycalima.models.dust_charge.dust_photoelectric_heating import photoelectric_yield_graphite, photoelectric_yield_silicate
        if material == 'graphite':
            wav = c_cgs / np.asarray(nu)
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
            wav = c_cgs / np.asarray(nu)
            Y_ref = np.zeros_like(E_eV)
            for e in range(len(E_eV)):
                E = E_eV[e]
                Y_ref[e] = photoelectric_yield_silicate(silicate_work_function,0,0.4,
                                                        electron_escape_length,E,
                                                        wav[e],yield_params['Im'][e])
            ax.plot(E_eV, Y_ref, label='Draine func Z=0', color='red', linestyle='--')
            print(f'[debug] overplotting Draine function yield for {material} Z=0',Y_ref)
        ax.legend(ncol=2, fontsize='small')
        outname = _dust_charging_output_path(
            f'yields_debug_{material}_a{a:.3e}cm_Z{zlo}_{zhi}.png')
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
    from pycalima.models.dust_charge.dust_photoelectric_heating import compute_photoelectric_heating_rate,\
                                                                compute_recombination_cooling_rate,\
                                                                compute_autoionisation_cooling_rate

    # Build radiation_field expected by compute_photoelectric_heating_rate
    # E_eV, lambda_cm, I_E_surface (erg / s / cm^2 / eV)
    E_eV = h_cgs * np.asarray(nu) / eV2erg
    # wavelength in cm (c_cgs defined at module level)
    lambda_cm = c_cgs / np.asarray(nu)
    J_nu_cm = np.asarray(J_nu)
    # Energy flux per eV: J_nu_cm * nu * eV2erg  (erg / s / cm^2 / eV)
    I_E_surface = J_nu_cm * np.asarray(nu) * eV2erg
    radiation_field_for_heating = np.column_stack([E_eV, lambda_cm, I_E_surface])

    # absorption cross section for use in heating function
    C_abs_for_heating = np.asarray(C_abs_nu)

    # dielectric arrays required by compute_photoelectric_heating_rate are expected
    # in yield_params (we pass the whole object down when needed)
    Im_for_heating = None
    if isinstance(yield_params, dict):
        if 'Im' in yield_params:
            Im_for_heating = yield_params.get('Im')
        elif 'Imperp' in yield_params and 'Impar' in yield_params:
            Im_for_heating = np.column_stack([yield_params.get('Imperp'), yield_params.get('Impar')])

    Gamma_Z = np.zeros_like(Zs, dtype=float)
    Recomb_Z = np.zeros_like(Zs, dtype=float)
    for iZ, Zv in enumerate(Zs):
        if not wdb06:
            # photoelectric heating (includes photodetachment inside compute_photoelectric_heating_rate)
            args_pe = (int(Zv), a, radiation_field_for_heating, ( 'graphite' if material=='graphite' else 'silicate'), Im_for_heating, C_abs_for_heating)
            Gamma_Z[iZ] = float(compute_photoelectric_heating_rate(args_pe))
        # recombination cooling
        args_re = (int(Zv), a, n_e, T_e, ( 'graphite' if material=='graphite' else 'silicate'))
        Recomb_Z[iZ] = float(compute_recombination_cooling_rate(args_re))
    if wdb06:
        Gamma_Z = _wdb06_heating(Zs, a, nu, J_nu, C_abs_nu, yield_params, material)
        if yield_params.get('include_electron_transmission', True):
            # transmitted electrons remove only the energy they deposit
            Recomb_Z = Recomb_Z * wdb06_yields.electron_transmission(Zs, a, T_e, material)[1]
        if yield_params.get('include_gas_secondaries', True):
            # WDB06 eq. 28: gas-excited secondaries return energy to the gas
            Recomb_Z = Recomb_Z - (wdb06_yields.gas_secondary_rate(Zs, a, n_e, T_e, material)
                                   * wdb06_yields.gas_secondary_mean_energy(Zs, a, T_e, material) * eV2erg)

    Gamma_total = float(np.sum(P * Gamma_Z))
    Recomb_total = float(np.sum(P * Recomb_Z))

    # autoionisation cooling (if Zmin exists)
    auto_cooling = 0.0
    if material == 'graphite':
        Zmin = most_negative_allowed_charge_graphite(a)
    else:
        Zmin = most_negative_allowed_charge_silicate(a)
    if Zmin in Zs:
        idx_auto = int(np.where(Zs == Zmin)[0][0])
        prob_Zmin = P[idx_auto]
        if prob_Zmin > 0.0:
            args_ai = (Zmin, prob_Zmin, a, n_e, T_e, ( 'graphite' if material=='graphite' else 'silicate'))
            auto_cooling = float(compute_autoionisation_cooling_rate(args_ai))

    rates['Gamma_Z'] = Gamma_Z
    rates['Gamma_total'] = Gamma_total
    rates['Recomb_Z'] = Recomb_Z
    rates['Recomb_total'] = Recomb_total
    rates['Autoionisation_cooling'] = auto_cooling
    # print(Recomb_total+auto_cooling)
    # Compute absorbed power E_abs [erg/s] using the same radiation field used
    # for the Gamma_Z computation. radiation_field_for_heating contains columns
    # [E_eV, lambda_cm, I_E_surface(erg/s/cm^2/eV)] and C_abs_for_heating is
    # in cm^2 so their product integrates to erg/s.
    try:
        E_grid = np.asarray(radiation_field_for_heating[:, 0], dtype=float)
        I_E_surf = np.asarray(radiation_field_for_heating[:, 2], dtype=float)
        mask = np.isfinite(E_grid) & np.isfinite(I_E_surf)
        if np.any(mask):
            # integrate only over finite points and positive energies
            sel = mask & (E_grid >= 0.0)
            if np.any(sel):
                E_abs_val = float(np.trapezoid(C_abs_for_heating[sel] * I_E_surf[sel], E_grid[sel]))
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

    # Compute ion recombination rates and rate coefficients for each species in ion_species
    # Weingartner & Draine (2001) model: s_i is assumed to be 1.0.
    ion_recomb_rates = []
    ion_recomb_rate_coefficients = []
    if ion_species:
        if material == 'graphite':
            W = GRAPHITE_WORK_FUNCTION
        else:
            W = SILICATE_WORK_FUNCTION

        cross = np.pi * a * a
        for ion in ion_species:
            n_i = float(ion.get("n", 0.0))
            T_i = float(ion.get("T", 1.0))
            m_g = float(ion.get("m", 1.0))
            m_g_cgs = m_g * 1e3 if m_g < 1e-25 else m_g
            z_i = float(ion.get("z", 1.0))
            s_i = float(ion.get("s_i", 1.0))  # Default sticking coefficient is 1.0 (WD01)

            vth_i = np.sqrt(8.0 * kb_cgs * T_i / (np.pi * m_g_cgs))
            Jtilde_i = DS87_J_function_vec(Zs, np.array([z_i]), a, T_i)

            mode = recomb_mode
            if mode is None and isinstance(yield_params, dict):
                mode = yield_params.get('recomb_mode', None)

            if mode in ['case_a', 'case_b']:
                name = ion.get('name', '')
                el_key = None
                for el in ['He', 'Na', 'Mg', 'Si', 'Ca', 'Mn', 'Fe']:
                    if name.startswith(el):
                        el_key = el
                        break
                if el_key is None:
                    for el in ['H', 'C', 'S', 'K']:
                        if name.startswith(el):
                            el_key = el
                            break
                
                ATOMIC_RADII = {
                    'H': 0.37, 'He': 0.50, 'C': 0.77, 'Na': 1.86, 'Mg': 1.60,
                    'Si': 1.18, 'S': 1.03, 'K': 2.27, 'Ca': 1.97, 'Mn': 1.37, 'Fe': 1.24
                }
                IONIZATION_POTENTIALS = {
                    'H': 13.598, 'He': 24.587, 'C': 11.260, 'Na': 5.139, 'Mg': 7.646,
                    'Si': 8.151, 'S': 10.360, 'K': 4.341, 'Ca': 6.113, 'Mn': 7.434, 'Fe': 7.902
                }

                r0 = float(ion.get('r0', ATOMIC_RADII.get(el_key, 0.77) if el_key else 0.77))
                IP_X_im1 = float(ion.get('IP_X_im1', ion.get('IP_ion', ion.get('ionization_potential', IONIZATION_POTENTIALS.get(el_key, 13.6) if el_key else 13.6))))

                # Calculate grain ionization potential IP(a,Z) in eV (piecewise: valence IP for Z >= 0, EA of Z+1 for Z < 0)
                # Grain material from yield_params, as the photoemission code in
                # this module reads it; graphite if unspecified.
                grain_type = (yield_params.get('material', 'graphite')
                              if isinstance(yield_params, dict) else 'graphite')
                is_graphite = grain_type.lower().startswith('gra') or grain_type.lower().startswith('car')
                IP_a_Z = np.zeros(len(Zs), dtype=float)
                for idx, Z in enumerate(Zs):
                    if Z >= 0:
                        IP_a_Z[idx] = ionisation_potential_valence_vec(W, Z, a)
                    else:
                        if is_graphite:
                            IP_a_Z[idx] = electron_affinity_graphite_vec(Z + 1, a)
                        else:
                            IP_a_Z[idx] = electron_affinity_silicate_vec(Z + 1, a)

                if mode == 'case_a':
                    y = IP_X_im1 - IP_a_Z
                    theta = np.where(y >= 0.0, 1.0, 0.0)
                elif mode == 'case_b':
                    r0_cm = r0 * 1e-8
                    term1 = (z_i - Zs - 1.0) * (E_STATC**2) / (a + r0_cm)
                    term2 = (1.0 - 2.0 * z_i) * (E_STATC**2) * (a**3) / (2.0 * r0_cm * ((a + r0_cm)**2) * (2.0 * a + r0_cm))
                    dU = (term1 + term2) / EV2ERG
                    y = IP_X_im1 - IP_a_Z + dU   # as dust_ion_recombination: WD01 Eq. 6 has a sign typo
                    theta = np.where(y >= 0.0, 1.0, 0.0)
                
                Jtilde_i = Jtilde_i * theta
            
            # Rate coefficient (cm^3 s^-1 per grain): alpha_gr_i = sum_Z P(Z) * cross * vth * s_i * Jtilde
            alpha_i = float(np.sum(P * cross * vth_i * s_i * Jtilde_i))
            # Total rate per grain (s^-1): rate_i = n_i * alpha_gr_i
            rate_i = n_i * alpha_i
            
            ion_recomb_rates.append(rate_i)
            ion_recomb_rate_coefficients.append(alpha_i)

    rates['ion_recomb_rates'] = np.array(ion_recomb_rates)
    rates['ion_recomb_rate_coefficients'] = np.array(ion_recomb_rate_coefficients)

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
    use_calima_style()

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
        if not os.path.isabs(savefile):
            savefile = _dust_charging_output_path(savefile)
        fig.savefig(savefile, dpi=200)

    return fig, ax


def equilibrium_charge_for_grain(G0, ne, T, grain_type, a_cm,
                                 radiation_model='Mathis', rad_field=None,
                                 yield_params=None, ion_species=None, Z_start=0, debug=False,
                                 recomb_mode=None):
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
    a_cm : float
        Grain radius in cm.
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
    - spectral intensity J_nu is supplied as erg s^-1 cm^-2 Hz^-1
    - absorption cross section C_abs is supplied in cm^2

    The wrapper intentionally ignores ion capture rates (passes an empty
    ion_species list) because the current solver is configured to consider
    electrons only per your request.
    """
    # lazy imports from other modules in the repo to avoid top-level dependency issues
    from pycalima.models.dust_charge.dust_photoelectric_heating import get_radiation_field, read_dielectric_file
    from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections

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
    if rad0.ndim == 2 and rad0.shape[1] >= 3:
        # energy-first input: [E_eV, wavelength_nm, I_E (erg/s/cm^2/eV)]
        E_eV = rad0[:, 0].astype(float)
        wav_nm = rad0[:, 1].astype(float)
        I_E = rad0[:, 2].astype(float)

        # preserve increasing energy ordering for interpolation and diagnostics
        order = np.argsort(E_eV)
        E_eV = E_eV[order]
        wav_nm = wav_nm[order]
        I_E = I_E[order]

        lambda_cm = wav_nm * 1e-7
        nu = c_cgs / lambda_cm
        # I_E is per eV; convert to photon flux per Hz using dE_eV/dnu = h / eV2erg
        # Apply the requested field scaling so this branch is consistent with
        # the wavelength/intensity branch below.
        J_nu = (I_E * float(G0)) / np.maximum(nu * eV2erg, _TINY)
        E_for_interp = E_eV
        wav_cm_for_interp = lambda_cm

    elif rad0.ndim == 2 and rad0.shape[1] >= 2:
        # original: wavelength (nm) and wavelength_intensity (erg cm-2 s-1 nm-1 sr-1)
        wavelength_nm = rad0[:, 0].astype(float)
        wavelength_intensity = rad0[:, 1].astype(float)

        wav_nm_rev = wavelength_nm[::-1]         # nm
        wav_int_rev = wavelength_intensity[::-1] # erg / (s cm^2 nm sr)

        # (optional) compute E_eV for diagnostics / interpolation
        hc_eVnm = 1239.84193                      # eV·nm
        E_eV = hc_eVnm / wav_nm_rev               # eV

        # Convert wavelength to cm and intensity to per-cm spectral form.
        lambda_cm = wav_nm_rev * 1e-7
        I_lambda_per_cm = wav_int_rev * 1e7

        # 4) photon number flux per Hz per sr: n_nu = I_nu_energy / (h * nu)
        #    but nu = c / lambda, so combine: n_nu = I_nu_energy / (h * c / lambda) = I_nu_energy * lambda / (h * c)
        #    equivalently, simplifying earlier: n_nu = I_lambda_SI_per_m * lambda^3 / (h * c^2)
        n_nu_per_sr = I_lambda_per_cm * (lambda_cm**3) / (h_cgs * c_cgs**2)

        # 5) integrate over hemisphere (4π sr) if original quantity was per sr
        n_nu = n_nu_per_sr * (4 * np.pi)    # photons s^-1 cm^-2 Hz^-1

        # 6) apply scaling by G0 if you want to scale the field
        n_nu *= float(G0)   # if you intend to scale; earlier you used I_E *= G0

        # final arrays:
        # nu (Hz)
        nu = (c_cgs / lambda_cm)
        # J_nu photons: n_nu (photons / s / cm^2 / Hz)
        J_nu = n_nu

        # For absorption cross section interpolation we still need E_eV and wav (micron)
        E_for_interp = E_eV
        wav_cm_for_interp = lambda_cm
    else:
        raise ValueError('rad_field must be an array with at least two columns (wavelength_nm, intensity)')

    # 4) absorption cross section: interpolate compute_cross_sections output onto E grid
    # compute_cross_sections returns (a0, wav_cm, C_sca_cm2, C_abs_cm2, C_rp_cm2)
    a_micron = float(a_cm) * 1e4
    _, wav_cs, _, C_abs_cs, _ = interpolate_cross_sections(material, a_micron)
    # wav_cs is in cm; convert to energy in eV for interpolation
    optical_E = 1.2398 / (wav_cs * 1e4)  # wav_cs(cm) *1e4 -> microns
    # interpolate C_abs to E_for_interp
    C_abs_interp_cm2 = np.interp(E_for_interp, optical_E, C_abs_cs)
    # 5) prepare yield function parameters: pass material and dielectric if possible
    # read dielectric files and interpolate imaginary parts to wav grid used by yields
    if material == 'graphite':
        data_perp = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10')
        data_par = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10')
        # interpolate to wavelengths in micron
        # compute wav in microns expected by the dielectric tables
        wav_micron = wav_cm_for_interp * 1e4
        Im_perp = np.interp(wav_micron, data_perp['table']['wavelength_um'][::-1], data_perp['table']['Im_n'][::-1])
        Im_par = np.interp(wav_micron, data_par['table']['wavelength_um'][::-1], data_par['table']['Im_n'][::-1])
        yield_params_local = dict(yield_params)
        yield_params_local.update({'W': graphite_work_function, 'le': electron_escape_length, 'Imperp': Im_perp, 'Impar': Im_par, 'wav': wav_cm_for_interp})
    else:
        data = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/eps_suvSil')
        wav_micron = wav_cm_for_interp * 1e4
        Im = np.interp(wav_micron, data['table']['wavelength_um'][::-1], data['table']['Im_n'][::-1])
        yield_params_local = dict(yield_params)
        yield_params_local.update({'W': silicate_work_function, 'le': electron_escape_length, 'Im': Im, 'wav': wav_cm_for_interp})

    # 6) run equilibrium solver
    if ion_species is None:
        ion_species = []

    # Default cache budget is left to RpeCache default unless specified in yield_params
    max_cache_bytes = yield_params_local.get('max_cache_bytes', None)

    # Choose yield function based on material
    if material == 'graphite':
        yield_func = yield_graphite_vectorized
    else:
        yield_func = yield_silicate_vectorized

    yield_params_local['material'] = material

    if yield_params_local.get('yield_model', 'WD01') == 'WDB06':
        # WDB06: Q_abs from Draine (2003) up to 20 keV; the dielectric l_a is
        # kept below 20 eV, where the band yield is WD01's.
        yield_func = wdb06_yields.yield_wdb06_vectorized
        C_abs_interp_cm2 = np.pi * a_cm ** 2 * wdb06_yields.absorption_efficiency(E_for_interp, a_cm, material)
        if material == 'graphite':
            la = photon_attenuation_length_graphite_vec(wav_cm_for_interp, Im_perp, Im_par)
        else:
            la = photon_attenuation_length_silicate_vec(wav_cm_for_interp, Im)
        yield_params_local.update({'E_eV': np.asarray(E_for_interp, dtype=float), 'la': la})

    Zs, P, rates, Zmean, Zsigma = compute_equilibrium_charge_distribution_vectorized(
        a_cm, ne, T, ion_species, nu, J_nu, C_abs_interp_cm2,
        yield_func=yield_func, yield_params=yield_params_local, Z_start=Z_start,
        debug=debug, max_cache_bytes=max_cache_bytes, recomb_mode=recomb_mode
    )

    return Zs, P, rates, Zmean, Zsigma


def heating_and_cooling_for_tuple(G0, ne, T, grain_type, a,
                                  radiation_model='Mathis', yield_params=None,
                                  ion_species=None, Z_start=0, debug=False,
                                  recomb_mode=None):
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
    a : float
        Grain radius in cm.
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
        G0, ne, T, grain_type, a,
        radiation_model=radiation_model,
        rad_field=None,
        yield_params=yield_params,
        ion_species=ion_species,
        Z_start=Z_start,
        debug=debug,
        recomb_mode=recomb_mode
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
        'ion_recomb_rates': rates.get('ion_recomb_rates', np.array([])),
        'ion_recomb_rate_coefficients': rates.get('ion_recomb_rate_coefficients', np.array([])),
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
        field, i.e. divided by 1.68e-3 erg cm^-2 s^-1.
    power_erg_cm2_s : float
        Integrated power in erg cm^-2 s^-1 between E_min and E_max.
    """

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

        # d(lambda_nm)/dE_eV = hc(eV*nm) / E^2  (nm per eV)
        hc_eVnm = 1239.84193
        dlam_dE_nm_per_eV = hc_eVnm / np.maximum(E**2, _TINY)

        # I_E (erg / s / cm^2 / eV / sr) = I_lambda (erg / s / cm^2 / nm / sr) * dlambda_nm/dE_eV
        I_E_erg_per_s_cm2_eV_per_sr = I_lambda * dlam_dE_nm_per_eV

        # integrate over hemisphere (2π sr)
        I_E_erg_per_s_cm2_eV = I_E_erg_per_s_cm2_eV_per_sr * (4.0 * np.pi)

    # select energy range
    mask = (E >= E_min) & (E <= E_max)
    if not np.any(mask):
        return 0.0, 0.0

    # integrate I_E over E to get erg / s / cm^2
    integral_erg_per_s_cm2 = np.trapezoid(I_E_erg_per_s_cm2_eV[mask], E[mask])

    # G0 reference in cgs: 1.68e-6 W/m^2 = 1.68e-3 erg/cm^2/s
    G0_ref = 1.68e-3
    G0 = integral_erg_per_s_cm2 / G0_ref

    return float(G0), float(integral_erg_per_s_cm2)


def compute_G0_from_model(radiation_model='Draine', E_min=6.0, E_max=13.6):
    """Helper that obtains the radiation field via get_radiation_field and computes G0."""
    from pycalima.models.dust_charge.dust_photoelectric_heating import get_radiation_field
    rad, _ = get_radiation_field(radiation_model)
    # earlier get_radiation_field returns [wavelength_nm, intensity erg/cm2/s/nm/sr]
    return compute_G0_from_rad_field(rad, E_min=E_min, E_max=E_max)


def _prepare_gamma_scan_context(grain_type, a_cm, radiation_model='Mathis', yield_params=None):
    """Prepare invariant radiation/optical inputs once for a gamma scan."""
    from pycalima.models.dust_charge.dust_photoelectric_heating import get_radiation_field, read_dielectric_file
    from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections

    if yield_params is None:
        yield_params = {}

    material = yield_params.get('material', 'graphite' if grain_type.lower().startswith('gra') or grain_type.lower().startswith('car') else 'silicate')
    rad0, _ = get_radiation_field(radiation_model)
    rad0 = np.asarray(rad0)

    if rad0.ndim == 2 and rad0.shape[1] >= 3:
        E_eV = np.asarray(rad0[:, 0], dtype=float)
        wav_nm = np.asarray(rad0[:, 1], dtype=float)
        I_E = np.asarray(rad0[:, 2], dtype=float)
        order = np.argsort(E_eV)
        E_eV = E_eV[order]
        wav_cm = wav_nm[order] * 1e-7
        nu = c_cgs / wav_cm
        J_nu = I_E[order] / np.maximum(nu * eV2erg, _TINY)
        wav_cm_for_interp = wav_cm
        E_for_interp = E_eV
    elif rad0.ndim == 2 and rad0.shape[1] >= 2:
        wavelength_nm = np.asarray(rad0[:, 0], dtype=float)
        wavelength_intensity = np.asarray(rad0[:, 1], dtype=float)
        wav_nm_rev = wavelength_nm[::-1]
        wav_int_rev = wavelength_intensity[::-1]
        wav_cm = wav_nm_rev * 1e-7
        nu = c_cgs / wav_cm
        I_lambda_per_cm = wav_int_rev * 1e7
        n_nu_per_sr = I_lambda_per_cm * (wav_cm**3) / (h_cgs * c_cgs**2)
        J_nu = n_nu_per_sr * (4.0 * np.pi)
        E_for_interp = 1.2398 / (wav_nm_rev * 1e-3)
        wav_cm_for_interp = wav_cm
    else:
        raise ValueError('radiation model returned an unsupported field shape')

    a_micron = float(a_cm) * 1e4
    _, wav_cs, _, C_abs_cs, _ = interpolate_cross_sections(material, a_micron)
    optical_E = 1.2398 / (wav_cs * 1e4)
    C_abs_interp_cm2 = np.interp(E_for_interp, optical_E, C_abs_cs)

    if material == 'graphite':
        data_perp = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10')
        data_par = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10')
        wav_micron = wav_cm_for_interp * 1e4
        Im_perp = np.interp(wav_micron, data_perp['table']['wavelength_um'][::-1], data_perp['table']['Im_n'][::-1])
        Im_par = np.interp(wav_micron, data_par['table']['wavelength_um'][::-1], data_par['table']['Im_n'][::-1])
        la = photon_attenuation_length_graphite_vec(wav_cm_for_interp, Im_perp, Im_par)
        yield_params_local = dict(yield_params)
        yield_params_local.update({'W': graphite_work_function, 'le': electron_escape_length, 'Imperp': Im_perp, 'Impar': Im_par, 'wav': wav_cm_for_interp, 'la': la, 'E_eV': h_cgs * nu / eV2erg})
        yield_func = yield_graphite_vectorized
    else:
        data = read_dielectric_file(f'{PATH_OPTICS}/draine_lee_1984/eps_suvSil')
        wav_micron = wav_cm_for_interp * 1e4
        Im = np.interp(wav_micron, data['table']['wavelength_um'][::-1], data['table']['Im_n'][::-1])
        la = photon_attenuation_length_silicate_vec(wav_cm_for_interp, Im)
        yield_params_local = dict(yield_params)
        yield_params_local.update({'W': silicate_work_function, 'le': electron_escape_length, 'Im': Im, 'wav': wav_cm_for_interp, 'la': la, 'E_eV': h_cgs * nu / eV2erg})
        yield_func = yield_silicate_vectorized

    yield_params_local['material'] = material

    return {
        'material': material,
        'nu': np.asarray(nu, dtype=float),
        'J_nu': np.asarray(J_nu, dtype=float),
        'C_abs_nu': np.asarray(C_abs_interp_cm2, dtype=float),
        'yield_func': yield_func,
        'yield_params': yield_params_local,
    }


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
        integral_erg_per_s_cm2 = np.trapezoid(I_E[mask], E[mask])
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
        integral_per_sr = np.trapezoid(I_E_per_sr[mask], E_sorted[mask])
        # multiply by 4π to get integrated per-surface, divide by c to get energy density
        u = (4.0 * np.pi * integral_per_sr) / c_cgs_local
        return float(u)


def compute_energy_density_from_nu_J(nu, J_nu, E_min=6.0, E_max=13.6):
    """
    Compute energy density (erg/cm^3) from nu [Hz] and J_nu [photons cm^-2 s^-1 Hz^-1]
    integrated between E_min and E_max (eV).
    """
    nu = np.asarray(nu)
    J_nu = np.asarray(J_nu)
    E_eV = (h_cgs * nu) / eV2erg
    mask = (E_eV >= E_min) & (E_eV <= E_max)
    if not np.any(mask):
        return 0.0

    # energy flux S = ∫ J_nu * (h nu) dnu  [erg s^-1 cm^-2]
    S = np.trapezoid(J_nu[mask] * (h_cgs * nu[mask]), nu[mask])

    # energy density u = S / c [erg cm^-3]
    return float(S / c_cgs)


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