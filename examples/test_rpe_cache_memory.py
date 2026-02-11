#!/usr/bin/env python3
"""
Quick memory test for RpeCache: instantiate with a small max_cache_bytes and
request R_pe over a wide Z range to see how the cache behaves and what the
process peak RSS is.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import resource
import numpy as np
from dust_charging import RpeCache


def build_synthetic_nu_J(N_nu=500, G0=1.0):
    # create a synthetic UV-like frequency grid and photon flux
    c_SI = 2.99792458e8
    # frequencies spanning ~1e14 - 1e16 Hz
    nu = np.logspace(14, 16, N_nu)
    # simple power-law photon flux (photons / s / m^2 / Hz)
    J_nu = G0 * 1e8 * (nu / 1e15) ** -1.5
    # simple constant absorption cross section per frequency (m^2)
    C_abs_nu = np.full_like(nu, 1e-18)
    # return also a wavelength array in nm for compatibility
    wav_nm = (c_SI / nu) * 1e9
    return nu, J_nu, C_abs_nu, wav_nm


def main():
    # build synthetic radiation and inputs
    nu, J_nu, C_abs_nu, wav_nm = build_synthetic_nu_J(N_nu=800, G0=1.0)
    # pick a material and size
    a_micron = 0.1
    a_m = a_micron * 1e-6
    # simple yield function that returns zeros (safe and fast)
    def dummy_yield(nu_arr, Zs, a_val, params):
        nu_arr = np.asarray(nu_arr)
        Zs = np.asarray(Zs)
        return np.zeros((nu_arr.size, Zs.size), dtype=float)

    # instantiate RpeCache with small budget (20 MiB)
    max_cache = 1024
    rpc = RpeCache(nu, J_nu, C_abs_nu, a_m, dummy_yield, {'material': 'graphite'}, max_cache_bytes=max_cache)

    Zs = np.arange(-2000, 2001, dtype=int)
    print('Requesting R_pe for', Zs.size, 'Z states with max_cache_bytes =', max_cache)
    Rvals = rpc.get_Rpe_for_Zs(Zs)
    print('Computed R_pe array length:', len(Rvals))
    print('Cache size after call:', len(rpc.cache))
    # report peak RSS from resource and show a clear MB value
    usage = resource.getrusage(resource.RUSAGE_SELF)
    ru = usage.ru_maxrss
    # On macOS ru_maxrss is in bytes; on many Linux distros it's in kilobytes.
    if sys.platform.startswith('darwin'):
        ru_bytes = int(ru)
    else:
        # assume kilobytes
        ru_bytes = int(ru) * 1024
    ru_mb = ru_bytes / (1024.0 ** 2)
    print(f'ru_maxrss raw = {ru} (platform={sys.platform}); interpreted = {ru_bytes} bytes ({ru_mb:.2f} MB)')

if __name__ == '__main__':
    main()
