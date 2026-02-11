#!/usr/bin/env python3
"""
Profile memory (peak RSS) and wall time for representative dust_charging operations.
Saves a small JSON report and prints a summary table.
"""
import time
import json
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import resource
import numpy as np
from dust_charging import RpeCache, compute_Rpe_vectorized, compute_equilibrium_charge_distribution_vectorized


def mb_from_ru(ru):
    if sys.platform.startswith('darwin'):
        return ru / (1024.0 * 1024.0)
    else:
        return (ru * 1024.0) / (1024.0 * 1024.0)


def snapshot(label):
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'label': label, 'ru_raw': int(ru), 'ru_mb': float(mb_from_ru(ru)), 'time': time.time()}


def measure(fn, *args, **kwargs):
    before = snapshot('before')
    t0 = time.time()
    res = fn(*args, **kwargs)
    t1 = time.time()
    after = snapshot('after')
    return {'duration_s': t1 - t0, 'before': before, 'after': after, 'result_summary': str(type(res))}


def main():
    out = {}
    # build synthetic nu/J and cross section (like earlier test)
    c_SI = 2.99792458e8
    nu = np.logspace(14, 16, 800)
    J_nu = 1e8 * (nu / 1e15) ** -1.5
    C_abs_nu = np.full_like(nu, 1e-18)
    a_m = 0.1e-6

    # dummy yield returns zeros (fast) and a simple synthetic yield that is small
    def dummy_yield(nu_arr, Zs, a_val, params):
        nu_arr = np.asarray(nu_arr)
        Zs = np.asarray(Zs)
        return np.zeros((nu_arr.size, Zs.size), dtype=float)

    # 1) small vectorized compute
    Zs_small = np.arange(-10, 11, dtype=int)
    out['vec_small'] = measure(compute_Rpe_vectorized, nu, J_nu, C_abs_nu, a_m, Zs_small, dummy_yield, {'material': 'graphite'})

    # 2) large vectorized compute (should create big temporaries)
    Zs_large = np.arange(-2000, 2001, dtype=int)
    out['vec_large'] = measure(compute_Rpe_vectorized, nu, J_nu, C_abs_nu, a_m, Zs_large, dummy_yield, {'material': 'graphite'})

    # 3) RpeCache get with small budget (forces no persistent caching and chunked computation path)
    rpc = RpeCache(nu, J_nu, C_abs_nu, a_m, dummy_yield, {'material': 'graphite'}, max_cache_bytes=20 * 1024 * 1024)
    out['rpc_get'] = measure(rpc.get_Rpe_for_Zs, Zs_large)

    # 4) equilibrium solver on a realistic single case (uses the vectorized solver internally)
    # Build minimal fake radiative field expected by compute_equilibrium_charge_distribution_vectorized
    # Here we pass nu and J_nu as the actual arrays expected by the lower-level functions
    ion_species = []
    # use dummy_yield as yield_func so we don't require dielectric tables or W
    out['equilibrium_single'] = measure(
        compute_equilibrium_charge_distribution_vectorized,
        a_m, 1.0, 100.0, ion_species, nu, J_nu, C_abs_nu,
        dummy_yield, {'material': 'graphite'}, 0, 20, 1e-6, False
    )

    # write report
    outpath = os.path.join(os.path.dirname(__file__), 'profile_report.json')
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)

    # print table
    print('\nProfile summary:')
    for k, v in out.items():
        print(f"{k:20s} duration={v['duration_s']:.3f}s  ru_before={v['before']['ru_mb']:.2f}MB  ru_after={v['after']['ru_mb']:.2f}MB")
    print('\nReport written to', outpath)

if __name__ == '__main__':
    main()
