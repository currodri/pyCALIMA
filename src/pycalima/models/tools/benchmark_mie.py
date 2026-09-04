
import os
import numpy as np
import pandas as pd
from pycalima.models.tools.mie_theory import MieTheory, read_draine_q_table

def run_benchmark():
    mie = MieTheory()
    base_path = 'optical_props/draine_lee_1984'
    test_rad = 0.1
    test_wavs = [0.1, 0.5, 1.0, 10.0, 100.0]

    # --- Silicate Benchmark ---
    print("\n" + "="*70)
    print("BENCHMARKING SILICATE (radius = 0.1 um) vs Sil_81")
    print("="*70)
    mie.load_dielectric_constants(os.path.join(base_path, 'eps_Sil'), 'silicate')
    sil_benchmark = read_draine_q_table(os.path.join(base_path, 'Sil_81'))
    
    # Benchmark Silicate 10 micron
    print("\n" + "="*70)
    print("BENCHMARKING SILICATE (radius = 10.0 um) vs Sil_81")
    print("="*70)
    test_rad_large = 10.0
    if test_rad_large in sil_benchmark:
        bench_df = sil_benchmark[test_rad_large]
        # At 10 um, let's pick some UV wavelengths where x is very large
        test_wavs_large = [0.001, 0.01, 0.1, 1.0, 10.0]
        print(f"{'Wav (um)':>10} | {'Prop':>5} | {'Draine':>12} | {'Mie':>12} | {'Diff (%)':>10}")
        print("-" * 70)
        for w in test_wavs_large:
            idx = (bench_df['wavelength_um'] - w).abs().idxmin()
            w_actual = bench_df.loc[idx, 'wavelength_um']
            qabs_d, qsca_d, g_d = bench_df.loc[idx, ['Qabs', 'Qsca', 'g']]
            
            qabs_m, qsca_m, g_m = mie.compute_grain_properties(test_rad_large, w_actual, 'silicate')
            
            for prop, val_d, val_m in [('Qabs', qabs_d, qabs_m), ('Qsca', qsca_d, qsca_m), ('g', g_d, g_m)]:
                diff = 100 * (val_m - val_d) / val_d if val_d != 0 else 0
                print(f"{w_actual:10.4f} | {prop:5} | {val_d:12.4e} | {val_m:12.4e} | {diff:10.2f}%")
            print("-" * 70)

    # --- Graphite Benchmark ---
    print("\n" + "="*70)
    print("BENCHMARKING GRAPHITE (radius = 0.1 um, 1/3-2/3 approx) vs Gra_81")
    print("="*70)
    print("Note: Using 2003 dielectric constants (callindex.out) as proxy.")
    mie.load_dielectric_constants(os.path.join(base_path, 'callindex.out_CpaD03_0.01'), 'graphite_pa')
    mie.load_dielectric_constants(os.path.join(base_path, 'callindex.out_CpeD03_0.01'), 'graphite_pe')
    gra_benchmark = read_draine_q_table(os.path.join(base_path, 'Gra_81'))
    
    if test_rad in gra_benchmark:
        bench_df = gra_benchmark[test_rad]
        print(f"{'Wav (um)':>10} | {'Prop':>5} | {'Draine':>12} | {'Mie':>12} | {'Diff (%)':>10}")
        print("-" * 70)
        for w in test_wavs:
            idx = (bench_df['wavelength_um'] - w).abs().idxmin()
            w_actual = bench_df.loc[idx, 'wavelength_um']
            qabs_d, qsca_d, g_d = bench_df.loc[idx, ['Qabs', 'Qsca', 'g']]
            
            qabs_m, qsca_m, g_m = mie.compute_grain_properties(test_rad, w_actual, 
                                                               {'parallel': 'graphite_pa', 'perpendicular': 'graphite_pe'})
            
            for prop, val_d, val_m in [('Qabs', qabs_d, qabs_m), ('Qsca', qsca_d, qsca_m), ('g', g_d, g_m)]:
                diff = 100 * (val_m - val_d) / val_d if val_d != 0 else 0
                print(f"{w_actual:10.4f} | {prop:5} | {val_d:12.4e} | {val_m:12.4e} | {diff:10.2f}%")
            print("-" * 70)
    else:
        print(f"Radius {test_rad} not found in Gra_81.")

if __name__ == "__main__":
    run_benchmark()
