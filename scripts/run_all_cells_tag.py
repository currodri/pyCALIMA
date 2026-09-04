"""
Run pyCALIMA over every valid cell in one RAMSES galaxy snapshot.

For each cell a short list of (solver × IC × t_end) combinations is evaluated
and the results are written to results/comparison/per_cell_all_<TAG>.npz.

Usage
-----
    python scripts/run_all_cells_tag.py G10 [--njobs 8]
    python scripts/run_all_cells_tag.py G8  --njobs 16

The script runs all valid cells (non-zero metallicity, non-zero dust, finite
temperature and density) inside the analysis sphere.  Depending on the
simulation resolution this can range from a few thousand cells (G8 dwarf,
R=5 kpc) to hundreds of thousands (G10 MW-like, R=15 kpc).  Expect a
wall-clock time of roughly  N_cells × 6 methods / njobs  seconds.

Output keys
-----------
dtm_{key}, fs_{key}, fc_{key}  — DTM / f_small / f_carb per method per cell
DTM_sim, fs_sim, fc_sim        — simulation ground truth
T, nH, Z_abs, sig, mass        — cell gas properties
"""

import os, sys, json, tempfile, argparse
import numpy as np
from joblib import Parallel, delayed

PYCALIMA = os.path.expanduser('~/Documents/GitHub/pyCALIMA')
sys.path.insert(0, PYCALIMA)
os.chdir(PYCALIMA)

import yt
yt.set_log_level('critical')
from pycalima.solvers.run_grid import run_grid

# ── Physical constants ─────────────────────────────────────────────────────────
MH         = 1.6726e-24    # Hydrogen atom mass [g]
MU         = 1.4           # Mean molecular weight (primordial + 10% He by number)
ZSUN       = 0.0134        # Absolute solar metallicity (Asplund+2009)
SioverSil  = 0.163         # Si mass fraction inside silicate grain (SiO₂ correction)
T_SIM_MYR  = 399.6         # Simulation age at the analysed snapshot [Myr]
UNIT_V_KMS = 65.59         # RAMSES velocity code unit → km/s

# Zubko+2004 equilibrium dust mass fractions at solar metallicity
# Order: SmCarb (0.005 µm), LgCarb (0.1 µm), SmSil (0.005 µm), LgSil (0.1 µm)
ZUBKO_FRACS = np.array([0.071933, 0.253672, 0.104929, 0.569437])

OUTDIR = 'results/comparison'

# ── Simulation paths ───────────────────────────────────────────────────────────
RAMSES = dict(
    G8  = os.path.expanduser('~/Documents/RAMSES_dev/pyCALIMA_testing/test_outputs'
                              '/CorrectSN_0.03Zsun_DTMini1d-3_MR/output_00081'),
    G9  = os.path.expanduser('~/Documents/RAMSES_dev/pyCALIMA_testing/test_outputs'
                              '/CorrectSN_0.1Zsun_DTMini1d-3/output_00081'),
    G10 = os.path.expanduser('~/Documents/RAMSES_dev/pyCALIMA_testing/test_outputs'
                              '/CorrectSN_MR/output_00081'),
)
SPHERE_KPC = dict(G8=5.0, G9=10.0, G10=15.0)

# ── Solver configurations ──────────────────────────────────────────────────────
# NK: Newton-Krylov steady-state solver (no time integration)
NK_CFG = {'type': 'newton_krylov'}
# Anninos: quasi-implicit time-stepping — unconditionally stable for stiff cells
ANNINOS_CFG = {'type': 'anninos', 'errmax': 0.1, 'countmax': 500000,
               'h_max_Myr': 20.0, 'h_init_s': 1.0e8}

_NON_METALS = {'H', 'He'}


# ── IC prescriptions ───────────────────────────────────────────────────────────

def rr2014_dtm(Z_Zsun):
    """Rémy-Ruyer+2014 DTM–metallicity relation (power-law break at 0.20 Zsun)."""
    if Z_Zsun <= 0: return 0.0
    dtm_mw, z_break, slope_lo = 0.40, 0.20, 3.0
    dtg_mw  = dtm_mw * ZSUN
    dtg_brk = dtg_mw * z_break
    dtg = dtg_mw * Z_Zsun if Z_Zsun >= z_break else dtg_brk * (Z_Zsun / z_break) ** slope_lo
    return dtg / (Z_Zsun * ZSUN)

def dubois_dtm(Z_Zsun):
    """Dubois+2024 DTM–metallicity relation (shallow power-law break at 0.203 Zsun)."""
    if Z_Zsun <= 0: return 0.0
    dtm_mw, z_break, slope_lo = 0.276, 0.203, 1.866
    if Z_Zsun >= z_break: return dtm_mw
    return dtm_mw * (Z_Zsun / z_break) ** (slope_lo - 1.0)


# ── Config builder ─────────────────────────────────────────────────────────────

def _z_ref(cfg):
    """Total metal mass fraction in the template config."""
    return sum(v['mass_fraction'] for k, v in cfg['elemental_abundances'].items()
               if k not in _NON_METALS)

def make_config(Z_abs, sig_kms, nH, ic_dtm_fracs, solver_cfg):
    """Build a pyCALIMA JSON config for the given cell properties.

    Scales all metal elements proportionally from the G8-0.03Zsun template,
    then rebalances H so that mass fractions sum to 1.
    """
    with open('solvers/configs/ramses_G8_0.03Zsun.json') as f:
        cfg = json.load(f)
    cfg['solver'] = solver_cfg
    cfg['environment']['metallicity_absolute']        = float(Z_abs)
    cfg['environment']['velocity_dispersion_kms']     = float(sig_kms)
    cfg['environment']['hydrogen_number_density_cm3'] = float(nH)

    # Scale all metal elements by the ratio of requested Z to template Z
    z_scale = float(Z_abs) / _z_ref(cfg)
    for el, props in cfg['elemental_abundances'].items():
        if el not in _NON_METALS:
            props['mass_fraction'] = float(props['mass_fraction']) * z_scale

    # Rebalance H to conserve total mass fraction
    X_He     = cfg['elemental_abundances']['He']['mass_fraction']
    X_metals = sum(v['mass_fraction'] for k, v in cfg['elemental_abundances'].items()
                   if k not in _NON_METALS)
    cfg['elemental_abundances']['H']['mass_fraction'] = max(1.0 - X_He - X_metals, 0.0)

    # Set dust IC: rho_dust_bin = DTM_bin * Z_abs * rho_gas
    rho_ic = float(nH) * MH * MU
    for k, db in enumerate(cfg['dust_bins']):
        db['initial_mass_density_gcm3'] = float(ic_dtm_fracs[k] * Z_abs * rho_ic)
    return cfg


# ── Single-cell solver ─────────────────────────────────────────────────────────

def _call_solver(T, nH, Z_abs, sig, ic_fracs, solver_cfg, t_end_Myr):
    """Run one (T, nH, Z, sig) cell through the pyCALIMA solver.

    Writes a temporary JSON config, calls run_grid for a 1×1 grid, and
    returns (DTM, f_small, f_carb) at the end of the integration.
    """
    cfg = make_config(Z_abs, sig, nH, ic_fracs, solver_cfg)
    cfg['environment']['temperature_K'] = float(T)

    # Temporary config file — use the system temp dir (safe for parallel workers)
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False,
                                     dir=tempfile.gettempdir()) as f:
        json.dump(cfg, f)
        tmp = f.name
    try:
        g = run_grid(tmp, x_param='T', x_values=[float(T)],
                     y_param='nH', y_values=[float(nH)],
                     solver_type=solver_cfg['type'],
                     t_end_Myr=t_end_Myr, n_jobs=1, verbose=False)
    finally:
        os.unlink(tmp)

    rho_d   = g['rho_dust'][0, 0]          # shape (4,) — one entry per bin
    total_d = float(np.sum(rho_d))
    dtm     = total_d / max(Z_abs * nH * MH * MU, 1e-300)
    if total_d > 0:
        fs = float((rho_d[0] + rho_d[2]) / total_d)   # (SmC + SmSi) / total
        fc = float((rho_d[0] + rho_d[1]) / total_d)   # (SmC + LgC)  / total
    else:
        fs = fc = 0.0
    return dtm, fs, fc


def _do_cell(cell):
    """Evaluate all 6 solver/IC/time combinations for one cell.

    The 6 combinations span: solver (NK, Anninos) × IC (RR14, Dubois, Low)
    × integration time (t_ff or 400 Myr).  NaN is stored if the solver fails.
    """
    T, nH, Z, sig = cell['T'], cell['nH'], cell['Z_abs'], cell['sig']

    # Free-fall time [Myr] — upper bound for how long a cell can evolve
    # before gravitational collapse; used by Ann-tff methods.
    t_ff  = 43.6 / np.sqrt(max(nH, 1e-6))
    t_tff = float(min(t_ff, T_SIM_MYR))

    # IC dust mass per bin = Zubko fraction × DTM(IC) × Z_abs
    ic_rr = list(ZUBKO_FRACS * rr2014_dtm(Z / ZSUN))
    ic_du = list(ZUBKO_FRACS * dubois_dtm(Z / ZSUN))
    ic_lo = list(ZUBKO_FRACS * 1e-3)

    out = {}
    for key, ic, solver, t in [
        ('nk_dub',     ic_du, NK_CFG,      T_SIM_MYR),   # NK steady-state, Dubois IC
        ('ann_tff_rr', ic_rr, ANNINOS_CFG, t_tff),        # Anninos to t_ff, RR14 IC
        ('ann_tff_du', ic_du, ANNINOS_CFG, t_tff),        # Anninos to t_ff, Dubois IC
        ('ann400_du',  ic_du, ANNINOS_CFG, 400.0),        # Anninos 400 Myr, Dubois IC
        ('ann400_lo',  ic_lo, ANNINOS_CFG, 400.0),        # Anninos 400 Myr, Low IC
        ('nk_lo',      ic_lo, NK_CFG,      T_SIM_MYR),   # NK steady-state, Low IC
    ]:
        try:
            dtm, fs, fc = _call_solver(T, nH, Z, sig, ic, solver, t)
        except Exception:
            dtm = fs = fc = np.nan
        out[f'dtm_{key}'] = dtm
        out[f'fs_{key}']  = fs
        out[f'fc_{key}']  = fc
    return out


# ── yt field registration ──────────────────────────────────────────────────────

def _add_fields(ds):
    """Register per-bin dust mass density fields on a yt dataset.

    Silicate bins (03, 04) store Si mass in RAMSES; we divide by SioverSil
    to recover the full grain mass.
    """
    for k, (bn, corr) in enumerate([
            ('hydro_dust_bin01', 1.0), ('hydro_dust_bin02', 1.0),
            ('hydro_dust_bin03', SioverSil), ('hydro_dust_bin04', SioverSil)]):
        def _mk(bn=bn, corr=corr):
            return lambda field, data: (
                data[('ramses', bn)] * data[('gas', 'density')] / corr).to('g/cm**3')
        ds.add_field(('gas', f'rho_d{k+1}'), units='g/cm**3',
                     sampling_type='cell', force_override=True, function=_mk())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run pyCALIMA over all valid cells in one RAMSES galaxy snapshot.')
    parser.add_argument('tag', choices=['G8', 'G9', 'G10'],
                        help='Galaxy tag matching the RAMSES path dict above.')
    parser.add_argument('--njobs', type=int, default=8,
                        help='Number of parallel worker processes (default: 8).')
    args = parser.parse_args()
    tag  = args.tag

    os.makedirs(OUTDIR, exist_ok=True)

    # ── Load yt snapshot and extract cell arrays ───────────────────────────────
    print(f'\n[{tag}] Loading yt sphere (R={SPHERE_KPC[tag]} kpc)...', flush=True)
    ds  = yt.load(RAMSES[tag])
    _add_fields(ds)
    cen = ds.domain_center
    reg = ds.sphere(cen, (SPHERE_KPC[tag], 'kpc'))

    rho  = reg[('gas', 'density')].to('g/cm**3').v
    vols = reg[('index', 'cell_volume')].to('cm**3').v
    mass = rho * vols

    try:
        mf = reg[('ramses', 'Metallicity')].v
    except Exception:
        mf = np.zeros(len(rho))
    try:
        T_arr = reg[('gas', 'temperature')].to('K').v
    except Exception:
        T_arr = np.full(len(rho), np.nan)

    # Per-bin dust densities; shape (N, 4)
    rho_d      = np.column_stack([reg[('gas', f'rho_d{k+1}')].v for k in range(4)])
    total_dust = rho_d.sum(axis=1)
    nH_arr     = rho / (MH * MU)
    Z_abs_arr  = mf

    # Simulation ground-truth dust diagnostics
    with np.errstate(invalid='ignore', divide='ignore'):
        DTM_sim = np.where(mf * rho > 0, total_dust / (mf * rho), np.nan)
        fs_sim  = np.where(total_dust > 0, (rho_d[:,0]+rho_d[:,2]) / total_dust, np.nan)
        fc_sim  = np.where(total_dust > 0, (rho_d[:,0]+rho_d[:,1]) / total_dust, np.nan)

    # 1-D velocity dispersion estimate: |v| / sqrt(3), floored at 1 km/s
    try:
        vx = reg[('gas', 'velocity_x')].to('km/s').v
        vy = reg[('gas', 'velocity_y')].to('km/s').v
        vz = reg[('gas', 'velocity_z')].to('km/s').v
        sig_arr = np.maximum(np.sqrt((vx**2 + vy**2 + vz**2) / 3.0), 1.0)
    except Exception:
        sig_arr = np.ones(len(rho))

    # ── Quality filter — use all cells that satisfy basic physical constraints ──
    ok = (Z_abs_arr > 0) & (total_dust > 0) & np.isfinite(DTM_sim) & \
         np.isfinite(T_arr) & (T_arr > 0) & np.isfinite(nH_arr) & (nH_arr > 0)
    idx = np.where(ok)[0]
    print(f'  {len(idx):,} valid cells out of {len(rho):,} total', flush=True)

    # Build per-cell dicts for the parallel worker
    cells = [dict(T=float(T_arr[i]), nH=float(nH_arr[i]),
                  Z_abs=float(Z_abs_arr[i]), sig=float(sig_arr[i]))
             for i in idx]

    # ── Parallel solver evaluation ─────────────────────────────────────────────
    eta_h = len(cells) * 6 / args.njobs / 3600
    print(f'  Running {len(cells):,} cells × 6 methods ({args.njobs} workers) '
          f'— estimated wall time: {eta_h:.1f} h', flush=True)
    results = Parallel(n_jobs=args.njobs, prefer='processes', verbose=5)(
        delayed(_do_cell)(c) for c in cells)

    # ── Assemble and save output ───────────────────────────────────────────────
    keys = list(results[0].keys())
    save = {k: np.array([r[k] for r in results]) for k in keys}
    save['T']       = T_arr[idx]
    save['nH']      = nH_arr[idx]
    save['Z_abs']   = Z_abs_arr[idx]
    save['sig']     = sig_arr[idx]
    save['mass']    = mass[idx]
    save['DTM_sim'] = DTM_sim[idx]
    save['fs_sim']  = fs_sim[idx]
    save['fc_sim']  = fc_sim[idx]

    out_path = os.path.join(OUTDIR, f'per_cell_all_{tag}.npz')
    np.savez(out_path, **save)
    print(f'\n[{tag}] Saved {out_path}  ({len(cells):,} cells)', flush=True)

    # Summary statistics
    print(f"\n{'Method':22s}  {'valid':>6}  {'DTM median':>10}  {'f_small median':>14}")
    for k in keys:
        v = save[k]
        ok_k = np.isfinite(v)
        print(f"  {k:22s}  {ok_k.sum():6d}  {np.nanmedian(v):10.4f}  —")


if __name__ == '__main__':
    main()
