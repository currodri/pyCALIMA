"""
reproduce_andrews16_fig9.py
===========================
Reproduces Fig. 9 of Andrews et al. (2016): steady-state PAH hydrogenation
fractions on a 2-D (G0, nH) grid for coronene (C24H12), circumcoronene
(C54H18), and circumcircumcoronene (C96H24).

Nine panels (3 molecules × 3 hydrogenation categories):
  Columns: fully dehydrogenated (Nh=0), normal (Nh=Nh0), super-H (Nh>Nh0)
  Colour:  fraction of PAHs in that hydrogenation state, summed over all
           charge states, on a log colour scale.

Physical assumptions
--------------------
  Radiation field : Kurucz 15000 K stellar spectrum (Andrews 2016 choice)
  T_gas           : 500 K (fixed)
  Molecular H frac: fH2 = 0.5  →  n_H_atomic = (1 - fH2) × nH
  Electron density: n_e = x_C × nH,  x_C = 1.6e-4  (C+ ionisation)
  Charge network  : Z ∈ {-1, 0, +1, +2}, solved by Newton-Raphson

Usage
-----
    python -m models.PAH_photophysics.reproduce_andrews16_fig9
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count

from models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    recombination_rate_Tielens21,
    attachment_rate_Carelli13,
)
from models.PAH_photophysics.pah_h_state import compute_solo_duo_counts
from models.PAH_photophysics.pah_network_solver import PAHNetworkSolver
from models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from models.PAH_photophysics.pah_radiation import load_kurucz_u_E

# ─── PAH definitions ──────────────────────────────────────────────────────────
# Topology from NASA Ames PAHdb (catalogued uid, D6h/Ag symmetry isomers).
# IP values from IONISATION_POTENTIAL in pah_charge_utils.py.
PAH_DEFS = {
    'C24': dict(Nc=24,  Nh0=12, solo=0,  duo=12,
                modes='C24H12_0.dat', label='Coronene (C$_{24}$H$_{12}$)',
                IP1=7.20, IP2=11.50),
    'C54': dict(Nc=54,  Nh0=18, solo=6,  duo=12,
                modes='C54H18_0.dat', label='Circumcoronene (C$_{54}$H$_{18}$)',
                IP1=6.14, IP2=8.91),
    'C96': dict(Nc=96,  Nh0=24, solo=12, duo=12,
                modes='C96H24_0.dat', label='Circumcircumcoronene (C$_{96}$H$_{24}$)',
                IP1=5.68, IP2=8.24),
}
_PAH_ORDER = ['C24', 'C54', 'C96']

# ─── Andrews (2016) RRKM classes ──────────────────────────────────────────────
# Each entry: (class_name, E_act_H [eV], ΔS_H [J K^-1 mol^-1],
#                          E_act_H2 [eV], ΔS_H2 [J K^-1 mol^-1])
#
# RRKM Arrhenius formula (Tielens 2005):
#   k = e × (k_B T_e / h) × exp(ΔS/R) × exp(−E_act / k_B T_e)
#   T_e = T_m × (1 − 0.2 × E_act / E),   U_QHO(T_m) = E
#
# Class assignment (also see _build_kdis and compare_rrkm_rates.rrkm_class):
#   H_even_duo     : even Nh, at least one pair of adjacent H atoms → H2 loss possible
#   H_even_solo    : even Nh, all H atoms isolated → H2 loss suppressed (E_act_H2=100)
#   H_odd          : odd Nh (one lone H, lower barrier)
#   superH_neutral : Nh > Nh0, Z ≤ 0  (sp3-bonded extra H, very low barrier)
#   superH_cation  : Nh > Nh0, Z > 0
#
# E_act_H2 = 100 eV is a sentinel that numerically kills the H2-loss channel.
# All parameter values are from Andrews et al. (2016), originally Tielens (2005).
_RRKM_CLASSES = [
    ('H_even_duo',     4.60, 44.8,  3.52, -53.1),
    ('H_even_solo',    4.60, 44.8, 100.0,   0.0),
    ('H_odd',          4.10, 55.6, 100.0,   0.0),
    ('superH_neutral', 1.40, 55.6, 100.0,   0.0),
    ('superH_cation',  1.55, 55.6, 100.0,   0.0),
]

# ─── Grid ─────────────────────────────────────────────────────────────────────
NG0, NNH = 25, 25
G0_GRID = np.logspace(0, 5, NG0)
NH_GRID = np.logspace(0, 5, NNH)

# ─── Physical parameters ──────────────────────────────────────────────────────
T_GAS  = 500.0   # K  (fixed gas temperature)
F_H2   = 0.5     # molecular hydrogen fraction → n_H_atomic = (1 - F_H2) * nH
X_E    = 1.6e-4  # electron fraction (C+ ionisation) → n_e = X_E * nH

_STATES_DIR = ROOT / 'model_data' / 'PAH_states'
_HV_EV      = 1.23984193e-4   # h·c in eV·cm


# ─── Worker (module-level for pickling) ───────────────────────────────────────

def _worker(task: tuple) -> tuple:
    """
    Compute GD89 temperature distribution, RRKM branching rates, and
    photoionisation rates for one (PAH, G0) pair.

    Called via Pool.map; all arguments are packed in a single tuple so the
    function is picklable.

    Pipeline
    --------
    1. Scale the Kurucz 15 000 K radiation field to the target G0.
    2. Run the adaptive GD89 temperature distribution solver with the
       PAHdb vibrational modes and Li & Draine (2001) UV cross-section.
    3. For each RRKM class in _RRKM_CLASSES, integrate k_H × W_down / denom
       over f(T) to obtain the class-averaged H-loss and H2-loss rates.
    4. Compute photoionisation rates above IP1 and IP2 using the Jochims
       (1996) ionisation yield.

    Important caveat
    ----------------
    The UV absorption cross-section (xsect) is taken from the Li & Draine
    (2001) PAH grain tables scaled to the PAH radius a = afromNc(Nc), NOT
    from molecule-specific quantum-chemistry cross-sections as used by Andrews
    (2016).  Similarly, U_QHO(T) is derived from PAHdb modes rather than
    B3LYP/6-31G* DFT modes.  Both differences feed exponentially into k_H
    via T_m → T_e, causing size-dependent discrepancies vs. the Andrews
    polynomial fits (see compare_rrkm_rates.py for a quantitative comparison).

    Parameters (packed as tuple)
    ----------------------------
    pah_name  : str
    modes_path: str    — PAHdb .dat file with vibrational frequencies and A_i
    xsect     : ndarray (N, 2) — columns [E_eV, sigma_abs_cm²]
    G0        : float  — radiation field strength (Habing units)
    G0_base   : float  — G0 value of the raw Kurucz spectrum (compute_base_g0)
    IP1, IP2  : float  — first and second ionisation potentials [eV]

    Returns
    -------
    (pah_name, G0, rates_dict)
    rates_dict keys: 'k_ion_1', 'k_ion_2',
                     'H_even_duo', 'H_even_solo', 'H_odd',
                     'superH_neutral', 'superH_cation'
    Each RRKM key maps to (Y_H, Y_H2) in s^-1.
    """
    pah_name, modes_path, xsect, G0, G0_base, IP1, IP2 = task

    from models.PAH_photophysics.pah_radiation import load_kurucz_I_nu
    from models.PAH_photophysics.pah_temperature import (
        compute_adaptive_temperature_distribution,
    )
    from models.PAH_photophysics.pah_dissociation import (
        compute_branching_integrated_rates,
        compute_total_photoionisation_rate,
    )

    kurucz_I_nu = load_kurucz_I_nu(15000)

    def field(nu: float) -> float:
        return (G0 / G0_base) * kurucz_I_nu(nu)

    # ── Temperature distribution (GD89, expensive) ──
    t_centers, f_T = compute_adaptive_temperature_distribution(
        modes_path, field, xsect, t_min=15.0, num_bins=150,
    )

    # ── Photoionisation rates ──
    k_ion_1 = float(compute_total_photoionisation_rate(field, xsect, IP=IP1))
    k_ion_2 = float(compute_total_photoionisation_rate(field, xsect, IP=IP2))

    # ── RRKM branching rates for all Andrews (2016) classes ──
    rrkm: dict = {}
    for cls, E_H, dS_H, E_H2, dS_H2 in _RRKM_CLASSES:
        Y_H, Y_H2 = compute_branching_integrated_rates(
            modes_path, t_centers, f_T, E_H, dS_H, E_H2, dS_H2,
        )
        rrkm[cls] = (float(Y_H), float(Y_H2))

    rates: dict = {'k_ion_1': k_ion_1, 'k_ion_2': k_ion_2}
    rates.update(rrkm)
    return pah_name, float(G0), rates


# ─── Helper: build k_Hloss / k_H2loss arrays from rate dict ──────────────────

def _build_kdis(
    solver: PAHNetworkSolver,
    rates:  dict,
    solo:   int,
    duo:    int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map RRKM class rates onto the (nZ, nNh) photodissociation arrays."""
    Nh0 = solver.Nh0
    k_Hloss  = np.zeros((solver.nZ, solver.nNh))
    k_H2loss = np.zeros((solver.nZ, solver.nNh))

    Y_H_duo,  Y_H2    = rates['H_even_duo']
    Y_H_solo, _       = rates['H_even_solo']
    Y_H_odd,  _       = rates['H_odd']
    Y_superH_n, _     = rates['superH_neutral']
    Y_superH_c, _     = rates['superH_cation']

    for iz, Z in enumerate(solver.Z_vals):
        for inh, Nh in enumerate(solver.Nh_vals):
            if Nh == 0:
                continue   # bare carbon — no H to lose

            if Nh <= Nh0:
                # Normal / de-hydrogenated regime
                state = compute_solo_duo_counts(int(Nh), solo, duo)
                if Nh % 2 == 1:
                    k_Hloss[iz, inh] = Y_H_odd
                elif state['H2_loss_possible']:
                    k_Hloss[iz, inh]  = Y_H_duo
                    k_H2loss[iz, inh] = Y_H2
                else:
                    k_Hloss[iz, inh] = Y_H_solo
            else:
                # Super-hydrogenated regime
                if int(Z) > 0:
                    k_Hloss[iz, inh] = Y_superH_c
                else:
                    k_Hloss[iz, inh] = Y_superH_n

    return k_Hloss, k_H2loss


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Precompute G0_base from Kurucz 15000 K spectrum ──
    print("Computing Kurucz G0 base ...", flush=True)
    kurucz_u_E = load_kurucz_u_E(15000)
    G0_base    = float(compute_base_g0(kurucz_u_E))
    print(f"  G0_base = {G0_base:.4f}")

    # ── Build cross-section tables (one per PAH size) ──
    xsect_tables: dict[str, np.ndarray] = {}
    for name, pdef in PAH_DEFS.items():
        a0 = afromNc(pdef['Nc'])
        w, C_abs = get_absorption_cross_section(0, a0)
        E_cs = _HV_EV / w
        idx  = np.argsort(E_cs)
        xsect_tables[name] = np.column_stack([E_cs[idx], C_abs[idx]])

    # ── Build task list for multiprocessing ──
    tasks = []
    for name, pdef in PAH_DEFS.items():
        modes_path = str(_STATES_DIR / pdef['modes'])
        xsect      = xsect_tables[name]
        for G0 in G0_GRID:
            tasks.append((
                name, modes_path, xsect,
                float(G0), G0_base,
                pdef['IP1'], pdef['IP2'],
            ))

    n_tasks   = len(tasks)
    n_workers = max(1, min(cpu_count() - 1, n_tasks))
    print(f"\nRunning {n_tasks} rate-table tasks on {n_workers} workers ...", flush=True)

    with Pool(n_workers) as pool:
        raw_results = pool.map(_worker, tasks)

    # ── Collect into nested dict: rates_table[pah_name][G0_idx] ──
    rates_table: dict[str, dict] = {name: {} for name in PAH_DEFS}
    for pah_name, G0, rates in raw_results:
        ig0 = int(np.argmin(np.abs(G0_GRID - G0)))
        rates_table[pah_name][ig0] = rates

    print("Rate tables complete.\n", flush=True)

    # ── Solve network for each (PAH, G0, nH) grid point ──
    # fracs[pah_name][state] = ndarray (NG0, NNH)
    fracs: dict[str, dict] = {}

    for name, pdef in PAH_DEFS.items():
        Nc   = pdef['Nc']
        Nh0  = pdef['Nh0']
        solo = pdef['solo']
        duo  = pdef['duo']

        solver = PAHNetworkSolver(
            Nc=Nc, Nh0=Nh0, parent_solo=solo, parent_duo=duo,
        )
        a_cm = afromNc(Nc)

        # Charge-rate coefficients (T-dependent only → computed once)
        k_rec_coeff = recombination_rate_Tielens21(Nc, T_GAS, ne=1.0)
        k_att_coeff = attachment_rate_Carelli13(T_GAS, ne=1.0)
        k_rec = np.array([0.0, 0.0, k_rec_coeff, k_rec_coeff])

        inh_normal  = int(np.where(solver.Nh_vals == Nh0)[0][0])
        inh_superH  = solver.Nh_vals > Nh0

        f_dehydro = np.zeros((NG0, NNH))
        f_normal  = np.zeros((NG0, NNH))
        f_superH  = np.zeros((NG0, NNH))

        print(f"Solving network for {name}  ({solver.N} species) ...", flush=True)

        for ig0, G0 in enumerate(G0_GRID):
            rates = rates_table[name][ig0]
            k_Hloss, k_H2loss = _build_kdis(solver, rates, solo, duo)

            # Photoionisation: k_ion[iz] drives Z_vals[iz] → Z_vals[iz]+1
            # Z=-1→0: photodetachment ≈ k_ion_1 (rough approximation)
            # Z=0→1: first ionisation
            # Z=1→2: second ionisation
            # Z=2  : not in network
            k_ion = np.array([
                rates['k_ion_1'],   # photodetachment (Z=-1 → 0)
                rates['k_ion_1'],   # first ionisation (Z=0 → +1)
                rates['k_ion_2'],   # second ionisation (Z=+1 → +2)
                0.0,
            ])

            for inh_gas, nH_total in enumerate(NH_GRID):
                n_H = (1.0 - F_H2) * nH_total
                n_e = X_E * nH_total

                try:
                    n2d = solver.solve_equilibrium(
                        k_Hloss, k_H2loss,
                        k_ion, k_rec, k_att_coeff,
                        n_H=n_H, n_e=n_e, T=T_GAS, a_pah_cm=a_cm,
                        n_PAH_total=1.0, method='newton',
                    )
                except RuntimeError:
                    # Fall back to direct solve if Newton-Raphson fails
                    n2d = solver.solve_equilibrium(
                        k_Hloss, k_H2loss,
                        k_ion, k_rec, k_att_coeff,
                        n_H=n_H, n_e=n_e, T=T_GAS, a_pah_cm=a_cm,
                        n_PAH_total=1.0, method='direct',
                    )

                n_tot = n2d.sum()
                if n_tot > 0:
                    f_dehydro[ig0, inh_gas] = n2d[:, 0].sum()     / n_tot
                    f_normal [ig0, inh_gas] = n2d[:, inh_normal].sum() / n_tot
                    f_superH [ig0, inh_gas] = n2d[:, inh_superH].sum() / n_tot

        fracs[name] = {
            'dehydro': f_dehydro,
            'normal':  f_normal,
            'superH':  f_superH,
        }
        print(f"  done.", flush=True)

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(fracs)


def _plot(fracs: dict) -> None:
    col_titles = [
        'Fully dehydrogenated\n($N_H = 0$)',
        'Normal\n($N_H = N_{H,0}$)',
        'Super-hydrogenated\n($N_H > N_{H,0}$)',
    ]
    state_keys = ['dehydro', 'normal', 'superH']

    fig, axes = plt.subplots(
        3, 3, figsize=(14, 11),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    vmin, vmax = 1e-4, 1.0
    cmap  = plt.cm.viridis
    norm  = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    # nH on x-axis, G0 on y-axis
    NH_m, G0_m = np.meshgrid(NH_GRID, G0_GRID, indexing='ij')

    for row, name in enumerate(_PAH_ORDER):
        pdef = PAH_DEFS[name]
        for col, (key, col_title) in enumerate(zip(state_keys, col_titles)):
            ax  = axes[row, col]
            dat = fracs[name][key].copy()   # shape (NG0, NNH)
            dat = np.clip(dat, vmin, None)

            # dat.T has shape (NNH, NG0): NH_m[i,j]=NH[i], G0_m[i,j]=G0[j]
            pc = ax.pcolormesh(NH_m, G0_m, dat.T, norm=norm, cmap=cmap,
                               shading='nearest')

            # Contours: contour(x1d, y1d, Z) expects Z[j_G0, i_nH] = dat
            levels = [0.1, 0.5, 0.9]
            try:
                cs = ax.contour(NH_GRID, G0_GRID, dat,
                                levels=levels, colors='white',
                                linewidths=0.9, linestyles=['-', '--', '-'])
                ax.clabel(cs, fmt={l: f'{int(l*100):d}%' for l in levels},
                          fontsize=7, inline=True)
            except Exception:
                pass

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(NH_GRID[0], NH_GRID[-1])
            ax.set_ylim(G0_GRID[0], G0_GRID[-1])
            ax.tick_params(labelsize=9)

            if col == 0:
                ax.set_ylabel(f'{pdef["label"]}\n$G_0$', fontsize=9)
            if row == 0:
                ax.set_title(col_title, fontsize=10)
            if row == 2:
                ax.set_xlabel('$n_H$ [cm$^{-3}$]', fontsize=10)

            fig.colorbar(pc, ax=ax, label='Fraction', fraction=0.046, pad=0.04)

    fig.suptitle(
        'PAH hydrogenation fractions  '
        r'($T_{\rm gas}=500\,{\rm K}$, $f_{H_2}=0.5$, Kurucz 15 kK)',
        fontsize=12,
    )

    out = ROOT / 'andrews16_fig9_reproduction.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
