"""
reproduce_nh_vs_ngc7023.py
==========================
Compare the circumcoronene (C54H18) hydrogenation-state distribution
along the NGC7023 NW PDR computed by the full (Z, Nh) network against
the digitised Andrews et al. (2016) Fig. 7 data.

For each Av point the network is solved using the self-consistent
(G0, ne, nH, T) profile from the digitised NGC7023 data.  The marginalised
Nh fractions are

    f(Nh, Av) = sum_Z n(Z, Nh, Av) / sum_{Z,Nh} n(Z, Nh, Av)

and compared directly with the Fig. 7 digitised curves for Nh = 14–17.

Usage
-----
    python -m models.PAH_photophysics.reproduce_nh_vs_ngc7023
"""

from __future__ import annotations

import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from pycalima.models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    recombination_rate_Bakes1994,
    attachment_rate_Bakes1994,
    se_neutral_Andrews2016,
)
from pycalima.models.PAH_photophysics.pah_dissociation import compute_total_photoionisation_rate
from pycalima.models.PAH_photophysics.pah_network_solver import PAHNetworkSolver
from pycalima.models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from pycalima.models.PAH_photophysics.pah_radiation import load_kurucz_u_E, load_kurucz_I_nu
from pycalima.models.PAH_photophysics.reproduce_andrews16_fig9 import (
    PAH_DEFS as _PAH_DEFS_BASE, _STATES_DIR, _HV_EV,
    _build_kdis, _worker,
)
from pycalima.models.PAH_photophysics.reproduce_charge_vs_ngc7023 import (
    load_pdr_profiles,
)

# ── PAH: circumcoronene only ──────────────────────────────────────────────────
_EA = {'C54': 1.44}
_PDEF = {k: dict(**v, EA=_EA[k]) for k, v in _PAH_DEFS_BASE.items() if k == 'C54'}
PDEF = _PDEF['C54']
Nc54, Nh0_54 = PDEF['Nc'], PDEF['Nh0']

N_AV = 25

# Nh states to compare with Andrews+16 Fig. 7
NH_COMPARE = [14, 15, 16, 17, 18]

# ── Load digitised Andrews+16 Fig. 7 data ────────────────────────────────────
_PDR_DIR = ROOT / 'external_data' / 'NWPDR_NGC7023'


def load_andrews_fig7() -> dict[int, np.ndarray]:
    """
    Parse Nh_circumcoronene.csv.

    Returns {Nh: array(N,2)} with columns [Av, fraction].
    Skips empty cells that pad shorter columns.
    """
    raw = np.genfromtxt(
        _PDR_DIR / 'Nh_circumcoronene.csv',
        delimiter=',', skip_header=2,    # skip 'Nh=14,,Nh=15,...' and 'X,Y,...'
        filling_values=np.nan,
    )
    nh_cols = {14: (0, 1), 15: (2, 3), 16: (4, 5), 17: (6, 7)}
    out = {}
    for Nh, (cx, cy) in nh_cols.items():
        mask = np.isfinite(raw[:, cx]) & np.isfinite(raw[:, cy])
        if mask.any():
            out[Nh] = raw[np.ix_(mask, [cx, cy])]
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading PDR profiles ...", flush=True)
    av_grid, G0_grid, ne_grid, nH_grid, T_grid, gamma_grid = load_pdr_profiles(N_AV)
    print(f"  Av=[{av_grid[0]:.3f},{av_grid[-1]:.3f}]  "
          f"G0=[{G0_grid.min():.1e},{G0_grid.max():.1e}]  "
          f"T=[{T_grid.min():.0f},{T_grid.max():.0f}] K", flush=True)

    print("\nComputing Kurucz G0 base ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G0_base = float(compute_base_g0(load_kurucz_u_E(15000)))
    print(f"  G0_base = {G0_base:.4e}", flush=True)

    # Cross-section table for C54
    a0 = afromNc(Nc54)
    w, C_abs = get_absorption_cross_section(0, a0)
    E_cs = _HV_EV / w
    idx  = np.argsort(E_cs)
    xsect = np.column_stack([E_cs[idx], C_abs[idx]])

    modes_path = str(_STATES_DIR / PDEF['modes'])

    # Build worker tasks
    tasks = [(
        'C54', modes_path, xsect, float(G0_grid[iav]), G0_base,
        PDEF['IP1'], PDEF['IP2'],
    ) for iav in range(N_AV)]

    n_workers = max(1, min(cpu_count() - 1, N_AV))
    print(f"\nRunning {N_AV} rate-table tasks on {n_workers} workers ...", flush=True)
    with Pool(n_workers) as pool:
        raw = pool.map(_worker, tasks)

    rates_by_iav = {iav: raw[iav][2] for iav in range(N_AV)}
    print("Rate tables complete.\n", flush=True)

    # Photodetachment reference rate
    print("Computing photodetachment reference rate ...", flush=True)
    kurucz_I = load_kurucz_I_nu(15000)
    def _field_ref(nu): return kurucz_I(nu) / G0_base
    k_det_ref = float(compute_total_photoionisation_rate(_field_ref, xsect, IP=PDEF['EA']))
    print(f"  k_det_ref = {k_det_ref:.3e}  (EA={PDEF['EA']} eV)", flush=True)

    # Build solver
    solver = PAHNetworkSolver(Nc=Nc54, Nh0=Nh0_54,
                              parent_solo=PDEF['solo'], parent_duo=PDEF['duo'])
    print(f"Solver: {solver.N} species  "
          f"Nh_vals = {solver.Nh_vals[0]}..{solver.Nh_vals[-1]}", flush=True)

    se_c = se_neutral_Andrews2016(Nc54, PDEF['EA'])
    print(f"se_calib = {se_c:.4f}\n", flush=True)

    # fNh[iav, iNh_in_solver] marginalized over Z
    fNh = np.zeros((N_AV, solver.nNh))

    for iav in range(N_AV):
        G0 = G0_grid[iav]
        ne = ne_grid[iav]
        nH = nH_grid[iav]
        T  = T_grid[iav]

        n_H = nH     # nH.csv is already n(H) atomic hydrogen
        n_e = ne

        k_rec1 = recombination_rate_Bakes1994(Nc54, Z=1, se=1.0, T=T, ne=1.0)
        k_rec2 = recombination_rate_Bakes1994(Nc54, Z=2, se=1.0, T=T, ne=1.0)
        k_rec  = np.array([0.0, 0.0, k_rec1, k_rec2])

        k_att = attachment_rate_Bakes1994(Nc54, se=se_c, T=T, ne=1.0)

        rates    = rates_by_iav[iav]
        k_Hloss, k_H2loss = _build_kdis(solver, rates, PDEF['solo'], PDEF['duo'])
        k_ion = np.array([
            k_det_ref * G0,
            rates['k_ion_1'],
            rates['k_ion_2'],
            0.0,
        ])

        try:
            n2d = solver.solve_equilibrium(
                k_Hloss, k_H2loss, k_ion, k_rec, k_att,
                n_H=n_H, n_e=n_e, T=T, a_pah_cm=a0,
                n_PAH_total=1.0, method='newton',
            )
        except RuntimeError:
            n2d = solver.solve_equilibrium(
                k_Hloss, k_H2loss, k_ion, k_rec, k_att,
                n_H=n_H, n_e=n_e, T=T, a_pah_cm=a0,
                n_PAH_total=1.0, method='direct',
            )

        n_tot = n2d.sum()
        if n_tot > 0:
            for inh in range(solver.nNh):
                fNh[iav, inh] = n2d[:, inh].sum() / n_tot

        if iav % 5 == 0:
            dominant = solver.Nh_vals[np.argmax(fNh[iav])]
            print(f"  Av={av_grid[iav]:.2f}  G0={G0:.1e}  T={T:.0f}K  "
                  f"dominant Nh={dominant}  "
                  f"f(18)={fNh[iav, 18]:.3f}  f(17)={fNh[iav, 17]:.3f}  "
                  f"f(16)={fNh[iav, 16]:.3f}  f(15)={fNh[iav, 15]:.3f}  "
                  f"f(14)={fNh[iav, 14]:.3f}", flush=True)

    _plot(av_grid, fNh, solver)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(av_grid, fNh, solver) -> None:
    andrews = load_andrews_fig7()

    colors  = {14: '#d62728', 15: '#ff7f0e', 16: '#2ca02c', 17: '#1f77b4', 18: '#9467bd'}
    labels  = {Nh: f'$N_H={Nh}$' for Nh in NH_COMPARE}

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for Nh in NH_COMPARE:
        inh = list(solver.Nh_vals).index(Nh)
        ax.plot(av_grid, fNh[:, inh],
                color=colors[Nh], lw=2.0, label=labels[Nh] + ' (model)')

    for Nh, arr in andrews.items():
        if Nh in colors:
            ax.scatter(arr[:, 0], arr[:, 1],
                       color=colors[Nh], marker='o', s=35, zorder=5,
                       edgecolors='k', linewidths=0.5,
                       label=labels[Nh] + ' (Andrews+16)')

    ax.set_xlabel(r'$A_V$ [mag]', fontsize=12)
    ax.set_ylabel('Hydrogenation-state fraction', fontsize=12)
    ax.set_xlim(av_grid[0], av_grid[-1])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which='both', alpha=0.25, lw=0.5)
    ax.set_title(
        r'Circumcoronene ($C_{54}H_{18}$) hydrogenation-state fractions — NGC7023 NW PDR'
        '\n'
        r'Lines: full $(Z,N_H)$ network with NGC7023 $(G_0, n_e, n_H, T)(A_V)$  '
        r'[calibrated $s_e$, Kurucz 15 kK]'
        '\nCircles: Andrews et al. (2016) Fig. 7 [digitised]',
        fontsize=9.5,
    )

    out = ROOT / 'nh_distribution_vs_ngc7023_pdr.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
