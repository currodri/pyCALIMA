"""
compare_nh_rrkm_vs_andrews_poly.py
====================================
Recompute the C54H18 hydrogenation-state distribution f(Nh) vs Av along the
NGC7023 NW PDR using the Andrews et al. (2016) Table B.2 polynomial fits for
k_Hloss / k_H2loss in place of our stochastic RRKM computation.

For states NOT covered by Table B.2 we fall back to the RRKM class rates.
Both model variants are overlaid on the same axes together with the digitised
Andrews+16 Fig. 7 data, so we can directly see whether the H-loss rate
discrepancy is the primary cause of the f(Nh) mismatch.

Usage
-----
    python -m models.PAH_photophysics.compare_nh_rrkm_vs_andrews_poly
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    recombination_rate_Bakes1994,
    attachment_rate_Bakes1994,
    se_neutral_Andrews2016,
)
from models.PAH_photophysics.pah_dissociation import compute_total_photoionisation_rate
from models.PAH_photophysics.pah_network_solver import PAHNetworkSolver
from models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from models.PAH_photophysics.pah_radiation import load_kurucz_u_E, load_kurucz_I_nu
from models.PAH_photophysics.pah_h_state import compute_solo_duo_counts
from models.PAH_photophysics.reproduce_andrews16_fig9 import (
    _STATES_DIR, _HV_EV, _worker, _build_kdis,
)
from models.PAH_photophysics.reproduce_charge_vs_ngc7023 import load_pdr_profiles
from models.PAH_photophysics.reproduce_nh_vs_ngc7023 import load_andrews_fig7

_EXT = ROOT / 'external_data'

# ── C54H18 definition ─────────────────────────────────────────────────────────
Nc54, Nh0_54, solo_54, duo_54 = 54, 18, 6, 12
EA_54 = 1.44   # eV
IP1_54, IP2_54 = 6.14, 8.91

N_AV = 25
NH_COMPARE = [14, 15, 16, 17, 18]

# ── Andrews+16 Table B.2 polynomial coefficients ──────────────────────────────
# Keyed by (Z, Nh).  Each entry: {'H': [p0..p4], 'H2': [p0..p4] or None}
_TABLE_B2: dict[tuple, dict] = {
    (-1, 18): {'H': [-13.972,  0.541,  0.498, -0.087,  0.007],
               'H2':[-14.226,  1.179, -0.231,  0.079, -0.005]},
    (-1, 19): {'H': [ -2.676,  0.973,  0.040, -0.017,  0.003], 'H2': None},
    (-1, 20): {'H': [ -2.866,  1.275, -0.130,  0.019,  0.0  ], 'H2': None},
    ( 0, 17): {'H': [-10.438,  0.597,  0.375, -0.058,  0.005], 'H2': None},
    ( 0, 18): {'H': [-14.148,  1.962, -0.031, -0.009,  0.003],
               'H2':[-13.527,  1.051, -0.121,  0.060, -0.004]},
    ( 0, 19): {'H': [ -0.624,  0.979,  0.031, -0.014,  0.002], 'H2': None},
    ( 0, 20): {'H': [ -0.982,  1.228, -0.115,  0.021, -0.001], 'H2': None},
    ( 1,  1): {'H': [-10.821,  0.825,  0.101,  0.015, -0.001], 'H2': None},
    ( 1, 18): {'H': [-13.752,  0.537,  0.503, -0.089,  0.007],
               'H2':[-14.036,  1.179, -0.230,  0.078, -0.005]},
    ( 1, 19): {'H': [ -2.904,  0.961,  0.058, -0.026,  0.004], 'H2': None},
}


def _poly_rate(coeffs: list[float], G0: float) -> float:
    """k(G0) = 10^(Σ p_i × (log10 G0)^i)."""
    x = np.log10(max(G0, 1e-10))
    return float(10.0 ** sum(c * x**i for i, c in enumerate(coeffs)))


def _build_kdis_poly(solver: PAHNetworkSolver, rates_rrkm: dict,
                     G0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build k_Hloss / k_H2loss using Andrews+16 Table B.2 polynomials where
    available, RRKM class rates elsewhere.
    """
    Nh0  = solver.Nh0
    k_Hloss  = np.zeros((solver.nZ, solver.nNh))
    k_H2loss = np.zeros((solver.nZ, solver.nNh))

    Y_H_duo,  Y_H2 = rates_rrkm['H_even_duo']
    Y_H_solo, _    = rates_rrkm['H_even_solo']
    Y_H_odd,  _    = rates_rrkm['H_odd']
    Y_sHn, _       = rates_rrkm['superH_neutral']
    Y_sHc, _       = rates_rrkm['superH_cation']

    for iz, Z in enumerate(solver.Z_vals):
        for inh, Nh in enumerate(solver.Nh_vals):
            if Nh == 0:
                continue

            key = (int(Z), int(Nh))
            if key in _TABLE_B2:
                k_Hloss[iz, inh] = _poly_rate(_TABLE_B2[key]['H'], G0)
                if _TABLE_B2[key]['H2'] is not None:
                    k_H2loss[iz, inh] = _poly_rate(_TABLE_B2[key]['H2'], G0)
            else:
                # RRKM fallback
                if Nh > Nh0:
                    k_Hloss[iz, inh] = Y_sHc if Z > 0 else Y_sHn
                elif Nh % 2 == 1:
                    k_Hloss[iz, inh] = Y_H_odd
                else:
                    state = compute_solo_duo_counts(int(Nh), solo_54, duo_54)
                    if state['H2_loss_possible']:
                        k_Hloss[iz, inh]  = Y_H_duo
                        k_H2loss[iz, inh] = Y_H2
                    else:
                        k_Hloss[iz, inh] = Y_H_solo

    return k_Hloss, k_H2loss


def _solve_network(solver, k_Hloss, k_H2loss, k_ion, k_rec, k_att,
                   n_H, n_e, T, a_cm):
    try:
        return solver.solve_equilibrium(
            k_Hloss, k_H2loss, k_ion, k_rec, k_att,
            n_H=n_H, n_e=n_e, T=T, a_pah_cm=a_cm,
            n_PAH_total=1.0, method='newton',
        )
    except RuntimeError:
        return solver.solve_equilibrium(
            k_Hloss, k_H2loss, k_ion, k_rec, k_att,
            n_H=n_H, n_e=n_e, T=T, a_pah_cm=a_cm,
            n_PAH_total=1.0, method='direct',
        )


def main() -> None:
    print("Loading PDR profiles ...", flush=True)
    av_grid, G0_grid, ne_grid, nH_grid, T_grid, _ = load_pdr_profiles(N_AV)
    print(f"  Av=[{av_grid[0]:.3f},{av_grid[-1]:.3f}]  "
          f"G0=[{G0_grid.min():.1e},{G0_grid.max():.1e}]", flush=True)

    print("\nComputing Kurucz G0 base ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G0_base = float(compute_base_g0(load_kurucz_u_E(15000)))
    print(f"  G0_base = {G0_base:.4e}", flush=True)

    a0 = afromNc(Nc54)
    w, C_abs = get_absorption_cross_section(0, a0)
    E_cs = _HV_EV / w
    idx  = np.argsort(E_cs)
    xsect = np.column_stack([E_cs[idx], C_abs[idx]])
    modes_path = str(_STATES_DIR / 'C54H18_0.dat')

    # ── Run workers for every Av point (same as existing script) ─────────────
    tasks = [
        ('C54', modes_path, xsect, float(G0_grid[iav]), G0_base, IP1_54, IP2_54)
        for iav in range(N_AV)
    ]
    n_workers = max(1, min(cpu_count() - 1, N_AV))
    print(f"\nRunning {N_AV} rate-table tasks on {n_workers} workers ...", flush=True)
    with Pool(n_workers) as pool:
        raw = pool.map(_worker, tasks)
    rates_by_iav = {iav: raw[iav][2] for iav in range(N_AV)}
    print("Rate tables complete.\n", flush=True)

    # Photodetachment reference rate
    kurucz_I = load_kurucz_I_nu(15000)
    def _field_ref(nu): return kurucz_I(nu) / G0_base
    k_det_ref = float(compute_total_photoionisation_rate(_field_ref, xsect, IP=EA_54))
    print(f"  k_det_ref = {k_det_ref:.3e}  (EA={EA_54} eV)", flush=True)

    solver = PAHNetworkSolver(Nc=Nc54, Nh0=Nh0_54,
                              parent_solo=solo_54, parent_duo=duo_54)
    se_c = se_neutral_Andrews2016(Nc54, EA_54)
    print(f"  se_calib = {se_c:.4f}\n", flush=True)

    # ── Solve at each Av under both rate prescriptions ────────────────────────
    fNh_rrkm = np.zeros((N_AV, solver.nNh))
    fNh_poly = np.zeros((N_AV, solver.nNh))

    for iav in range(N_AV):
        G0 = G0_grid[iav]
        ne = ne_grid[iav]
        nH = nH_grid[iav]   # n(H) atomic, direct from nH.csv
        T  = T_grid[iav]

        k_rec1 = recombination_rate_Bakes1994(Nc54, Z=1, se=1.0, T=T, ne=1.0)
        k_rec2 = recombination_rate_Bakes1994(Nc54, Z=2, se=1.0, T=T, ne=1.0)
        k_rec  = np.array([0.0, 0.0, k_rec1, k_rec2])
        k_att  = attachment_rate_Bakes1994(Nc54, se=se_c, T=T, ne=1.0)
        k_ion  = np.array([k_det_ref * G0, rates_by_iav[iav]['k_ion_1'],
                            rates_by_iav[iav]['k_ion_2'], 0.0])

        # --- RRKM rates (existing approach) ---
        k_Hloss_r, k_H2loss_r = _build_kdis(solver, rates_by_iav[iav], solo_54, duo_54)
        n2d_r = _solve_network(solver, k_Hloss_r, k_H2loss_r, k_ion, k_rec,
                               k_att, nH, ne, T, a0)
        n_tot = n2d_r.sum()
        if n_tot > 0:
            fNh_rrkm[iav] = n2d_r.sum(axis=0) / n_tot

        # --- Andrews+16 polynomial rates (for covered states) ---
        k_Hloss_p, k_H2loss_p = _build_kdis_poly(solver, rates_by_iav[iav], G0)
        n2d_p = _solve_network(solver, k_Hloss_p, k_H2loss_p, k_ion, k_rec,
                               k_att, nH, ne, T, a0)
        n_tot = n2d_p.sum()
        if n_tot > 0:
            fNh_poly[iav] = n2d_p.sum(axis=0) / n_tot

        if iav % 5 == 0:
            inh18 = list(solver.Nh_vals).index(18)
            inh17 = list(solver.Nh_vals).index(17)
            print(f"  Av={av_grid[iav]:.2f}  G0={G0:.1e}  T={T:.0f}K")
            print(f"    RRKM:  f(18)={fNh_rrkm[iav,inh18]:.4f}  "
                  f"f(17)={fNh_rrkm[iav,inh17]:.4f}")
            print(f"    Poly:  f(18)={fNh_poly[iav,inh18]:.4f}  "
                  f"f(17)={fNh_poly[iav,inh17]:.4f}", flush=True)

    _plot(av_grid, fNh_rrkm, fNh_poly, solver)


def _plot(av_grid, fNh_rrkm, fNh_poly, solver) -> None:
    andrews = load_andrews_fig7()

    colors  = {14: '#d62728', 15: '#ff7f0e', 16: '#2ca02c', 17: '#1f77b4', 18: '#9467bd'}
    labels  = {Nh: f'$N_H={Nh}$' for Nh in NH_COMPARE}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True,
                             constrained_layout=True)

    titles = ['RRKM class rates (our model)',
              'Andrews+16 Table B.2 polynomials\n(RRKM fallback for unlisted states)']

    for col, (ax, fNh, title) in enumerate(zip(axes, [fNh_rrkm, fNh_poly], titles)):
        for Nh in NH_COMPARE:
            inh = list(solver.Nh_vals).index(Nh)
            ax.plot(av_grid, fNh[:, inh],
                    color=colors[Nh], lw=2.0,
                    label=labels[Nh] + ' (model)' if col == 0 else labels[Nh])

        for Nh, arr in andrews.items():
            if Nh in colors:
                ax.scatter(arr[:, 0], arr[:, 1],
                           color=colors[Nh], marker='o', s=35, zorder=5,
                           edgecolors='k', linewidths=0.5,
                           label=labels[Nh] + ' (Andrews+16)' if col == 0 else None)

        ax.set_xlabel(r'$A_V$ [mag]', fontsize=12)
        if col == 0:
            ax.set_ylabel('Hydrogenation-state fraction', fontsize=12)
        ax.set_xlim(av_grid[0], av_grid[-1])
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        r'$C_{54}H_{18}$ hydrogenation fractions — NGC7023 NW PDR'
        '\n'
        r'Circles: Andrews+16 Fig. 7 [digitised]  '
        r'Lines: this work  '
        r'[calibrated $s_e$, Kurucz 15 kK, $n(H)$ from nH.csv]',
        fontsize=10,
    )

    out = ROOT / 'nh_rrkm_vs_andrews_poly_ngc7023.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
