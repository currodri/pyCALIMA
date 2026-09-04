"""
diagnose_temperature_distribution.py
=====================================
Diagnostic script for understanding why C54H18 photodissociation rates differ
from Andrews (2016) quantum-chemical benchmarks.

Produces four figures:
  1. U(T): QHO modes vs Tielens (2005) E-T table vs DustEM DL07 heat capacity
  2. P_IR(T): QHO mode-sum vs spectral ∫ C_abs × B_λ dλ (DustEM/SHIVA approach)
  3. f(T): all three CALIMA solvers at G0 = 1, 100, 1000 (Kurucz 15000 K field)
  4. H-loss and H2-loss rates vs G0: Methods B, C (GD89+Andrews k(E)), E vs Andrews

Usage
-----
    python models/PAH_photophysics/diagnose_temperature_distribution.py
"""

from __future__ import annotations

from pathlib import Path


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pycalima.models.PAH_photophysics.pah_mol_data import load_pah_modes
from pycalima.models.PAH_photophysics.pah_temperature import (
    _qho_energy, _qho_cv,
    compute_base_g0,
    get_absorption_cross_section,
    compute_gd89_temperature_distribution,
    compute_adaptive_temperature_distribution,
    compute_bakes_temperature_distribution,
    compute_spectral_gd89_distribution,
    compute_dustem_poweriter_distribution,
)
from pycalima.models.PAH_photophysics.pah_dissociation import (
    compare_dissociation_methods,
    compute_andrews_direct_branching,
    compute_branching_integrated_rates,
    print_method_comparison,
)
from pycalima.models.PAH_photophysics.pah_mol_data import compute_thermal_ir_rate
from pycalima.models.PAH_photophysics.pah_charge_utils import afromNc
from pycalima.models.PAH_photophysics.pah_radiation import load_kurucz_I_nu, load_kurucz_u_E
from pycalima.plotting_style import use_calima_style
from pycalima import _paths
from pycalima.models.grain_size_config import get_model_data_dir

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXT       = _paths.get_external_data_path()
_STATES    = get_model_data_dir() / 'PAH_states'

def _dustem_file():
    """The DustEM heat-capacity table used for the f(T) cross-check.

    Ships with DustEM (https://www.ias.u-psud.fr/DUSTEM/), not with pyCALIMA.
    Point $CALIMA_DUSTEM_FILE at your own copy of ``hcap/C_PAH0_DL07.DAT``.
    Returns None when unset, which the caller reports as a skipped comparison.
    """
    import os

    raw = os.environ.get('CALIMA_DUSTEM_FILE')
    return Path(raw).expanduser() if raw else None


PAH_FILE   = str(_STATES / 'C54H18_0.dat')
ET_TABLE   = str(_EXT / 'E-Trelation_neutral_circumcoronene_Tielens2005.csv')
KH_TABLE   = str(_EXT / 'H-loss_C54H18_Andrews16.csv')
KH2_TABLE  = str(_EXT / 'H2-loss_C54H18_Andrews16.csv')
GH_CSV     = str(_EXT / 'H-loss_G0_C54H18_Andrews16.csv')
GH2_CSV    = str(_EXT / 'H2-loss-G0_C54H18_Andrews16.csv')

NC   = 54
A0   = afromNc(NC)      # cm,  ≈ 6.61e-8 cm
EV2ERG = 1.602176634e-12
KB_EV  = 8.61733326e-5
HC_EV  = 1.23984193e-4   # h·c in eV·cm
H_CGS  = 6.62607015e-27  # erg·s
C_CGS  = 2.99792458e10   # cm/s
KB_CGS = 1.380649e-16    # erg/K

# ---------------------------------------------------------------------------
# Kurucz 15000 K field factory
# ---------------------------------------------------------------------------
_kurucz_I_nu  = load_kurucz_I_nu(15000)   # I_nu(nu [Hz]) — used in rate integrals
_kurucz_u_E   = load_kurucz_u_E(15000)    # u_E(E [eV])   — used for G0 normalisation

BASE_G0 = float(compute_base_g0(_kurucz_u_E))
print(f"Kurucz 15000K base G0 = {BASE_G0:.3e}")


def field_factory(g0: float):
    """Return I_nu(nu) scaled to the requested G0."""
    scale = g0 / BASE_G0
    def _field(nu):
        return scale * _kurucz_I_nu(nu)
    return _field


# ---------------------------------------------------------------------------
# Cross-section for C54H18 neutral
# ---------------------------------------------------------------------------
wav_cm, C_abs_cm2 = get_absorption_cross_section(Z=0, a0=A0)


# ===========================================================================
# PANEL 1 — U(T) comparison
# ===========================================================================

def _u_qho(freq_ev, T_arr):
    return np.array([_qho_energy(freq_ev, T) for T in T_arr])


def _load_tielens_et():
    """Return T [K] and U_per_molecule [eV] from Tielens (2005) E-T table."""
    data = np.loadtxt(ET_TABLE, delimiter=',')
    T    = data[:, 0]
    U_nc = data[:, 1]      # eV per carbon atom
    return T, U_nc * NC    # scale to full molecule


def _load_dl07_u():
    """
    Read DustEM C_PAH0_DL07.DAT and return (T_K, U_per_molecule_eV).

    File format: comment lines (#), then:
      n_sizes (int), sizes (n_sizes floats), n_T (int),
      then n_T data rows each with: log10(T)  log10(C_1) ... log10(C_ns)

    C is in erg/K/cm³; U per molecule = ∫ C dT × V_grain → eV.
    """
    dustem = _dustem_file()
    if dustem is None:
        print("  Warning: DustEM heat-capacity table not configured; "
              "set $CALIMA_DUSTEM_FILE to hcap/C_PAH0_DL07.DAT from a DustEM "
              "install to enable this comparison.")
        return None, None
    if not dustem.exists():
        print(f"  Warning: DustEM file not found: {dustem}")
        return None, None

    # Collect all numeric tokens from non-comment lines
    tokens = []
    with open(dustem) as fh:
        for line in fh:
            if line.strip().startswith('#') or not line.strip():
                continue
            for tok in line.split():
                try:
                    tokens.append(float(tok))
                except ValueError:
                    pass

    idx = 0
    n_sizes = int(tokens[idx]); idx += 1
    sizes   = tokens[idx:idx+n_sizes]; idx += n_sizes
    n_T     = int(tokens[idx]); idx += 1
    # Each data row: log10(T) + n_sizes values of log10(C)
    n_cols  = 1 + n_sizes
    data    = np.array(tokens[idx:idx + n_T * n_cols]).reshape(n_T, n_cols)

    logT  = data[:, 0]
    logC  = data[:, 1]     # use first grain size column (all equal for PAH0)
    T_K   = 10.0**logT
    C_erg = 10.0**logC    # erg / K / cm³

    # Volume for C54H18 grain
    V_grain = (4.0 / 3.0) * np.pi * A0**3   # cm³

    # Integrate C dT (trapezoidal) to get U [erg/molecule]
    # Note: T grid is monotonic from file order
    idx = np.argsort(T_K)
    T_K = T_K[idx]; C_erg = C_erg[idx]
    U_erg = np.zeros(len(T_K))
    for i in range(1, len(T_K)):
        U_erg[i] = U_erg[i-1] + 0.5*(C_erg[i]+C_erg[i-1])*(T_K[i]-T_K[i-1]) * V_grain

    return T_K, U_erg / EV2ERG    # eV


def plot_u_of_t():
    use_calima_style()
    print("\n=== Panel 1: U(T) comparison ===")
    freq_ev, _ = load_pah_modes(PAH_FILE)

    T_grid = np.logspace(1, 4, 200)   # 10 K to 10000 K
    U_qho  = _u_qho(freq_ev, T_grid)

    T_tiel, U_tiel = _load_tielens_et()
    T_dl07, U_dl07 = _load_dl07_u()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.loglog(T_grid, U_qho,   'b-',  lw=2, label='QHO modes (PAHdb, this work)')
    if T_tiel is not None:
        mask = (T_tiel > 5) & (T_tiel < 12000)
        ax.loglog(T_tiel[mask], U_tiel[mask], 'r--', lw=2, label='Tielens (2005) E-T table ×54')
    if T_dl07 is not None:
        mask = (T_dl07 > 5) & (T_dl07 < 12000)
        ax.loglog(T_dl07[mask], U_dl07[mask], 'g:',  lw=2, label='DustEM DL07 C(T) integral')
    ax.set_xlabel('T [K]', fontsize=13)
    ax.set_ylabel('U(T) [eV / molecule]', fontsize=13)
    ax.set_title('Internal energy U(T) for C$_{54}$H$_{18}$', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(10, 10000)
    fig.tight_layout()
    fig.savefig('diagnostic_U_of_T.png', dpi=150)
    plt.close(fig)
    print("  Saved: diagnostic_U_of_T.png")

    # Print sample values for comparison
    for T_check in [100, 500, 1000, 2000, 5000]:
        u_q = float(np.interp(T_check, T_grid, U_qho))
        u_t = float(np.interp(T_check, T_tiel, U_tiel)) if T_tiel is not None else float('nan')
        u_d = float(np.interp(T_check, T_dl07, U_dl07)) if T_dl07 is not None else float('nan')
        print(f"  T={T_check:5d}K:  U_QHO={u_q:.4f} eV,  U_Tielens={u_t:.4f} eV,  U_DL07={u_d:.4f} eV")


# ===========================================================================
# PANEL 2 — P_IR(T) comparison
# ===========================================================================

def _pir_qho(freq_ev, einstein_A, T):
    """IR cooling power from QHO mode sum [eV/s]."""
    x = freq_ev / (KB_EV * T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
        occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
    return float(np.sum(freq_ev * einstein_A * occ))   # eV/s


def _planck_lambda(T, wav_cm):
    """Planck function B_λ(T) [erg/s/cm²/cm/sr]."""
    x = H_CGS * C_CGS / (wav_cm * KB_CGS * T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        bnu = np.where(x > 500.0, 0.0,
                       2.0 * H_CGS * C_CGS**2 / wav_cm**5 / (np.expm1(x) + 1e-300))
    return bnu


def _pir_spectral(T, wav_cm, C_abs):
    """IR cooling power = 4π ∫ C_abs(λ) B_λ(T) dλ  [erg/s]."""
    B = _planck_lambda(T, wav_cm)
    # sort by wavelength for integration
    idx = np.argsort(wav_cm)
    return float(4.0 * np.pi * np.trapezoid(C_abs[idx] * B[idx], wav_cm[idx]))


def plot_pir():
    use_calima_style()
    print("\n=== Panel 2: P_IR(T) comparison ===")
    freq_ev, einstein_A = load_pah_modes(PAH_FILE)
    active = einstein_A > 0
    freq_a  = freq_ev[active]
    A_a     = einstein_A[active]

    T_grid = np.logspace(1.5, 4, 80)   # ~30 K to 10000 K

    P_qho = np.array([_pir_qho(freq_a, A_a, T) * EV2ERG for T in T_grid])
    P_spec = np.array([_pir_spectral(T, wav_cm, C_abs_cm2) for T in T_grid])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax = axes[0]
    ax.loglog(T_grid, P_qho,  'b-',  lw=2, label='QHO modes (PAHdb)')
    ax.loglog(T_grid, P_spec, 'r--', lw=2, label=r'Spectral: $4\pi\int C_\mathrm{abs} B_\lambda\,d\lambda$')
    ax.set_xlabel('T [K]', fontsize=13)
    ax.set_ylabel('P$_{IR}$(T) [erg s$^{-1}$]', fontsize=13)
    ax.set_title('Cooling rate P$_{IR}$(T) for C$_{54}$H$_{18}$', fontsize=13)
    ax.legend(fontsize=11)

    ax = axes[1]
    ratio = P_spec / np.maximum(P_qho, 1e-40)
    ax.semilogx(T_grid, ratio, 'k-', lw=2)
    ax.axhline(1.0, color='grey', ls=':')
    ax.set_xlabel('T [K]', fontsize=13)
    ax.set_ylabel('P_IR(spectral) / P_IR(QHO)', fontsize=13)
    ax.set_title('Ratio: spectral vs QHO cooling', fontsize=13)

    fig.tight_layout()
    fig.savefig('diagnostic_P_IR.png', dpi=150)
    plt.close(fig)
    print("  Saved: diagnostic_P_IR.png")

    for T_check in [100, 500, 1000, 3000]:
        p_q = float(np.interp(T_check, T_grid, P_qho))
        p_s = float(np.interp(T_check, T_grid, P_spec))
        print(f"  T={T_check:5d}K:  P_QHO={p_q:.3e} erg/s,  P_spec={p_s:.3e} erg/s,  ratio={p_s/max(p_q,1e-40):.2f}")


# ===========================================================================
# PANEL 3 — f(T) comparison
# ===========================================================================

def plot_ft():
    use_calima_style()
    print("\n=== Panel 3: f(T) comparison for GD89, Adaptive, Bakes ===")

    G0_vals  = [1.0, 100.0, 1000.0]
    colors   = {'gd89': 'blue', 'adaptive': 'green', 'bakes': 'red'}

    fig, axes = plt.subplots(1, len(G0_vals), figsize=(5*len(G0_vals), 5), dpi=150, sharey=False)

    xsect = np.column_stack([wav_cm[::-1] * 1e4 / (HC_EV * 1e4), C_abs_cm2[::-1]])
    # Wait — we need xsect as (E_eV, sigma_cm2). Build it from wavelengths.
    E_eV_xsect = HC_EV / wav_cm   # eV  (HC_EV in eV*cm)
    # sort ascending energy
    idx = np.argsort(E_eV_xsect)
    xsect = np.column_stack([E_eV_xsect[idx], C_abs_cm2[idx]])

    for i, G0 in enumerate(G0_vals):
        ax   = axes[i]
        field = field_factory(G0)

        print(f"  G0 = {G0:.0f}")

        # GD89
        print("    GD89 ...", flush=True)
        T_gd, f_gd = compute_gd89_temperature_distribution(
            PAH_FILE, field, xsect, t_min=1.0, num_bins=150)
        ax.loglog(T_gd, f_gd, color=colors['gd89'], lw=2, label='GD89 (single-photon)')

        # Adaptive
        print("    Adaptive ...", flush=True)
        T_ad, f_ad = compute_adaptive_temperature_distribution(
            PAH_FILE, field, xsect, t_min=15.0, num_bins=150)
        ax.loglog(T_ad, f_ad, color=colors['adaptive'], lw=2, ls='--', label='Adaptive (GD89+matrix)')

        # Bakes
        print("    Bakes/Dwek ...", flush=True)
        T_bk, f_bk = compute_bakes_temperature_distribution(
            PAH_FILE, field, xsect, ET_TABLE, Nc=NC, T0=2.73, num_bins=300)
        ax.loglog(T_bk, f_bk, color=colors['bakes'], lw=2, ls=':', label='Bakes/Dwek')

        ax.set_xlabel('T [K]', fontsize=12)
        ax.set_ylabel('f(T) [K$^{-1}$]', fontsize=12)
        ax.set_title(f'C$_{{54}}$H$_{{18}}$, G0={G0:.0f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_xlim(10, 10000)

    fig.suptitle('Temperature distribution f(T) — C$_{54}$H$_{18}$ (Kurucz 15000 K)', fontsize=13)
    fig.tight_layout()
    fig.savefig('diagnostic_fT.png', dpi=150)
    plt.close(fig)
    print("  Saved: diagnostic_fT.png")


# ===========================================================================
# PANEL 4 — Dissociation rates vs G0
# ===========================================================================

def _build_kir_table(pah_file: str, tmp_path: str, n_E: int = 200) -> str:
    """
    Build k_IR(E) table [E_eV, K_thermal s^-1] from PAHdb modes and write to CSV.

    K_thermal(E) = Σ_i eps_i A_i n_i(T_can(E)) / E
    where T_can(E) satisfies U_QHO(T) = E.

    Energies sampled from 0.5 eV to 13.6 eV (log-spaced).
    Returns the path to the written CSV.
    """
    from pycalima.models.PAH_photophysics.pah_mol_data import load_pah_modes
    from scipy.optimize import root_scalar

    freq_ev, einstein_A = load_pah_modes(pah_file)

    # Extend slightly past 13.6 eV to avoid a table-edge divide-by-zero in Bakes/Volterra
    E_grid = np.logspace(np.log10(0.5), np.log10(15.0), n_E)
    K_arr  = np.zeros(n_E)

    def u_qho(T):
        x = freq_ev / (KB_EV * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
        return np.sum(freq_ev * occ)

    # Maximum energy: U_QHO at T=100000 K
    U_max = u_qho(1e5)

    for i, E in enumerate(E_grid):
        if E > U_max:
            break
        try:
            sol = root_scalar(lambda T: u_qho(T) - E, bracket=[1.0, 1e5], method='brentq')
            T_can = sol.root
        except ValueError:
            continue
        x = freq_ev / (KB_EV * T_can)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
        K_arr[i] = np.sum(freq_ev * einstein_A * occ) / E if E > 0 else 0.0

    # Keep only rows where K > 0
    mask = K_arr > 0
    np.savetxt(tmp_path, np.column_stack([E_grid[mask], K_arr[mask]]),
               delimiter=',')
    print(f"  k_IR table: {mask.sum()} energy points → {tmp_path}")
    return tmp_path


def plot_rates_vs_g0():
    use_calima_style()
    print("\n=== Panel 4: Dissociation rates vs G0 ===")

    G0_GRID = np.logspace(0, 5, 20)

    # Cross-section in (E_eV, sigma_cm2) format
    E_eV_xsect = HC_EV / wav_cm
    idx = np.argsort(E_eV_xsect)
    xsect = np.column_stack([E_eV_xsect[idx], C_abs_cm2[idx]])

    # Andrews RRKM parameters for C54H18 normal (H_even_duo class)
    E_act_H  = 4.60   # eV
    dS_H     = 44.8   # J/K/mol
    E_act_H2 = 3.52   # eV
    dS_H2    = -53.1  # J/K/mol

    # Build our PAHdb-based k_IR(E) table to enable Methods C/D/E
    import os
    import tempfile
    tmp_dir  = tempfile.mkdtemp(prefix='calima_kir_')
    kir_path = os.path.join(tmp_dir, 'C54H18_kIR_PAHdb.csv')
    print("  Building k_IR(E) table from PAHdb modes...")
    _build_kir_table(PAH_FILE, kir_path)

    print("  Running compare_dissociation_methods (may take a few minutes)...")
    results = compare_dissociation_methods(
        pah_file             = PAH_FILE,
        cross_section_table  = xsect,
        field_factory        = field_factory,
        g0_grid              = G0_GRID,
        E_act_H              = E_act_H,
        dS_H                 = dS_H,
        E_act_H2             = E_act_H2,
        dS_H2                = dS_H2,
        andrews_H_table      = KH_TABLE,
        andrews_H2_table     = KH2_TABLE,
        andrews_IR_table     = kir_path,   # our PAHdb k_IR enables Methods C/D/E
        num_bins             = 150,
        t_min                = 1.0,
    )

    # Andrews G0-dependent reference data
    gh_dat  = np.loadtxt(GH_CSV,  delimiter=',')
    gh2_dat = np.loadtxt(GH2_CSV, delimiter=',')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    meth_styles = [
        ('B_H',  'B_H2',  'b-',  'o', 'Method B (GD89 f(T) + RRKM)'),
        ('C_H',  'C_H2',  'g--', 's', 'Method C (GD89 f(T) + Andrews k(E))'),
        ('D_H',  'D_H2',  'm:',  '^', 'Method D (Bakes/Dwek)'),
        ('E_H',  'E_H2',  'r-.', 'v', 'Method E (direct energy, Andrews)'),
    ]

    g0 = results['g0']

    for col, (kH_key, kH2_key, style, mk, label) in enumerate(meth_styles):
        rH  = results[kH_key]
        rH2 = results[kH2_key]
        if np.all(rH == 0) and np.all(rH2 == 0):
            continue
        pos = rH > 0
        if pos.any():
            axes[0].loglog(g0[pos], rH[pos],  style, lw=2, marker=mk, ms=5, label=label)
        pos2 = rH2 > 0
        if pos2.any():
            axes[1].loglog(g0[pos2], rH2[pos2], style, lw=2, marker=mk, ms=5, label=label)

    # Andrews reference
    axes[0].loglog(gh_dat[:,0],  gh_dat[:,1],  'k-', lw=2.5, label='Andrews (2016) reference')
    axes[1].loglog(gh2_dat[:,0], gh2_dat[:,1], 'k-', lw=2.5, label='Andrews (2016) reference')

    for ax, title in zip(axes, ['H-loss rate', 'H$_2$-loss rate']):
        ax.set_xlabel('G0', fontsize=13)
        ax.set_ylabel('k [s$^{-1}$]', fontsize=13)
        ax.set_title(f'C$_{{54}}$H$_{{18}}$ {title}', fontsize=13)
        ax.legend(fontsize=9)

    note = ("Methods C/D/E use Andrews k_H/k_H2 tables + our PAHdb-based k_IR(E).\n"
            "Method B uses our RRKM k_H/k_H2 + PAHdb k_IR. Andrews ref: digitised from Fig. 9.")
    fig.text(0.5, 0.01, note, ha='center', fontsize=9, color='grey')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig('diagnostic_rates_vs_G0.png', dpi=150)
    plt.close(fig)
    print("  Saved: diagnostic_rates_vs_G0.png")

    print("\n  Method comparison table:")
    print_method_comparison(results, andrews_H_csv=GH_CSV, andrews_H2_csv=GH2_CSV)

    return results


# ===========================================================================
# PANEL 5 — f(T) convergence: QHO-GD89 vs Spectral-GD89 vs DustEM power-iter
# ===========================================================================

def plot_ft_dustem_comparison():
    """
    Compare three f(T) solvers that differ in physics inputs and algorithm:

      1. GD89-QHO      : GD89 recursion + QHO U(T) + QHO P_IR  (our current default)
      2. GD89-spectral : GD89 recursion + DL07 U(T) + spectral P_IR  (DustEM inputs)
      3. DustEM-iter   : Desert+86 power iteration + DL07 U(T) + spectral P_IR

    In the single-photon regime (2) and (3) must converge to the same answer
    because the GD89 recursion is the analytic solution of the stationary
    distribution that (3) reaches by iteration.
    In the multi-photon regime only (3) is correct (τ correction handles it);
    (1) and (2) deviate.

    Four subplots: G0 = 1, 100, 1000, 10000 (covers single- and multi-photon).
    """
    use_calima_style()
    print("\n=== Panel 5: f(T) convergence — QHO-GD89 vs Spectral-GD89 vs DustEM-iter ===")

    # Use the Tielens (2005) E-T table, which is specific to circumcoronene
    # (C54H18). The DustEM DL07 heat-capacity file (C_PAH0_DL07.DAT) is for
    # a graphite grain with more atoms than C54H18; it gives U(1000K)=23 eV
    # which exceeds the Lyman limit, making f(T>1000K)≈0 incorrectly.
    T_dl07, U_dl07 = _load_tielens_et()
    if T_dl07 is None:
        print("  WARNING: Tielens E-T table not found; skipping Panel 5")
        return

    G0_vals = [1.0, 100.0, 1000.0, 10000.0]

    # Cross-section table for spectral solvers (E in eV, sigma in cm²)
    E_eV_xsect = HC_EV / wav_cm
    idx_s = np.argsort(E_eV_xsect)
    xsect = np.column_stack([E_eV_xsect[idx_s], C_abs_cm2[idx_s]])

    # QHO vibrational frequencies for bin-structure mode (freq_ev_qho parameter)
    freq_ev_modes, _ = load_pah_modes(PAH_FILE)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    axes = axes.ravel()

    for panel, G0 in enumerate(G0_vals):
        ax    = axes[panel]
        field = field_factory(G0)
        print(f"\n  G0 = {G0:.0f}")

        # --- GD89-QHO : QHO U(T) + QHO P_IR ---
        print("    GD89-QHO ...", flush=True)
        T_qho, f_qho = compute_gd89_temperature_distribution(
            PAH_FILE, field, xsect, t_min=5.0, num_bins=150)
        ax.loglog(T_qho, f_qho, 'b-', lw=2, label='GD89 (QHO U + QHO P_IR)', zorder=3)

        # --- GD89-spectral : QHO U(T) + spectral P_IR  (isolates P_IR effect) ---
        print("    GD89-spectral [QHO U + spec P_IR] ...", flush=True)
        T_sp, f_sp = compute_spectral_gd89_distribution(
            field, xsect, wav_cm, C_abs_cm2,
            T_dl07, U_dl07, t_min=5.0, num_bins=150,
            freq_ev_qho=freq_ev_modes)
        ax.loglog(T_sp, f_sp, 'r--', lw=2, label='GD89 (QHO U + spec P_IR)', zorder=3)

        # --- DustEM power-iter : QHO U(T) + spectral P_IR ---
        print("    DustEM power-iter [QHO U + spec P_IR] ...", flush=True)
        T_pi, f_pi = compute_dustem_poweriter_distribution(
            field, xsect, wav_cm, C_abs_cm2,
            T_dl07, U_dl07, t_min=5.0, num_bins=150, n_iter=150,
            freq_ev_qho=freq_ev_modes)
        ax.loglog(T_pi, f_pi, 'g:', lw=2.5, label='DustEM iter (QHO U + spec P_IR)', zorder=2)

        ax.set_xlabel('T [K]', fontsize=12)
        ax.set_ylabel('f(T) [K$^{-1}$]', fontsize=12)
        ax.set_title(f'C$_{{54}}$H$_{{18}}$, G0 = {G0:.0f}', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(5, 15000)

        # Print peak T and high-T tail values for each method
        T_probe = [500, 1000, 2000, 5000]
        for lbl, T_arr, f_arr in [('QHO', T_qho, f_qho), ('Spec', T_sp, f_sp), ('PI', T_pi, f_pi)]:
            if np.any(f_arr > 0):
                T_peak = T_arr[np.argmax(f_arr)]
                tail_vals = [float(np.interp(Tp, T_arr, f_arr, left=0.0, right=0.0))
                             for Tp in T_probe]
                tail_str = "  ".join(f"f({Tp}K)={v:.2e}" for Tp, v in zip(T_probe, tail_vals))
                print(f"      {lbl} peak: T={T_peak:.0f} K  |  {tail_str}")

    fig.suptitle('f(T) convergence: C$_{54}$H$_{18}$ — Kurucz 15000 K field', fontsize=13)
    fig.tight_layout()
    fig.savefig('diagnostic_ft_dustem_comparison.png', dpi=150)
    plt.close(fig)
    print("\n  Saved: diagnostic_ft_dustem_comparison.png")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PAH Temperature Distribution Diagnostic — C54H18 (circumcoronene)")
    print("=" * 70)
    print(f"  PAHdb modes:   {PAH_FILE}")
    print(f"  E-T table:     {ET_TABLE}")
    print(f"  Grain radius:  {A0*1e8:.2f} Å")

    plot_u_of_t()
    plot_pir()
    plot_ft()
    plot_rates_vs_g0()
    plot_ft_dustem_comparison()

    print("\nDone. Check diagnostic_*.png files.")
