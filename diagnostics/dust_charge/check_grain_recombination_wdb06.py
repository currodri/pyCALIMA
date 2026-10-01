#!/usr/bin/env python3
"""Grain-assisted ion recombination: WDB06 charging vs Weingartner & Draine (2001b).

(A) Same grains as WD01b: alpha per H nucleus [cm^3 s^-1] integrated over the
    WD01 size distribution (R_V = 3.1, b_C = 6e-5; check_grain_size_distribution),
    from full-spectrum charge distributions in the Mathis ISRF at G = 1 (Habing,
    6-13.6 eV) with WD01 and with WDB06 photoemission yields, case A thresholds
    (dust_ion_recombination), against the WD01b fit (their Table 2).
(B) What CALIMA would use: the dust bins of the grain-size configuration through
    the RT-group solver (rt_group_charging, UV background kernels, no RT photons)
    plus the PAH bins as small graphite grains (full-spectrum WDB06), at the
    Milky Way mass fractions of the RAMSES test setups, against the WD01b fit as
    RAMSES RTZ applies it (x 3.3/6, rtz/dust_recombination_module.f90).

Charge distributions include H+ capture with n(H+) = n_e. psi = G sqrt(T) / n_e.

    python check_grain_recombination_wdb06.py [--workers 8] [--plot-only]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import rt_group_charging as rg

sys.path.insert(0, str(Path(__file__).parent))
import check_grain_size_distribution as gsd  # noqa: E402  (WD01 size distribution, WD01b fit)

M_P = 1.67262192e-24
T_GRID = (100.0, 1000.0, 6000.0)
PSI_GRID = np.logspace(2.0, 6.0, 9)
SIZES = np.geomspace(gsd.A_MIN, 1e-4, 32)          # 3.5 A - 1 um
IONS = ("H+", "He+", "C+", "Mg+", "Si+", "S+", "Fe+")  # WD01b fits among RECOMB_IONS
ION_INDEX = {f"{el}+": k for k, (el, _, _) in enumerate(rg.RECOMB_IONS)}
# Milky Way mass fractions of the RAMSES RTZ test setups (per unit gas mass)
MW_DUST = {"DustBin_01": 5.106243e-4, "DustBin_02": 5.106243e-4,
           "DustBin_03": 1.835089e-3, "DustBin_04": 1.835089e-3}
MW_PAH = {"PAHBin_01": (54, 1.582947e-4), "PAHBin_02": (418, 1.582947e-4)}   # N_C, mass fraction
X_H = 0.73738783
RHO_GR = {"graphite": 2.2, "silicate": 3.3}


def mathis_G():
    """Habing G of the unscaled Mathis ISRF that equilibrium_charge_for_grain uses at G0 = 1."""
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field
    E = np.geomspace(6.0, 13.6, 400)
    uE = np.array([Mathis83_radiation_field(e) for e in E])     # erg cm^-3 eV^-1
    return float(rg.C_CGS * np.trapezoid(uE, E) / rg.HABING_FLUX)


def _alpha_full(args):
    """alpha_i [cm^3 s^-1 per grain] of RECOMB_IONS, full-spectrum solve."""
    G0_mathis, ne, T, material, a, yield_model = args
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
    from pycalima.models.dust_charge.dust_ion_recombination import compute_ion_recombination_coefficients
    Zs, P, _, _, _ = equilibrium_charge_for_grain(
        G0_mathis, ne, T, material, a, yield_params={"yield_model": yield_model},
        ion_species=[{"n": ne, "T": T, "m": M_P, "z": 1.0}])
    ions = [{"name": f"{el}+", "z": 1.0, "T": T, "m": m * rg.M_U} for el, m, _ in rg.RECOMB_IONS]
    _, al = compute_ion_recombination_coefficients(Zs, P, a, ions, grain_type=material, recomb_mode="case_a")
    return np.array(al)


def compute(workers):
    G = mathis_G()
    out = {"G_mathis": G, "T": list(T_GRID), "psi": PSI_GRID.tolist(), "sizes": SIZES.tolist()}
    dn = {"graphite": np.array([gsd.graphite_dn_da(a) for a in SIZES]),
          "silicate": np.array([gsd.silicate_dn_da(a) for a in SIZES])}
    tasks, keys = [], []
    for iT, T in enumerate(T_GRID):
        for ip, psi in enumerate(PSI_GRID):
            ne = np.sqrt(T) / psi                  # G = 1 Habing
            for ym in ("WD01", "WDB06"):
                for mat in ("graphite", "silicate"):
                    for ia, a in enumerate(SIZES):
                        tasks.append((1.0 / G, ne, T, mat, a, ym))
                        keys.append((iT, ip, ym, mat, ia))
    print(f"(A) {len(tasks)} full-spectrum solves on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_alpha_full, tasks, chunksize=8))
    per_grain = {}
    for k, r in zip(keys, res):
        per_grain[k] = r
    A = {ym: np.zeros((len(T_GRID), PSI_GRID.size, len(rg.RECOMB_IONS))) for ym in ("WD01", "WDB06")}
    for iT in range(len(T_GRID)):
        for ip in range(PSI_GRID.size):
            for ym in A:
                for mat in ("graphite", "silicate"):
                    al = np.array([per_grain[(iT, ip, ym, mat, ia)] for ia in range(SIZES.size)])
                    A[ym][iT, ip] += np.trapezoid(al * dn[mat][:, None], SIZES, axis=0)
    out["A"] = {ym: v.tolist() for ym, v in A.items()}
    out["fit"] = [[[float(gsd.get_analytical_recomb_coefficient(ion, 1.0, np.sqrt(T) / psi, T)) for ion in IONS]
                   for psi in PSI_GRID] for T in T_GRID]

    # (B) CALIMA bins
    from pycalima.models.grain_size_config import get_bins, get_lognormal_parameters
    tabdir = _paths.get_model_data_dir() / "dust_charging_rtgroups_data"
    B_dust = np.zeros((len(T_GRID), PSI_GRID.size, len(rg.RECOMB_IONS)))
    for b in sorted(get_bins(is_pah=False), key=lambda b: b["index"]):
        tab = rg.read_bin_table(tabdir / f"dust_charging_rtgroups_{b['id']}.dat")
        gt = rg.GroupChargingTable(tab)
        n_per_H = MW_DUST[b["id"]] * M_P / X_H / (4.0 / 3.0 * np.pi * tab["a"] ** 3 * RHO_GR[tab["material"]])
        for iT, T in enumerate(T_GRID):
            for ip, psi in enumerate(PSI_GRID):
                ne = np.sqrt(T) / psi
                r = gt.solve(np.zeros(tab["L0"].size), 0.0, T, ne, [(ne, 1.0, M_P)], G0_bg=1.0, recomb="exact")
                B_dust[iT, ip] += n_per_H * r["alpha_rec"]
    tasks, keys = [], []
    for pid, (nc, _) in MW_PAH.items():
        a = 1e-7 * (nc / 468.0) ** (1.0 / 3.0)
        for iT, T in enumerate(T_GRID):
            for ip, psi in enumerate(PSI_GRID):
                tasks.append((1.0 / G, np.sqrt(T) / psi, T, "graphite", a, "WDB06"))
                keys.append((pid, a, iT, ip))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_alpha_full, tasks, chunksize=4))
    B_pah = np.zeros_like(B_dust)
    for (pid, a, iT, ip), r in zip(keys, res):
        n_per_H = MW_PAH[pid][1] * M_P / X_H / (4.0 / 3.0 * np.pi * a ** 3 * 2.0)   # spah = 2 g/cm^3
        B_pah[iT, ip] += n_per_H * r
    out["B_dust"], out["B_pah"] = B_dust.tolist(), B_pah.tolist()
    return out


def plot(d, path):
    T, psi = d["T"], np.array(d["psi"])
    fit = np.array(d["fit"])
    A = {k: np.array(v) for k, v in d["A"].items()}
    Bd, Bp = np.array(d["B_dust"]), np.array(d["B_pah"])
    ions_plot = ("H+", "C+", "Mg+", "Si+", "Fe+")
    fig, axes = plt.subplots(len(ions_plot), 2 * len(T), figsize=(4.0 * 2 * len(T), 2.9 * len(ions_plot)),
                             sharex=True, squeeze=False)
    for r, ion in enumerate(ions_plot):
        k, kf = ION_INDEX[ion], IONS.index(ion)
        for c, TT in enumerate(T):
            ax = axes[r, c]
            ax.loglog(psi, fit[c, :, kf], "k--", lw=1.6, label="WD01b fit")
            ax.loglog(psi, A["WD01"][c, :, k], color="#eb6834", lw=2, label="pyCALIMA, WD01 yields")
            ax.loglog(psi, A["WDB06"][c, :, k], color="#2a78d6", lw=2, label="pyCALIMA, WDB06 yields")
            ax2 = axes[r, len(T) + c]
            ax2.loglog(psi, fit[c, :, kf] * 3.3 / 6.0, "k--", lw=1.6, label="RAMSES now: WD01b fit x 3.3/6")
            ax2.loglog(psi, Bd[c, :, k] + Bp[c, :, k], color="#2a78d6", lw=2, label="CALIMA bins + PAH bins (WDB06)")
            ax2.loglog(psi, Bd[c, :, k], color="#2a78d6", lw=1.2, ls=":", label="CALIMA dust bins only")
            for a_ in (ax, ax2):
                a_.grid(color="0.92", lw=0.6)
            if r == 0:
                ax.set_title(f"(A) WD01 sizes, T = {TT:g} K", fontsize=10)
                ax2.set_title(f"(B) CALIMA bins, T = {TT:g} K", fontsize=10)
            if c == 0:
                ax.set_ylabel(rf"$\alpha_{{\rm gr}}$({ion}) [cm$^3$ s$^{{-1}}$ H$^{{-1}}$]")
            if r == len(ions_plot) - 1:
                ax.set_xlabel(r"$\psi = G\sqrt{T}/n_e$")
                ax2.set_xlabel(r"$\psi = G\sqrt{T}/n_e$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, len(T)].legend(frameon=False, fontsize=7)
    fig.suptitle("Grain-assisted recombination: WDB06 charging vs Weingartner & Draine (2001b); "
                 f"Mathis ISRF, G = 1 Habing; case A", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("Saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    js = _paths.get_results_dir() / "grain_recombination_wdb06.json"
    if not args.plot_only:
        d = compute(args.workers)
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(d))
    d = json.loads(js.read_text())
    plot(d, _paths.get_plots_dir() / "grain_recombination_wdb06.png")


if __name__ == "__main__":
    main()
