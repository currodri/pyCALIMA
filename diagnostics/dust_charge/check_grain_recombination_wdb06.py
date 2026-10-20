#!/usr/bin/env python3
"""Grain-assisted ion recombination: WDB06 charging vs Weingartner & Draine (2001b).

What CALIMA would use: the dust bins of the grain-size configuration through the
RT-group solver (rt_group_charging, UV background kernels, no RT photons, case A
thresholds) plus the PAH bins as small graphite grains through the same solver
(RT-group tables built here for their sizes), at the Milky Way mass fractions of
the RAMSES test setups, against the WD01b fit (their Table 2) as RAMSES RTZ
applies it (x 3.3/6, rtz/dust_recombination_module.f90).

Charge distributions include H+ capture with n(H+) = n_e. psi = G sqrt(T) / n_e,
G = 1 Habing (Mathis ISRF).

    python check_grain_recombination_wdb06.py [--plot-only]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import rt_group_charging as rg

sys.path.insert(0, str(Path(__file__).parent))
import check_grain_size_distribution as gsd  # noqa: E402  (WD01b fit)

M_P = 1.67262192e-24
T_GRID = (100.0, 1000.0, 6000.0)
PSI_GRID = np.logspace(2.0, 6.0, 9)
IONS = ("H+", "He+", "C+", "Mg+", "Si+", "S+", "Fe+")  # WD01b fits among RECOMB_IONS
ION_INDEX = {f"{el}+": k for k, (el, _, _) in enumerate(rg.RECOMB_IONS)}
# Milky Way mass fractions of the RAMSES RTZ test setups (per unit gas mass)
MW_DUST = {"DustBin_01": 5.106243e-4, "DustBin_02": 5.106243e-4,
           "DustBin_03": 1.835089e-3, "DustBin_04": 1.835089e-3}
MW_PAH = {"PAHBin_01": (54, 1.582947e-4), "PAHBin_02": (418, 1.582947e-4)}   # N_C, mass fraction
X_H = 0.73738783
RHO_GR = {"graphite": 2.2, "silicate": 3.3}


def _alpha_per_H(gt, n_per_H):
    """n_per_H alpha_i [cm^3 s^-1 per H] of RECOMB_IONS on (T_GRID, PSI_GRID): UV background
    G0_bg = 1, no RT photons, exact recombination sum."""
    out = np.zeros((len(T_GRID), PSI_GRID.size, len(rg.RECOMB_IONS)))
    for iT, T in enumerate(T_GRID):
        for ip, psi in enumerate(PSI_GRID):
            ne = np.sqrt(T) / psi
            r = gt.solve(np.zeros(gt.t["L0"].size), 0.0, T, ne, [(ne, 1.0, M_P)], G0_bg=1.0, recomb="exact")
            out[iT, ip] = n_per_H * r["alpha_rec"]
    return out


def compute():
    out = {"T": list(T_GRID), "psi": PSI_GRID.tolist()}
    out["fit"] = [[[float(gsd.get_analytical_recomb_coefficient(ion, 1.0, np.sqrt(T) / psi, T)) for ion in IONS]
                   for psi in PSI_GRID] for T in T_GRID]

    from pycalima.models.grain_size_config import get_bins, get_lognormal_parameters
    tabdir = _paths.get_model_data_dir() / "dust_charging_rtgroups_data"
    B_dust = np.zeros((len(T_GRID), PSI_GRID.size, len(rg.RECOMB_IONS)))
    for b in sorted(get_bins(is_pah=False), key=lambda b: b["index"]):
        tab = rg.read_bin_table(tabdir / f"dust_charging_rtgroups_{b['id']}.dat")
        n_per_H = MW_DUST[b["id"]] * M_P / X_H / (4.0 / 3.0 * np.pi * tab["a"] ** 3 * RHO_GR[tab["material"]])
        B_dust += _alpha_per_H(rg.GroupChargingTable(tab), n_per_H)
    B_pah = np.zeros_like(B_dust)
    for pid, (nc, frac) in MW_PAH.items():
        a = 1e-7 * (nc / 468.0) ** (1.0 / 3.0)
        n_per_H = frac * M_P / X_H / (4.0 / 3.0 * np.pi * a ** 3 * 2.0)   # spah = 2 g/cm^3
        B_pah += _alpha_per_H(rg.GroupChargingTable(rg.build_bin_table("graphite", a, label=pid)), n_per_H)
    out["B_dust"], out["B_pah"] = B_dust.tolist(), B_pah.tolist()
    return out


def plot(d, path):
    T, psi = d["T"], np.array(d["psi"])
    fit = np.array(d["fit"])
    Bd, Bp = np.array(d["B_dust"]), np.array(d["B_pah"])
    ions_plot = ("H+", "C+", "Mg+", "Si+", "Fe+")
    fig, axes = plt.subplots(len(ions_plot), len(T), figsize=(4.0 * len(T), 2.9 * len(ions_plot)),
                             sharex=True, squeeze=False)
    for r, ion in enumerate(ions_plot):
        k, kf = ION_INDEX[ion], IONS.index(ion)
        for c, TT in enumerate(T):
            ax = axes[r, c]
            ax.loglog(psi, fit[c, :, kf] * 3.3 / 6.0, "k--", lw=1.6, label="RAMSES now: WD01b fit x 3.3/6")
            ax.loglog(psi, Bd[c, :, k] + Bp[c, :, k], color="#2a78d6", lw=2, label="CALIMA bins + PAH bins (WDB06)")
            ax.loglog(psi, Bd[c, :, k], color="#2a78d6", lw=1.2, ls=":", label="CALIMA dust bins only")
            ax.grid(color="0.92", lw=0.6)
            if r == 0:
                ax.set_title(f"CALIMA bins, T = {TT:g} K", fontsize=10)
            if c == 0:
                ax.set_ylabel(rf"$\alpha_{{\rm gr}}$({ion}) [cm$^3$ s$^{{-1}}$ H$^{{-1}}$]")
            if r == len(ions_plot) - 1:
                ax.set_xlabel(r"$\psi = G\sqrt{T}/n_e$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Grain-assisted recombination: WDB06 charging vs Weingartner & Draine (2001b); "
                 f"Mathis ISRF, G = 1 Habing; case A", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("Saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    js = _paths.get_results_dir() / "grain_recombination_wdb06.json"
    if not args.plot_only:
        d = compute()
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(d))
    d = json.loads(js.read_text())
    plot(d, _paths.get_plots_dir() / "grain_recombination_wdb06.png")


if __name__ == "__main__":
    main()
