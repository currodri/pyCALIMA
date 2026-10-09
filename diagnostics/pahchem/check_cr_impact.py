#!/usr/bin/env python3
"""PAHs hit by cosmic rays: energy deposited, destruction, ionisation and IR emission.

Primary CRs only. The secondary-electron spectrum is still missing
(pahchem-data-requests.md). Prints and plots:

1. Gate G1: the TRK sum rule and the stopping against NIST STAR polystyrene.
2. Gate G4: Micelotta+11 from its card (Table 2; the CR-vs-shock crossover sizes).
3. Per passage, the internal energy left in C54H18 by each projectile group,
   against the energies of the ISRF photons a PAH absorbs.
4. Lifetimes, ionisation and H-loss rates of C24H12, C54H18 and C96H24 against
   N(H2), for the Padovani+18 L and H spectra. Two C-loss channels bracket the
   carbon loss: Micelotta+10b C2 (4.6 eV) and West+18 acenaphthylene+ C2H2 (5.21 eV).
5. The IR spectrum of C54H18 excited by CR passages, against the ISRF-excited
   one: band ratios per unit deposited energy.

Run: python diagnostics/pahchem/check_cr_impact.py --states-dir <model_data>/PAH_states
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pycalima import _paths  # noqa: E402
from pycalima.pahchem import cosmic_rays as cr  # noqa: E402
from pycalima.pahchem import dos, emission, energy_me, fields, impact, photo  # noqa: E402
from pycalima.pahchem import micelotta2011 as mjt  # noqa: E402
from pycalima.pahchem.andrews16 import IONISATION  # noqa: E402
from pycalima.pahchem.baseline import E_PH  # noqa: E402
from pycalima.pahchem.family_emission import BANDS_UM, band_fluxes  # noqa: E402
from pycalima.pahchem.molecule import PAHSpecies  # noqa: E402

MOLECULES = (("C24", "C24H12", 24, 12), ("C54", "C54H18", 54, 18), ("C96", "C96H24", 96, 24))
C_LOSS = {"Micelotta+10b C2": impact.Channel("C2", 4.6, 41.84, 1.0, 1600.0, 2, 0),
          "West+18 C2H2": impact.Channel("C2H2", 5.21, 46.0, 1.0, 1600.0, 2, 2)}
COLUMNS = (0.0, 1e21, 1e22, 1e23)
GROUPS = {"H": ("H",), "He": ("He",), "C-O..Ca": ("Li-B", "C-O", "F-Ne", "Na-Mg", "Al-Si", "P-S", "Cl-Ar", "K-Ca"),
          "Sc-Ni": ("Sc-Mn", "Fe-Ni"), "e (primary)": ("e",)}
YR = 3.15576e7


def channels(nh, c_loss):
    return [impact.Channel("H", 4.41, 12.0, nh, 3050.0, 0, 1), impact.Channel("H2", 4.58, 40.0, 2.0, 3050.0, 0, 2), c_loss]


def gate_g1():
    w = impact.w_grid()
    print("G1  TRK sum rule and Bethe stopping (no correction) against NIST polystyrene, per electron")
    for _, formula, nc, nh in MOLECULES:
        f = impact.oscillator_strength(nc, nh, 0, w)
        print(f"    {formula}: int f = {np.trapezoid(f, w):.1f} / {6 * nc + nh};  I = {impact.mean_excitation_energy(nc, nh):.1f} eV")
    f = impact.oscillator_strength(54, 18, 0, w)
    for proj, prog, ts in ((impact.PROTON, "pstar", (1e6, 1e7, 1e8)), (impact.ALPHA, "astar", (4e6, 4e7, 4e8)),
                           (impact.ELECTRON, "estar", (1e4, 1e6, 1e8))):
        r = [impact.collisions(proj, t, f, w).stopping / (6 * 54 + 18) / float(impact.polystyrene_stopping_per_electron(prog, t))
             for t in ts]
        print(f"    {proj.name:2s} model/STAR at T = {', '.join(f'{t:.0e}' for t in ts)} eV: {', '.join(f'{x:.3f}' for x in r)}")


def gate_g4():
    t = mjt.CARD["targets"]
    print("G4  Micelotta+11 from method_cards/micelotta2011.toml")
    for t0, key in ((4.5, "electron_lifetime_t0_4p5"), (7.5, "electron_lifetime_t0_7p5"), (15.0, "electron_lifetime_t0_15")):
        print(f"    electrons T0 = {t0:4.1f} eV: {mjt.electron_lifetime_yr(t0, orientation=1.0):.2e} yr "
              f"(Table 2 {t[key]['value']:.1e}; with eq. 31's 0.5: {mjt.electron_lifetime_yr(t0):.2e})")
    for f, key in ((0.5, "crossover_nc_f0p5"), (1.0, "crossover_nc_f1")):
        print(f"    tau_CR = 150 Myr at N_C = {mjt.crossover_nc(1.5e8, f=f):.0f} for f = {f} (paper: ~{t[key]['value']:.0f})")


def deposition_figure(out_dir, spectra_by_model, isrf_e, isrf_h):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (label, spec) in zip(axes, spectra_by_model.items()):
        for group, names in GROUPS.items():
            P = sum(spec[n].P.sum(axis=0) for n in names if n in spec)
            if np.ndim(P) == 0:
                continue
            e = spec["H"].e
            tot = P.sum() * spec["H"].de
            if tot > 0:
                ax.step(e, P / tot, where="mid", label=f"{group} ({tot:.1e} s$^{{-1}}$)")
        ax.plot(isrf_e, isrf_h / np.trapezoid(isrf_h, isrf_e), "k--", label="ISRF photons absorbed (heating)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 600)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("internal energy left per passage E* [eV]")
        ax.set_title(f"C54H18: {label}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("dP/dE* [eV$^{-1}$] (passages with a collision)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cr_impact_deposition.png", dpi=120)
    plt.close(fig)


def lifetimes(states_dir, out_dir):
    print("Lifetimes (primary CRs), ionisation and H loss per molecule, neutral parents")
    print("    molecule  model  N(H2)     tau [yr] (C2 / C2H2)    k_ion [s^-1]  H lost [s^-1]  heat [eV/s]  main destroyers")
    res = {}
    spectra = {}
    for parent, formula, nc, nh in MOLECULES:
        sp = PAHSpecies.from_states_file(Path(states_dir) / f"{formula}_0.dat")
        ip, ip2 = IONISATION[parent][0], IONISATION[parent][1]
        yf = lambda E, ip=ip: photo.yield_jochims1996(E, ip)  # noqa: E731
        chains = {k: impact.FragmentationChain(sp, channels(nh, c)) for k, c in C_LOSS.items()}
        for model in ("L", "H"):
            for n in COLUMNS:
                spec = impact.passage_spectrum(nc, nh, 0, ip, ip2, cr.projectile_spectra(model, n), yf)
                if formula == "C54H18" and n == 1e22:
                    spectra[f"Padovani+18 {model}, N(H2) = 1e22"] = spec
                taus = []
                for k, ch in chains.items():
                    r = impact.destruction_rates(spec, nc, nh, 0, ch)
                    taus.append(impact.lifetime_yr(r, nc))
                    res[(formula, model, n, k)] = taus[-1]
                tot = lambda key: sum(v[key] for v in r.values())  # noqa: E731
                top = sorted(r, key=lambda x: -(r[x]["destroyed"] + r[x]["carbon"] / (nc / 3)))[:2]
                print(f"    {formula:8s}  {model}      {n:7.0e}   {taus[0]:.2e} / {taus[1]:.2e}     "
                      f"{tot('ionised'):.2e}      {tot('hydrogen'):.2e}      {tot('heat'):.2e}    {', '.join(top)}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    cols = np.array([max(c, 1e19) for c in COLUMNS])
    for ax, (_, formula, nc, _) in zip(axes, MOLECULES):
        for model, color in (("L", "C0"), ("H", "C3")):
            for k, ls in zip(C_LOSS, ("-", "--")):
                ax.plot(cols, [res[(formula, model, n, k)] for n in COLUMNS], ls, color=color, marker="o",
                        label=f"{model}, {k}")
        ax.axhline(mjt.ion_lifetime_yr(max(nc, 50), 4.58, 1.0), color="k", lw=0.8,
                   label=f"Micelotta+11 ions, N_C = {max(nc, 50)}, f = 1")
        ax.axhline(1.5e8, color="grey", ls=":", lw=0.8, label="shock lifetime (MJTa)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N(H$_2$) [cm$^{-2}$] (first point: unattenuated)")
        ax.set_title(formula)
    axes[0].set_ylabel("lifetime against primary CRs [yr]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cr_impact_lifetimes.png", dpi=120)
    plt.close(fig)
    return spectra


def ir_spectra(states_dir, out_dir, spectra):
    """C54H18: IR band ratios after CR passages and after ISRF photons, per unit deposited energy."""
    nc, nh = 54, 18
    grid = np.arange(500.0, 3300.0, 1.0)
    rows = {}
    cms = {}
    for z in (0, 1):
        sp = PAHSpecies.from_states_file(Path(states_dir) / f"C54H18_{z}.dat")
        cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=40.0 * dos.CM_PER_EV, de_cm=50.0)
        ec = cm.energy_cm / dos.CM_PER_EV
        sinks = {c.name: np.interp(ec, ec, impact.rrkm_whitten_rabinovitch(sp.freqs_cm, c, ec))
                 for c in channels(nh, C_LOSS["Micelotta+10b C2"])}
        cms[z] = (cm, emission.cascade_photons(cm, sinks), sinks)
    field = fields.draine1978(g0=1.0)
    h_uv = {}
    for z in (0, 1):
        cm, _, sinks = cms[z]
        ip = IONISATION["C54"][z]
        y = photo.yield_wenzel2020(E_PH, ip, nc) if z == 1 else photo.yield_jochims1996(E_PH, ip)
        h = photo.heating_spectrum(field, photo.sigma_li_draine(nc, z, E_PH), y, E_PH)
        h_uv[z] = h
        n_uv = emission.single_photon_emission(cm, E_PH, h, sinks)
        rows[f"ISRF photons, Z = {z}"] = (n_uv[:cm.n_ir], cm.freqs_cm[:cm.n_ir], 0.0)
    for label, spec in spectra.items():
        for z, rowsel, tag in ((0, slice(0, 1), "neutral products"), (1, slice(1, None), "ionised products")):
            cmz, N, _ = cms[z]
            tot_n = np.zeros(cmz.n_ir)
            for ps in spec.values():
                pe = (ps.P * ps.de)[rowsel].sum(axis=0)
                bins = np.rint(ps.e * dos.CM_PER_EV / cmz.de_cm).astype(int)
                ok = bins < cmz.n
                tot_n += N[:cmz.n_ir][:, bins[ok]] @ pe[ok]
            rows[f"CR {label.split(',')[0][-1]}: {tag}"] = (tot_n, cmz.freqs_cm[:cmz.n_ir], 0.0)
        frac = sum((ps.P * ps.de)[1:].sum() for ps in spec.values()) / sum((ps.P * ps.de).sum() for ps in spec.values())
        print(f"    {label}: {100 * frac:.0f}% of the passages with a collision ionise")
    h = h_uv[0]
    print("IR band ratios of C54H18 (neutral and cation cascades, dissociating molecules stop emitting)")
    print("    excitation                              " + "  ".join(f"{b}/11.2" for b in ("3.3", "6.2", "7.7", "8.6")))
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, (n_i, freqs, norm) in rows.items():
        s = emission.spectrum(freqs, n_i * freqs, grid)
        s = s / np.trapezoid(s, grid)
        b = band_fluxes(grid, s, BANDS_UM)
        print(f"    {label:40s} " + "  ".join(f"{b[k] / b['11.2']:8.3f}" for k in ("3.3", "6.2", "7.7", "8.6")))
        ax.plot(1e4 / grid, s, label=label)
    ax.set_xlim(3, 15)
    ax.set_xlabel("wavelength [um]")
    ax.set_ylabel("normalised power per cm$^{-1}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cr_impact_ir_spectra.png", dpi=120)
    plt.close(fig)
    return field, h


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    args = ap.parse_args()
    out_dir = _paths.get_plots_dir("pahchem")
    gate_g1()
    gate_g4()
    spectra = lifetimes(args.states_dir, out_dir)
    _, h = ir_spectra(args.states_dir, out_dir, spectra)
    deposition_figure(out_dir, spectra, E_PH, h)
    print(f"figures in {out_dir}")


if __name__ == "__main__":
    main()
