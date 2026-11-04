#!/usr/bin/env python3
"""C 1s absorption and Auger decay in X-ray-bright fields (gas-coupling section 2.2c, stage 3): pahchem's K-shell branching
against one ionisation per absorption and against WDB06 grains of the same carbon.

For C24H12, C54H18, C96H24, C150H30 and C216H36 (as ``check_charge_ladder``: PAHdb v4.00, the calibrated IPs, the defaults) in
power laws F_nu ~ nu^-alpha, alpha = 1 and 1.5, from 1 eV to 10 keV (AGN-like; ``fields.powerlaw``), G0 = 1e4 Habing, T = 1e4 K,
n(H+) = n_e, psi = G0 sqrt(T) / n_e = 1e2-1e6:
- pahchem with ``Params.kshell = "auger"`` (the default: per C 1s absorption +1 to +3, ``population.kshell_branching``) and
  "single" (one electron per absorption at the valence yield to 500 eV, none above: before 2026-11-03): <Z>, the share of the
  ionisation electrons from C 1s, Gamma_PE and the Coulomb-explosion rate;
- WDB06 (pyCALIMA's ``yield_model="WDB06"``, which carries the inner-shell photo- and Auger electrons of grains) for one graphite
  sphere of the same carbon in the same field, with H+ capture.
Nothing is fitted.

Result (2026-11-03, with the outgoing electrons' collisions; 93 s on 4 cores with the bases built, 1.9 GB): with the far UV
in the field, C 1s makes 0.2-0.5 % of the ionisation electrons at psi <= 1e4 and 1.3-6.4 % at psi = 1e6 (alpha = 1, C24-C150;
<= 1.2 % for alpha = 1.5). "auger" raises <Z> over "single" by <= 0.24 (C150 at psi = 1e6, 19.55 against 19.30); Gamma_PE by
0.5 % at low psi and by -2.7 % (C150) to +0.7 % (C24) at psi = 1e6, the electrons leaving part of their energy in the molecule.
pahchem stays below WDB06 up to psi ~ 1e5 (the cation sticking) and reaches it at 1e6 for N_C >= 96 (alpha = 1); C24 and C54
sit at their cold limits there, WDB06's grains do not have one.

    check_kshell_auger.py [--workers 4] [--out DIR]
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import check_wd01_per_size as WS
import fit_pdrs4all_population as FP

from pycalima import _paths
from pycalima.models.dust_charge import dust_charging as DC
from pycalima.pahchem import fields
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment
from pycalima.plotting_style import use_calima_style

G0 = 1e4
T = 1e4
PSI = np.logspace(2.0, 6.0, 9)
ALPHAS = (1.0, 1.5)
E_MAX = 1e4
M_H = 1.67262192e-24
MODES = ("auger", "single")


def env(psi, alpha):
    ne = G0 * np.sqrt(T) / psi
    return Environment("xray", G0, 1e-3 * ne, ne, T, None, field_e_max_ev=E_MAX, ions=(("H+", ne),), field_powerlaw=alpha)


def rad_field(alpha):
    """The power law at G0 = 1 as pyCALIMA's [E_eV, wavelength_nm, I_E (erg s^-1 cm^-2 eV^-1)], I_E = E F(E)."""
    f = fields.powerlaw(alpha, 1.0, e_min=1.0, e_max=E_MAX)
    return np.column_stack([f.energy_ev, fields.HC_EV_NM / f.energy_ev, f.energy_ev * f.photon_flux * fields.EV_ERG])


def wdb06_job(job):
    """(<Z>, Gamma [erg s^-1]) of one WDB06 graphite grain."""
    nc, psi, alpha = job
    ne = G0 * np.sqrt(T) / psi
    _, _, rates, zbar, _ = DC.equilibrium_charge_for_grain(G0, ne, T, "graphite", float(WS.W.radius_cm(nc)), rad_field=rad_field(alpha),
                                                           yield_params={"yield_model": "WDB06"},
                                                           ion_species=[{"n": ne, "T": T, "m": M_H, "z": 1.0}])
    return zbar, rates["Gamma_total"]


def pahchem(pop):
    """{quantity: (molecule, alpha, psi)}: <Z>, the C 1s share of the ionisation electrons, Gamma_PE, explosions."""
    n = len(pop.library)
    q = {k: np.zeros((n, len(ALPHAS), PSI.size)) for k in ("z", "k_share", "gamma", "explode")}
    zs = np.array(pop.zs, dtype=float)
    for i, alpha in enumerate(ALPHAS):
        shape = P.field_shape(env(PSI[0], alpha))
        ks = pop._kshell_stack(shape)
        k_up = pop._stack(shape)[0]
        for j, psi in enumerate(PSI):
            e = env(psi, alpha)
            f = pop.charge_matrix(e)
            q["z"][:, i, j] = f @ zs
            q["gamma"][:, i, j] = pop.photoelectric_heating(e)[0]
            q["explode"][:, i, j] = pop.chain_overflow(e)
            if ks is not None:
                k_el = (ks["k"] * np.arange(1, ks["k"].shape[0] + 1)[:, None, None]).sum(axis=0)
                valence = np.maximum(k_up - ks["over"], 0.0)
                q["k_share"][:, i, j] = (f * k_el).sum(axis=1) / (f * (k_el + valence)).sum(axis=1)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    q = {}
    for mode in MODES:
        pop = P.Population(L.library(), L.WINDOWS, params=replace(P.Params(), kshell=mode), cache_dir=FP.BASIS_CACHE)
        for alpha in ALPHAS:
            pop.prepare(P.field_shape(env(PSI[0], alpha)), workers=args.workers)
        q[mode] = pahchem(pop)
        names = [m.name for m in pop.library]
        ncs = [m.nc for m in pop.library]
        del pop
    jobs = [(nc, psi, alpha) for nc in ncs for alpha in ALPHAS for psi in PSI]
    with ProcessPoolExecutor(args.workers) as ex:
        ref = np.array(list(ex.map(wdb06_job, jobs))).reshape(len(ncs), len(ALPHAS), PSI.size, 2)
    for s, name in enumerate(names):
        print(f"\n{name}:  psi " + " ".join(f"{p:8.0e}" for p in PSI))
        for i, alpha in enumerate(ALPHAS):
            rows = ((f"<Z> auger, alpha {alpha:g}", q["auger"]["z"][s, i]), ("<Z> single", q["single"]["z"][s, i]),
                    ("<Z> WDB06", ref[s, i, :, 0]), ("C 1s share of electrons", q["auger"]["k_share"][s, i]),
                    ("Gamma auger/single", q["auger"]["gamma"][s, i] / q["single"]["gamma"][s, i]),
                    ("Gamma auger/WDB06", q["auger"]["gamma"][s, i] / ref[s, i, :, 1]),
                    ("explosions auger [s^-1]", q["auger"]["explode"][s, i]), ("explosions single [s^-1]", q["single"]["explode"][s, i]))
            for label, a in rows:
                fmt = "{:8.2e}" if label.startswith("explosions") else "{:8.3f}"
                print(f"  {label:28s}" + " ".join(fmt.format(v) for v in a))
    plot(names, q, ref, (args.out or _paths.get_plots_dir("pahchem")) / "pahchem_kshell_auger.png")


def plot(names, q, ref, path):
    use_calima_style()
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.6), gridspec_kw={"hspace": 0.3, "wspace": 0.3})
    styles = {1.0: "-", 1.5: "--"}
    for s, name in enumerate(names):
        for i, alpha in enumerate(ALPHAS):
            lab = s == 0
            axes[0, s].semilogx(PSI, q["auger"]["z"][s, i], color="#2a78d6", ls=styles[alpha], lw=2.0,
                                label=rf"pahchem, Auger, $\alpha = {alpha:g}$" if lab else None)
            axes[0, s].semilogx(PSI, q["single"]["z"][s, i], color="#7f7f7f", ls=styles[alpha], lw=1.4,
                                label=rf"pahchem, one electron per C 1s, $\alpha = {alpha:g}$" if lab else None)
            axes[0, s].semilogx(PSI, ref[s, i, :, 0], color="#eb6834", ls=styles[alpha], lw=1.4,
                                label=rf"WDB06 graphite, $\alpha = {alpha:g}$" if lab else None)
            axes[1, s].semilogx(PSI, q["auger"]["k_share"][s, i], color="#2a78d6", ls=styles[alpha], lw=1.8)
        axes[0, s].set_title(name.replace(" (", ", UID ").rstrip(")"), fontsize=11)
        axes[1, s].set_xlabel(r"$\psi = G_0\sqrt{T}/n_e$")
        for a in axes[:, s]:
            a.minorticks_on()
            a.tick_params(which="both", direction="in", top=True, right=True)
    axes[0, 0].set_ylabel(r"$\langle Z\rangle$")
    axes[1, 0].set_ylabel("C 1s share of the ionisation electrons")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(r"$F_\nu \propto \nu^{-\alpha}$, 1 eV to 10 keV, $G_0 = 10^{4}$, $T = 10^{4}$ K, $n({\rm H^+}) = n_e$",
                 fontsize=13, y=1.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
