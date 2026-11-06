#!/usr/bin/env python3
"""The charge ladder to the cold Coulomb limit (gas-coupling section 2.2c) per molecule: against the library's ladder and WD01.

For C24H12, C54H18, C96H24, C150H30 and C216H36 (PAHdb v4.00, the calibrated IPs ``W.IP_SHIFTS``, the defaults) in a Draine field at
G0 = 1e4 Habing, T = 100 and 1000 K, psi = G0 sqrt(T) / n_e = 1e2-1e6 and n(H+) = n_e (as ``check_wd01_per_size``):
- <Z>, the share of molecules above the highest computed charge, alpha(H+), Gamma_PE and Lambda_rec per molecule, with
  ``Params.charge_ladder`` "cold_limit" (the default) and "library";
- WD01 for one graphite sphere of the same carbon count, a = 10 A (N_C/468)^(1/3) (``check_wd01_per_size.wd01_job``), in the Mathis
  field at the same Habing G0. The field shapes differ; in ``check_wd01_ingredients`` the photon side moved <Z> by <= 0.1 for N_C >= 54;
- the share of the IR power emitted by stand-in states (``Molecule.stand_ins``, the highest computed charge's modes);
- the UV sensitivity: <Z> and Gamma_PE with the stand-ins' 5-13.6 eV cross section lowered by 5 and 10 % per charge above +2, the
  trend of Malloci's TD-DFT and our Octopus library through +2, which the model does not extrapolate (no basis cache: the cache key
  does not see the change).

Nothing is fitted.

Result (2026-11-02; live in this field: +2, +3, +4, +5, +6 for C24-C216):
- Regression: identical to the library ladder where the field reaches no state above the computed ones (C24 and C54 at every psi,
  every molecule at psi <= 3e3); at psi = 1e4 <Z> moves <= 0.006.
- Where it matters, <Z> at T = 100 K, library -> cold limit: C96 2.60 -> 2.69 at psi = 1e6; C150 1.28 -> 1.62 (1e5), 1.89 -> 3.50
  (1e6); C216 1.35 -> 1.92 (1e5), 1.90 -> 4.16 (1e6). 1000 K within 0.04.
- Against WD01: at psi = 1e6 pahchem now reaches WD01's charge for N_C >= 96 (2.69 against 2.81, 3.50 against 3.27, 4.16 against
  3.70); the library ladder stopped near +1.9. Below psi ~ 3e5 pahchem stays 0.4-0.9 below WD01: the cation sticking (section 2.2b).
  alpha(H+)/WD01 at psi = 1e6 falls from 486 to 40 (C150) and 1725 to 41 (C216): the library ladder's pile-up at +2 had kept
  neutrals that the ladder now ionises. Gamma_PE/WD01 is 7.2-7.5 for N_C >= 96 there (9.3 before); Lambda_rec rises x1.6-1.9.
- Stand-ins: the share of molecules above the highest computed charge, and of the IR power, is 6 % (C96), 92 % (C150) and 98 %
  (C216) at psi = 1e6, and 19-30 % for N_C >= 150 at psi = 1e5. Above psi ~ 1e5 the emission of the largest PAHs rests on the
  stand-in modes.
- UV: the stand-ins' 5-13.6 eV cross section lowered 5-10 % per charge moves <Z> by <= 0.09 and Gamma_PE by <= 3 % (C216 at
  psi = 1e6): a minor uncertainty against the sticking.
- No Coulomb explosions in a field cut at 13.6 eV (the top is out of reach).
- C24's IR share is undefined (nan) at most psi: at G0 = 1e4 it is stripped below the emitting ladder N_H0 ... N_H0-3.

    check_charge_ladder.py [--workers 8] [--out DIR]
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_wd01_per_size as WS
import fit_pdrs4all_population as FP

from pycalima import _paths
from pycalima.pahchem import photo
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params
from pycalima.plotting_style import use_calima_style

NAMES = ("C24H12 (18)", "C54H18 (37)", "C96H24 (108)", "C150H30 (612)", "C216H36 (615)")
G0 = 1e4
N_H = 1e4                                  # only the hydrogenation (the emission weights) sees it
T_GRID = (100.0, 1000.0)
PSI = np.logspace(2.0, 6.0, 9)
WINDOWS = {"all": (1.0, 1e4)}
LADDERS = {"library": ("0.5", "--"), "cold_limit": ("#2a78d6", "-")}
UV_DROPS = (0.05, 0.10)


def env(psi, t):
    ne = G0 * np.sqrt(t) / psi
    return Environment("ladder", G0, N_H, ne, t, None, ions=(("H+", ne),))


def library():
    lib = P.shift_thresholds(FP.library("v400"), WS.W.IP_SHIFTS)
    return [next(m for m in lib if m.name == n) for n in NAMES]


def measure(pop):
    """{quantity: array (molecule, T, psi)} of one population."""
    n = len(pop.library)
    q = {k: np.zeros((n, len(T_GRID), PSI.size)) for k in ("z", "above", "alpha", "gamma", "lam", "explode", "ir_stand_in")}
    zs = np.array(pop.zs, dtype=float)
    for i, t in enumerate(T_GRID):
        for j, psi in enumerate(PSI):
            e = env(psi, t)
            f = pop.charge_matrix(e)
            q["z"][:, i, j] = f @ zs
            q["gamma"][:, i, j] = pop.photoelectric_heating(e)[0]
            q["lam"][:, i, j] = pop.recombination_cooling(e)[0]
            q["explode"][:, i, j] = pop.chain_overflow(e)
            w, E = pop.emission_weights(e), pop.state_powers(P.field_shape(e))
            power = (w * E).sum(axis=2)
            for s, m in enumerate(pop.library):
                q["above"][s, i, j] = sum(f[s, z - pop.zmin] for z in m.stand_ins)
                q["alpha"][s, i, j] = sum(pop.gas_rates(e, np.eye(n)[s])[k]["H+"] for k in ("alpha_mn", "alpha_ct"))
                q["ir_stand_in"][s, i, j] = sum(power[s, z - pop.zmin] for z in m.stand_ins) / power[s].sum()
    return q


def wd01(workers):
    gm = WS.W.mathis_G()
    jobs = [(G0 / gm, G0 * np.sqrt(t) / psi, t, float(WS.W.radius_cm(m_nc))) for m_nc in [int(n[1:n.index("H")]) for n in NAMES]
            for t in T_GRID for psi in PSI]
    with ProcessPoolExecutor(workers) as ex:
        out = np.array(list(ex.map(WS.wd01_job, jobs, chunksize=4)))
    out = out.reshape(len(NAMES), len(T_GRID), PSI.size, -1)
    return {"alpha": out[..., 0], "gamma": out[..., 1], "lam": out[..., 2], "z": out[..., 3]}


def uv_sensitivity(lib):
    """{drop: (<Z>, Gamma) (molecule, T, psi)} with sigma_uv x (1 - drop)^(Z - 2) at 5-13.6 eV for Z > 2 (serial, no cache)."""
    original = photo.sigma_uv
    out = {}
    for drop in UV_DROPS:
        def lowered(nc, charge, energy_ev, model="octopus_library", _d=drop):
            s = original(nc, charge, energy_ev, model)
            if charge <= 2:
                return s
            E = np.asarray(energy_ev, dtype=float)
            return np.where((E >= 5.0) & (E <= 13.6), s * (1.0 - _d) ** (charge - 2), s)
        photo.sigma_uv = lowered
        try:
            pop = P.Population(lib, WINDOWS)
            pop.prepare(P.field_shape(env(PSI[0], T_GRID[0])), workers=1)
            z, g = np.zeros((len(lib), len(T_GRID), PSI.size)), np.zeros((len(lib), len(T_GRID), PSI.size))
            for i, t in enumerate(T_GRID):
                for j, psi in enumerate(PSI):
                    e = env(psi, t)
                    z[:, i, j] = pop.charge_matrix(e) @ np.array(pop.zs, dtype=float)
                    g[:, i, j] = pop.photoelectric_heating(e)[0]
            out[drop] = (z, g)
        finally:
            photo.sigma_uv = original
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    lib = library()
    res = {}
    for ladder in LADDERS:
        pop = P.Population(lib, WINDOWS, params=Params(charge_ladder=ladder), cache_dir=FP.BASIS_CACHE)
        pop.prepare(P.field_shape(env(PSI[0], T_GRID[0])), workers=args.workers)
        res[ladder] = measure(pop)
        if ladder == "cold_limit":
            live = pop._live_charges
            print("live in this field: " + "; ".join(f"{m.name} {m.charges[0]}..+{max(live(m, P.field_shape(env(PSI[0], 100.0))))}"
                                                     f" (cold limit +{max(m.charges)})" for m in pop.library))
    ref = wd01(args.workers)
    uv = uv_sensitivity(lib)
    new, old = res["cold_limit"], res["library"]
    for s, name in enumerate(NAMES):
        for i, t in enumerate(T_GRID):
            print(f"\n{name}, T = {t:g} K:  psi " + " ".join(f"{p:8.0e}" for p in PSI))
            rows = (("<Z> library", old["z"]), ("<Z> cold limit", new["z"]), ("<Z> WD01", ref["z"]),
                    ("f above computed", new["above"]), ("IR from stand-ins", new["ir_stand_in"]),
                    ("alpha(H+) lib/WD01", old["alpha"] / ref["alpha"]), ("alpha(H+) cold/WD01", new["alpha"] / ref["alpha"]),
                    ("Gamma lib/WD01", old["gamma"] / ref["gamma"]), ("Gamma cold/WD01", new["gamma"] / ref["gamma"]),
                    ("Lambda cold/lib", new["lam"] / old["lam"]), ("explosions [s^-1]", new["explode"]),
                    *((f"<Z> UV -{int(d * 100)} %/charge", uv[d][0]) for d in UV_DROPS),
                    *((f"Gamma UV -{int(d * 100)} %/cold", uv[d][1] / new["gamma"]) for d in UV_DROPS))
            for label, a in rows:
                fmt = "{:8.2e}" if label.startswith("explosions") else "{:8.3f}"
                print(f"  {label:22s}" + " ".join(fmt.format(v) for v in a[s, i]))
    plot(res, ref, (args.out or _paths.get_plots_dir("pahchem")) / "pahchem_charge_ladder.png")


def plot(res, ref, path):
    use_calima_style()
    fig, axes = plt.subplots(3, len(NAMES), figsize=(3.3 * len(NAMES), 9.0), sharex=True, gridspec_kw={"hspace": 0.08, "wspace": 0.3})
    for s, name in enumerate(NAMES):
        for i, (t, ls_t) in enumerate(zip(T_GRID, ("-", "--"))):
            axes[0, s].semilogx(PSI, ref["z"][s, i], color="#eb6834", ls=ls_t, lw=1.5, label=rf"WD01, {t:g} K" if s == 0 else None)
            for ladder, (c, _) in LADDERS.items():
                q = res[ladder]
                lab = f"pahchem, {'cold limit' if ladder == 'cold_limit' else 'library ladder'}, {t:g} K" if s == 0 else None
                axes[0, s].semilogx(PSI, q["z"][s, i], color=c, ls=ls_t, lw=2.0, label=lab)
                axes[1, s].loglog(PSI, q["alpha"][s, i] / ref["alpha"][s, i], color=c, ls=ls_t, lw=2.0)
                axes[2, s].loglog(PSI, q["gamma"][s, i] / ref["gamma"][s, i], color=c, ls=ls_t, lw=2.0)
        axes[0, s].set_title(name.replace(" (", ", UID ").rstrip(")"), fontsize=11)
        axes[2, s].set_xlabel(r"$\psi = G_0\sqrt{T}/n_e$")
        for a in axes[1:, s]:
            a.axhline(1.0, color="0.7", lw=1, zorder=0)
        for a in axes[:, s]:
            a.minorticks_on()
            a.tick_params(which="both", direction="in", top=True, right=True)
    axes[0, 0].set_ylabel(r"$\langle Z\rangle$")
    axes[1, 0].set_ylabel(r"$\alpha({\rm H^+})$ / WD01")
    axes[2, 0].set_ylabel(r"$\Gamma_{\rm PE}$ / WD01")
    fig.legend(frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
