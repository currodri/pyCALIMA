#!/usr/bin/env python3
"""The population's 3.29 um band from the lab temperature law, with a size-dependent slope (2026-10-21).

The size factor is AD HOC: an exploratory option, not part of the model, until the anharmonic coupling of the
CH stretch against size is derived (VPT2 or quantum-corrected MD of compact PAHs, C24-C96). s_large = 1 is the
physical default (pyrene's law unchanged).

The full fit (``fit_pdrs4all_spectra --fit-shape``) gives the 3.29 um band one symmetric Gaussian per mode,
shifted -28 cm^-1 with FWHM ~34 cm^-1 (fitted). With pyrene's lab law instead, the band came out ~11 cm^-1
too blue and too narrow (spec, 3.4 um study). Mai+25's ML-MD spectra shift the CH-stretch peak of N_C >= 60
1.4-1.9x as far as that of C24-C39 at equal energy per mode (``check_mai2025_ch.py``). This tests whether
that factor closes the gap, with no free shift:

- each aromatic CH photon of state (molecule, Z) emitted at bin j takes nu_i + ANCHOR_CM + s(N_C) chi1' T_j and
  the pyrene FWHM law at T_j, pseudo-Voigt (CosmicPAH-IRDB pyrene, dataset ID 22: chi1' = -0.0216 cm^-1/K);
- photons per mode and bin: the state's residence time P_j (``fit_34um_sidegroups`` tables) times the mode's
  harmonic rate on its density of states (``cooling.mode_emission_rates``, as the tables);
- s = 1 below N_C = 40, rising linearly to s_large at N_C = 60 and above;
- fitted with the CH2 / CH3 side groups and a linear continuum on 3.1-3.65 um (all >= 0, the aromatic scale
  free), against the same fit with the full fit's own 3 um spectrum.

    fit_33um_sizelaw.py [--s-large 1.0 1.5 1.9]
"""
import argparse
import pickle

import numpy as np

import fit_34um_sidegroups as S
from pycalima import _paths
from pycalima.pahchem import conditions, cooling, cosmicpah, dos, population

ANCHOR_CM = -6.0          # PAHdb v4.00 pyrene CH stretches minus the Demyk+26 0 K intercepts (fit_34um_profiles)


def size_factor(nc, s_large):
    return 1.0 + (s_large - 1.0) * float(np.clip((nc - 40.0) / 20.0, 0.0, 1.0))


def aromatic_spectrum(mol, z, entry, nu, law, s_large):
    """Power per cm^-1 on ``nu`` of the state's aromatic CH stretches (lab law, slope x size factor)."""
    inp = population._basis_inputs(mol, z, S.FSHAPE, S.DE_CM, S.E_MAX_CM, S.PARAMS)
    f, a = np.asarray(inp["freqs"], float), np.asarray(inp["einstein_a"], float)
    ch = (f >= S.CH_CM[0]) & (f < S.CH_CM[1])
    P, T = entry["P"].astype(float), entry["T"].astype(float)
    energy = np.arange(P.size) * S.DE_CM
    E, counts = dos.beyer_swinehart(f, energy[-1] + S.WINDOW_CM, S.GRAIN_CM)
    k = cooling.mode_emission_rates(E, counts, f[ch], a[ch], window_cm=S.WINDOW_CM, at_cm=energy)
    X = k * P[None, :]                                    # photons per mode and bin
    live = P > 1e-10 * P.max()
    X, T = X[:, live], T[live]
    slope = law.position_pieces[-1].c1 * size_factor(mol.nc, s_large)
    pos = f[ch][:, None] + ANCHOR_CM + slope * T[None, :]
    fw = np.broadcast_to(law.fwhm(T)[None, :], pos.shape)
    w = (X * pos).ravel()
    keep = w > 1e-8 * w.max()
    prof = cosmicpah.pseudo_voigt(nu[None, :], pos.ravel()[keep][:, None], fw.ravel()[keep][:, None], law.lorentz)
    return w[keep] @ prof


def band_stats(lam, y, lo=3.22, hi=3.36):
    """(peak [um], FWHM [cm^-1]) of the 3.29 um band in ``y`` (ascending lam not required)."""
    o = np.argsort(lam)
    l, v = lam[o], y[o]
    w = (l > lo) & (l < hi)
    l, v = l[w], v[w]
    i = int(np.argmax(v))
    half = l[v >= 0.5 * v[i]]
    return float(l[i]), float(1e4 / half.min() - 1e4 / half.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s-large", type=float, nargs="+", default=[1.0, 1.5, 1.9])
    args = ap.parse_args()
    lib = S.library()
    tab = S.tables(lib, 8)
    with open(_paths.get_results_dir() / f"pahchem_spectral_shape_{S.FIT}.pkl", "rb") as fh:
        fit = pickle.load(fh)
    grid = fit["grid"]
    seg = grid.segment == 0
    nu, lam = grid.nu[seg], grid.lam_um[seg]
    pop = population.Population(lib, S.FP.WINDOWS, cache_dir=S.FP.BASIS_CACHE, params=S.PARAMS)
    pop.prepare(S.FSHAPE, workers=1)
    card = conditions.load("orion_bar")
    teff = int(card["region"]["field_teff_k"]["value"])
    law = cosmicpah.band("pyrene", 3051.3)
    sg = S.SideGroups(lib, tab, nu)
    w33 = (lam >= 3.20) & (lam < 3.36)
    out = {}
    for t in S.C.TEMPLATES:
        row, env = S.best_cell(fit, t, card, teff)
        f = pop.charge_matrix(env)
        y = fit[t]["observed"][seg]
        side = [env.g0 * sg.column(row["weights"], f, pop.zmin, g) for g in S.GROUPS]
        cases = {"full-fit shape (-28 / 34)": row["pah"][seg]}
        active = [(s_, m) for s_, m in enumerate(lib) if row["weights"][s_] > 0]
        for s_large in args.s_large:
            col = np.zeros(nu.size)
            for s_, m in active:
                for z in m.charges:
                    fz = f[s_, z - pop.zmin]
                    if fz > 0:
                        col += row["weights"][s_] * fz * aromatic_spectrum(m, z, tab[(m.uids[0], z)], nu, law, s_large)
            cases[f"lab law, s_large {s_large:.1f}"] = env.g0 * col
        ok = np.isfinite(y)
        res = {}
        for name, arom in cases.items():
            x, model, cont = S.fit_template(y, lam, arom, side)
            band = y - cont
            r33 = float(np.sqrt(np.nanmean((band - model)[w33 & ok] ** 2)) / np.nanmean(y[w33 & ok]))
            pk_m, fw_m = band_stats(lam, x[0] * arom)
            pk_o, fw_o = band_stats(lam[ok], band[ok] - (model - x[0] * arom)[ok])
            res[name] = {"x": x, "model": model, "cont": cont, "r33": r33}
            print(f"{t:6s} {name:28s}: aromatic scale {x[0]:.2f}; 3.20-3.36 residual {r33:.3f}; 3.29 band peak "
                  f"{pk_m:.4f} um (obs {pk_o:.4f}), FWHM {fw_m:.1f} cm^-1 (obs {fw_o:.1f})", flush=True)
        out[t] = {"y": y, "res": res, "cases": cases}
    with open(_paths.get_results_dir() / "pahchem_33um_sizelaw.pkl", "wb") as fh:
        pickle.dump({"lam": lam, "out": out}, fh)
    plot(lam, out, _paths.get_plots_dir("pahchem"))


def plot(lam, out, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycalima.plotting_style import use_calima_style

    use_calima_style(usetex=False)
    plt.rcParams.update({"font.serif": ["cmr10", "DejaVu Serif", "serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "legend.frameon": False})
    fig, axes = plt.subplots(1, len(out), figsize=(3.2 * len(out), 3.6), sharey=False)
    o = np.argsort(lam)
    for ax, (t, v) in zip(axes, out.items()):
        first = next(iter(v["res"].values()))
        ax.plot(lam[o], (v["y"] - first["cont"])[o], color="k", lw=1.0, label="observed - continuum")
        for (name, r), col in zip(v["res"].items(), ("0.55", "C0", "C3", "C2", "C1")):
            ax.plot(lam[o], r["model"][o], color=col, lw=1.1, ls="--" if "full-fit" in name else "-",
                    label=f"{name.replace('lab law, s_large', 'lab law, s =')} ({r['r33']:.3f})")
        ax.set_xlim(3.18, 3.40)
        ax.set_title(t)
        ax.set_xlabel(r"$\lambda$ [$\mu$m]")
        ax.legend(fontsize=5.5)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in")
    fig.tight_layout()
    fig.savefig(path / "sizelaw_33um.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path / 'sizelaw_33um.png'}")


if __name__ == "__main__":
    main()
