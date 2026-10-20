#!/usr/bin/env python3
"""The 3.4 um complex from aliphatic side groups on the fitted Orion Bar population (2026-10-20).

The population fit (``fit_pdrs4all_spectra --fit-shape``, aromatic library, best Gaussian cell per template)
fixes which molecules emit and their charges. Each molecule is then given side groups, CH2 (as in
6H-pyrene) or CH3 (as in 1-methylpyrene), that radiate from that molecule's own cascade:

- **Cascade.** Per state, the residence time in each energy bin, P_j = X_ij / k_ij (the photons a mode emits
  from bin j over its rate there; the same for every mode), from the state's cascade under the Kurucz 40 kK
  field with recombination heating, as the population bases (``state_table``). A side-group band of
  frequency nu_b and Einstein A_b emits P_j A_b rho(E_j - h nu_b) / rho(E_j) photons from bin j: the
  harmonic rate on the state's own density of states. First order in the side groups: their modes are not
  added to the density of states, nor their cooling to the cascade, and the aromatic emission is not
  reduced (a hydrogenated or methylated site loses its aromatic H; the free scale of the aromatic column
  takes that).
- **Bands.** The aliphatic CH-stretch bands of Demyk+26 (Table 5): 2942.7, 2916.3, 2876.8, 2846.1 cm^-1
  (CH2, 6H-pyrene) and 2981.1, 2956.9, 2936.8, 2908.4, 2877.8 cm^-1 (CH3, Me-pyrene), each photon at the
  CosmicPAH-IRDB law position and FWHM at the bin's microcanonical T, pseudo-Voigt (``pahchem.demyk2026``,
  whose single-molecule profiles reproduce theirs). Intensity per aliphatic H = the lab ratio to the
  aromatic CH stretch per H of the same molecule (CH2 1.63, CH3 0.95 summed over bands) times the state's
  own aromatic CH-stretch intensity per H ("relative"; cations weaken both, as PAHdb's CH2 stretches do), or
  times that of the neutral ("neutral", an upper bracket for cations).
- **Fit** of 3.1-3.65 um per template: y = a AROM + x2 CH2 + x3 CH3 + linear continuum, all >= 0, with AROM
  the full fit's own 3 um PAH spectrum and CH2/CH3 the side-group emission of the whole population for one
  aliphatic H per aromatic H. x2, x3 are then the aliphatic H per aromatic H in each group.

Figure ``sidegroups_34um`` (plots/pahchem); printed: x2, x3, the 3.36-3.60 um residual before/after, and the
observed and model (3.36-3.60)/(3.20-3.36) ratios.

    fit_34um_sidegroups.py [--processes 8]      (needs the 392313363fe8 fit and the cosmicpah-irdb dataset)
"""
import argparse
import hashlib
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import compare_pdrs4all as C
import fit_pdrs4all_population as FP
import fit_pdrs4all_spectra as FS
from pycalima import _paths
from pycalima.pahchem import conditions, cooling, cosmicpah, demyk2026 as D, dos, emission, energy_me, photo, population
from pycalima.pahchem.baseline import Environment, Params
from pycalima.pahchem.molecule import _EINSTEIN_A_PER_KM_MOL

IP_SHIFTS = {-1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64}
PARAMS = Params(stretch_scale=population.LAB_STRETCH_SCALE)
FSHAPE = ("kurucz", 40000)
DE_CM, E_MAX_CM, WINDOW_CM, GRAIN_CM = 100.0, 16.0 * dos.CM_PER_EV, 100.0, 2.0
CH_CM = (2600.0, 3300.0)
FIT = "392313363fe8"
#: group -> (Demyk+26 molecule, its aliphatic bands, aliphatic H, aromatic CH band, aromatic H)
GROUPS = {"CH2": ("6h-pyrene", (2942.7, 2916.3, 2876.8, 2846.1), 12, 3028.0, 4),
          "CH3": ("me-pyrene", (2981.1, 2956.9, 2936.8, 2908.4, 2877.8), 3, 3049.9, 9)}
BANDS = [b for g in GROUPS.values() for b in g[1]]
TABLE_VERSION = 1


def library():
    return population.shift_thresholds(FP.library("v400"), IP_SHIFTS)


# -- per-state tables ----------------------------------------------------------------------------
def state_table(job):
    """((uid, z), residence P_j, side-band rates per unit A K[b, j], microcanonical T_j, aromatic CH km/mol per H,
    and the CH-mode power check (ours, the basis's))."""
    mol, z = job
    inp = population._basis_inputs(mol, z, FSHAPE, DE_CM, E_MAX_CM, PARAMS)
    field1 = population._unit_field(inp["shape"])
    sigma = photo.sigma_li_draine(mol.nc, z, population.E_PH) * PARAMS.sigma_scale
    h = photo.heating_spectrum(field1, sigma, inp["yield_"], population.E_PH)
    k_up = float(photo.ionisation_rate(field1, sigma, inp["yield_"], population.E_PH))
    f, a = np.asarray(inp["freqs"], float), np.asarray(inp["einstein_a"], float)
    cm = energy_me.CascadeModel(f, a, E_MAX_CM, DE_CM, GRAIN_CM, WINDOW_CM)
    e = population.E_PH * dos.CM_PER_EV / DE_CM
    lo = np.floor(e).astype(int)
    w = e - lo
    dE = np.gradient(population.E_PH)
    s = np.zeros(cm.n)
    np.add.at(s, lo, (1 - w) * h * dE)
    np.add.at(s, lo + 1, w * h * dE)
    if inp["recomb_ev"] > 0:
        x = inp["recomb_ev"] * dos.CM_PER_EV / DE_CM
        i0 = int(np.floor(x))
        s[i0] += k_up * (1 - (x - i0))
        s[i0 + 1] += k_up * (x - i0)
    X = emission.emission_by_energy(cm, s)
    ch = (f >= CH_CM[0]) & (f < CH_CM[1])
    kc = cm.k_modes[:cm.n_ir][ch].sum(axis=0)
    P = np.divide(X[ch].sum(axis=0), kc, out=np.zeros(cm.n), where=kc > 0)
    E, counts = dos.beyer_swinehart(f, cm.energy_cm[-1] + WINDOW_CM, GRAIN_CM)
    K = cooling.mode_emission_rates(E, counts, BANDS, np.ones(len(BANDS)), window_cm=WINDOW_CM, at_cm=cm.energy_cm)
    T = D.temperature(f, cm.energy_cm)
    i_aro_h = float(np.sum(a[ch] / (_EINSTEIN_A_PER_KM_MOL * f[ch] ** 2)) / mol.nh)
    ours = float((X[ch].sum(axis=1) * f[ch]).sum())
    return (mol.uids[0], z), {"P": P.astype(np.float32), "K": K.astype(np.float32), "T": T.astype(np.float32),
                              "i_aro_h": i_aro_h, "ch_power": ours}


def tables(lib, processes):
    stamp = hashlib.sha1(repr((TABLE_VERSION, population.BASIS_VERSION, PARAMS, IP_SHIFTS, DE_CM, E_MAX_CM,
                               WINDOW_CM, BANDS, [(m.name, m.charges) for m in lib])).encode()).hexdigest()[:12]
    path = _paths.get_model_data_dir() / "pahchem_cascade_ch" / f"sidegroups_{stamp}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    jobs = [(m, z) for m in lib for z in m.charges]
    with Pool(processes) as p:
        tab = dict(p.imap_unordered(state_table, jobs, chunksize=8))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tab, fh)
    return tab


# -- side-group spectra --------------------------------------------------------------------------
class SideGroups:
    """Per state and group, the side-group power per cm^-1 on ``nu`` for one aliphatic H per aromatic H."""

    def __init__(self, lib, tab, nu, below="hold"):
        self.lib, self.tab, self.nu, self.below = lib, tab, nu, below
        self.energy = np.arange(next(iter(tab.values()))["P"].size) * DE_CM
        self.bands = {}
        for g, (key, bands, n_ali, aro, n_aro) in GROUPS.items():
            t5 = D.table5(key)
            inten = dict(zip(t5.band_cm, t5.intensity_km_mol))
            ratio = [(inten[b] / n_ali) / (inten[aro] / n_aro) for b in bands]
            laws = [cosmicpah.band(key, b) for b in bands]
            self.bands[g] = [(BANDS.index(b), r, law) for b, r, law in zip(bands, ratio, laws)]
        self._cache = {}

    def spectrum(self, mol, z, group, intensity="relative"):
        k = (mol.uids[0], z, group, intensity)
        if k not in self._cache:
            e = self.tab[(mol.uids[0], z)]
            ref = e if intensity == "relative" else self.tab[(mol.uids[0], 0)]
            P, T = e["P"].astype(float), e["T"].astype(float)
            live = P > 1e-12 * P.max() if P.max() > 0 else P > 0
            out = np.zeros(self.nu.size)
            for i, r, law in self.bands[group]:
                a_b = _EINSTEIN_A_PER_KM_MOL * BANDS[i] ** 2 * r * ref["i_aro_h"] * mol.nh
                photons = P[live] * a_b * e["K"][i].astype(float)[live]
                t = T[live]
                pos, fw = law.position(t, self.below), law.fwhm(t, self.below)
                prof = cosmicpah.pseudo_voigt(self.nu[None, :], pos[:, None], fw[:, None], law.lorentz)
                out += (photons * pos) @ prof
            self._cache[k] = out
        return self._cache[k]

    def column(self, weights, f, zmin, group, intensity="relative"):
        out = np.zeros(self.nu.size)
        for s, m in enumerate(self.lib):
            if weights[s] <= 0:
                continue
            for z in m.charges:
                if f[s, z - zmin] > 0:
                    out += weights[s] * f[s, z - zmin] * self.spectrum(m, z, group, intensity)
        return out


def best_cell(fit, t, card, teff):
    row = min(fit[t]["gaussian"]["cells"], key=lambda r: r["sse"])
    temp = next(c[3] for c in FS._cells(card, t) if np.isclose(c[0], row["g0"]) and np.isclose(c[1], row["n_h"])
                and np.isclose(c[2], row["n_e"]))
    return row, Environment(t, row["g0"], row["n_h"], row["n_e"], temp, teff)


def fit_template(y, lam, arom, cols):
    """NNLS of y on [arom, cols..., linear continuum]; returns (coefficients, model, continuum)."""
    from scipy.optimize import nnls

    lin = np.stack([3.65 - lam, lam - 3.1], axis=1)
    A = np.column_stack([arom] + list(cols) + [lin])
    ok = np.isfinite(y)
    scale = np.where(A[ok].max(axis=0) > 0, A[ok].max(axis=0), 1.0)
    x, _ = nnls(A[ok] / scale, y[ok])
    x /= scale
    cont = lin @ x[-2:]
    return x[:-2], A[:, :-2] @ x[:-2], cont


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--below", choices=("hold", "extrapolate"), default="hold",
                    help="band laws below the lab data: Demyk+26's lowest-T value, or continued to 0 K (Mackie+18)")
    args = ap.parse_args()
    lib = library()
    tab = tables(lib, args.processes)
    with open(_paths.get_results_dir() / f"pahchem_spectral_shape_{FIT}.pkl", "rb") as fh:
        fit = pickle.load(fh)
    grid = fit["grid"]
    seg = grid.segment == 0
    nu, lam = grid.nu[seg], grid.lam_um[seg]
    pop = population.Population(lib, FP.WINDOWS, cache_dir=FP.BASIS_CACHE, params=PARAMS)
    pop.prepare(FSHAPE, workers=1)
    # the check: CH-mode power of the tables against the basis
    worst = 0.0
    for m in lib[::97]:
        for z in m.charges:
            b = pop.basis(m, z, FSHAPE)
            ch = (b.freqs_cm >= CH_CM[0]) & (b.freqs_cm < CH_CM[1])
            worst = max(worst, abs(tab[(m.uids[0], z)]["ch_power"] / float(b.emitted[ch].sum()) - 1))
    print(f"CH-stretch power, tables vs population bases (sample of states): max |rel diff| {worst:.3f}")
    card = conditions.load("orion_bar")
    teff = int(card["region"]["field_teff_k"]["value"])
    sg = SideGroups(lib, tab, nu, args.below)
    w33, w34 = (lam >= 3.20) & (lam < 3.36), (lam >= 3.36) & (lam < 3.60)
    results = {}
    for t in C.TEMPLATES:
        row, env = best_cell(fit, t, card, teff)
        f = pop.charge_matrix(env)
        y = fit[t]["observed"][seg]
        arom = row["pah"][seg]
        cols = {g: env.g0 * sg.column(row["weights"], f, pop.zmin, g, "relative") for g in GROUPS}
        cols_n = {g: env.g0 * sg.column(row["weights"], f, pop.zmin, g, "neutral") for g in GROUPS}
        res = {}
        for name, cs in (("aromatic only", []), ("CH2 (relative)", [cols["CH2"]]), ("CH3 (relative)", [cols["CH3"]]),
                         ("CH2 + CH3 (relative)", [cols["CH2"], cols["CH3"]]),
                         ("CH2 + CH3 (neutral)", [cols_n["CH2"], cols_n["CH3"]])):
            x, model, cont = fit_template(y, lam, arom, cs)
            band = y - cont
            ok = np.isfinite(y)
            rel34 = float(np.sqrt(np.nanmean((band - model)[w34 & ok] ** 2)) / np.nanmean(y[w34 & ok]))
            res[name] = {"x": x, "model": model, "cont": cont, "rms34": rel34,
                         "ratio_obs": float(np.nansum(band[w34]) / np.nansum(band[w33])),
                         "ratio_model": float(model[w34].sum() / model[w33].sum())}
            xs = ", ".join(f"x_{g} {v:.3f}" for g, v in zip([k for k in GROUPS if k in name or "CH2 + CH3" in name], x[1:]))
            print(f"{t:6s} {name:22s}: aromatic scale {x[0]:.2f}{', ' + xs if xs else ''}; 3.36-3.60 residual "
                  f"{rel34:.2f}; (3.36-3.60)/(3.20-3.36) obs {res[name]['ratio_obs']:.2f}, model {res[name]['ratio_model']:.2f}",
                  flush=True)
        results[t] = {"y": y, "arom": arom, "res": res, "cols": cols}
    out = args.out or _paths.get_plots_dir("pahchem")
    suffix = "" if args.below == "hold" else f"_{args.below}"
    with open(_paths.get_results_dir() / f"pahchem_sidegroups_34um{suffix}.pkl", "wb") as fh:
        pickle.dump({"lam": lam, "results": results}, fh)
    plot(lam, results, out, suffix)


def plot(lam, results, out, suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycalima.plotting_style import use_calima_style

    use_calima_style(usetex=False)
    plt.rcParams.update({"font.serif": ["cmr10", "DejaVu Serif", "serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "legend.frameon": False})
    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.6 * len(results)), sharex=True)
    o = np.argsort(lam)
    for ax, (t, r) in zip(axes, results.items()):
        base = r["res"]["aromatic only"]
        both = r["res"]["CH2 + CH3 (relative)"]
        band = r["y"] - both["cont"]
        ax.plot(lam[o], band[o], color="k", lw=1.0, label="observed - continuum")
        ax.plot(lam[o], (r["y"] - base["cont"])[o] * 0 + base["model"][o], color="0.6", lw=1.0, ls="--",
                label=f"aromatic only (3.36-3.60 resid. {base['rms34']:.2f})")
        ax.plot(lam[o], both["model"][o], color="C3", lw=1.2,
                label=f"+ CH$_2$, CH$_3$ side groups (resid. {both['rms34']:.2f})")
        x = both["x"]
        ax.fill_between(lam[o], 0, (x[1] * r["cols"]["CH2"])[o], color="C0", alpha=0.3, lw=0,
                        label=f"CH$_2$: {x[1]:.2f} H per aromatic H")
        ax.fill_between(lam[o], 0, (x[2] * r["cols"]["CH3"])[o], color="C2", alpha=0.3, lw=0,
                        label=f"CH$_3$: {x[2]:.2f} H per aromatic H")
        for c in (3.29, 3.395, 3.403, 3.424, 3.465, 3.516, 3.561):
            ax.axvline(c, color="0.85", lw=0.5, ls=":")
        ax.set_ylabel(f"{t}")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlim(3.15, 3.62)
    axes[-1].set_xlabel(r"$\lambda$ [$\mu$m] (dotted: Peeters+24 components)")
    for a in fig.axes:
        a.minorticks_on()
        a.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    path = Path(out) / f"sidegroups_34um{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
