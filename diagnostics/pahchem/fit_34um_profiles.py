#!/usr/bin/env python3
"""The 3.4 um aliphatic complex of the PDRs4All Orion Bar templates: carriers and band profiles (2026-10-07).

Four parts, each a figure in plots/pahchem:

1. ``aliphatic_34um_survey``: the observed 3.15-3.65 um templates over a linear
   continuum, and where PAHdb v4.00 (N_C >= 21) puts the CH2 / CH3 / aromatic
   stretches with the fitted -28 cm^-1 shift and 37 cm^-1 Gaussians.
2. ``cascade_profile_prototype``: single-molecule profiles built from the cooling
   cascade (``emission.emission_by_energy``): each photon at nu_i + slope T(E)
   with FWHM w0 + wslope T(E). Shown with the Mai+25 classical MD laws at the
   quantum (canonical) and classical (E / s k) temperatures.
3. ``refit_3um_twostep_<template>``: the 3.1-3.65 um segment refitted per band
   law, in two steps that keep the 3.29 band from driving the aliphatic weights:
   (1) the aromatic population of the full fit (``fit_pdrs4all_spectra
   --fit-shape --aliphatic free``, best Gaussian cell: weights, charges) as ONE
   column with a free scale, plus a linear continuum, on 3.1-3.36 and
   3.60-3.65 um; (2) the aliphatic species (NNLS) on the whole residual, so
   they pay for their own 3.29 emission. Laws:
   - "symmetric -28/37": the shape of the full fit;
   - "Mai+25 classical T": -0.07 cm^-1/K, FWHM 10 + 0.06 T (classical T);
   - "Demyk+26 lab laws": Demyk+26 (arXiv:2607.16018) Table 4, pyrene aromatic
     CH (-0.0216, 10.0, 0.0159) for modes >= 3000 cm^-1 and 6H-pyrene
     asymmetric CH2 (-0.0115, 10.2, 0.0084; 523-673 K) below, quantum T; PAHdb
     positions moved by ANCHOR_CM, the offset of PAHdb pyrene / 6H-pyrene
     from the Demyk 0 K intercepts;
   - "Demyk+26, aromatic law for all": the control.
4. The best 3.4 / 3.29 power ratio any aliphatic molecule of the library can
   give, against the observed (3.36-3.60) / (3.20-3.36) over the linear
   continuum (printed).

Why two steps: fitted alone on 3.1-3.65 um, NNLS lets neutral aliphatic
molecules carry the 3.29 band (their CH stretch is strong, the cation-dominated
population's weak); with the full-fit continuum (one cubic per segment) the
continuum sags under the 3.3-3.5 um bands and takes part of the complex.

The per-state cascade tables (CH modes only) are kept in
model_data/pahchem_cascade_ch (about 0.6 GB).

    fit_34um_profiles.py --templates .../cds_vdp25_J_A+A_701_A111/orionbar.dat
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
from pycalima.pahchem import conditions, dos, emission, energy_me, photo, population, spectral_fit
from pycalima.pahchem.baseline import Environment, Params

IP_SHIFTS = {-1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64}
PARAMS = Params(stretch_scale=population.LAB_STRETCH_SCALE)
DE_CM, E_MAX_CM = 100.0, 16.0 * dos.CM_PER_EV
K_CM = 0.6950348                                     # k_B [cm^-1 / K]
CH_CM = (2600.0, 3300.0)
#: PAHdb v4.00 minus Demyk+26 chi0': pyrene aromatic 3067.6 / 3070.9 vs 3062.5, 6H-pyrene (UID 336) aromatic
#: 3048.1 vs 3041.1, asymmetric CH2 ~2954 vs 2947.1-2948.7: about -6 cm^-1 for both classes
ANCHOR_CM = -6.0
PYRENE_ARO = (-0.0216, 10.0, 0.0159)
HHP_CH2 = (-0.0115, 10.2, 0.0084)
LAWS = {
    "symmetric -28/37": {"kind": "symmetric", "shift": -28.0, "fwhm": 37.0},
    "Mai+25 classical T": {"kind": "md", "slope": -0.07, "w0": 10.0, "wslope": 0.06},
    "Demyk+26 lab laws": {"kind": "lab", "aro": PYRENE_ARO, "ali": HHP_CH2},
    "Demyk+26, aromatic law for all": {"kind": "lab", "aro": PYRENE_ARO, "ali": PYRENE_ARO},
}
#: Peeters+24 Table G.1 component centres [um]
COMPONENTS_UM = (3.29, 3.395, 3.403, 3.424, 3.465, 3.516, 3.561)
TABLE_VERSION = 1


def library():
    lib = population.library_v400(include_superh=True, cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    return population.shift_thresholds(lib, IP_SHIFTS)


# -- cascade tables ------------------------------------------------------------------------------
def _state_table(job):
    """(key, (CH freqs, X_ch [photons s^-1 per mode and bin at G0 = 1], T_quantum, T_classical))."""
    mol, z = job
    inp = population._basis_inputs(mol, z, ("kurucz", 40000), DE_CM, E_MAX_CM, PARAMS)
    field1 = population._unit_field(inp["shape"])
    sigma = photo.sigma_li_draine(mol.nc, z, population.E_PH) * PARAMS.sigma_scale
    h = photo.heating_spectrum(field1, sigma, inp["yield_"], population.E_PH)
    k_up = float(photo.ionisation_rate(field1, sigma, inp["yield_"], population.E_PH))
    cm = energy_me.CascadeModel(inp["freqs"], inp["einstein_a"], E_MAX_CM, DE_CM)
    # start: the absorbed heating spectrum plus recombination heating, as population.build_state_basis
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
    f = np.asarray(inp["freqs"])
    ch = (f >= CH_CM[0]) & (f < CH_CM[1])
    return (mol.uids[0], z), (f[ch], X[ch].astype(np.float32), dos.canonical_temperature(f, cm.energy_cm),
                              cm.energy_cm / (f.size * K_CM))


def cascade_tables(lib, processes=8):
    stamp = hashlib.sha1(repr((TABLE_VERSION, population.BASIS_VERSION, PARAMS, IP_SHIFTS, DE_CM, E_MAX_CM,
                               [(m.name, m.charges) for m in lib])).encode()).hexdigest()[:12]
    path = _paths.get_model_data_dir() / "pahchem_cascade_ch" / f"tables_{stamp}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    jobs = [(m, z) for m in lib for z in m.charges]
    with Pool(processes) as p:
        tab = dict(p.imap_unordered(_state_table, jobs, chunksize=8))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tab, fh)
    return tab


# -- band laws -----------------------------------------------------------------------------------
def state_profile(entry, law, nu):
    """Emitted power per cm^-1 on ``nu`` of one state's CH modes under ``law``."""
    f, X, tq, tc = entry
    X = X.astype(float)
    if law["kind"] == "symmetric":
        c = (f + law["shift"])[:, None] * np.ones_like(X)
        s = np.full_like(X, law["fwhm"] / 2.3548)
    elif law["kind"] == "lab":
        ar = (f >= 3000.0)[:, None]
        p = [np.where(ar, a, b) for a, b in zip(law["aro"], law["ali"])]
        c = f[:, None] + ANCHOR_CM + p[0] * tq[None, :]
        s = np.maximum(p[1] + p[2] * tq[None, :], 1.0) / 2.3548 * np.ones_like(X)
    else:
        c = f[:, None] + law["slope"] * tc[None, :]
        s = (law["w0"] + law["wslope"] * tc[None, :]) / 2.3548 * np.ones_like(X)
    wgt = X * f[:, None]
    if not wgt.any():
        return np.zeros_like(nu)
    live = wgt > 1e-12 * wgt.max()
    c, s, wgt = c[live], s[live], wgt[live]
    g = np.exp(-0.5 * ((nu[:, None] - c[None, :]) / s[None, :]) ** 2) / (np.sqrt(2 * np.pi) * s[None, :])
    return g @ wgt


# -- the two-step refit --------------------------------------------------------------------------
class Refit:
    def __init__(self, lib, tab, templates):
        self.lib, self.tab = lib, tab
        self.pop = population.Population(lib, FP.WINDOWS, cache_dir=FP.BASIS_CACHE, params=PARAMS)
        self.grid = spectral_fit.Grid.build()
        self.seg = self.grid.segment == 0
        self.nu = self.grid.nu[self.seg]
        self.lam_um = 1e4 / self.nu
        self.linear = np.stack([3.65 - self.lam_um, self.lam_um - 3.1], axis=1)
        self.obs_lam, self.tmpl = C.load_templates(templates)
        self.aliph = np.array([m.edges.get("aliphatic", 0) > 0 for m in lib])
        self.n_ali = np.array([m.edges.get("aliphatic", 0) for m in lib], float)
        self.nh = np.array([m.nh for m in lib], float)
        self._prof = {}

    def profiles(self, name):
        if name not in self._prof:
            self._prof[name] = {k: state_profile(v, LAWS[name], self.nu) for k, v in self.tab.items()}
        return self._prof[name]

    def columns(self, name, w_full, f):
        prof = self.profiles(name)
        aro = np.zeros(self.nu.size)
        P = np.zeros((self.nu.size, len(self.lib)))
        for s_, m in enumerate(self.lib):
            for z in m.charges:
                p = f[s_, z - self.pop.zmin] * prof[(m.uids[0], z)]
                if self.aliph[s_]:
                    P[:, s_] += p
                elif w_full[s_] > 0:
                    aro += w_full[s_] * p
        return aro, P

    def two_step(self, t, name, w_full, f):
        y = FS.observed_on_grid(self.obs_lam, self.tmpl[t], self.grid)[self.seg]
        ok = np.isfinite(y)
        l_ = self.lam_um
        outside = ok & ((l_ < 3.36) | (l_ >= 3.60))
        inside = ok & (l_ >= 3.36) & (l_ < 3.60)
        w33 = ok & (l_ >= 3.20) & (l_ < 3.36)
        aro, P = self.columns(name, w_full, f)
        s1 = spectral_fit.fit_spectrum(np.where(outside, y, np.nan), aro[:, None], self.linear)
        resid = y - s1.pah - s1.cont
        s2 = spectral_fit.fit_spectrum(np.where(ok, resid, np.nan), P[:, self.aliph], np.zeros((self.nu.size, 0)))
        ali = P[:, self.aliph] @ s2.weights
        w = np.zeros(len(self.lib))
        w[~self.aliph] = w_full[~self.aliph] * s1.weights[0]
        w[self.aliph] = s2.weights
        total = s1.pah + ali
        band = y - s1.cont
        rel = lambda m, v: float(np.sqrt(np.mean(v[m] ** 2)) / y[m].mean())
        return {"rms": rel(ok, band - total), "rms34": rel(inside, band - total), "rms34_aro": rel(inside, resid),
                "ali_h": float(w @ self.n_ali / (w @ self.nh)), "ali_33_share": float(ali[w33].sum() / total[w33].sum()),
                "obs_ratio": float(band[inside].sum() / band[w33].sum()),
                "y": y, "band": band, "aro": s1.pah, "ali": ali, "total": total}


def full_fit(templates):
    """The ``--fit-shape --aliphatic free`` result of fit_pdrs4all_spectra (run it first; it is cached)."""
    args = argparse.Namespace(templates=Path(templates), library="v400", ip_shift="-1:0.37,0:0.6,1:0.64,2:0.64",
                              capture_scale=1.0, capture_model="bt94", stretch_scale="lab", intact_min=0.5,
                              aliphatic="free", plot=False, fit_shape=True)
    path = _paths.get_results_dir() / f"pahchem_spectral_shape_{FS._result_key(args)}.pkl"
    if not path.exists():
        raise SystemExit(f"run fit_pdrs4all_spectra.py --fit-shape --aliphatic free first ({path.name} missing)")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# -- figures -------------------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_survey(R, lib, out_dir):
    plt = _plt()
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [2, 1.3]})
    s = (R.obs_lam > 3.1) & (R.obs_lam < 3.66)
    for i, t in enumerate(C.TEMPLATES):
        l, f = R.obs_lam[s], R.tmpl[t][s]
        a, b = (l > 3.12) & (l < 3.17), (l > 3.62) & (l < 3.66)
        y = f - np.interp(l, [l[a].mean(), l[b].mean()], [np.nanmedian(f[a]), np.nanmedian(f[b])])
        ax[0].plot(l, y / np.nanmax(y[(l > 3.25) & (l < 3.33)]), lw=0.8, color=f"C{i}",
                   label=f"{t} (3.4/3.29 Peeters+24 {C.OBS_34_329[t]:.2f})")
    nu = np.linspace(1e4 / 3.66, 1e4 / 3.12, 2000)
    groups = {"methyl CH3": [], "super-H / other off-edge H": []}
    for m in lib:
        if m.edges.get("aliphatic", 0) == 0:
            continue
        key = "methyl CH3" if m.edges["ch3"] > 0 else "super-H / other off-edge H"
        groups[key] += [(m.species[z], z) for z in (0, 1) if z in m.species]
    for (k, v), c in zip(groups.items(), ("C1", "C3")):
        for z, ls in ((0, "-"), (1, "--")):
            sel = [sp for sp, zz in v if zz == z]
            if not sel:
                continue
            prof = sum(emission.spectrum(sp.freqs_cm[(sp.freqs_cm > 2600) & (sp.freqs_cm < 3300)],
                                         sp.einstein_a[(sp.freqs_cm > 2600) & (sp.freqs_cm < 3300)], nu, fwhm_cm=37.0,
                                         shift_cm=-28.0, profile="gaussian") for sp in sel) / len(sel)
            ax[1].plot(1e4 / nu, prof / prof.max(), color=c, ls=ls, label=f"{k}, Z={z} (n={len(sel)})")
    for a_ in ax:
        for x in COMPONENTS_UM:
            a_.axvline(x, color="0.8", lw=0.5, ls=":")
        a_.set_xlim(3.15, 3.65)
    ax[0].set_ylim(-0.05, 1.1)
    ax[0].set_ylabel("observed - linear continuum / 3.29 peak")
    ax[0].legend(fontsize=7)
    ax[1].set_ylabel("mean PAHdb CH-stretch\n(A-weighted, peak = 1)")
    ax[1].set_xlabel("wavelength [um] (dotted: Peeters+24 components)")
    ax[1].legend(fontsize=7)
    ax[1].set_title("PAHdb v4.00 aliphatic species of the library, -28 cm$^{-1}$ / 37 cm$^{-1}$ Gaussians", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "aliphatic_34um_survey.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_prototype(lib, tab, R, out_dir):
    plt = _plt()
    by = {m.uids[0]: m for m in lib}
    cases = [(37, 0, "C54H18"), (37, 1, "C54H18+"), (108, 1, "C96H24+"), (316, 0, "C55H20 (methyl)"),
             (682, 0, "C96H25 (super-H)"), (190, 0, "C96H28 (super-H, adjacent CH2)")]
    nu = np.linspace(2700, 3200, 1000)
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 2.3 * len(cases)), sharex=True)
    l, f = R.obs_lam, R.tmpl["DF2"]
    s = (l > 3.12) & (l < 3.66)
    l, f = l[s], f[s]
    a, b = (l > 3.12) & (l < 3.17), (l > 3.62) & (l < 3.66)
    obs = (f - np.interp(l, [l[a].mean(), l[b].mean()], [np.nanmedian(f[a]), np.nanmedian(f[b])])) * l**2
    obs /= np.nanmax(obs[(l > 3.25) & (l < 3.33)])
    for ax, (uid, z, label) in zip(axes, cases):
        entry = tab[(uid, z)]
        ax.plot(l, obs, "k", lw=0.6, label="DF2 observed")
        for name, c, ls in (("symmetric -28/37", "0.5", "--"), ("Mai+25 classical T", "C0", "-"),
                            ("Demyk+26 lab laws", "C3", "-")):
            p = state_profile(entry, LAWS[name], nu)
            ax.plot(1e4 / nu, p / p.max(), color=c, ls=ls, lw=1, label=name)
        ax.set_title(f"{label}, UID {uid}, Z = {z}", fontsize=8)
        ax.set_ylim(-0.05, 1.15)
    axes[0].legend(fontsize=7)
    axes[-1].set_xlim(3.15, 3.65)
    axes[-1].set_xlabel("wavelength [um]")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cascade_profile_prototype.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_refit(R, res, out_dir):
    plt = _plt()
    o = np.argsort(R.lam_um)
    for t, rows in res.items():
        fig, axes = plt.subplots(len(LAWS), 1, figsize=(10, 2.7 * len(LAWS)), sharex=True)
        for ax, name in zip(axes, LAWS):
            r = rows[name]
            ax.plot(R.lam_um[o], r["band"][o], "k", lw=0.7, label="observed - linear continuum")
            ax.plot(R.lam_um[o], r["aro"][o], "C0", lw=0.9, label="aromatic population (full-fit mix)")
            ax.plot(R.lam_um[o], r["ali"][o], "C1", lw=0.9, label="aliphatic species")
            ax.plot(R.lam_um[o], r["total"][o], "C3", lw=1.0, label="total")
            ax.plot(R.lam_um[o], (r["band"] - r["total"])[o], "0.6", lw=0.5, label="residual")
            for x in COMPONENTS_UM:
                ax.axvline(x, color="0.85", lw=0.5, ls=":")
            ax.set_title(f"{t}, {name}: 3.36-3.60 residual {r['rms34_aro']:.3f} (aromatic only) -> {r['rms34']:.3f}; "
                         f"off-edge H {r['ali_h']:.3f}; aliphatic share of 3.2-3.36 {r['ali_33_share']:.2f}", fontsize=8)
        axes[0].legend(fontsize=7, ncol=3)
        axes[-1].set_xlim(3.15, 3.62)
        axes[-1].set_xlabel("wavelength [um] (dotted: Peeters+24 components)")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"refit_3um_twostep_{t}.png", dpi=115, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    full = full_fit(args.templates)
    lib = library()
    if [m.name for m in lib] != [x[0] for x in full["library"]]:
        raise SystemExit("library differs from the full fit's")
    tab = cascade_tables(lib, args.processes)
    R = Refit(lib, tab, args.templates)
    out_dir = _paths.get_plots_dir("pahchem")
    plot_survey(R, lib, out_dir)
    plot_prototype(lib, tab, R, out_dir)
    card = conditions.load("orion_bar")
    res = {}
    for t in C.TEMPLATES:
        b = min(full[t]["gaussian"]["cells"], key=lambda c: c["sse"])
        env = Environment(t, b["g0"], b["n_h"], b["n_e"], FS._cells(card, t)[0][3], 40000)
        f = R.pop.charge_matrix(env)
        res[t] = {}
        for name in LAWS:
            r = R.two_step(t, name, b["weights"], f)
            res[t][name] = r
            print(f"{t:6s} {name:32s} 3.36-3.60 residual: aromatic only {r['rms34_aro']:.3f} -> + aliphatics "
                  f"{r['rms34']:.3f}; off-edge H {r['ali_h']:.4f}; aliphatic share of 3.20-3.36 {r['ali_33_share']:.2f}; "
                  f"segment rms {r['rms']:.3f}", flush=True)
    plot_refit(R, res, out_dir)
    # the best ratio a library molecule can give (DF3 cell charges)
    t = "DF3"
    b = min(full[t]["gaussian"]["cells"], key=lambda c: c["sse"])
    f = R.pop.charge_matrix(Environment(t, b["g0"], b["n_h"], b["n_e"], FS._cells(card, t)[0][3], 40000))
    w34 = (R.lam_um >= 3.36) & (R.lam_um < 3.60)
    w33 = (R.lam_um >= 3.20) & (R.lam_um < 3.36)
    for name in ("Demyk+26 lab laws", "symmetric -28/37"):
        prof = R.profiles(name)
        ratios = sorted(((sum(f[s_, z - R.pop.zmin] * prof[(m.uids[0], z)] for z in m.charges), m)
                         for s_, m in enumerate(lib) if R.aliph[s_]), key=lambda x: -x[0][w34].sum() / x[0][w33].sum())
        vals = [p[w34].sum() / p[w33].sum() for p, _ in ratios]
        print(f"{name}: library aliphatic molecules, (3.36-3.60)/(3.20-3.36): median {np.median(vals):.3f}; best "
              + ", ".join(f"{m.name} {v:.2f}" for (_, m), v in zip(ratios[:5], vals[:5])))
    print("observed (3.36-3.60)/(3.20-3.36) over the linear continuum: "
          + ", ".join(f"{t} {res[t]['symmetric -28/37']['obs_ratio']:.2f}" for t in res))
    print("wrote figures to", out_dir)


if __name__ == "__main__":
    main()
