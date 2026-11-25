"""PAHrt study S7: track the hydrogenation of the PAH bins, or take it in equilibrium? (rerun for the GSD's molecular bins)

The November study (docs/physics/pah-hydrogen.md, T3-T6) tracked one scalar per bin, <N_H>, with a closure for the rest
of the distribution, and found it no better than equilibrium (all-or-nothing stripping, the alternating super-H ladder,
mixing of a bimodal state); the user then chose equilibrium tables (2026-11-15). The per-cell solver of the PAHrt study
changes what tracking could carry: the populations of the hydrogenation classes, which mix exactly and keep the ladder in
the network. This study scores, per molecule and then per GSD molecular bin (molecules summed, ``common.molecules``):

- **truth**: each molecule's full joint (Z, N_H) chain evolved exactly through each segment of a history (conditions held
  per segment, ``expm``);
- **E**: the equilibrium of the segment's conditions (today's decision);
- **TC**: tracked class populations (``s3_states.CLASSES``: bare, deep, N_H0-3, -2, -1, intact, +1 ... +4); at each
  segment the full distribution is rebuilt as each class's population times its local equilibrium (the steady state of
  the network restricted to the class's states), evolved exactly, and summed back into classes. What RAMSES would carry:
  9 numbers per bin;
- **TN**: the tracked N_H distribution with the charge in equilibrium at every N_H (the charge steps of that column),
  evolved and summed back over charge: a direct test of "charge is fast".

Histories (all conditions are illustrative inputs or published profiles, labelled):
- A, PDR crossings: NGC 7023 NW (Meudon profile, Kurucz 15 kK), the Orion Bar and the Horsehead (Murga et al. 2020;
  40 and 30 kK), inward and outward at 0.3, 1, 3 km/s (``check_pah_hydrogen_tracking``'s paths); ions: C+ at n_e;
- B, square cycles: CNM <-> WNM (``common.PHASES``) and G0 x10 at n(H I) = 10 cm^-3 (100 K, n_e = 1.4e-3) across
  G0 = 0.1-1, 10-100, 1e3-1e4 (the bins' transitions), periods 1e3, 1e4, 1e5 yr, 4 cycles;
- C, random histories: log G0 and log n(H I) as Ornstein-Uhlenbeck processes (sigma 1 and 0.5 dex, correlation time
  tau = 1e2 ... 1e5 yr, steps tau/5, 10 tau long; n_e and the ions scale with n(H I), T fixed) around the CNM, WNM,
  translucent and PDR phases (Draine shape);
- D, mixing: two parcels at equilibrium mixed 1:1 and held at the conditions in between (geometric means), sampled
  from 1 yr to 1e6 yr: the truth evolves the mixture, E is the equilibrium there, TC/TN carry the mixed classes.

Scores per bin and history (``score``): rates as factor errors max(r, 1/r) - 1 where the truth exceeds 1e-3 of its
maximum over the history (carbon also where the bin loses its carbon within 1 Gyr); the electron source as the absolute
error over its largest magnitude; <Z> and the shares as absolute errors. Reported: the worst over the history and the
time-weighted mean ("budget").

    python diagnostics/pahchem/pahrt/s7_tracking.py --workers 4 [--per-bin 5]
    python diagnostics/pahchem/pahrt/s7_tracking.py --analyse

Figures: ``s7_paths_bin<b>.png`` (truth, E, TC, TN along six histories), ``s7_errors.png`` (each scheme's budget and
worst error against the history's time scale, per bin and output), ``s7_mixing.png`` (after mixing, against time).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "gsd"))
import common as C  # noqa: E402
import s1_drivers as S1  # noqa: E402
import s3_states as S3  # noqa: E402

OUT = Path("results/pahrt_study/s7")
YR = 3.15576e7
MURGA_ROOT = Path.home() / "Documents/GitHub/pyCALIMA/reference_data/pahchem/murga2020"
SCHEMES = ("E", "TC", "TN")
SCORED = ("heating", "cooling", "carbon", "h2", "e_source", "alpha_Cp", "z_mean", "bare", "dnh")
TAUS = (1e2, 1e3, 1e4, 1e5)
PERIODS = (1e3, 1e4, 1e5)
MIX_T = np.geomspace(1.0, 1e6, 25) * YR


# ---------------------------------------------------------------------------------------------------------------------
# histories

def _pdr_phase(q, i, teff):
    n_e = float(q["n_e"][i])
    return C.Phase(float(q["g0"][i]), float(q["n_h"][i]), float(q["t"][i]), n_e, (("C+", n_e),), f"kurucz{teff}")


def histories():
    """[(category, name, time scale [yr], [Phase], durations [s])]."""
    import check_pah_hydrogen_tracking as P2

    out = []
    nodes = {"NGC 7023": P2.ngc7023_nodes(), "Orion Bar": P2.murga_nodes("Orion Bar", MURGA_ROOT),
             "Horsehead": P2.murga_nodes("Horsehead", MURGA_ROOT)}
    for site, (av, q) in nodes.items():
        teff = P2.SITES[site]
        n_mid = np.sqrt(q["n_nuc"][1:] * q["n_nuc"][:-1])
        for v in P2.SPEEDS_KMS:
            dt = np.diff(av) * P2.NH_PER_AV / n_mid / (v * 1e5)
            ph = [_pdr_phase(q, i, teff) for i in range(av.size)]
            for inward in (True, False):
                phs, ds = (ph[:-1], dt) if inward else (ph[::-1][:-1], dt[::-1])
                out.append(("A", f"{site} {'in' if inward else 'out'} {v:g} km/s", float(dt.sum() / YR), phs, ds))
    cnm, wnm = C.PHASES["CNM"], C.PHASES["WNM"]
    for period in PERIODS:
        out.append(("B", f"CNM<->WNM {period:.0e} yr", period, [cnm, wnm] * 4, np.full(8, 0.5 * period * YR)))
        for g in (0.1, 10.0, 1e3):
            lo = C.Phase(g, 10.0, 100.0, 1.4e-3, (("C+", 1.4e-3),))
            out.append(("B", f"G0 {g:g}<->{10 * g:g} {period:.0e} yr", period, [lo, replace(lo, g0=10 * g)] * 4,
                        np.full(8, 0.5 * period * YR)))
    rng = np.random.default_rng(20261124)
    for base in ("CNM", "WNM", "translucent", "PDR"):
        ph0 = replace(C.PHASES[base], field="draine")
        for tau in TAUS:
            n = 50
            a = np.exp(-0.2)
            xg, xn = np.zeros(n), np.zeros(n)
            for k in range(1, n):
                xg[k] = xg[k - 1] * a + 1.0 * np.sqrt(1 - a * a) * rng.standard_normal()
                xn[k] = xn[k - 1] * a + 0.5 * np.sqrt(1 - a * a) * rng.standard_normal()
            phs = []
            for k in range(n):
                fn = 10.0 ** xn[k]
                g = min(ph0.g0 * 10.0 ** xg[k], 1e6)
                phs.append(replace(ph0, g0=g, n_hi=ph0.n_hi * fn, n_e=ph0.n_e * fn, ions=C.scaled_ions(ph0, fn)))
            out.append(("C", f"random {base} tau {tau:.0e} yr", tau, phs, np.full(n, 0.2 * tau * YR)))
    return out


def mixtures():
    """[(name, Phase a, Phase b, Phase of the mixture)]: parcels of one field shape."""
    import check_pah_hydrogen_tracking as P2

    def mid(a, b):
        g = lambda x, y: float(np.sqrt(x * y))                                       # noqa: E731
        return replace(a, g0=g(a.g0, b.g0), n_hi=g(a.n_hi, b.n_hi), T=g(a.T, b.T), n_e=g(a.n_e, b.n_e),
                       ions=tuple((n, g(x, dict(b.ions).get(n, x))) for n, x in a.ions))

    cnm, wnm = C.PHASES["CNM"], C.PHASES["WNM"]
    bright = replace(wnm, g0=170.0)                                  # bin 1-2 molecules stripped
    av, q = P2.murga_nodes("Orion Bar", MURGA_ROOT)
    edge, deep = _pdr_phase(q, 0, 40000), _pdr_phase(q, av.size - 1, 40000)
    return [("CNM + WNM", cnm, wnm, mid(cnm, wnm)), ("bright WNM + CNM", bright, cnm, mid(bright, cnm)),
            ("Orion Bar edge + shielded end", edge, deep, mid(edge, deep))]


# ---------------------------------------------------------------------------------------------------------------------
# per molecule

class Cell:
    """A molecule's network at one condition, with what the schemes need (cached per condition)."""

    def __init__(self, truth, ph):
        self.env = ph.environment()
        self.net, self.W = truth.network(self.env)
        self.F = self.net.steady_state_closed()
        self.ion = truth.ion_rates(self.env)
        self.Q = self.R = None                     # dense matrices: built per segment (``load``), then dropped

    def load(self):
        self.Q = self.net.generator().toarray()
        self.R = self.net.transition_matrix()

    def drop(self):
        self.Q = self.R = None

    def out(self, nh0, f):
        o = C.outputs(self.net, self.W, nh0, self.ion, self.env, f)
        return [o[k] for k in C.OUTPUTS]

    def evolve(self, f, dt):
        g = np.clip(scipy.linalg.expm(self.Q * dt) @ f.ravel(), 0.0, None)
        return (g / g.sum()).reshape(self.net.shape)


def _local(R, idx):
    """The stationary distribution of the chain restricted to the states ``idx`` (None if it is not irreducible)."""
    from pycalima.pahchem.network import gth_stationary

    if len(idx) == 1:
        return np.ones(1)
    return gth_stationary(R[np.ix_(idx, idx)])


def class_rebuild(cell, cls_of_n, pops, n_classes):
    """f[Z, N_H] from class populations: each class's population over its local equilibrium (fallback: the global
    steady state restricted to it, else uniform)."""
    nz, nn = cell.net.shape
    f = np.zeros((nz, nn))
    states = np.arange(nz * nn).reshape(nz, nn)
    for k in range(n_classes):
        if pops[k] <= 0.0:
            continue
        cols = np.nonzero(cls_of_n == k)[0]
        if cols.size == 0:
            continue
        idx = states[:, cols].ravel()
        loc = _local(cell.R, idx)
        if loc is None or not np.all(np.isfinite(loc)) or loc.sum() <= 0:
            loc = cell.F[:, cols].ravel()
            loc = loc / loc.sum() if loc.sum() > 0 else np.full(idx.size, 1.0 / idx.size)
        f.ravel()[idx] += pops[k] * loc
    return f / f.sum()


def column_rebuild(cell, p_n):
    """f[Z, N_H] from the N_H distribution, the charge in equilibrium within each column."""
    nz, nn = cell.net.shape
    f = np.zeros((nz, nn))
    states = np.arange(nz * nn).reshape(nz, nn)
    for n in range(nn):
        if p_n[n] <= 0.0:
            continue
        loc = _local(cell.R, states[:, n])
        if loc is None or loc.sum() <= 0:
            loc = cell.F[:, n] / cell.F[:, n].sum() if cell.F[:, n].sum() > 0 else np.full(nz, 1.0 / nz)
        f[:, n] = p_n[n] * loc
    return f / f.sum()


def run_molecule(mol_name):
    mols = {m.name: m for ms in C.molecules(S1.ARGS_PER_BIN).values() for m in ms}
    mol = mols[mol_name]
    truth = C.Truth(mol)
    nh0 = mol.nh
    cache = {}
    errors = {}

    def cell(ph):
        if ph not in cache:
            cache[ph] = Cell(truth, ph)
        return cache[ph]

    t0 = time.time()
    hist = histories()
    res = []
    for _cat, name, _, phs, ds in hist:
        arr = np.full((len(phs), 1 + len(SCHEMES), len(C.OUTPUTS)), np.nan)
        try:
            c0 = cell(phs[0])
            cls = S3.class_of(c0.net.nh_values, nh0)
            nc = len(S3.CLASSES)
            f_t = c0.F.copy()
            f_c, f_n = c0.F.copy(), c0.F.copy()
            for k, (ph, dt) in enumerate(zip(phs, ds, strict=True)):
                c = cell(ph)
                c.load()
                f_t = c.evolve(f_t, dt)
                pops = np.bincount(cls, weights=f_c.sum(axis=0), minlength=nc)
                f_c = c.evolve(class_rebuild(c, cls, pops, nc), dt)
                f_n = c.evolve(column_rebuild(c, f_n.sum(axis=0)), dt)
                arr[k, 0] = c.out(nh0, f_t)
                arr[k, 1] = c.out(nh0, c.F)
                arr[k, 2] = c.out(nh0, f_c)
                arr[k, 3] = c.out(nh0, f_n)
                c.drop()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc(limit=3)
        res.append(arr)
    mix = []
    for name, a, b, m in mixtures():
        arr = np.full((MIX_T.size, 1 + len(SCHEMES), len(C.OUTPUTS)), np.nan)
        try:
            ca, cb, cm = cell(a), cell(b), cell(m)
            cm.load()
            if ca.net.shape != cb.net.shape or ca.net.shape != cm.net.shape:
                raise ValueError("parcels on different networks")
            f0 = 0.5 * (ca.F + cb.F)
            cls = S3.class_of(cm.net.nh_values, nh0)
            nc = len(S3.CLASSES)
            fc0 = class_rebuild(cm, cls, np.bincount(cls, weights=f0.sum(axis=0), minlength=nc), nc)
            fn0 = column_rebuild(cm, f0.sum(axis=0))
            for k, t in enumerate(MIX_T):
                arr[k, 0] = cm.out(nh0, cm.evolve(f0, t))
                arr[k, 1] = cm.out(nh0, cm.F)
                arr[k, 2] = cm.out(nh0, cm.evolve(fc0, t))
                arr[k, 3] = cm.out(nh0, cm.evolve(fn0, t))
            cm.drop()
        except Exception as exc:
            errors[f"mix {name}"] = f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc(limit=3)
        mix.append(arr)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{S1.safe(mol_name)}.npz"
    np.savez(path, **{f"h{i}": a for i, a in enumerate(res)}, **{f"m{i}": a for i, a in enumerate(mix)},
             nc=mol.nc, nh=mol.nh, name=mol_name, seconds=time.time() - t0, errors=json.dumps(errors))
    return str(path), time.time() - t0, len(errors)


def run(workers, per_bin):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    OUT.mkdir(parents=True, exist_ok=True)
    mols = [m for ms in C.molecules(per_bin).values() for m in ms]
    todo = sorted((m for m in mols if not (OUT / f"{S1.safe(m.name)}.npz").exists()), key=lambda m: -m.nc)
    h = histories()
    print(f"{len(h)} histories ({sum(len(x[3]) for x in h)} segments), {len(mixtures())} mixtures; {len(todo)} of "
          f"{len(mols)} molecules on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers, initializer=S1._init, initargs=(per_bin,)) as ex:
        futs = {ex.submit(run_molecule, m.name): m.name for m in todo}
        for f in as_completed(futs):
            try:
                path, sec, nerr = f.result()
                print(f"{futs[f]}: {sec:.0f} s, {nerr} errors -> {path}", flush=True)
            except Exception as exc:
                print(f"{futs[f]}: FAILED {type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# analysis

def load(per_bin):
    """{bin: (per-molecule history arrays [mol][h] -> [seg, scheme, out], mixture arrays, N_C list)}."""
    out = {}
    for b, ms in C.molecules(per_bin).items():
        H, M, ncs = [], [], []
        for m in ms:
            p = OUT / f"{S1.safe(m.name)}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            H.append([d[f"h{i}"] for i in range(len([k for k in d.files if k.startswith("h")]))])
            M.append([d[f"m{i}"] for i in range(len([k for k in d.files if k.startswith("m")]))])
            ncs.append(float(d["nc"]))
        out[b] = (H, M, ncs)
    return out


def bin_sum(arrs):
    """The bin's totals of one history: molecules summed (rates and shares summed, then shares and <Z> divided by the
    number of molecules, i.e. equal numbers of each)."""
    a = np.nansum(np.array(arrs), axis=0)
    n = len(arrs)
    oi = {k: i for i, k in enumerate(C.OUTPUTS)}
    for k in C.OUTPUTS:
        if k not in C.RATES and k != "e_source":
            a[..., oi[k]] /= n
    return a


def score(a, durs, nc_tot):
    """{output: (worst, time-weighted mean)} for each scheme of one bin history a[seg, 1 + scheme, out]."""
    oi = {k: i for i, k in enumerate(C.OUTPUTS)}
    w = np.asarray(durs, float) / np.sum(durs)
    out = {}
    for k in SCORED:
        t = a[:, 0, oi[k]]
        res = []
        for s in range(len(SCHEMES)):
            x = a[:, 1 + s, oi[k]]
            if k in C.RATES:
                ok = np.isfinite(t) & np.isfinite(x) & (t > 1e-3 * np.nanmax(t)) & (t > 0) & (x > 0)
                if k == "carbon":
                    ok &= t > nc_tot / (1e9 * YR)
                e = np.where(ok, np.maximum(x / np.where(ok, t, 1), np.where(ok, t, 1) / np.maximum(x, 1e-300)) - 1, 0)
            elif k == "e_source":
                e = np.abs(x - t) / max(np.nanmax(np.abs(t)), 1e-300)
            else:
                e = np.abs(x - t)
            e = np.nan_to_num(e)
            res.append((float(e.max()), float(w @ e)))
        out[k] = res
    return out


def analyse(per_bin):
    data = load(per_bin)
    hist = histories()
    mixes = mixtures()
    lines = ["PAHrt S7: track the hydrogenation of the PAH bins, or take it in equilibrium?",
             "Schemes against the exact evolution of each molecule's full joint chain (molecules summed per bin):",
             "E = equilibrium; TC = tracked class populations (local equilibrium within each class); TN = tracked N_H with",
             "the charge in equilibrium per N_H. Errors: rates factor - 1, e_source over its max, <Z> and shares absolute;",
             "'worst / time-weighted mean' over each history.", ""]
    table = {}
    for b, (H, _, ncs) in data.items():
        if not H:
            continue
        lines.append(f"=== bin {b + 1} ({len(H)} molecules)")
        for i, (cat, name, _tscale, _, ds) in enumerate(hist):
            a = bin_sum([h[i] for h in H])
            sc = score(a, ds, sum(ncs))
            table[(b, i)] = sc
            cells = []
            for k in ("heating", "carbon", "h2", "z_mean", "bare"):
                cells.append(f"{k} " + " ".join(f"{SCHEMES[s]} {sc[k][s][0]:.2g}/{sc[k][s][1]:.2g}"
                                                for s in range(len(SCHEMES))))
            lines.append(f"  [{cat}] {name:38s} " + " | ".join(cells))
        lines.append("")
    lines.append("Summary per bin and category: the largest time-weighted error of each scheme (and its worst cell)")
    for b in data:
        for cat, label in (("A", "PDR crossings"), ("B", "square cycles"), ("C", "random histories")):
            rows = [i for i, h in enumerate(hist) if h[0] == cat and (b, i) in table]
            if not rows:
                continue
            cells = []
            for k in ("heating", "cooling", "carbon", "h2", "e_source", "z_mean", "bare"):
                cells.append(f"{k} " + " ".join(
                    f"{SCHEMES[s]} {max(table[(b, i)][k][s][1] for i in rows):.2g}"
                    f"({max(table[(b, i)][k][s][0] for i in rows):.2g})" for s in range(len(SCHEMES))))
            lines.append(f"  bin {b + 1} {label:16s} " + "; ".join(cells))
    lines.append("")
    lines.append("Mixing: error at 1 yr, 1e3 yr and 1e5 yr after mixing (heating, carbon, h2, bare), per bin and scheme")
    oi = {k: i for i, k in enumerate(C.OUTPUTS)}
    for b, (_, M, _) in data.items():
        if not M:
            continue
        for j, (name, *_rest) in enumerate(mixes):
            a = bin_sum([m[j] for m in M])
            cells = []
            for k in ("heating", "carbon", "h2", "bare"):
                t = a[:, 0, oi[k]]
                vals = []
                for s in range(len(SCHEMES)):
                    x = a[:, 1 + s, oi[k]]
                    e = (np.maximum(x / np.maximum(t, 1e-300), t / np.maximum(x, 1e-300)) - 1 if k in C.RATES
                         else np.abs(x - t))
                    vals.append("/".join(f"{np.interp(np.log(tt), np.log(MIX_T / YR), e):.2g}" for tt in (1, 1e3, 1e5)))
                cells.append(f"{k} " + " ".join(f"{SCHEMES[s]} {vals[s]}" for s in range(len(SCHEMES))))
            lines.append(f"  bin {b + 1} {name:30s} " + " | ".join(cells))
    txt = "\n".join(lines)
    (OUT.parent / "s7_tracking.txt").write_text(txt + "\n")
    print(txt)
    figures(data, hist, mixes, table)


def figures(data, hist, mixes, table):
    import plots as PL

    plt = PL.setup()
    oi = {k: i for i, k in enumerate(C.OUTPUTS)}
    scol = {"truth": "k", "E": "C3", "TC": "C0", "TN": "C2"}
    sls = {"truth": "-", "E": "--", "TC": "-", "TN": ":"}
    pick = ("Orion Bar in 1 km/s", "Horsehead out 1 km/s", "CNM<->WNM 1e+03 yr", "G0 10<->100 1e+03 yr",
            "random CNM tau 1e+03 yr", "random PDR tau 1e+03 yr")
    outs = (("heating", r"$\Gamma_{\rm PE}$ [erg s$^{-1}$]", True), ("carbon", r"C loss [s$^{-1}$]", True),
            ("h2", r"H$_2$ [s$^{-1}$]", True), ("z_mean", r"$\langle Z\rangle$", False),
            ("dnh", r"$\langle N_{\rm H}\rangle - N_{\rm H,0}$", False))
    names = [h[1] for h in hist]
    for b, (H, _, _) in data.items():
        if not H:
            continue
        fig, axes = plt.subplots(len(outs), len(pick), figsize=(20, 13), sharex="col")
        for j, nm in enumerate(pick):
            i = names.index(nm)
            a = bin_sum([h[i] for h in H])
            t = np.cumsum(hist[i][4]) / YR
            for r, (k, lab, logy) in enumerate(outs):
                ax = axes[r, j]
                for s, sch in enumerate(("truth", *SCHEMES)):
                    y = a[:, s, oi[k]]
                    if logy:
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(t, y, sls[sch], color=scol[sch], lw=1.3 if sch == "truth" else 1.1,
                            label=sch if (r == 0 and j == 0) else None, drawstyle="steps-pre")
                PL.style(ax, logx=hist[i][0] == "A", logy=logy)
                if r == 0:
                    ax.set_title(nm.replace("<->", r" $\leftrightarrow$ "), fontsize=10)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=11)
                if r == len(outs) - 1:
                    ax.set_xlabel("time [yr]", fontsize=11)
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle(PL.BIN_LABELS[b] + ": truth (black), equilibrium E (red dashed), tracked classes TC (blue), tracked "
                     r"$N_{\rm H}$ with fast charge TN (green dotted)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT.parent / f"s7_paths_bin{b + 1}.png", dpi=170)
        plt.close(fig)
    # errors against the history's time scale
    souts = (("heating", r"$\Gamma_{\rm PE}$"), ("cooling", r"$\Lambda_{\rm rec}$"), ("carbon", "C loss"),
             ("h2", r"H$_2$"), ("z_mean", r"$|\Delta\langle Z\rangle|$"), ("bare", r"$|\Delta$ bare$|$"))
    marks = {"A": "o", "B": "s", "C": "^"}
    fig, axes = plt.subplots(4, len(souts), figsize=(20, 14), sharex=True)
    for b in range(4):
        for j, (k, lab) in enumerate(souts):
            ax = axes[b, j]
            for i, h in enumerate(hist):
                if (b, i) not in table:
                    continue
                for s, sch in enumerate(SCHEMES):
                    worst, mean = table[(b, i)][k][s]
                    x = h[2] * (1.0 + 0.12 * (s - 1))
                    ax.plot(x, max(mean, 1e-6), marks[h[0]], color=scol[sch], ms=4, alpha=0.8)
                    ax.plot([x, x], [max(mean, 1e-6), max(worst, 1e-6)], "-", color=scol[sch], lw=0.5, alpha=0.5)
            PL.style(ax, logx=True, logy=True)
            ax.set_ylim(1e-6, 1e3)
            ax.axhline(0.1, color="0.5", lw=0.8, ls=":")
            if b == 3:
                ax.set_xlabel("time scale [yr]", fontsize=11)
            if j == 0:
                ax.set_ylabel(PL.BIN_LABELS[b] + "\nerror (mean; bar to worst)", fontsize=10)
            if b == 0:
                ax.set_title(lab, fontsize=12)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=scol[s], marker="o", ls="", label=s) for s in SCHEMES]
    handles += [Line2D([], [], color="0.3", marker=m, ls="", label=lab)
                for m, lab in (("o", "PDR crossings"), ("s", "square cycles"), ("^", "random histories"))]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=10,
               title="time scale: PDR crossing time, cycle period, or the random history's correlation time; "
                     "dotted: 0.1", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT.parent / "s7_errors.png", dpi=170)
    plt.close(fig)
    # mixing
    fig, axes = plt.subplots(len(mixes), 4, figsize=(17, 4 * len(mixes)), sharex=True)
    axes = np.atleast_2d(axes)
    bcol = PL.bin_colors(plt)
    for j, (name, *_r) in enumerate(mixes):
        for c, (k, lab) in enumerate((("heating", r"$\Gamma_{\rm PE}$ factor $-1$"), ("carbon", r"C loss factor $-1$"),
                                      ("h2", r"H$_2$ factor $-1$"), ("bare", r"$|\Delta$ bare$|$"))):
            ax = axes[j, c]
            for b, (_, M, _) in data.items():
                if not M:
                    continue
                a = bin_sum([m[j] for m in M])
                t = a[:, 0, oi[k]]
                for s, sch in enumerate(SCHEMES):
                    x = a[:, 1 + s, oi[k]]
                    e = (np.maximum(x / np.maximum(t, 1e-300), t / np.maximum(x, 1e-300)) - 1 if k in C.RATES
                         else np.abs(x - t))
                    ax.plot(MIX_T / YR, np.maximum(e, 1e-6), sls[sch], color=bcol[b], lw=1.2,
                            label=f"bin {b + 1} {sch}" if (j == 0 and c == 0) else None)
            PL.style(ax, logx=True, logy=True)
            ax.set_ylim(1e-6, 1e3)
            if j == 0:
                ax.set_title(lab, fontsize=11)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
            if j == len(mixes) - 1:
                ax.set_xlabel("time after mixing [yr]", fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, title="dashed E, solid TC, dotted TN", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT.parent / "s7_mixing.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-bin", type=int, default=5)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args(argv)
    S1._init(a.per_bin)
    if a.analyse:
        analyse(a.per_bin)
    else:
        run(a.workers, a.per_bin)


if __name__ == "__main__":
    main()
