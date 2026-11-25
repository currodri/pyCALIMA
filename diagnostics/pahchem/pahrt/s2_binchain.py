"""PAHrt study S2: one averaged network per PAH bin, against the mean of its molecules.

The per-cell solver will hold one network per bin, not one per molecule (decided 2026-11-24: tracked hydrogenation
classes for bins 1-2, equilibrium for bins 3-4). Each molecule's full joint (Z, N_H) chain is first lumped onto
(Z, class) aggregates (``s3_states.CLASSES``: bare, deep, N_H0-3 ... N_H0+4, so that molecules with different N_H0
share one grid): inside each class the states are weighted by the class's local equilibrium (the chain restricted to
the class, as ``s7_tracking``'s TC rebuilds it), the rates between aggregates are those weights times the full rates,
and every per-state output (heating, cooling, absorbed power, carbon loss, H2, H uptake, the electrons each state
makes, <N_H> - N_H0) is weighted the same way. The bin's network is the molecules' lumped rates and outputs averaged
state by state (equal weights; each aggregate over the molecules that have it), as ``group_tables`` averages kernels.

Two tests, against the mean over the bin's molecules (equal numbers, ``common.molecules``):
- equilibrium (the scheme of bins 3-4, reported for all): the bin network's steady state at every node of S1's design
  (``s1_drivers.design``), against the mean of the molecules' full equilibria (S1's results);
- tracking (bins 1-2): the bin network evolved through S7's histories (``s7_tracking.histories``, rates held per
  segment), against the exact evolution of each molecule (S7's truth) and against each molecule's own class tracking
  (S7's TC): what averaging adds to tracking.

Per molecule the lumped quantities are computed in a worker and stored in ``results/pahrt_study/s2/<molecule>.npz``;
``--analyse`` averages them per bin, solves and scores (as ``s7_tracking.score``), and draws ``s2_equilibrium.png``
(bin network against the molecules' mean over S1's sweeps), ``s2_tracking_bin<b>.png`` (histories) and
``s2_errors.png``.

    python diagnostics/pahchem/pahrt/s2_binchain.py --workers 4
    python diagnostics/pahchem/pahrt/s2_binchain.py --analyse
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "gsd"))
import common as C  # noqa: E402
import s1_drivers as S1  # noqa: E402
import s3_states as S3  # noqa: E402
import s7_tracking as S7  # noqa: E402

OUT = Path("results/pahrt_study/s2")
YR = S7.YR
WKEYS = ("heating", "cooling", "absorbed", "carbon", "h2", "h_uptake")
NCL = len(S3.CLASSES)
TRACKED_BINS = (0, 1)


def electrons_per_state(net, ion_rates, env):
    """e[Z, N_H]: free electrons made per second by a molecule in each state (``common.outputs``' bookkeeping)."""
    z = np.asarray(net.charges)
    ion_up = np.zeros(z.size)
    for ion, n in env.ions:
        k_mn, k_ct = ion_rates[ion]
        ion_up += n * np.where(z == -1, k_mn, 0.0) + n * np.where(z == 0, k_ct, 0.0)
    e = np.zeros(net.shape)
    for p in net.processes:
        per = C.electrons_per_event(p.name, p.dz)
        if per == 0:
            continue
        r = p.rate
        if p.name == "ionisation (dZ +1)":
            r = np.maximum(r - ion_up[:, None], 0.0)
        e += per * r
    return e


def lump(cell, nh0):
    """The molecule's lumped (Z, class) quantities at one condition: (charges, R[a, b], {key: w[a]}, present[a])
    with aggregates a = iz * NCL + class."""
    net = cell.net
    nz, nn = net.shape
    cls = S3.class_of(net.nh_values, nh0)
    R = net.transition_matrix()
    states = np.arange(nz * nn).reshape(nz, nn)
    na = nz * NCL
    P = np.zeros((na, nz * nn))                      # P[a, state]: the within-aggregate weights
    present = np.zeros(na, bool)
    for k in range(NCL):
        cols = np.nonzero(cls == k)[0]
        if cols.size == 0:
            continue
        idx = states[:, cols].ravel()
        loc = S7._local(R, idx)
        if loc is None or not np.all(np.isfinite(loc)) or loc.sum() <= 0:
            loc = cell.F[:, cols].ravel()
            if loc.sum() <= 0:
                loc = np.ones(idx.size)
        loc = loc.reshape(nz, cols.size)
        for iz in range(nz):
            w = loc[iz]
            if w.sum() <= 0:
                w = np.ones(cols.size)
            a = iz * NCL + k
            P[a, states[iz, cols]] = w / w.sum()
            present[a] = True
    agg = np.zeros(nz * nn, int)                     # the aggregate of every state
    for iz in range(nz):
        agg[states[iz]] = iz * NCL + cls
    M = np.zeros((nz * nn, na))
    M[np.arange(nz * nn), agg] = 1.0
    Rl = P @ R @ M
    np.fill_diagonal(Rl, 0.0)
    w = {k: P @ cell.W[k].ravel() for k in WKEYS}
    w["e"] = P @ electrons_per_state(net, cell.ion, cell.env).ravel()
    w["dnh"] = P @ np.repeat((np.asarray(net.nh_values) - nh0)[None, :], nz, axis=0).ravel()
    return np.asarray(net.charges), Rl, w, present


def run_molecule(mol_name):
    mols = {m.name: (b, m) for b, ms in C.molecules(S1.ARGS_PER_BIN).items() for m in ms}
    b, mol = mols[mol_name]
    truth = C.Truth(mol)
    t0 = time.time()
    store = {}

    def put(tag, cell):
        z, Rl, w, pres = lump(cell, mol.nh)
        store[f"{tag}_z"] = z
        store[f"{tag}_R"] = Rl.astype(np.float64)
        store[f"{tag}_p"] = pres
        for k, v in w.items():
            store[f"{tag}_{k}"] = v
        for ion, (k_mn, k_ct) in cell.ion.items():
            store[f"{tag}_k{ion}"] = np.array([k_mn, k_ct])

    for i, (_, _, _, ph) in enumerate(S1.design()):
        try:
            cell = S7.Cell(truth, ph)
            put(f"e{i}", cell)
        except Exception as exc:
            store[f"e{i}_err"] = np.array(str(exc)[:200])
    if b in TRACKED_BINS:
        cache = {}
        for j, (_, _, _, phs, _) in enumerate(S7.histories()):
            for k, ph in enumerate(phs):
                try:
                    if ph not in cache:
                        cache[ph] = len(cache)
                        put(f"c{cache[ph]}", S7.Cell(truth, ph))
                    store[f"h{j}_{k}"] = np.array(cache[ph])
                except Exception as exc:
                    store[f"h{j}_{k}_err"] = np.array(str(exc)[:200])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{S1.safe(mol_name)}.npz"
    np.savez_compressed(path, nc=mol.nc, nh=mol.nh, name=mol_name, **store)
    return str(path), time.time() - t0, 0


def run(workers, per_bin):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    OUT.mkdir(parents=True, exist_ok=True)
    mols = [m for ms in C.molecules(per_bin).values() for m in ms]
    todo = sorted((m for m in mols if not (OUT / f"{S1.safe(m.name)}.npz").exists()), key=lambda m: -m.nc)
    print(f"{len(todo)} of {len(mols)} molecules on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers, initializer=S1._init, initargs=(per_bin,)) as ex:
        futs = {ex.submit(run_molecule, m.name): m.name for m in todo}
        for f in as_completed(futs):
            try:
                path, sec, _ = f.result()
                print(f"{futs[f]}: {sec:.0f} s -> {path}", flush=True)
            except Exception as exc:
                print(f"{futs[f]}: FAILED {type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# the bin network

class BinChain:
    """The bin's averaged lumped network at one condition, from its molecules' stores (``tag`` of each)."""

    def __init__(self, items, ions):
        zs = sorted({int(z) for d, t in items for z in d[f"{t}_z"]})
        self.z = np.array(zs)
        na = self.z.size * NCL
        R = np.zeros((na, na))
        cnt = np.zeros(na)
        w = {k: np.zeros(na) for k in (*WKEYS, "e", "dnh")}
        kion = {ion: np.zeros(2) for ion in ions}
        for d, t in items:
            zm = d[f"{t}_z"]
            sel = np.concatenate([(zs.index(int(z)) * NCL + np.arange(NCL)) for z in zm])   # molecule -> bin aggregates
            pres = d[f"{t}_p"]
            R[np.ix_(sel, sel)] += d[f"{t}_R"] * pres[:, None]
            cnt[sel] += pres
            for k in w:
                w[k][sel] += d[f"{t}_{k}"] * pres
            for ion in ions:
                kion[ion] += d[f"{t}_k{ion}"] / len(items)
        n = np.maximum(cnt, 1)
        self.R = R / n[:, None]
        self.w = {k: v / n for k, v in w.items()}
        self.present = cnt > 0
        self.kion = kion
        self.zag = np.repeat(self.z, NCL)
        self.cls = np.tile(np.arange(NCL), self.z.size)

    def steady(self):
        from pycalima.pahchem.network import closed_class_stationary

        keep = np.nonzero(self.present)[0]
        f = np.zeros(self.R.shape[0])
        f[keep] = closed_class_stationary(self.R[np.ix_(keep, keep)])
        return f

    def evolve(self, f, dt):
        Q = self.R.T.copy()
        Q[np.diag_indices_from(Q)] = -self.R.sum(axis=1)
        g = np.clip(scipy.linalg.expm(Q * dt) @ f, 0.0, None)
        return g / g.sum()

    def outputs(self, f, env):
        o = {k: float(f @ self.w[k]) for k in WKEYS}
        fz = {c: float(f[self.zag == c].sum()) for c in self.z}
        o.update(f_m1=fz.get(-1, 0.0), f_0=fz.get(0, 0.0), f_p1=fz.get(1, 0.0), f_p2=fz.get(2, 0.0),
                 z_mean=float(f @ self.zag), e_source=float(f @ self.w["e"]), bare=float(f[self.cls == 0].sum()),
                 dehydrogenated=float(f[(self.cls >= 1) & (self.cls <= 4)].sum()), intact=float(f[self.cls == 5].sum()),
                 superh=float(f[self.cls >= 6].sum()), dnh=float(f @ self.w["dnh"]))
        for ion, key in (("C+", "alpha_Cp"), ("H+", "alpha_Hp")):
            k = self.kion.get(ion)
            o[key] = 0.0 if k is None else o["f_m1"] * k[0] + o["f_0"] * k[1]
        return [o[k] for k in C.OUTPUTS]


def _ions(env):
    return [ion for ion, _ in env.ions]


def analyse(per_bin):
    mols = C.molecules(per_bin)
    nodes = S1.design()
    hist = S7.histories()
    lines = ["PAHrt S2: one averaged lumped (Z, class) network per bin against the mean of its molecules.",
             "Equilibrium: S1's nodes (bin network's steady state vs the molecules' mean full equilibrium). Tracking",
             "(bins 1-2): S7's histories (bin network evolved vs the molecules' exact evolution; their own class",
             "tracking TC shown for reference). Errors as in S7 (rates factor - 1 where they matter, e_source over its",
             "max, <Z> and shares absolute): worst / mean over the nodes (equilibrium) or the time-weighted mean.", ""]
    eq = {}
    for b, ms in mols.items():
        ds = [np.load(OUT / f"{S1.safe(m.name)}.npz") for m in ms if (OUT / f"{S1.safe(m.name)}.npz").exists()]
        s1 = [np.load(S1.OUT / f"{S1.safe(m.name)}.npz")["res"] for m in ms]
        if len(ds) != len(ms):
            continue
        truth = np.nanmean(np.array(s1), axis=0)                           # [node, out]: equal-weight mean
        binv = np.full_like(truth, np.nan)
        for i, (_, _, _, ph) in enumerate(nodes):
            if any(f"e{i}_err" in d.files for d in ds):
                continue
            env = ph.environment()
            ch = BinChain([(d, f"e{i}") for d in ds], _ions(env))
            try:
                binv[i] = ch.outputs(ch.steady(), env)
            except Exception:
                continue
        eq[b] = (truth, binv)
        e = S3.errors_of(truth, binv)
        cells = []
        for k in ("z_mean", "f_m1", "f_p1", "heating", "cooling", "carbon", "h2", "e_source", "alpha_Cp", "bare", "dnh"):
            x = e[k][np.isfinite(e[k])]
            if x.size:
                v = x - 1 if k in C.RATES else x
                cells.append(f"{k} {np.max(v):.2g}/{np.mean(v):.2g}")
        lines.append(f"equilibrium bin {b + 1}: " + "; ".join(cells))
    lines.append("")
    trk = {}
    for b in TRACKED_BINS:
        ms = mols[b]
        ds = [np.load(OUT / f"{S1.safe(m.name)}.npz") for m in ms]
        s7 = [np.load(S7.OUT / f"{S1.safe(m.name)}.npz") for m in ms]
        rows = []
        for j, (cat, name, _, phs, durs) in enumerate(hist):
            truth = S7.bin_sum([d[f"h{j}"] for d in s7])                      # [seg, 1 + scheme, out] (molecules summed)
            n = len(ms)
            arr = np.full((len(phs), 3, len(C.OUTPUTS)), np.nan)              # truth, TC (per molecule), bin network
            arr[:, 0], arr[:, 1] = truth[:, 0], truth[:, 2]
            f = None
            try:
                for k, ph in enumerate(phs):
                    env = ph.environment()
                    ch = BinChain([(d, f"c{int(d[f'h{j}_{k}'])}") for d in ds], _ions(env))
                    if f is None or f.size != ch.R.shape[0]:
                        f = ch.steady()
                    f = ch.evolve(f, durs[k])
                    out = np.array(ch.outputs(f, env))
                    oi = {kk: ii for ii, kk in enumerate(C.OUTPUTS)}
                    for kk in (*C.RATES, "e_source"):        # the bin network is the mean molecule: x n = the sum
                        out[oi[kk]] *= n
                    arr[k, 2] = out
            except Exception as exc:
                lines.append(f"  {name}: {type(exc).__name__}: {exc}"[:200])
            sc = S7.score(arr, durs, float(sum(d["nc"] for d in ds)))
            rows.append((cat, name, sc))
            trk[(b, j)] = arr
        for cat, label in (("A", "PDR crossings"), ("B", "square cycles"), ("C", "random histories")):
            sel = [r for r in rows if r[0] == cat]
            cells = []
            for k in ("heating", "cooling", "carbon", "h2", "e_source", "z_mean", "bare"):
                tc = max(r[2][k][0][1] for r in sel)
                bn = max(r[2][k][1][1] for r in sel)
                cells.append(f"{k} TC {tc:.2g} bin {bn:.2g} (worst {max(r[2][k][1][0] for r in sel):.2g})")
            lines.append(f"tracking bin {b + 1} {label:16s} " + "; ".join(cells))
    txt = "\n".join(lines)
    (OUT.parent / "s2_binchain.txt").write_text(txt + "\n")
    print(txt)
    figures(eq, trk, nodes, hist)


def figures(eq, trk, nodes, hist):
    import plots as PL

    plt = PL.setup()
    oi = {k: i for i, k in enumerate(C.OUTPUTS)}
    key = [(n[0], n[1]) for n in nodes]
    outs = (("z_mean", r"$\langle Z\rangle$", False), ("heating", r"$\Gamma_{\rm PE}$", True),
            ("cooling", r"$\Lambda_{\rm rec}$", True), ("carbon", r"C loss", True), ("h2", r"H$_2$", True),
            ("dnh", r"$\langle N_{\rm H}\rangle - N_{\rm H,0}$", False))
    # equilibrium: the bin network (dashed) against the molecules' mean (solid) along every phase's G0 and n(H I) sweep
    pcol = PL.phase_colors(plt, list(C.PHASES))
    for b, (truth, binv) in eq.items():
        fig, axes = plt.subplots(len(outs), 2, figsize=(11, 15), sharex="col")
        for j, inp in enumerate(("g0", "n_hi")):
            for ph in C.PHASES:
                rows = [i for i, kk in enumerate(key) if kk == (ph, inp)]
                x = np.array([nodes[i][2] for i in rows], float)
                for r, (k, lab, logy) in enumerate(outs):
                    ax = axes[r, j]
                    yt, yb = truth[rows, oi[k]], binv[rows, oi[k]]
                    if logy:
                        yt, yb = np.where(yt > 0, yt, np.nan), np.where(yb > 0, yb, np.nan)
                    ax.plot(x, yt, "-", color=pcol[ph], lw=1.4, label=ph if (r == 0 and j == 0) else None)
                    ax.plot(x, yb, "--", color=pcol[ph], lw=1.2)
                    PL.style(ax, logx=True, logy=logy)
                    if j == 0:
                        ax.set_ylabel(lab, fontsize=11)
            axes[-1, j].set_xlabel(r"$G_0$" if inp == "g0" else r"$n({\rm H\,I})$ [cm$^{-3}$]", fontsize=12)
        axes[0, 0].legend(frameon=False, fontsize=8, title="solid: molecules' mean; dashed: bin network",
                          title_fontsize=8)
        fig.suptitle(PL.BIN_LABELS[b] + ": equilibrium of the averaged bin network against its molecules", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(OUT.parent / f"s2_equilibrium_bin{b + 1}.png", dpi=170)
        plt.close(fig)
    # tracking: six histories per tracked bin
    pick = ("Orion Bar in 1 km/s", "Horsehead out 1 km/s", "CNM<->WNM 1e+03 yr", "G0 10<->100 1e+03 yr",
            "random CNM tau 1e+03 yr", "random PDR tau 1e+03 yr")
    names = [h[1] for h in hist]
    for b in TRACKED_BINS:
        fig, axes = plt.subplots(len(outs), len(pick), figsize=(20, 14), sharex="col")
        for j, nm in enumerate(pick):
            i = names.index(nm)
            if (b, i) not in trk:
                continue
            a = trk[(b, i)]
            t = np.cumsum(hist[i][4]) / YR
            for r, (k, lab, logy) in enumerate(outs):
                ax = axes[r, j]
                for s, (sch, col, ls) in enumerate((("truth", "k", "-"), ("TC per molecule", "C0", "-"),
                                                    ("bin network", "C1", "--"))):
                    y = a[:, s, oi[k]]
                    if logy:
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(t, y, ls, color=col, lw=1.2, drawstyle="steps-pre", label=sch if (r == 0 and j == 0) else None)
                PL.style(ax, logx=hist[i][0] == "A", logy=logy)
                if r == 0:
                    ax.set_title(nm.replace("<->", r" $\leftrightarrow$ "), fontsize=10)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=11)
                if r == len(outs) - 1:
                    ax.set_xlabel("time [yr]", fontsize=11)
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle(PL.BIN_LABELS[b] + ": tracked classes with one averaged network for the bin (orange) against the "
                     "truth (black) and per-molecule class tracking (blue); molecules summed", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT.parent / f"s2_tracking_bin{b + 1}.png", dpi=170)
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
