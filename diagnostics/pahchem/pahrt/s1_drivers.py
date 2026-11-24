"""PAHrt study S1: what drives the PAH equilibrium state (the inputs a per-cell solver needs).

Around each base phase of ``common.PHASES`` (illustrative inputs, labelled), one input at a time is swept and every
RAMSES output (``common.OUTPUTS``) of each molecule's full joint chain is computed (``common.Truth``):
- G0 x 10^(-2 ... +2) in 0.5 dex (at most 1e6, pahchem's table limit), at the base field shape;
- the field shape at the base G0: Draine (1978) far-UV, Draine with the solar-circle old-star light, Kurucz 10, 20,
  30, 40 kK (the hottest template; all cut at 13.6 eV);
- n_e x 10^(-2 ... +2) and n(H I) x 10^(-2 ... +2) in 0.5 dex, each alone (n_e apart from the ions: the electrons'
  own effect);
- T x (0.3, 0.5, 1, 2, 3);
- the ions x (0, 0.3, 1, 3, 10).

Molecules: ``common.molecules(per_bin)``. Each molecule runs in one worker, checkpointed to
``results/pahrt_study/s1/<molecule>.npz`` (a rerun skips the done ones); a node that raises is recorded as NaN with
its error, so one failure costs one node. ``--analyse`` writes the report (``s1_drivers.txt``) and the figures:
per bin (molecules equally weighted; the in-bin shape is S5's question), the range of every output over each sweep,
its logarithmic slope at the base, and two scaling checks: the charge against gamma = G0 sqrt(T) / n_e (the G0 and n_e
sweeps should fall on one curve if gamma is the charge's variable) and the hydrogenation against G0 / n(H I).
Figures: ``s1_<phase>.png`` (every sweep of the phase, per bin), ``s1_scaling.png`` (the scaling collapses),
``s1_fields.png`` (the field shapes against the Draine far-UV at the base G0).

    python diagnostics/pahchem/pahrt/s1_drivers.py --workers 4 [--per-bin 5]
    python diagnostics/pahchem/pahrt/s1_drivers.py --analyse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path("results/pahrt_study/s1")
LOG_STEPS = np.arange(-2.0, 2.0 + 1e-9, 0.5)
FIELDS = ("draine", "draine+old", "kurucz10000", "kurucz20000", "kurucz30000", "kurucz40000")
T_FACTORS = (0.3, 0.5, 1.0, 2.0, 3.0)
ION_FACTORS = (0.0, 0.3, 1.0, 3.0, 10.0)
G0_MAX = 1e6


def design():
    """[(phase name, input, value, Phase)]: the one-at-a-time sweeps around every phase."""
    nodes = []
    for name, ph in C.PHASES.items():
        for s in LOG_STEPS:
            g = ph.g0 * 10.0 ** s
            if g <= G0_MAX:
                nodes.append((name, "g0", float(g), replace(ph, g0=g)))
        for f in FIELDS:
            nodes.append((name, "field", f, replace(ph, field=f)))
        for s in LOG_STEPS:
            nodes.append((name, "n_e", float(ph.n_e * 10.0 ** s), replace(ph, n_e=ph.n_e * 10.0 ** s)))
        for s in LOG_STEPS:
            nodes.append((name, "n_hi", float(ph.n_hi * 10.0 ** s), replace(ph, n_hi=ph.n_hi * 10.0 ** s)))
        for x in T_FACTORS:
            nodes.append((name, "T", float(ph.T * x), replace(ph, T=ph.T * x)))
        for x in ION_FACTORS:
            nodes.append((name, "ions", float(x), replace(ph, ions=C.scaled_ions(ph, x))))
    return nodes


def run_molecule(mol_name):
    """Every node of ``design()`` for one molecule; saves and returns the path."""
    mols = {m.name: m for ms in C.molecules(ARGS_PER_BIN).values() for m in ms}
    mol = mols[mol_name]
    nodes = design()
    res = np.full((len(nodes), len(C.OUTPUTS)), np.nan)
    errors = {}
    t0 = time.time()
    truth = C.Truth(mol)
    for i, (_, _, _, ph) in enumerate(nodes):
        try:
            o = truth.outputs(ph.environment())
            res[i] = [o[k] for k in C.OUTPUTS]
        except Exception as exc:                       # one node, not the run (hot_k8's lesson)
            errors[i] = f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc(limit=2)
    path = OUT / f"{safe(mol_name)}.npz"
    np.savez(path, res=res, nc=mol.nc, nh=mol.nh, name=mol_name, seconds=time.time() - t0,
             errors=json.dumps(errors))
    return str(path), time.time() - t0, len(errors)


def safe(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


ARGS_PER_BIN = 5


def _init(per_bin):
    global ARGS_PER_BIN
    ARGS_PER_BIN = per_bin


def run(workers, per_bin):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    OUT.mkdir(parents=True, exist_ok=True)
    mols = [m for ms in C.molecules(per_bin).values() for m in ms]
    todo = sorted((m for m in mols if not (OUT / f"{safe(m.name)}.npz").exists()), key=lambda m: -m.nc)
    print(f"{len(design())} nodes per molecule; {len(todo)} of {len(mols)} molecules to run on {workers} workers",
          flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(per_bin,)) as ex:
        futs = {ex.submit(run_molecule, m.name): m.name for m in todo}
        for f in as_completed(futs):
            try:
                path, sec, nerr = f.result()
                print(f"{futs[f]}: {sec:.0f} s, {nerr} failed nodes -> {path}", flush=True)
            except Exception as exc:
                print(f"{futs[f]}: FAILED {type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# analysis

INPUT_LABEL = {"g0": r"$G_0$", "n_e": r"$n_e$", "n_hi": r"$n({\rm H\,I})$", "T": r"$T$", "ions": r"ions $\times$",
               "field": "field"}


def load(per_bin):
    mols = C.molecules(per_bin)
    data = {}
    for b, ms in mols.items():
        arr = []
        for m in ms:
            p = OUT / f"{safe(m.name)}.npz"
            if p.exists():
                arr.append(np.load(p)["res"])
        data[b] = np.array(arr) if arr else None
    return data


def bin_mean(arr):
    """Molecules equally weighted: rates and shares are means over the bin's molecules."""
    return np.nanmean(arr, axis=0)


def analyse(per_bin):
    nodes = design()
    data = load(per_bin)
    idx = {k: i for i, k in enumerate(C.OUTPUTS)}
    key = [(n[0], n[1]) for n in nodes]
    lines = ["PAHrt S1: what drives the PAH equilibrium state (one input at a time around illustrative phases).",
             "Per bin, molecules equally weighted. Rates: range = max/min over the sweep, slope = d ln O / d ln X at",
             "the base; shares and <Z>: range = max - min, slope = d O / d ln X. Inputs swept: G0, n_e, n(H I) over",
             "+-2 dex; T x0.3-3; ions x0-10 (slope from x0.3-3); field shapes at the base G0 (range over the six).", ""]
    show = ("z_mean", "f_m1", "f_p1", "heating", "cooling", "carbon", "h2", "e_source", "alpha_Cp", "bare", "dnh")
    for b, arr in data.items():
        if arr is None:
            continue
        m = bin_mean(arr)
        lines.append(f"=== bin {b + 1} ({arr.shape[0]} molecules)")
        for ph in C.PHASES:
            base_rows = [i for i, kk in enumerate(key) if kk == (ph, "ions")]
            base = m[base_rows[ION_FACTORS.index(1.0)]]
            lines.append(f"  {ph}: base " + ", ".join(f"{k} {fmt(base[idx[k]])}" for k in show))
            for inp in ("g0", "n_e", "n_hi", "T", "ions", "field"):
                rows = [i for i, kk in enumerate(key) if kk == (ph, inp)]
                xs = np.array([nodes[i][2] for i in rows], dtype=object)
                cells = []
                for k in show:
                    y = m[rows, idx[k]]
                    cells.append(f"{k} {sweep_metric(k, xs, y, inp)}")
                lines.append(f"    {inp:5s}: " + "; ".join(cells))
        lines.append("")
    lines += scaling_checks(nodes, data, idx)
    lines = lines[:5] + summary(nodes, data, idx) + [""] + lines[5:]
    txt = "\n".join(lines)
    (OUT.parent / "s1_drivers.txt").write_text(txt + "\n")
    print(txt)
    figures(nodes, data, idx)
    fig_scaling(nodes, data, idx)
    fig_fields(nodes, data, idx)


def summary(nodes, data, idx):
    """Per bin and output, how far each input moves it at most over the five phases: rates in dex (log10 max/min over
    the sweep), shares and <Z> in absolute terms; 'field' at fixed G0 over the six shapes."""
    key = [(n[0], n[1]) for n in nodes]
    inputs = ("g0", "field", "n_e", "n_hi", "T", "ions")
    outs = ("z_mean", "f_m1", "heating", "cooling", "carbon", "h2", "alpha_Cp", "bare", "dnh")
    out = ["Summary: the largest change of each output over each input's sweep, over all phases (rates: dex;",
           "<Z>, f(-1), bare: absolute; dnh: H atoms). Columns: " + ", ".join(inputs) + "."]
    for b, arr in data.items():
        if arr is None:
            continue
        m = bin_mean(arr)
        out.append(f"  bin {b + 1}")
        for k in outs:
            cells = []
            for inp in inputs:
                best = 0.0
                for ph in C.PHASES:
                    rows = [i for i, kk in enumerate(key) if kk == (ph, inp)]
                    y = m[rows, idx[k]]
                    y = y[np.isfinite(y)]
                    if y.size < 2:
                        continue
                    if k in C.RATES:
                        y = y[y > 0]
                        r = np.log10(y.max() / y.min()) if y.size > 1 else 0.0
                    else:
                        r = y.max() - y.min()
                    best = max(best, r)
                cells.append(f"{best:7.2f}")
            out.append(f"    {k:9s}" + "".join(cells))
    return out


def fmt(v):
    return f"{v:.3g}" if np.isfinite(v) else "nan"


def sweep_metric(k, xs, y, inp):
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return "-"
    y = y[ok]
    if inp == "field":
        r = (np.nanmax(y) / np.nanmin(y) if k in C.RATES and np.nanmin(y) > 0 else np.nanmax(y) - np.nanmin(y))
        return f"{'x' if k in C.RATES else '+-'}{r:.2g}"
    x = np.asarray(xs[ok], float)
    if inp == "ions":
        sel = (x > 0)
        x, y = x[sel], y[sel]
    lx = np.log(x)
    i0 = int(np.argmin(np.abs(lx - np.log(1.0 if inp == "ions" else np.exp(np.median(lx))))))
    if k in C.RATES:
        if np.nanmin(y) <= 0:
            return "0"
        ly = np.log(y)
        slope = np.gradient(ly, lx)[i0]
        return f"x{np.nanmax(y) / np.nanmin(y):.2g} ({slope:+.2f})"
    slope = np.gradient(y, lx)[i0]
    return f"+-{np.nanmax(y) - np.nanmin(y):.2g} ({slope:+.2f})"


def scaling_checks(nodes, data, idx):
    """The charge against gamma = G0 sqrt(T)/n_e and the hydrogenation against G0/n(H I): the G0 sweep and the n_e
    (n(H I)) sweep of each phase compared at equal gamma (G0/n(H I)), interpolated in log; reported: the largest
    difference of <Z> (dnh) between the two sweeps where they overlap."""
    out = ["Scaling checks: largest difference between the G0 sweep and the n_e sweep at equal gamma (<Z>, f(-1)),",
           "and between the G0 sweep and the n(H I) sweep at equal G0/n(H I) (dnh, bare); '-' = no overlap."]
    key = [(n[0], n[1]) for n in nodes]
    for b, arr in data.items():
        if arr is None:
            continue
        m = bin_mean(arr)
        cells = []
        for ph_name, ph in C.PHASES.items():
            rg = [i for i, kk in enumerate(key) if kk == (ph_name, "g0")]
            re = [i for i, kk in enumerate(key) if kk == (ph_name, "n_e")]
            rh = [i for i, kk in enumerate(key) if kk == (ph_name, "n_hi")]
            g_gamma = np.log10([nodes[i][2] * np.sqrt(ph.T) / ph.n_e for i in rg])
            e_gamma = np.log10([ph.g0 * np.sqrt(ph.T) / nodes[i][2] for i in re])
            g_ratio = np.log10([nodes[i][2] / ph.n_hi for i in rg])
            h_ratio = np.log10([ph.g0 / nodes[i][2] for i in rh])
            dz = overlap_diff(g_gamma, m[rg, idx["z_mean"]], e_gamma, m[re, idx["z_mean"]])
            dm = overlap_diff(g_gamma, m[rg, idx["f_m1"]], e_gamma, m[re, idx["f_m1"]])
            dh = overlap_diff(g_ratio, m[rg, idx["dnh"]], h_ratio, m[rh, idx["dnh"]])
            db = overlap_diff(g_ratio, m[rg, idx["bare"]], h_ratio, m[rh, idx["bare"]])
            cells.append(f"{ph_name}: <Z> {dz}, f(-1) {dm}, dnh {dh}, bare {db}")
        out.append(f"  bin {b + 1}: " + " | ".join(cells))
    return out


def overlap_diff(x1, y1, x2, y2):
    o1, o2 = np.argsort(x1), np.argsort(x2)
    x1, y1, x2, y2 = np.asarray(x1)[o1], np.asarray(y1)[o1], np.asarray(x2)[o2], np.asarray(y2)[o2]
    lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    if hi <= lo:
        return "-"
    x = np.linspace(lo, hi, 25)
    d = np.interp(x, x1, y1) - np.interp(x, x2, y2)
    return f"{np.nanmax(np.abs(d)):.2g}"


def figures(nodes, data, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    key = [(n[0], n[1]) for n in nodes]
    rows = (("z_mean", r"$\langle Z\rangle$", False), ("heating", r"$\Gamma_{\rm PE}$ [erg s$^{-1}$]", True),
            ("carbon", r"C loss [s$^{-1}$]", True), ("h2", r"H$_2$ [s$^{-1}$]", True),
            ("dnh", r"$\langle N_{\rm H}\rangle - N_{\rm H,0}$", False))
    inputs = ("g0", "n_e", "n_hi", "T", "ions")
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, 4))
    for ph in C.PHASES:
        fig, axes = plt.subplots(len(rows), len(inputs), figsize=(13, 11), sharex="col")
        for j, inp in enumerate(inputs):
            r = [i for i, kk in enumerate(key) if kk == (ph, inp)]
            x = np.array([nodes[i][2] for i in r], float)
            for i, (k, lab, logy) in enumerate(rows):
                ax = axes[i, j]
                for b, arr in data.items():
                    if arr is None:
                        continue
                    y = bin_mean(arr)[r, idx[k]]
                    if logy:
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(np.where(x > 0, x, np.nan), y, "-o", ms=2.5, lw=1.2, color=colors[b],
                            label=f"bin {b + 1}")
                ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                if j == 0:
                    ax.set_ylabel(lab, fontsize=11)
                if i == len(rows) - 1:
                    ax.set_xlabel(INPUT_LABEL[inp], fontsize=12)
                ax.minorticks_on()
                ax.tick_params(which="both", direction="in", top=True, right=True, bottom=True, left=True)
                ax.tick_params(which="major", length=4)
                ax.tick_params(which="minor", length=2)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{ph}: one input at a time (bins' molecules equally weighted)", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT.parent / f"s1_{ph}.png", dpi=200)
        plt.close(fig)


def fig_scaling(nodes, data, idx):
    """s1_scaling.png: the scaling variables. Per bin (rows): <Z> against gamma = G0 sqrt(T)/n_e and f(-1) against
    n_e/G0 (from the G0 sweep, solid, and the n_e sweep, dashed), and the H offset <N_H> - N_H0 and the bare share
    against G0/n(H I) (from the G0 sweep, solid, and the n(H I) sweep, dashed); one colour per phase. Curves of one
    phase that fall on each other mean the variable carries that input's whole effect."""
    import plots as PL

    plt = PL.setup()
    key = [(n[0], n[1]) for n in nodes]
    cols = PL.phase_colors(plt, list(C.PHASES))
    panels = (("z_mean", "gamma", r"$\log_{10}\,G_0\sqrt{T}/n_e$", r"$\langle Z\rangle$", False),
              ("f_m1", "ne_g0", r"$\log_{10}\,n_e/G_0$", r"$f(-1)$", False),
              ("dnh", "g0_nh", r"$\log_{10}\,G_0/n({\rm H\,I})$", r"$\langle N_{\rm H}\rangle - N_{\rm H,0}$", False),
              ("bare", "g0_nh", r"$\log_{10}\,G_0/n({\rm H\,I})$", "bare share", True))
    fig, axes = plt.subplots(4, 4, figsize=(15, 13))
    for b in range(4):
        arr = data.get(b)
        for j, (k, var, xl, yl, logy) in enumerate(panels):
            ax = axes[b, j]
            if arr is None:
                continue
            m = bin_mean(arr)
            for ph_name, ph in C.PHASES.items():
                other = "n_e" if var in ("gamma", "ne_g0") else "n_hi"
                for inp, ls in (("g0", "-"), (other, "--")):
                    r = [i for i, kk in enumerate(key) if kk == (ph_name, inp)]
                    v = np.array([nodes[i][2] for i in r], float)
                    g0 = v if inp == "g0" else np.full(v.size, ph.g0)
                    ne = v if inp == "n_e" else np.full(v.size, ph.n_e)
                    nh = v if inp == "n_hi" else np.full(v.size, ph.n_hi)
                    x = {"gamma": g0 * np.sqrt(ph.T) / ne, "ne_g0": ne / g0, "g0_nh": g0 / nh}[var]
                    y = m[r, idx[k]]
                    if logy:
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(np.log10(x), y, ls, color=cols[ph_name], lw=1.3,
                            label=f"{ph_name}" if (inp == "g0" and b == 0 and j == 0) else None)
            PL.style(ax, logy=logy)
            if b == 3:
                ax.set_xlabel(xl, fontsize=12)
            if j == 0:
                ax.set_ylabel(PL.BIN_LABELS[b] + "\n" + yl, fontsize=11)
            else:
                ax.set_ylabel(yl, fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=8, title="solid: G0 sweep; dashed: n_e or n(H I) sweep",
                      title_fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT.parent / "s1_scaling.png", dpi=200)
    plt.close(fig)


def fig_fields(nodes, data, idx):
    """s1_fields.png: the field shape at the base G0. Per bin (rows), per output (columns), each phase's value for the
    six shapes, relative to the Draine far-UV shape (ratio for rates, difference for <Z> and shares)."""
    import plots as PL

    plt = PL.setup()
    key = [(n[0], n[1]) for n in nodes]
    cols = PL.phase_colors(plt, list(C.PHASES))
    outs = (("z_mean", r"$\Delta\langle Z\rangle$"), ("heating", r"$\Gamma_{\rm PE}$ / Draine"),
            ("absorbed", r"absorbed / Draine"), ("carbon", r"C loss / Draine"), ("h2", r"H$_2$ / Draine"),
            ("dnh", r"$\Delta(\langle N_{\rm H}\rangle - N_{\rm H,0})$"))
    labels = ["Draine", "+old stars", "10 kK", "20 kK", "30 kK", "40 kK"]
    fig, axes = plt.subplots(4, len(outs), figsize=(17, 12), sharex=True)
    xs = np.arange(len(FIELDS))
    for b in range(4):
        arr = data.get(b)
        for j, (k, yl) in enumerate(outs):
            ax = axes[b, j]
            if arr is None:
                continue
            m = bin_mean(arr)
            for ph_name in C.PHASES:
                r = [i for i, kk in enumerate(key) if kk == (ph_name, "field")]
                y = m[r, idx[k]]
                y = y / y[0] if k in C.RATES else y - y[0]
                ax.plot(xs, y, "-o", ms=3, lw=1.1, color=cols[ph_name], label=ph_name if (b == 0 and j == 0) else None)
            PL.style(ax, logy=k in C.RATES)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            if j == 0:
                ax.set_ylabel(PL.BIN_LABELS[b] + "\n" + yl, fontsize=10)
            else:
                ax.set_ylabel(yl, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT.parent / "s1_fields.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-bin", type=int, default=5)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args(argv)
    _init(a.per_bin)
    if a.analyse:
        analyse(a.per_bin)
    else:
        run(a.workers, a.per_bin)


if __name__ == "__main__":
    main()
