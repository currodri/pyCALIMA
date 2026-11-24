"""PAHrt study S3: the smallest state set that reproduces each molecule's full joint (Z, N_H) chain.

A molecule's full chain has the charges -1 ... +2 (in neutral gas) times N_H = 0 ... N_H0 + 4 (main's pahchem: the
Andrews super-H chain capped at +4): 4 x (N_H0 + 5) states, ~180 for the largest molecules, too many for a per-cell
solve in RAMSES (~1 us per bin per call, the WDB06rt envelope). pahchem's own rates suggest the reduction: only
N_H0 - 3 ... N_H0 carry their own rates; the deep states 1 ... N_H0 - 4 reuse them by parity (with the H loss scaled
by N_H), and stripping is nearly all-or-nothing, so the deep states mostly pass the flux between N_H0 - 3 and the bare
skeleton.

Per molecule and condition (the S1 sweeps on the Draine shape), this reports:
1. where the population and each output sit: the share of every state class (bare; deep 1 ... N_H0 - 4; N_H0 - 3,
   -2, -1; intact; super-H +1 ... +4) in the population, the heating, cooling, absorbed power, carbon loss, H2, H uptake
   and the electron source;
2. the error of reduced networks against the full chain, on every RAMSES output (``common.OUTPUTS``):
   - "cap+k": super-H cut at N_H0 + k (k = 2, 1, 0), the boundary reflecting;
   - "Z<=+1": the charge +2 dropped (reflecting);
   - "deep": the deep segment collapsed into effective transitions (below), the rest exact;
   - "deep,cap+2", "deep,cap+2,Z<=+1": the combinations (4 x 7 = 28 and 3 x 7 = 21 states).

The deep collapse (what a per-cell solver could do at O(N_H0) cost): in each deep column the charge is taken in its
own equilibrium (the charge steps at that N_H, as ``group_tables``' chain), the H steps (all processes with dN_H != 0,
deposits included) are averaged over it, and the resulting one-dimensional chain in N_H (steps -2, -1, +1) gives, from
each deep state, the probability of reaching the bare skeleton before N_H0 - 3 (``splitting``). A transition of the
kept states into the deep segment is replaced by two, to the bare skeleton and back to N_H0 - 3 (keeping the
process's charge step), weighted by that probability; the bare skeleton's H addition reaches N_H0 - 3 with the
complementary probability. Exact when the charge equilibrates within each deep column faster than H steps.

Figures: ``s3_classes_bin<b>.png`` (the share of each state class in the population, heating, carbon loss and H2 along
each phase's G0 sweep), ``s3_errors.png`` (every reduced network's error, every molecule and condition, per bin and
output), ``s3_deep_bin<b>.png`` (the full chain against "deep, cap +2" along the CNM and PDR sweeps, per molecule).

    python diagnostics/pahchem/pahrt/s3_states.py --workers 4 [--per-bin 5]
    python diagnostics/pahchem/pahrt/s3_states.py --analyse
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import s1_drivers as S1

OUT = Path("results/pahrt_study/s3")
VARIANTS = ("cap+2", "cap+1", "cap+0", "Z<=+1", "deep", "deep,cap+2", "deep,cap+2,Z<=+1")
CLASSES = ("bare", "deep", "-3", "-2", "-1", "intact", "+1", "+2", "+3", "+4")
SHARE_OF = ("population", "heating", "cooling", "absorbed", "carbon", "h2", "h_uptake")


def design():
    """The S1 sweeps on the Draine shape (the field sweep is S1's and S4's)."""
    return [n for n in S1.design() if n[1] != "field"]


def class_of(nh, nh0):
    """Class index (``CLASSES``) of every N_H value."""
    nh = np.asarray(nh)
    out = np.empty(nh.size, int)
    for i, n in enumerate(nh):
        d = n - nh0
        if n == 0:
            out[i] = 0
        elif d <= -4:
            out[i] = 1
        elif d < 0:
            out[i] = 2 + (d + 3)
        elif d == 0:
            out[i] = 5
        else:
            out[i] = 5 + min(d, 4)
    return out


def splitting(net, nh0, entries):
    """{deep N_H: (p_bare, p_top)} for the deep states in ``entries``: the probabilities of reaching N_H = 0, or N_H0 - 3
    (and above), first, on the charge-averaged chain in N_H (module doc). Computed by eliminating the other deep states
    one by one (GTH state reduction: no subtractions, so a probability of 1e-40 keeps its digits; 1 - p would lose it
    and make the bare state a trap). Empty without a deep segment."""
    lo, hi = 1, nh0 - 4
    if hi < lo:
        return {}
    z = np.asarray(net.charges)
    nz = z.size
    deep = list(range(lo, hi + 1))
    m = len(deep)
    R = np.zeros((m + 2, m + 2))                        # 0: bare, 1..m: deep, m + 1: top (both absorbing)
    for i, n in enumerate(deep, start=1):
        up = np.zeros(nz)
        down = np.zeros(nz)
        for p in net.processes:
            if p.dnh == 0 and p.dz == 1:
                up += p.rate[:, n]
            elif p.dnh == 0 and p.dz == -1:
                down += p.rate[:, n]
        lnr = np.log(np.maximum(up[:-1], 1e-300)) - np.log(np.maximum(down[1:], 1e-300))
        lnf = np.concatenate([[0.0], np.cumsum(lnr)])
        f = np.exp(lnf - lnf.max())
        f /= f.sum()
        for p in net.processes:
            if p.dnh == 0:
                continue
            r = float(f @ p.rate[:, n])
            t = n + p.dnh
            j = 0 if t <= 0 else (m + 1 if t >= nh0 - 3 else deep.index(t) + 1)
            R[i, j] += r
    out = {}
    for e in entries:
        if e not in deep:
            continue
        Q = R.copy()
        ke = deep.index(e) + 1
        for k in range(1, m + 1):
            if k == ke:
                continue
            s = Q[k].sum() - Q[k, k]
            if s > 0.0:
                Q += np.outer(Q[:, k], Q[k, :]) / s
            Q[k, :] = 0.0
            Q[:, k] = 0.0
            np.fill_diagonal(Q, 0.0)
        tot = Q[ke, 0] + Q[ke, m + 1]
        out[e] = (Q[ke, 0] / tot, Q[ke, m + 1] / tot) if tot > 0.0 else (0.0, 0.0)
    return out


#: a collapsed transition less likely than this is dropped (a state it alone fed becomes transient)
P_NEGLIGIBLE = 1e-200


def reduce_network(net, nh0, cap=None, zmax=None, deep=False):
    """(reduced network, kept charges' rows, kept N_H values in order): the variant of ``net`` (module doc)."""
    from pycalima.pahchem.network import PAHNetwork

    z = np.asarray(net.charges)
    nh = np.asarray(net.nh_values)
    top = nh.max() if cap is None else min(nh.max(), nh0 + cap)
    zkeep = z if zmax is None else z[z <= zmax]
    use_deep = deep and nh0 - 4 >= 1
    kept = [n for n in nh if n <= top and not (use_deep and 1 <= n <= nh0 - 4)]
    new = {n: i for i, n in enumerate(kept)}
    zi = {c: i for i, c in enumerate(zkeep)}
    entries = set()
    if use_deep:                                       # where transitions of the kept states land in the segment
        for p in net.processes:
            for n in kept:
                t = n + p.dnh
                if 1 <= t <= nh0 - 4 and np.any(p.rate[:, n] > 0):
                    entries.add(t)
    split = splitting(net, nh0, entries) if use_deep else {}
    procs = {}                                          # (name, dz, dnh_new, h2) -> rate array on the reduced grid

    def add(name, dz, dnew, h2, iz, j, r):
        key = (name, dz, dnew, h2)
        if key not in procs:
            procs[key] = np.zeros((len(zkeep), len(kept)))
        procs[key][iz, j] += r

    for p in net.processes:
        for a, c in enumerate(z):
            if c not in zi:
                continue
            c2 = c + p.dz
            if c2 not in zi:
                continue                                # leaves the charge window: dropped (reflecting)
            for n in kept:
                r = float(p.rate[a, n])
                if r <= 0.0:
                    continue
                t = n + p.dnh
                if t > top:
                    continue                            # above the super-H cap: dropped
                if t in new:
                    add(p.name, p.dz, new[t] - new[n], p.h2_released, zi[c], new[n], r)
                elif t in split:                        # into the deep segment
                    pb, pt = split[t]                   # below P_NEGLIGIBLE a channel carries nothing (and its
                    if pb > P_NEGLIGIBLE:               # ~1e-300 rates would underflow the GTH pivots)
                        add(p.name, p.dz, new[0] - new[n], p.h2_released, zi[c], new[n], r * pb)
                    if pt > P_NEGLIGIBLE and (nh0 - 3) in new:
                        add(p.name, p.dz, new[nh0 - 3] - new[n], p.h2_released, zi[c], new[n], r * pt)
    out = PAHNetwork(tuple(int(c) for c in zkeep), tuple(range(len(kept))))
    for (name, dz, dnew, h2), rate in procs.items():
        if np.any(rate > 0):
            out.add(name, rate, dz, dnew, h2)
    return out, [int(np.where(z == c)[0][0]) for c in zkeep], kept


def reduced_outputs(truth, env, net, W, variant):
    nh0 = truth.mol.nh
    kw = {"cap": None, "zmax": None, "deep": False}
    for part in variant.split(","):
        if part.startswith("cap+"):
            kw["cap"] = int(part[4:])
        elif part == "Z<=+1":
            kw["zmax"] = 1
        elif part == "deep":
            kw["deep"] = True
    red, rows, kept = reduce_network(net, nh0, **kw)
    Wr = {k: v[np.ix_(rows, kept)] for k, v in W.items()}
    F = red.steady_state_closed()
    return C.outputs(_Relabelled(red, kept), Wr, nh0, truth.ion_rates(env), env, F)


class _Relabelled:
    """A reduced network seen with its true N_H values (for the hydrogenation outputs)."""

    def __init__(self, net, kept):
        self.charges, self.nh_values, self.processes = net.charges, tuple(kept), net.processes


def run_molecule(mol_name):
    mols = {m.name: m for ms in C.molecules(S1.ARGS_PER_BIN).values() for m in ms}
    mol = mols[mol_name]
    nodes = design()
    nv = len(VARIANTS)
    full = np.full((len(nodes), len(C.OUTPUTS)), np.nan)
    red = np.full((len(nodes), nv, len(C.OUTPUTS)), np.nan)
    shares = np.full((len(nodes), len(SHARE_OF), len(CLASSES)), np.nan)
    sizes = np.zeros(nv + 1, int)
    errors = {}
    t0 = time.time()
    truth = C.Truth(mol)
    for i, (_, _, _, ph) in enumerate(nodes):
        env = ph.environment()
        try:
            net, W = truth.network(env)
            F = net.steady_state_closed()
            o = C.outputs(net, W, mol.nh, truth.ion_rates(env), env, F)
            full[i] = [o[k] for k in C.OUTPUTS]
            sizes[0] = F.size
            cls = class_of(net.nh_values, mol.nh)
            for s, what in enumerate(SHARE_OF):
                x = F if what == "population" else W[what] * F
                per_n = x.sum(axis=0)
                tot = per_n.sum()
                if tot > 0:
                    shares[i, s] = np.bincount(cls, weights=per_n, minlength=len(CLASSES)) / tot
            for v, var in enumerate(VARIANTS):
                try:
                    ov = reduced_outputs(truth, env, net, W, var)
                    red[i, v] = [ov[k] for k in C.OUTPUTS]
                except Exception as exc:
                    errors[f"{i}:{var}"] = f"{type(exc).__name__}: {exc}"[:200]
        except Exception as exc:
            errors[str(i)] = f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc(limit=2)
    path = OUT / f"{S1.safe(mol_name)}.npz"
    np.savez(path, full=full, red=red, shares=shares, nc=mol.nc, nh=mol.nh, name=mol_name,
             seconds=time.time() - t0, errors=json.dumps(errors))
    return str(path), time.time() - t0, len(errors)


def run(workers, per_bin):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    OUT.mkdir(parents=True, exist_ok=True)
    mols = [m for ms in C.molecules(per_bin).values() for m in ms]
    todo = sorted((m for m in mols if not (OUT / f"{S1.safe(m.name)}.npz").exists()), key=lambda m: -m.nc)
    print(f"{len(design())} nodes x {len(VARIANTS)} variants per molecule; {len(todo)} of {len(mols)} molecules",
          flush=True)
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

RELEVANT = 1e-6


def errors_of(full, red):
    """Per output, where it matters: for rates the factor error max(r, 1/r) where the truth is within ``RELEVANT`` of
    its largest value over the molecule's conditions (a rate 1e-300 is no cell's concern); for the electron source the
    absolute error over its largest magnitude (it is a small net of large fluxes); for shares and charges the absolute
    difference."""
    out = {}
    for j, k in enumerate(C.OUTPUTS):
        a, b = full[..., j], red[..., j]
        if k in C.RATES:
            top = np.nanmax(a) if np.any(np.isfinite(a)) else 0.0
            ok = (a > RELEVANT * top) & (a > 0) & (b > 0)
            r = np.where(ok, b / np.where(ok, a, 1.0), np.nan)
            out[k] = np.where(ok, np.maximum(r, 1.0 / r), np.nan)
        elif k == "e_source":
            scale = np.nanmax(np.abs(a)) if np.any(np.isfinite(a)) else 1.0
            out[k] = np.abs(b - a) / max(scale, 1e-300)
        else:
            out[k] = np.abs(b - a)
    return out


def analyse(per_bin):
    mols = C.molecules(per_bin)
    nodes = design()
    lines = ["PAHrt S3: the smallest state set (per molecule, against its full joint chain; the S1 sweeps on the Draine",
             "shape). Rates: worst factor error where the rate is within 1e-6 of its maximum over the molecule's",
             "conditions; e_source: worst error over its largest magnitude; shares and <Z>: worst absolute error.",
             f"Variants: {', '.join(VARIANTS)}.", ""]
    show = ("z_mean", "f_m1", "f_p1", "heating", "cooling", "carbon", "h2", "h_uptake", "e_source", "alpha_Cp", "bare",
            "dnh")
    lines.append("1. Where the population and the outputs sit: largest share of each state class over all conditions")
    lines.append("   (per bin, worst molecule): " + ", ".join(CLASSES))
    allres = {}
    for b, ms in mols.items():
        res = [np.load(OUT / f"{S1.safe(m.name)}.npz") for m in ms if (OUT / f"{S1.safe(m.name)}.npz").exists()]
        if not res:
            continue
        allres[b] = res
        sh = np.nanmax(np.array([r["shares"] for r in res]), axis=(0, 1))      # [what, class]
        for s, what in enumerate(SHARE_OF):
            lines.append(f"   bin {b + 1} {what:10s} " + " ".join(f"{x:6.3f}" for x in sh[s]))
    lines.append("")
    lines.append("2. Reduced networks: worst error over the molecules of a bin and all conditions")
    for b, res in allres.items():
        lines.append(f"   bin {b + 1} ({len(res)} molecules, N_H0 {min(int(r['nh']) for r in res)}-"
                     f"{max(int(r['nh']) for r in res)})")
        for v, var in enumerate(VARIANTS):
            errs = [errors_of(r["full"], r["red"][:, v]) for r in res]
            cells = []
            for k in show:
                e = np.concatenate([x[k].ravel() for x in errs])
                e = e[np.isfinite(e)]
                cells.append(f"{k} {fmt_err(k, e)}")
            lines.append(f"     {var:18s} " + "; ".join(cells))
    lines.append("")
    lines.append("3. Where 'deep,cap+2' errs most (per bin: the output, its error, the phase and input of that node)")
    v = VARIANTS.index("deep,cap+2")
    for b, res in allres.items():
        for k in ("heating", "carbon", "h2", "z_mean", "bare"):
            worst = (0.0, None, None)
            for r in res:
                e = errors_of(r["full"], r["red"][:, v])[k]
                if np.any(np.isfinite(e)):
                    i = int(np.nanargmax(e))
                    if e[i] > worst[0]:
                        worst = (float(e[i]), str(r["name"]), nodes[i][:3])
            lines.append(f"   bin {b + 1} {k:8s} {worst[0]:.3g} {worst[1]} {worst[2]}")
    txt = "\n".join(lines)
    (OUT.parent / "s3_states.txt").write_text(txt + "\n")
    print(txt)
    figures(allres, nodes)


VARIANT_LABELS = {"cap+2": r"cap $+2$", "cap+1": r"cap $+1$", "cap+0": r"cap $+0$", "Z<=+1": r"$Z\le+1$",
                  "deep": "deep", "deep,cap+2": r"deep, cap $+2$", "deep,cap+2,Z<=+1": r"deep, cap $+2$, $Z\le+1$"}
CLASS_LABELS = ("bare", r"deep", r"$-3$", r"$-2$", r"$-1$", "intact", r"$+1$", r"$+2$", r"$+3$", r"$+4$")


def figures(allres, nodes):
    """The S3 figures (module doc): s3_classes_bin<b>.png, s3_errors.png, s3_deep_bin<b>.png."""
    import plots as PL

    plt = PL.setup()
    key = [(n[0], n[1]) for n in nodes]
    ccols = plt.get_cmap("tab10")(np.arange(len(CLASSES)))
    # 1. where the population and the outputs sit, along each phase's G0 sweep (bin means)
    whats = ("population", "heating", "carbon", "h2")
    for b, res in allres.items():
        sh = np.nanmean(np.array([r["shares"] for r in res]), axis=0)          # [node, what, class]
        fig, axes = plt.subplots(len(whats), len(C.PHASES), figsize=(16, 10), sharex="col", sharey=True)
        for j, ph in enumerate(C.PHASES):
            rows = [i for i, kk in enumerate(key) if kk == (ph, "g0")]
            x = np.array([nodes[i][2] for i in rows], float)
            for i, what in enumerate(whats):
                ax = axes[i, j]
                y = np.nan_to_num(sh[rows, SHARE_OF.index(what), :])
                ax.stackplot(x, y.T, colors=ccols, labels=CLASS_LABELS if (i == 0 and j == 0) else (),
                             alpha=0.9)
                PL.style(ax, logx=True)
                ax.set_ylim(0, 1)
                if i == 0:
                    ax.set_title(ph, fontsize=12)
                if j == 0:
                    ax.set_ylabel(f"share of {what}", fontsize=11)
                if i == len(whats) - 1:
                    ax.set_xlabel(r"$G_0$", fontsize=12)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=len(CLASSES), frameon=False,
                   fontsize=9, title="state class (N_H relative to the intact molecule)", title_fontsize=9)
        fig.suptitle(PL.BIN_LABELS[b] + ": where the population and the outputs sit (G0 sweeps; each molecule's shares,"
                     " averaged over the bin's molecules)", fontsize=12)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(OUT.parent / f"s3_classes_bin{b + 1}.png", dpi=200)
        plt.close(fig)
    # 2. the error of each reduced network, every molecule and condition (dots) and the worst (diamond)
    outs = (("z_mean", r"$|\Delta\langle Z\rangle|$"), ("f_p1", r"$|\Delta f(+1)|$"),
            ("heating", r"$\Gamma_{\rm PE}$ factor $-1$"), ("cooling", r"$\Lambda_{\rm rec}$ factor $-1$"),
            ("carbon", r"C loss factor $-1$"), ("h2", r"H$_2$ factor $-1$"), ("e_source", r"$e^-$ source rel.\ error"),
            ("bare", r"$|\Delta$ bare$|$"))
    fig, axes = plt.subplots(len(allres), len(outs), figsize=(19, 3.0 * len(allres) + 1), sharey=True)
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(1)
    for bi, (b, res) in enumerate(allres.items()):
        for j, (k, xl) in enumerate(outs):
            ax = axes[bi, j]
            for v in range(len(VARIANTS)):
                e = np.concatenate([errors_of(r["full"], r["red"][:, v])[k].ravel() for r in res])
                e = e[np.isfinite(e)]
                if k in C.RATES:
                    e = e - 1.0
                e = np.maximum(e, 1e-8)
                ax.plot(e, v + 0.25 * (rng.random(e.size) - 0.5), ".", ms=1.5, alpha=0.35, color="0.35")
                if e.size:
                    ax.plot(e.max(), v, "D", ms=5, color="C3")
            PL.style(ax, logx=True)
            ax.set_xlim(1e-8, 1e2)
            ax.axvline(0.1, color="C0", lw=0.8, ls=":")
            ax.set_yticks(range(len(VARIANTS)))
            ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=8)
            if bi == len(allres) - 1:
                ax.set_xlabel(xl, fontsize=10)
            if j == 0:
                ax.set_ylabel(PL.BIN_LABELS[b], fontsize=10)
    fig.suptitle("Reduced networks against the full chain (grey: every molecule and condition; red: the worst; "
                 "dotted: 0.1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT.parent / "s3_errors.png", dpi=200)
    plt.close(fig)
    # 3. full (solid) against 'deep, cap +2' (dashed) along the CNM and PDR sweeps, per molecule
    v = VARIANTS.index("deep,cap+2")
    outs = (("carbon", r"C loss [s$^{-1}$]", True), ("bare", "bare share", True),
            ("dnh", r"$\langle N_{\rm H}\rangle - N_{\rm H,0}$", False), ("heating", r"$\Gamma_{\rm PE}$", True),
            ("h2", r"H$_2$ [s$^{-1}$]", True), ("z_mean", r"$\langle Z\rangle$", False))
    sweeps = (("CNM", "g0"), ("CNM", "n_hi"), ("PDR", "g0"), ("PDR", "n_hi"))
    oi = {k: i for i, k in enumerate(C.OUTPUTS)}
    for b, res in allres.items():
        fig, axes = plt.subplots(len(outs), len(sweeps), figsize=(14, 15), sharex="col")
        mcols = plt.get_cmap("plasma")(np.linspace(0.05, 0.85, len(res)))
        for j, (ph, inp) in enumerate(sweeps):
            rows = [i for i, kk in enumerate(key) if kk == (ph, inp)]
            x = np.array([nodes[i][2] for i in rows], float)
            for i, (k, yl, logy) in enumerate(outs):
                ax = axes[i, j]
                for r, c in zip(res, mcols, strict=True):
                    yf, yr = r["full"][rows, oi[k]], r["red"][rows, v, oi[k]]
                    if logy:
                        yf, yr = np.where(yf > 0, yf, np.nan), np.where(yr > 0, yr, np.nan)
                    ax.plot(x, yf, "-", color=c, lw=1.4, label=str(r["name"]).split()[0] if (i == 0 and j == 0) else None)
                    ax.plot(x, yr, "--", color=c, lw=1.2)
                PL.style(ax, logx=True, logy=logy)
                if i == 0:
                    ax.set_title(f"{ph}, {'G0' if inp == 'g0' else 'n(H I)'} sweep", fontsize=11)
                if j == 0:
                    ax.set_ylabel(yl, fontsize=11)
                if i == len(outs) - 1:
                    ax.set_xlabel(r"$G_0$" if inp == "g0" else r"$n({\rm H\,I})$ [cm$^{-3}$]", fontsize=12)
        axes[0, 0].legend(frameon=False, fontsize=8, title="solid: full; dashed: deep, cap +2", title_fontsize=8)
        fig.suptitle(PL.BIN_LABELS[b] + ": the full chain against the reduced one (deep collapse, super-H cap +2)",
                     fontsize=12, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(OUT.parent / f"s3_deep_bin{b + 1}.png", dpi=200)
        plt.close(fig)


def fmt_err(k, e):
    if e.size == 0:
        return "-"
    if k in C.RATES:
        return f"x{np.max(e):.3g}"
    if k == "e_source":
        return f"{np.max(e):.2g}"
    return f"{np.max(e):.2g}"


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
