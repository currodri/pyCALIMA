"""Per-cell cost of the per-bin RT group cross-sections in a RAMSES-like cool step (numba, one thread).

Per cell: 8 carbon + 8 silicate bins, 27 per-gram values per bin (8 groups x csa, css, csr, and 3 shielding bands;
the count RAMSES copies today). Schemes, per cell-call (once per cool step, frozen over its substeps):

* copy: today's RAMSES, the star-polled group values copied into the cell;
* pseudo-bins (n = 5, adopted 2026-11-18; and 9): the cell's s, its bracket among n Chebyshev states, one linear
  blend of two pseudo-bins (``check_rt_group_optics.py`` scores the accuracy);
* PL2M s tables: the same blend on the A2 s tables (31-77 states per bin), i.e. PL2M at s almost exactly;
* EA exact: the EA pieces of the cell, the s table where a bin is pure base (EA = PL2M there) and the exact node
  integral (``optics.node_table``, tolerance 3e-3) over each front.

Then the IR group on every substep: today's per-bin Planck and Rosseland lookups at T_rad, and the pseudo-bin version
(one shared T_rad index, the two bracketing pseudo-bins blended with the cell-call's weight).

The tables hold random numbers: only the cost is measured. Cell states are MRN-like segments, half spanning the grid
and half ending inside it (internal fronts, which make EA dearer).

    python diagnostics/gsd/bench_rt_group_optics.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_node_tables as NT      # noqa: E402
import check_optics_tables as A2    # noqa: E402
from pycalima.gsd import closures as gc   # noqa: E402
from pycalima.gsd.optics import node_u    # noqa: E402

NCOL = 27
N_CELL = 2000
N_T = 200                                                                                  # T_rad table length
NODES = {"carbon": (97, 65, 33, 17, 17, 17, 25, 65), "silicate": (17, 17, 17, 17, 25, 49, 97, 65)}   # builder 3e-3
S_TAB = {"carbon": (77, 60, 40, 35, 34, 31, 35, 46), "silicate": (32, 32, 32, 33, 35, 41, 47, 46)}   # A2 s tables 3e-3
COPY, PSEUDO, EA = 0, 1, 2
_ea_nb = gc._ea_nb                     # numba kernel of the EA pieces


@njit(cache=False)
def node_piece(u, lnu, K, p, lo, hi, wgt, acc):
    """acc[:] += wgt * (average of K over u^p du on [lo, hi]), K linear in u between nodes. One exp per node."""
    nu = u.size
    if abs(p + 1.0) > 1e-12:
        norm = (hi ** (p + 1.0) - lo ** (p + 1.0)) / (p + 1.0)
    else:
        norm = np.log(hi / lo)
    if not norm > 0.0:
        return
    k0 = 0
    while k0 < nu - 2 and u[k0 + 1] <= lo:
        k0 += 1
    a = lo
    la = np.log(lo)
    ea = np.exp((p + 1.0) * la)
    for k in range(k0, nu - 1):
        if u[k] >= hi:
            break
        b = u[k + 1] if u[k + 1] < hi else hi
        lb = lnu[k + 1] if u[k + 1] < hi else np.log(hi)
        eb = np.exp((p + 1.0) * lb)
        i0 = (eb - ea) / (p + 1.0) if abs(p + 1.0) > 1e-12 else lb - la
        i1 = (eb * b - ea * a) / (p + 2.0) if abs(p + 2.0) > 1e-12 else lb - la
        h = u[k + 1] - u[k]
        w0 = wgt * (u[k + 1] * i0 - i1) / h / norm
        w1 = wgt * (i1 - u[k] * i0) / h / norm
        for c in range(K.shape[1]):
            acc[c] += w0 * K[k, c] + w1 * K[k + 1, c]
        a, la, ea = b, lb, eb


@njit(cache=False)
def bracket(sg, s):
    lo, hi = 0, sg.size - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if sg[m] <= s:
            lo = m
        else:
            hi = m
    return lo, (s - sg[lo]) / (sg[hi] - sg[lo])


@njit(cache=False)
def blend(T, k, t, out):
    for c in range(T.shape[1]):
        out[c] = (1.0 - t) * T[k, c] + t * T[k + 1, c]


@njit(cache=False)
def cells(scheme, Ns, Ms, lams, deltas, udel, mL, uR, da, uN, lnN, KN, offN, sgs, TS, offS, out, kb, tb):
    """One cell-call per cell; tables of all bins concatenated, with offsets. kb, tb: each bin's bracket and weight,
    kept for the substeps (pseudo-bins)."""
    ncell, nb = Ns.shape
    acc = np.zeros(NCOL)
    for ic in range(ncell):
        N = Ns[ic]
        if scheme == COPY:
            for i in range(nb):
                for c in range(NCOL):
                    out[i, c] = KN[offN[i], c]
            continue
        if scheme == PSEUDO:
            for i in range(nb):
                if N[i] > 0.0:
                    s = (Ms[ic, i] / (N[i] * mL[i]) - 1.0) / (uR[i] ** 3 - 1.0)
                    s = min(max(s, 0.0), 1.0)
                    sg = sgs[offS[i]:offS[i + 1]]
                    k, t = bracket(sg, s)
                    kb[i] = k
                    tb[i] = t
                    blend(TS[offS[i]:offS[i + 1]], k, t, out[i])
            continue
        r = Ms[ic] / (N * mL + 1e-300)
        n, lam, u1, u2, sp = _ea_nb(N, r, uR, da, lams[ic], deltas[ic], udel[ic], 1e-6, -3.5, False)
        for i in range(nb):
            if N[i] <= 0.0:
                continue
            s = min(max((Ms[ic, i] / (N[i] * mL[i]) - 1.0) / (uR[i] ** 3 - 1.0), 0.0), 1.0)
            sg = sgs[offS[i]:offS[i + 1]]
            pure = n[i, 1] == 0.0 and n[i, 2] == 0.0 and n[i, 3] == 0.0 and not sp[i, 0]
            if pure:
                k, t = bracket(sg, s)
                blend(TS[offS[i]:offS[i + 1]], k, t, out[i])
                continue
            for c in range(NCOL):
                acc[c] = 0.0
            u = uN[offN[i]:offN[i + 1]]
            lu = lnN[offN[i]:offN[i + 1]]
            K = KN[offN[i]:offN[i + 1]]
            tot = 0.0
            for j in range(4):
                if n[i, j] == 0.0:
                    continue
                if sp[i, j]:                       # a spike: one lerp, negligible
                    tot += n[i, j]
                    continue
                if j == 0:                         # the base piece from the s table, weighted by its share
                    k, t = bracket(sg, s)
                    blend(TS[offS[i]:offS[i + 1]], k, t, out[i])
                    for c in range(NCOL):
                        acc[c] += n[i, 0] * out[i, c]
                else:
                    node_piece(u, lu, K, lam[i, j] + 3.0, u1[i, j], u2[i, j], n[i, j], acc)
                tot += n[i, j]
            for c in range(NCOL):
                out[i, c] = acc[c] / tot
    return out


@njit(cache=False)
def ir_today(logT, lt0, dlt, P, R, Ns, out):
    """Per bin, its own Planck and Rosseland lookups at log T_rad (as Planck_tab / Rosseland_tab%interpolate)."""
    for ic in range(Ns.shape[0]):
        for i in range(Ns.shape[1]):
            x = (logT[ic] - lt0) / dlt
            j = min(max(int(x), 0), P.shape[1] - 2)
            w = x - j
            out[i, 0] = (1.0 - w) * P[i, j] + w * P[i, j + 1]
            x = (logT[ic] - lt0) / dlt
            j = min(max(int(x), 0), R.shape[1] - 2)
            w = x - j
            out[i, 1] = (1.0 - w) * R[i, j] + w * R[i, j + 1]


@njit(cache=False)
def ir_pseudo(logT, lt0, dlt, P, R, Ns, offS, kb, tb, out):
    """One shared T_rad index; per bin the two bracketing pseudo-bins, blended with the cell-call's weight."""
    for ic in range(Ns.shape[0]):
        x = (logT[ic] - lt0) / dlt
        j = min(max(int(x), 0), P.shape[1] - 2)
        w = x - j
        for i in range(Ns.shape[1]):
            k = offS[i] + kb[i]
            t = tb[i]
            p0 = (1.0 - w) * P[k, j] + w * P[k, j + 1]
            p1 = (1.0 - w) * P[k + 1, j] + w * P[k + 1, j + 1]
            r0 = (1.0 - w) * R[k, j] + w * R[k, j + 1]
            r1 = (1.0 - w) * R[k + 1, j] + w * R[k + 1, j + 1]
            out[i, 0] = (1.0 - t) * p0 + t * p1
            out[i, 1] = (1.0 - t) * r0 + t * r1


def chebyshev_s(n):
    return (1.0 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2.0


def states(g, rng):
    out = []
    for k in range(N_CELL):
        q = rng.uniform(-4.5, -2.0)
        lo = g.a_edges[0] * (1.0 if k % 2 == 0 else rng.uniform(1.0, 3.0))
        hi = g.a_edges[-1] * (1.0 if k % 2 == 0 else rng.uniform(0.05, 0.6))
        out.append(NT.segment(g, lo, hi, q))
    return np.array([s[0] for s in out]), np.array([s[1] for s in out])


def timed(f, *args, rep=5):
    f(*args)                                                   # compile
    t0 = time.perf_counter()
    for _ in range(rep):
        f(*args)
    return (time.perf_counter() - t0) / rep / N_CELL * 1e6     # us per cell


def main():
    rng = np.random.default_rng(1)
    runs = (("copy (today)", COPY, None), ("5 pseudo-bins (adopted)", PSEUDO, 5), ("9 pseudo-bins", PSEUDO, 9),
            ("PL2M s tables", PSEUDO, "A2"), ("EA exact (s tables + front node integrals)", EA, "A2"))
    cost = {name: 0.0 for name, _, _ in runs}
    ir = {"IR today": 0.0, "IR 5 pseudo-bins": 0.0}
    for mat, g, _ in A2.materials():
        nb = g.n_bins
        Ns, Ms = states(g, rng)
        lams = np.zeros_like(Ns)
        deltas = np.zeros(Ns.shape, np.bool_)
        udel = np.ones_like(Ns)
        for ic in range(N_CELL):
            prof = gc.BinProfile(g, Ns[ic], Ms[ic])
            lams[ic], deltas[ic], udel[ic] = prof.lam, prof.delta, prof.u_delta
        uN = [node_u(e, n - 1) for e, n in zip(g.etas, NODES[mat])]
        uNc = np.concatenate(uN)
        offN = np.concatenate([[0], np.cumsum([u.size for u in uN])])
        KN = rng.uniform(0.5, 1.5, (uNc.size, NCOL))
        mL, uR, da = g.m_edges[:-1].copy(), g.etas ** (1.0 / 3.0), np.diff(g.a_edges)
        out = np.zeros((nb, NCOL))
        kb, tb = np.zeros(nb, np.int64), np.zeros(nb)
        for name, sch, sp in runs:
            sg = [np.linspace(0.0, 1.0, n) for n in S_TAB[mat]] if sp in (None, "A2") else [chebyshev_s(sp)] * nb
            sgc = np.concatenate(sg)
            offS = np.concatenate([[0], np.cumsum([s.size for s in sg])])
            TS = rng.uniform(0.5, 1.5, (sgc.size, NCOL))
            cost[name] += timed(cells, sch, Ns, Ms, lams, deltas, udel, mL, uR, da, uNc, np.log(uNc), KN, offN,
                                sgc, TS, offS, out, kb, tb)
            if sp == 5:
                logT = rng.uniform(0.5, 3.0, N_CELL)
                P5, R5 = rng.uniform(0.5, 1.5, (2, sgc.size, N_T))
                P1, R1 = rng.uniform(0.5, 1.5, (2, nb, N_T))
                ir["IR today"] += timed(ir_today, logT, 0.0, 0.02, P1, R1, Ns, out)
                ir["IR 5 pseudo-bins"] += timed(ir_pseudo, logT, 0.0, 0.02, P5, R5, Ns, offS, kb, tb, out)
    lines = [f"Per-cell cost, 8 carbon + 8 silicate bins, {NCOL} values per bin, numba, one thread, {N_CELL} cells "
             "(MRN-like segments, half with internal fronts):",
             "  per cell-call (once per cool step):"]
    lines += [f"    {k:44s} {v:6.2f} us" for k, v in cost.items()]
    lines += ["  IR group, per substep (Planck csa and Rosseland csr per bin):"]
    lines += [f"    {k:44s} {v:6.2f} us" for k, v in ir.items()]
    print("\n".join(lines))
    out_path = Path("results/gsd_overview/figs/rt_group_optics_cost.txt")
    out_path.write_text("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
