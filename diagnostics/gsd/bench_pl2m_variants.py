"""Benchmark of PL2M variants (``pl2m_variants``) against PL2M on the existing GSD tests and on stress tests.

Every variant keeps two scalars per bin (N, M). Subcommands write ``<name>.pkl`` (results) and ``<name>.txt``
(tables) into ``--out`` (default results/gsd_overview/variants); ``figures`` reads them.

  growth      McKinnon+18 growth convergence (vs bins, vs steps), sinusoid, edge shapes, overflow (Fig. 11),
              EA parameter scans
  collisions  McKinnon+18 shattering (8/16 bins) and coagulation (8/32), threshold sweeps, constant and additive
              (Golovin) kernels against their analytic solutions
  stress      narrow population grown x10/x100, 10 growth-shrink periods, equilibrium-radius pile-up, edge sweep,
              mixing of cells, RAMSES-like steps on the silicate and carbon grids
  a0          the A0 cases of check_bin_convergence at 4/6/8/12 bins (processes in parallel), against the stored
              references in results/binconv_v4
  rates       grain rates (PE heating, recombination cooling, surface) on the A0 final states
  drift       one drifting bin (check_inbin_drift) with per-moment velocities from each local closure
  cost        time per call of closure, remap, nodes and collision terms at 8 bins
  figures     the figures (PDF)

Run from the worktree root with PYTHONPATH=src:diagnostics/gsd.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_mckinnon18_tests as MK      # noqa: E402
import pl2m_variants as V                # noqa: E402

UM, GYR, MYR = MK.UM, MK.GYR, MK.MYR
OUT = Path("results/gsd_overview/variants")
A0_REF = Path("results/binconv_v4/gsd_bin_convergence")

REMAP_VARIANTS = ["pl2m", "tc:16", "tc:8", "tc:4", "ea:1e-2", "ea:1e-6", "ea:1e-2:8", "aff", "exp"]
LABEL = {"pl2m": "PL2M", "tc:16": "TC, $L=16$", "tc:8": "TC, $L=8$", "tc:4": "TC, $L=4$",
         "ea:1e-2": r"EA, $\epsilon=10^{-2}$", "ea:1e-6": r"EA, $\epsilon=10^{-6}$",
         "ea:1e-2:8": r"EA on TC8", "aff": "affine+switch", "exp": "max-entropy (exp)"}


def save(out, name, obj, lines):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh)
    (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------------
# shared metrics
# ---------------------------------------------------------------------------------------------------------------------

def binned(grid, a, dN):
    """(N, M) per bin of samples (a, dN); outside the grid dropped."""
    k = grid.bin_of_radius(a)
    ok = (k >= 0) & (k < grid.n_bins)
    return (np.bincount(k[ok], dN[ok], grid.n_bins),
            np.bincount(k[ok], dN[ok] * grid.mass(a[ok]), grid.n_bins))


def l1(x, ref):
    return float(np.abs(x - ref).sum() / max(np.abs(ref).sum(), 1e-300))


def cdf_closure(grid, N, M, closure, x):
    """Cumulative number and mass of the reconstruction below the radii x."""
    P = closure.pieces(grid, N, M)
    cN = np.concatenate([[0.0], np.cumsum(np.asarray(N, float))])
    cM = np.concatenate([[0.0], np.cumsum(np.asarray(M, float))])
    k = np.clip(grid.bin_of_radius(x), -1, grid.n_bins)
    FN, FM = np.zeros(x.size), np.zeros(x.size)
    for j, (kk, xx) in enumerate(zip(k, x)):
        if kk < 0:
            continue
        if kk >= grid.n_bins:
            FN[j], FM[j] = cN[-1], cM[-1]
            continue
        c, m = V._cum(P.n, P.A, P.lam, P.u1, P.u2, P.spike, kk, xx / P.aL[kk])
        FN[j], FM[j] = cN[kk] + c, cM[kk] + m * P.mL[kk]
    return FN, FM


def w1_lna(grid, N, M, closure, a_s, dN_s, n_pts=3000):
    """Wasserstein-1 distance in ln a between the reconstruction and samples, for the number and the mass
    distribution (normalised): the mean displacement in ln a."""
    lo, hi = grid.a_edges[0], grid.a_edges[-1]
    x = np.geomspace(lo, hi, n_pts)
    FN, FM = cdf_closure(grid, N, M, closure, x)
    o = np.argsort(a_s)
    a_s, dN_s = a_s[o], dN_s[o]
    inside = (a_s >= lo) & (a_s < hi)
    a_s, dN_s = a_s[inside], dN_s[inside]
    cn = np.concatenate([[0.0], np.cumsum(dN_s)])
    cm = np.concatenate([[0.0], np.cumsum(dN_s * grid.mass(a_s))])
    idx = np.searchsorted(a_s, x, side="right")
    EN, EM = cn[idx], cm[idx]
    dl = np.diff(np.log(x))
    tN, tM = max(cn[-1], 1e-300), max(cm[-1], 1e-300)
    wN = np.sum(0.5 * (np.abs(FN / max(FN[-1], 1e-300) - EN / tN)[1:]
                       + np.abs(FN / max(FN[-1], 1e-300) - EN / tN)[:-1]) * dl)
    wM = np.sum(0.5 * (np.abs(FM / max(FM[-1], 1e-300) - EM / tM)[1:]
                       + np.abs(FM / max(FM[-1], 1e-300) - EM / tM)[:-1]) * dl)
    return float(wN), float(wM)


def mrn_samples(lo, hi, p=-3.5, n=200001):
    """(a, dN) of dn/da ~ a^p on [lo, hi] (trapezoid in ln a)."""
    a = np.geomspace(lo, hi, n)
    w = np.full(n, np.log(hi / lo) / (n - 1))
    w[[0, -1]] *= 0.5
    return a, a ** p * a * w


def boxcar_samples(lo, hi, n=200001):
    return mrn_samples(lo, hi, p=0.0, n=n)


# ---------------------------------------------------------------------------------------------------------------------
# growth
# ---------------------------------------------------------------------------------------------------------------------

def _grow(g, N, M, cl, da, steps):
    for _ in range(steps):
        N, M, *_ = V.remap(g, N, M, lambda x, d=da: x + d, cl)
    return N, M


def cmd_growth(args):
    out = Path(args.out)
    res, lines = {}, []
    a_cut = MK.A_MIN * (MK.A_MAX / MK.A_MIN) ** 0.25
    da_tot = 0.005 * UM / GYR * 5.0 * GYR
    exact = np.pi * MK.RHO / 3.0 * ((a_cut + da_tot) ** 4 - (MK.A_MIN + da_tot) ** 4)

    # --- convergence vs bins (100 steps) ---
    ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    conv = {"N": ns}
    for name in ("PC", "PL"):
        e = []
        for n in ns:
            a, _ = MK.grid_of(n)
            N0, M0 = MK.boxcar_moments(a, MK.A_MIN, a_cut)
            b = MK.LinearBins(a, N0, M0, constant=(name == "PC"))
            for _ in range(100):
                b.shift(da_tot / 100)
            e.append(abs(b.mass_total() / exact - 1.0))
        conv[name] = e
    for spec in REMAP_VARIANTS:
        cl = V.make_closure(spec)
        e = []
        for n in ns:
            a, g = MK.grid_of(n)
            N, M = _grow(g, *MK.boxcar_moments(a, MK.A_MIN, a_cut), cl, da_tot / 100, 100)
            e.append(abs(M.sum() / exact - 1.0))
        conv[spec] = e
    res["conv_bins"] = conv
    lines.append("Growth convergence (McKinnon+18 Fig. 10; boxcar 1-5.6 nm, 0.005 um/Gyr, 5 Gyr, 100 steps): "
                 "|M/M_exact - 1| at 5 Gyr")
    lines.append(f"{'variant':12s} " + " ".join(f"{n:>9d}" for n in ns))
    for k in ["PC", "PL"] + REMAP_VARIANTS:
        lines.append(f"{k:12s} " + " ".join(f"{x:9.2e}" for x in conv[k]))

    # --- convergence vs steps ---
    steps_list = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    cs = {"steps": steps_list}
    for n in (8, 16, 32):
        a, g = MK.grid_of(n)
        N0, M0 = MK.boxcar_moments(a, MK.A_MIN, a_cut)
        for spec in REMAP_VARIANTS:
            cl = V.make_closure(spec)
            cs[(spec, n)] = [abs(_grow(g, N0, M0, cl, da_tot / k, k)[1].sum() / exact - 1.0) for k in steps_list]
        for name in ("PC", "PL"):
            e = []
            for k in steps_list:
                b = MK.LinearBins(a, N0, M0, constant=(name == "PC"))
                for _ in range(k):
                    b.shift(da_tot / k)
                e.append(abs(b.mass_total() / exact - 1.0))
            cs[(name, n)] = e
    res["conv_steps"] = cs
    lines.append("")
    lines.append("Growth test vs number of steps (same test): |M/M_exact - 1|")
    lines.append(f"{'variant':12s} {'bins':>4s} " + " ".join(f"{k:>9d}" for k in steps_list))
    for n in (8, 16, 32):
        for k in ["PC", "PL"] + REMAP_VARIANTS:
            lines.append(f"{k:12s} {n:4d} " + " ".join(f"{x:9.2e}" for x in cs[(k, n)]))

    # --- EA parameter scans (16 and 8 bins, 100 steps) ---
    scan = {}
    lines.append("")
    lines.append("EA parameter scans on the growth test (100 steps): |M/M_exact - 1| at 8 / 16 / 32 bins")
    for spec in ["ea:1e-8", "ea:1e-6", "ea:1e-4", "ea:1e-2", "ea:1e-1", "ea:3e-1",
                 "ea:1e-2:inf:-3.5", "ea:1e-2:inf:-1", "ea:1e-2:inf:0", "ea:1e-2:inf:2",
                 "ea:1e-2:4", "ea:1e-2:8", "ea:1e-2:16", "eaf:1e-6:-3.5", "eaf:1e-6:0", "eaf:1e-6:-1"]:
        cl = V.make_closure(spec)
        e = []
        for n in (8, 16, 32):
            a, g = MK.grid_of(n)
            e.append(abs(_grow(g, *MK.boxcar_moments(a, MK.A_MIN, a_cut), cl, da_tot / 100, 100)[1].sum()
                         / exact - 1.0))
        scan[spec] = e
        lines.append(f"  {spec:18s} " + " ".join(f"{x:9.2e}" for x in e) + f"   ({cl.params})")
    res["ea_scan"] = scan

    # --- sinusoid (Fig. 13), 1 period, snapshots ---
    sin = {}
    lines.append("")
    lines.append("Sinusoid (Fig. 13), one period, 200 steps: bins beyond the left analytic edge (mean dn/da > 1e-3 "
                 "of the plateau), mass error at 1 Gyr, W1 (ln a) of number and mass vs the analytic boxcar")
    for n in (128, 16):
        sin[n] = sinusoid(n, REMAP_VARIANTS, periods=1, steps_per_period=200, snaps=(0.25, 0.5, 0.75, 1.0))
        for spec in REMAP_VARIANTS + ["PC", "PL"]:
            d = sin[n][spec]
            lines.append(f"  N = {n:3d} {spec:12s} spread {d['spread'][-1]:3d}  mass err {d['merr'][-1]:.2e}  "
                         f"W1_N {(d['w1N'] or [np.nan])[-1]:.3e}  W1_M {(d['w1M'] or [np.nan])[-1]:.3e}")
    res["sinusoid"] = sin

    # --- one-step edge shapes (Fig. 9-like) ---
    res["fig9"] = fig9_states()

    # --- overflow, Fig. 11 test, carried at the mean size ---
    lines.append("")
    lines.append("Overflow (McKinnon+18 Fig. 11; dn/da flat (theirs) or ~a^-3.5 up to 1 um, grown 0.3 um/Gyr for 8 Gyr, "
                 "160 steps; the overflow carried as a spike at its mean size, or as a power law truncated above "
                 "a_max with the top bin's exponent, 'tpl'): m_d(8 Gyr)/exact - 1")
    ov = {}
    for ic in ("boxcar", "mrn"):
        for mode in ("spike", "tpl"):
            for spec in (REMAP_VARIANTS if (ic, mode) == ("boxcar", "spike") else ["pl2m", "ea:1e-2"]):
                ov[(ic, mode, spec)] = boundary_carry(V.make_closure(spec), mode=mode, ic=ic)
                lines.append(f"  {ic:6s} {mode:5s} {spec:12s} " + "  ".join(
                    f"a_max={k:.0f} um {v[1][-1] / v[2][-1] - 1:+.4f}" for k, v in ov[(ic, mode, spec)].items()))
    res["overflow"] = ov
    save(out, "growth", res, lines)


def sinusoid(n, variants, periods=1, steps_per_period=200, snaps=(), amp=0.015 * UM / GYR, period=1.0 * GYR,
             ic="boxcar"):
    """McKinnon's growth-shrinking test, generalised: ``periods`` periods; per variant the history of the mass error,
    left spread and W1 distances at the end of each period, and (N, M) at the snapshot phases of the first period."""
    a, g = MK.grid_of(n)
    lo, hi = a[3 * n // 8], a[5 * n // 8]
    if ic == "boxcar":
        N0, M0 = MK.boxcar_moments(a, lo, hi)
        a_s, dN_s = boxcar_samples(lo, hi)
    else:
        a_s, dN_s = mrn_samples(lo, hi)
        N0, M0 = binned(g, a_s, dN_s)
    plateau = (N0 / np.diff(a))[3 * n // 8 + 1: 5 * n // 8 - 1].max()
    dt = period / steps_per_period
    shift = lambda t: amp * period / (2 * np.pi) * (1.0 - np.cos(2 * np.pi * t / period))   # noqa: E731
    out = {"a": a, "lo": lo, "hi": hi, "N0": N0, "M0": M0}
    for spec in variants + ["PC", "PL"]:
        if spec in ("PC", "PL"):
            if ic != "boxcar":
                continue
            b = MK.LinearBins(a, N0, M0, constant=(spec == "PC"))
            cl = None
        else:
            cl = V.make_closure(spec)
            N, M = N0.copy(), M0.copy()
        d = {"merr": [], "spread": [], "w1N": [], "w1M": [], "l1M": [], "snaps": {}}
        for p in range(periods):
            for k in range(1, steps_per_period + 1):
                dd = shift(k * dt) - shift((k - 1) * dt)
                if cl is None:
                    b.shift(dd)
                else:
                    N, M, *_ = V.remap(g, N, M, lambda x, d_=dd: x + d_, cl)
                ph = k / steps_per_period
                if p == 0 and any(abs(ph - s) < 1e-9 for s in snaps):
                    d["snaps"][round(ph, 6)] = ((b.N.copy(), b.s.copy()) if cl is None else (N.copy(), M.copy()),
                                                (lo + shift(k * dt), hi + shift(k * dt)))
            if cl is None:
                Mtot = b.mass_total()
                Nb = b.N
                Mb = None
            else:
                Mtot, Nb, Mb = M.sum(), N, M
                wN, wM = w1_lna(g, N, M, cl, a_s, dN_s)
                d["w1N"].append(wN)
                d["w1M"].append(wM)
                d["l1M"].append(l1(M, M0))
            d["merr"].append(abs(Mtot / M0.sum() - 1.0))
            d["spread"].append(MK.left_spread(a, Nb / np.diff(a), lo, plateau))
        out[spec] = d
    return out


def fig9_states():
    """One growth step of a boxcar with its right edge inside a bin (McKinnon+18 Fig. 9 setup) and 50 steps more."""
    n = 5
    a = np.geomspace(0.02 * UM, 0.06 * UM, n + 1)
    from pycalima.gsd.grid import MassGrid
    g = MassGrid(MK.mass(a), MK.RHO)
    lo, hi = a[0] + 0.3 * (a[1] - a[0]), a[2] + 0.45 * (a[3] - a[2])
    N0, M0 = MK.boxcar_moments(a, lo, hi)
    da = 0.55 * (a[1] - a[0]) / 10
    out = {"a": a, "lo": lo, "hi": hi, "da": da, "grid": g, "N0": N0, "M0": M0}
    for spec in ["pl2m", "tc:8", "ea:1e-2", "aff", "exp"]:
        cl = V.make_closure(spec)
        N, M = N0.copy(), M0.copy()
        st = {0: (N.copy(), M.copy())}
        for k in range(1, 21):
            N, M, *_ = V.remap(g, N, M, lambda x: x + da, cl)
            if k in (10, 20):
                st[k] = (N.copy(), M.copy())
        out[spec] = st
    return out


def _grow_overflow(oN, oM, lam, a_max, d, mode):
    """Grow the carried overflow (oN, oM) by d. 'spike': all grains at the mean mass. 'tpl': dn/da ~ a^lam on
    [a_max, a_u], a_u from oM/oN (the overflow continues the top bin's slope from a_max upwards)."""
    if oN <= 0:
        return oM
    if mode == "spike":
        return oN * MK.mass(((3.0 * oM / oN / (4.0 * np.pi * MK.RHO)) ** (1.0 / 3.0)) + d)
    r = oM / (oN * MK.mass(a_max))
    uR = 1e3
    if r <= 1.0 + 1e-12:
        return oN * MK.mass(a_max + d)
    uc = V._solve_cut(r, lam, uR, True)
    mom = [V._mom(lam, 1.0, uc, float(k)) * a_max ** k for k in range(4)]
    a3 = mom[3] + 3 * d * mom[2] + 3 * d * d * mom[1] + d ** 3
    return oN * 4.0 / 3.0 * np.pi * MK.RHO * a3


def boundary_carry(cl, steps=160, adot=0.3 * UM / GYR, t_end=8.0 * GYR, mode="spike", ic="boxcar"):
    """McKinnon's Fig. 11 test with the overflow carried ('spike' or 'tpl'), for a variant closure; ic 'boxcar'
    (theirs) or 'mrn' (dn/da ~ a^-3.5 on the same range)."""
    from pycalima.gsd.grid import MassGrid
    a_min = 0.015625 * UM
    delta = (1.0 * UM / a_min) ** (1.0 / 60.0)
    a_l = a_min * delta ** 45
    t = np.linspace(0.0, t_end, steps + 1)
    p = 0.0 if ic == "boxcar" else -3.5
    a_s, dN_s = mrn_samples(a_l, 1.0 * UM, p=p, n=200001)
    m0 = np.sum(dN_s * MK.mass(a_s))
    exact = np.array([np.sum(dN_s * MK.mass(a_s + adot * tt)) for tt in t]) / m0
    out = {}
    for n in (60, 70, 80):
        a = a_min * delta ** np.arange(n + 1)
        g = MassGrid(MK.mass(a), MK.RHO)
        N0, M0 = binned(g, a_s, dN_s)
        N, M = N0.copy(), M0.copy()
        oN = oM = 0.0
        y = [1.0]
        d = adot * t_end / steps
        for _ in range(steps):
            P = cl.pieces(g, N, M)
            top = P.lam[-1, 0] if (N[-1] > 0 and not P.spike[-1, 0]) else 0.0
            oM = _grow_overflow(oN, oM, top, a[-1], d, mode)
            N, M, _, dN, dM = V.remap(g, N, M, lambda x, d=d: x + d, cl)
            oN += dN
            oM += dM
            y.append((M.sum() + oM) / M0.sum())
        out[round(a[-1] / UM)] = (t, np.array(y), exact)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# collisions
# ---------------------------------------------------------------------------------------------------------------------

COLL_VARIANTS = [("pl2m", "median", 1), ("pl2m", "gauss", 1), ("pl2m", "gaussm", 1), ("pl2m", "median", 2),
                 ("pl2m", "median", 4), ("pl2m", "median", 8), ("pl2m", "gauss", 4), ("tc:8", "median", 1),
                 ("ea:1e-2", "median", 1), ("aff", "median", 1), ("exp", "median", 1)]
# ka > 1: kernel averaged over ka x ka sub-slices of the two nodes, products and the donor split rate-weighted
# over the same sub-pairs (pl2m_variants._kernel_averaged_terms, products=True)


def cv_name(v):
    cl, rule, ka = v
    s = cl + ("" if rule == "median" else "+" + rule) + ("" if ka == 1 else f"+ka{ka}")
    return s


def collide_variant(v, g, N, M, model, dt, errmax=1e-4):
    from pycalima.gsd.collisions import collide
    cl, rule, ka = v
    with V.patched(V.make_closure(cl) if cl != "pl2m" else None, rule=rule, ka=ka):
        return collide(g, N, M, model, dt, errmax=errmax)


def bin_errors(g, N, M, a_e, nref_k, m_of):
    """McKinnon-test metrics on the coarse bins against a fine reference (number per fine bin)."""
    a_c = np.sqrt(a_e[:-1] * a_e[1:])
    x = np.log(g.a_edges / a_e[0]) / np.log(a_e[1] / a_e[0])
    idx = np.rint(x).astype(int)
    if np.max(np.abs(x - idx)) > 1e-6:
        raise ValueError("coarse edges are not fine edges")
    M_ref = np.add.reduceat(nref_k * m_of(a_c), idx[:-1])
    N_ref = np.add.reduceat(nref_k, idx[:-1])
    good = M_ref > 1e-2 * M_ref.sum()
    return dict(worstM=float(np.max(np.abs(M[good] / M_ref[good] - 1.0))),
                worstN=float(np.max(np.abs(N[good] / N_ref[good] - 1.0))),
                misplaced=float(np.sum(np.abs(M - M_ref)) / M_ref.sum()),
                mass=float(M.sum() / M_ref.sum() - 1.0))


def fine_ref_general(kind, dnda0, rho_d, t_out, n_fine=400, a_thr=0.1 * UM, v_coag=1.1e3):
    """MK.fine_reference with the threshold as a parameter (shattering above a_thr; coagulation below v_coag)."""
    from scipy.integrate import solve_ivp
    a_e = np.geomspace(MK.A_MIN, MK.A_MAX, n_fine + 1)
    a_c = np.sqrt(a_e[:-1] * a_e[1:])
    m_c = MK.mass(a_c)
    N0, M0 = MK.moments(a_e, dnda0, n_sub=40)
    n0 = N0 * rho_d / M0.sum()
    A1, A2 = a_c[:, None], a_c[None, :]
    if kind == "coag":
        Kk = MK.coag_kernel(m_c[:, None], m_c[None, :], v_coag=v_coag)
        mp = (m_c[:, None] + m_c[None, :]).ravel()
        L = np.clip(np.searchsorted(m_c, mp, side="right") - 1, 0, n_fine - 2)
        f = np.clip((m_c[L + 1] - mp) / (m_c[L + 1] - m_c[L]), 0.0, 1.0)

        def rhs(t, n):
            R = 0.5 * (Kk * np.outer(n, n)).ravel()
            gain = np.bincount(L, R * f, n_fine) + np.bincount(L + 1, R * (1.0 - f), n_fine)
            return gain - n * (Kk @ n)
    else:
        # thresholds at fine-bin resolution: the fraction of each fine bin above a_thr (uniform in ln a)
        fa = np.clip(np.log(a_e[1:] / a_thr) / np.log(a_e[1:] / a_e[:-1]), 0.0, 1.0)
        Kk = np.pi * (A1 + A2) ** 2 * 3e5 * fa[:, None] * fa[None, :]
        frac = (a_e[1:] ** 0.7 - a_e[:-1] ** 0.7) / (MK.A_MAX ** 0.7 - MK.A_MIN ** 0.7)
        mbar = MK.mass(1.0) * (-2.3 / 0.7) * (a_e[1:] ** 0.7 - a_e[:-1] ** 0.7) / (a_e[1:] ** -2.3 - a_e[:-1] ** -2.3)

        def rhs(t, n):
            loss = n * (Kk @ n)
            return np.sum(loss * m_c) * frac / mbar - loss
    sol = solve_ivp(rhs, (0.0, t_out[-1]), n0, method="LSODA", t_eval=t_out, rtol=1e-8, atol=1e-30 * n0.sum())
    return a_e, a_c, sol.y.T


def mk_collision_case(kind, n, v, t_out, a_thr=0.1 * UM, v_coag=1.1e3):
    lognormal = lambda x: np.exp(-np.log(x / (0.1 * UM)) ** 2 / (2 * 0.6 ** 2)) / x   # noqa: E731
    mrn = lambda x: x ** -3.5                                                          # noqa: E731
    f0, rho_d = (lognormal, 3.7e-3 * 1.4 * MK.M_H * 0.4) if kind == "shat" else (mrn, 0.01 * 1.4 * MK.M_H * 1e5)
    a, g = MK.grid_of(n)
    N, M = MK.moments(a, f0)
    N, M = N * rho_d / M.sum(), M * rho_d / M.sum()
    if kind == "shat":
        model = MK.FullShattering(a_thr=a_thr)
    else:
        from pycalima.gsd.collisions import CoagulationModel
        model = CoagulationModel(lambda m1, m2: MK.coag_kernel(m1, m2, v_coag=v_coag))
    states, tp, steps = [], 0.0, 0
    t0 = time.perf_counter()
    for t in t_out:
        if t > tp:
            r = collide_variant(v, g, N, M, model, t - tp)
            N, M = r["N"], r["M"]
            steps += r["n_steps"]
        states.append((N.copy(), M.copy()))
        tp = t
    return g, states, dict(wall=time.perf_counter() - t0, steps=steps)


def golovin(x, T):
    """n(m) m0 / N0 of the additive kernel with an exponential start (Golovin 1963), x = m / m0."""
    from scipy.special import ive
    if T <= 0:
        return np.exp(-x)
    z = 2.0 * x * np.sqrt(T)
    # I1(z) e^{-(1+T) x} = ive(1, z) e^{z - (1+T) x}
    return (1.0 - T) / (x * np.sqrt(T)) * ive(1, z) * np.exp(z - (1.0 + T) * x)


def kernel_case(kernel, n, v, taus, x_lo=1e-3, x_hi=1e5):
    """Coagulation with the constant ('const') or additive ('golovin') kernel from an exponential, m0 = 1, N0 = 1,
    rho = 1; errors of the binned N and M against the analytic solutions at the given tau."""
    from pycalima.gsd.collisions import CoagulationModel
    from pycalima.gsd.grid import MassGrid
    g = MassGrid(np.geomspace(x_lo, x_hi, n + 1), 1.0)
    me = g.m_edges

    def exact_bins(tau):
        xs = np.geomspace(x_lo, x_hi, 400001)
        if kernel == "const":
            f = (2.0 / (2.0 + tau)) ** 2 * np.exp(-2.0 * xs / (2.0 + tau))
        else:
            f = golovin(xs, 1.0 - np.exp(-tau))
        y = f * xs
        lx = np.log(xs)
        cN = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(lx))])
        cM = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] * xs[1:] + y[:-1] * xs[:-1]) * np.diff(lx))])
        return np.diff(np.interp(np.log(me), lx, cN)), np.diff(np.interp(np.log(me), lx, cM))

    N, M = exact_bins(0.0)
    K0 = 1.0
    model = CoagulationModel((lambda m1, m2: K0 + 0.0 * m1 * m2) if kernel == "const"
                             else (lambda m1, m2: K0 * (m1 + m2)))
    model.collector = "smallest"
    out = []
    tp = 0.0
    for tau in taus:
        t = tau / K0           # N0 = 1, m0 = 1: tau = K0 N0 t (const), B N0 m0 t (additive)
        r = collide_variant(v, g, N, M, model, t - tp)
        N, M = r["N"], r["M"]
        tp = t
        Ne, Me = exact_bins(tau)
        good = Me > 1e-2 * Me.sum()
        out.append(dict(tau=tau, misplaced=float(np.abs(M - Me).sum() / Me.sum()),
                        worstM=float(np.max(np.abs(M[good] / Me[good] - 1.0))),
                        Ntot=float(N.sum() / Ne.sum() - 1.0), N=N.copy(), M=M.copy(), Ne=Ne, Me=Me,
                        over=float(r["overflow_M"])))
    return g, out


def collisions_from_log(path):
    """The McKinnon and threshold-sweep results of a collisions run, read back from its log (the lines carry every
    metric the report and figures use)."""
    import re
    res = {"mk": {}, "sweep": {}}
    pat_mk = re.compile(r"^  (shat|coag) N=\s*(\d+) (\S+)\s+worst M ([\d.]+) N ([\d.]+)\s+misplaced ([\d.]+)\s+mass "
                        r"(\S+)\s+wall ([\d.]+) s\s+steps (\d+)")
    pat_sw = re.compile(r"^  (shat|coag) f=([\d.]+) (\S+)\s+misplaced ([\d.]+) worst M ([\d.]+)")
    for line in Path(path).read_text().splitlines():
        m = pat_mk.match(line)
        if m:
            kind, n, nm = m.group(1), int(m.group(2)), m.group(3)
            e = dict(worstM=float(m.group(4)), worstN=float(m.group(5)), misplaced=float(m.group(6)),
                     mass=float(m.group(7)))
            res["mk"][(kind, n, nm)] = dict(errs=[e], info=dict(wall=float(m.group(8)), steps=int(m.group(9))))
            continue
        m = pat_sw.match(line)
        if m:
            res["sweep"][(m.group(1), float(m.group(2)), m.group(3))] = dict(misplaced=float(m.group(4)),
                                                                            worstM=float(m.group(5)))
    return res


def cmd_kernels(args):
    """Only the constant/Golovin section of ``collisions``, merged with the McKinnon and sweep results read from
    collisions.log (a complete earlier run of those sections)."""
    out = Path(args.out)
    res = collisions_from_log(out / "collisions_mk_sweep.log")
    res["kernels"] = {}
    lines = [l for l in (out / "collisions_mk_sweep.log").read_text().splitlines()
             if l.startswith("  shat") or l.startswith("  coag") or l.startswith("McKinnon") or
             l.startswith("Threshold")]
    lines.insert(0, "McKinnon+18 collision tests and threshold sweeps: lines from collisions_mk_sweep.log (complete run); "
                    "kernel tests rerun by the 'kernels' subcommand")
    lines.append("")
    lines.append("Constant and additive (Golovin) kernels from an exponential, 8 decades in mass: mass misplaced "
                 "and worst bin (>= 1 % of the mass) against the analytic solutions; total number error")
    kern_vars = [("pl2m", "median", 1), ("pl2m", "gauss", 1), ("pl2m", "median", 4), ("tc:8", "median", 1),
                 ("ea:1e-6", "median", 1), ("aff", "median", 1), ("exp", "median", 1)]
    for kernel, taus in (("const", [1.0, 10.0, 100.0]), ("golovin", [0.5, 1.0, 2.0, 3.0])):
        for n in (8, 16, 32):
            for v in kern_vars:
                g, rows = kernel_case(kernel, n, v, taus)
                res["kernels"][(kernel, n, cv_name(v))] = rows
                lines.append(f"  {kernel:7s} N={n:2d} {cv_name(v):16s} " + "  ".join(
                    f"tau={r['tau']:g}: {r['misplaced']:.4f}/{r['worstM']:.3f}/{r['Ntot']:+.3f}" for r in rows))
                print(lines[-1], flush=True)
    save(out, "collisions", res, lines)


def cmd_collisions(args):
    out = Path(args.out)
    res, lines = {"mk": {}, "sweep": {}, "kernels": {}}, []
    variants = COLL_VARIANTS if not args.quick else COLL_VARIANTS[:3]
    lognormal = lambda x: np.exp(-np.log(x / (0.1 * UM)) ** 2 / (2 * 0.6 ** 2)) / x   # noqa: E731
    mrn = lambda x: x ** -3.5                                                          # noqa: E731
    cases = {"shat": (lognormal, 3.7e-3 * 1.4 * MK.M_H * 0.4, np.array([0.0, 100.0]) * MYR, (8, 16)),
             "coag": (mrn, 0.01 * 1.4 * MK.M_H * 1e5, np.array([0.0, 2.0, 20.0]) * MYR, (8, 32))}
    lines.append("McKinnon+18 collision tests, final time (shattering 100 Myr, coagulation 20 Myr) against the fine "
                 "reference: worst bin (>= 1 % of the mass) in M and N, mass misplaced sum|dM|/M; wall [s], sub-steps")
    for kind, (f0, rho_d, t_out, n_list) in cases.items():
        a_e, a_c, nref = fine_ref_general(kind, f0, rho_d, t_out, n_fine=640)
        res["mk"][(kind, "ref")] = (a_e, a_c, nref, t_out)
        for n in n_list:
            for v in variants:
                g, states, info = mk_collision_case(kind, n, v, t_out)
                errs = [bin_errors(g, *states[k], a_e, nref[k], MK.mass) for k in range(1, len(t_out))]
                res["mk"][(kind, n, cv_name(v))] = dict(states=states, errs=errs, info=info, grid=g)
                e = errs[-1]
                lines.append(f"  {kind} N={n:2d} {cv_name(v):18s} worst M {e['worstM']:.3f} N {e['worstN']:.3f}  "
                             f"misplaced {e['misplaced']:.4f}  mass {e['mass']:+.1e}  "
                             f"wall {info['wall']:.1f} s  steps {info['steps']}")
                print(lines[-1], flush=True)

    # --- threshold sweeps ---
    lines.append("")
    lines.append("Threshold swept through one bin. Shattering (N = 8; both grains above a_t shatter; 100 Myr) and "
                 "coagulation (N = 16; sticking below v_coag, swept so the equal-size threshold crosses a bin; "
                 "5 Myr): mass misplaced / worst bin vs the threshold's position f in the bin (ln a)")
    sweep_vars = [v for v in [("pl2m", "median", 1), ("pl2m", "gauss", 1), ("pl2m", "median", 2),
                              ("pl2m", "median", 4), ("pl2m", "median", 8), ("pl2m", "gauss", 4)]
                  if v in variants or not args.quick]
    fs = np.linspace(0.05, 0.95, 7)
    a8, _ = MK.grid_of(8)
    jb = np.searchsorted(a8, 0.1 * UM) - 1
    a16, _ = MK.grid_of(16)
    jc = np.searchsorted(a16, 0.025 * UM) - 1
    for kind, n, j, a_ in (("shat", 8, jb, a8), ("coag", 16, jc, a16)):
        f0, rho_d = cases[kind][:2]
        t_out = np.array([0.0, 100.0 if kind == "shat" else 5.0]) * MYR
        for f in fs:
            a_t = a_[j] * (a_[j + 1] / a_[j]) ** f
            kw = dict(a_thr=a_t) if kind == "shat" else dict(v_coag=2.0 * 1.1e3 * np.sqrt(a_t / (0.1 * UM)))
            a_e, a_c, nref = fine_ref_general(kind, f0, rho_d, t_out, n_fine=640, **kw)
            for v in sweep_vars:
                g, states, info = mk_collision_case(kind, n, v, t_out, **kw)
                e = bin_errors(g, *states[-1], a_e, nref[-1], MK.mass)
                res["sweep"][(kind, f, cv_name(v))] = e
                lines.append(f"  {kind} f={f:.2f} {cv_name(v):18s} misplaced {e['misplaced']:.4f} worst M "
                             f"{e['worstM']:.3f}")
                print(lines[-1], flush=True)

    # --- constant and additive kernels ---
    lines.append("")
    lines.append("Constant and additive (Golovin) kernels from an exponential, 8 decades in mass: mass misplaced "
                 "and worst bin (>= 1 % of the mass) against the analytic solutions; total number error")
    kern_vars = [v for v in [("pl2m", "median", 1), ("pl2m", "gauss", 1), ("pl2m", "median", 4),
                             ("tc:8", "median", 1), ("ea:1e-2", "median", 1), ("aff", "median", 1),
                             ("exp", "median", 1)] if v in variants or not args.quick]
    for kernel, taus in (("const", [1.0, 10.0, 100.0]), ("golovin", [0.5, 1.0, 2.0, 3.0])):
        for n in (8, 16, 32):
            for v in kern_vars:
                g, rows = kernel_case(kernel, n, v, taus)
                res["kernels"][(kernel, n, cv_name(v))] = rows
                lines.append(f"  {kernel:7s} N={n:2d} {cv_name(v):16s} " + "  ".join(
                    f"tau={r['tau']:g}: {r['misplaced']:.4f}/{r['worstM']:.3f}/{r['Ntot']:+.3f}" for r in rows))
                print(lines[-1], flush=True)
    save(out, "collisions" if not args.quick else "collisions_quick", res, lines)


# ---------------------------------------------------------------------------------------------------------------------
# stress tests
# ---------------------------------------------------------------------------------------------------------------------

def carbon_grid():
    from pycalima.gsd.carbon import CarbonLayout
    return CarbonLayout().grid()


def silicate_grid8():
    from pycalima.gsd.grid import MassGrid
    rho = 3.3
    m = 4.0 / 3.0 * np.pi * rho * np.array([4.5e-8, 1e-4]) ** 3
    return MassGrid(np.geomspace(m[0], m[1], 9), rho)


def run_shift_test(g, a_s, dN_s, variants, da_total, steps, snap_every=None):
    """Constant da/dt: every variant over ``steps`` steps; exact = samples shifted. Returns per variant the final
    errors (mass, L1 of N and M, W1 of number and mass in ln a)."""
    N0, M0 = binned(g, a_s, dN_s)
    out = {}
    for spec in variants:
        cl = V.make_closure(spec)
        N, M = N0.copy(), M0.copy()
        hist = []
        for k in range(1, steps + 1):
            N, M, *_ = V.remap(g, N, M, lambda x, d=da_total / steps: x + d, cl)
            if snap_every and k % snap_every == 0:
                Ne, Me = binned(g, a_s + da_total * k / steps, dN_s)
                hist.append((k, abs(M.sum() / Me.sum() - 1.0), l1(N, Ne), l1(M, Me)))
        Ne, Me = binned(g, a_s + da_total, dN_s)
        wN, wM = w1_lna(g, N, M, cl, a_s + da_total, dN_s)
        out[spec] = dict(merr=abs(M.sum() / Me.sum() - 1.0), l1N=l1(N, Ne), l1M=l1(M, Me), w1N=wN, w1M=wM,
                         N=N, M=M, hist=hist, nbins=int(np.sum(N > 1e-3 * N.sum())),
                         nbins_exact=int(np.sum(Ne > 1e-3 * Ne.sum())))
    return out


def cmd_stress(args):
    out = Path(args.out)
    res, lines = {}, []
    variants = REMAP_VARIANTS

    # S1: narrow population grown x10 and x100
    lines.append("S1 narrow population: top-hat a0 (1 +- 0.01), a0 = 2 nm, constant da/dt to 10 a0 and 100 a0 "
                 "(McKinnon grid 1 nm - 1 um). Final: |dM/M|, L1 of binned N and M, W1 (ln a) of number and mass, "
                 "bins holding > 1e-3 of N (exact in brackets)")
    s1 = {}
    for n in (8, 16):
        a, g = MK.grid_of(n)
        a_s, dN_s = boxcar_samples(2e-7 * 0.99, 2e-7 * 1.01, n=20001)
        for fac, steps in ((10, 1000), (100, 1000), (100, 5000)):
            r = run_shift_test(g, a_s, dN_s, variants, 2e-7 * (fac - 1), steps)
            s1[(n, fac, steps)] = r
            for spec in variants:
                d = r[spec]
                lines.append(f"  N={n:2d} x{fac:3d} {steps:5d} steps {spec:10s} dM {d['merr']:.2e} L1N {d['l1N']:.3f} "
                             f"L1M {d['l1M']:.3f} W1N {d['w1N']:.3f} W1M {d['w1M']:.3f} bins {d['nbins']} "
                             f"({d['nbins_exact']})")
    res["S1"] = s1
    print("\n".join(lines), flush=True)

    # S2: 10 growth-shrink periods
    lines.append("")
    lines.append("S2 ten growth-shrink periods (McKinnon sinusoid), 200 steps per period: at the end of periods 1 and 10, "
                 "|dM/M|, L1 of M against the initial bins, W1_M (ln a), left spread")
    s2 = {}
    for n in (16, 128):
        for ic in ("boxcar", "mrn"):
            d = sinusoid(n, variants, periods=10, steps_per_period=200, ic=ic)
            s2[(n, ic)] = d
            for spec in variants + (["PC", "PL"] if ic == "boxcar" else []):
                x = d[spec]
                lines.append(f"  N={n:3d} {ic:6s} {spec:10s} dM {x['merr'][0]:.2e} -> {x['merr'][-1]:.2e}  L1M "
                             + (f"{x['l1M'][0]:.3f} -> {x['l1M'][-1]:.3f}  W1M {x['w1M'][0]:.3f} -> {x['w1M'][-1]:.3f}"
                                if x["l1M"] else "    -")
                             + f"  spread {x['spread'][0]} -> {x['spread'][-1]}")
    res["S2"] = s2
    print("\n".join(lines[-20:]), flush=True)

    # S3: equilibrium radius (pile-up), da/dt = v0 (a_eq / a - 1)
    lines.append("")
    lines.append("S3 equilibrium radius: da/dt = v0 (a_eq/a - 1), a_eq = 30 nm, MRN 5-250 nm, characteristics from the "
                 "velocity table (same for the reference: 2e5 grains moved exactly); t = 1, 3, 10 tau "
                 "(tau = a_eq/v0), 300 steps per tau. L1 of binned N and M, W1_M")
    from pycalima.gsd.remap import advance, velocity_table
    s3 = {}
    a_eq, v0 = 30e-7, 1e-7 / MYR
    rate = lambda x: v0 * (a_eq / np.maximum(x, 1e-30) - 1.0)    # noqa: E731
    tau = a_eq / v0
    for n in (8, 16):
        a, g = MK.grid_of(n)
        table = velocity_table(rate, g.a_edges[0] * 0.3, g.a_edges[-1] * 3.0, 2000)
        a_s, dN_s = mrn_samples(5e-7, 2.5e-5, n=200001)
        N0, M0 = binned(g, a_s, dN_s)
        dt = tau / 300
        for spec in variants:
            cl = V.make_closure(spec)
            N, M = N0.copy(), M0.copy()
            rows = []
            x = a_s.copy()
            for k in range(1, 3001):
                N, M, *_ = V.remap(g, N, M, lambda y: advance(table, y, dt), cl)
                x = advance(table, x, dt)
                if k in (300, 900, 3000):
                    Ne, Me = binned(g, x, dN_s)
                    wN, wM = w1_lna(g, N, M, cl, x, dN_s)
                    rows.append(dict(t=k / 300, l1N=l1(N, Ne), l1M=l1(M, Me), w1M=wM, dM=abs(M.sum() / Me.sum() - 1)))
            s3[(n, spec)] = rows
            lines.append(f"  N={n:2d} {spec:10s} " + "  ".join(
                f"t={r['t']:g}tau: L1N {r['l1N']:.3f} L1M {r['l1M']:.3f} W1M {r['w1M']:.3f}" for r in rows))
    res["S3"] = s3
    print("\n".join(lines[-20:]), flush=True)

    # S4: edge sweep
    lines.append("")
    lines.append("S4 sharp edges at fraction f of a bin (ln a): MRN segment x13.3 wide with both cut-offs at f, grown "
                 "(+, lower edge up by 2 bins of N = 8) or shrunk (-, upper edge down by 0.6 bin) at constant rate in "
                 "200 steps (N = 8, 16). Per variant |dM/M| / W1_N / W1_M")
    s4 = {}
    fs = (0.05, 0.25, 0.5, 0.75, 0.95)
    for n in (8, 16):
        a, g = MK.grid_of(n)
        r_bin = a[1] / a[0]
        for sign in (+1, -1):
            for f in fs:
                j0 = n // 8
                lo = a[j0] * r_bin ** f
                hi = lo * r_bin ** (3 * n // 8)          # the same physical width (x 13.3) on both grids
                a_s, dN_s = mrn_samples(lo, hi, n=100001)
                # growth moves the lower edge up by ~2 bins; shrinking moves the upper edge down by 0.6 of a bin
                da = lo * (r_bin ** (2 * n // 8) - 1.0) if sign > 0 else -0.6 * hi * (1.0 - 1.0 / r_bin)
                r = run_shift_test(g, a_s, dN_s, variants, da, 200)
                s4[(n, sign, f)] = r
                lines.append(f"  N={n:2d} {'+' if sign > 0 else '-'} f={f:.2f} " + " ".join(
                    f"{spec}:{r[spec]['merr']:.1e}/{r[spec]['w1N']:.3f}/{r[spec]['w1M']:.3f}" for spec in variants))
    res["S4"] = s4
    print("\n".join(lines[-20:]), flush=True)

    # S5: mixing of cells
    lines.append("")
    lines.append("S5 mixing two cells (an advection step): A = MRN 5-250 nm, B = A grown by d; mixed 50/50. Static: "
                 "errors of the mixture's surface S, int a dN and next-step mass gain vs exact. Cycled: 200 cycles of "
                 "(grow A by 0.05 nm; mix 50/50 into both), against a 512-bin PL2M reference: L1 of M, error of S")
    s5 = {}
    for n in (8, 16):
        a, g = MK.grid_of(n)
        aA, dA = mrn_samples(5e-7, 2.5e-5, n=100001)
        for dsh in (1e-7, 3e-7, 1e-6):
            NA, MA = binned(g, aA, dA)
            NB, MB = binned(g, aA + dsh, dA)
            Nm, Mm = 0.5 * (NA + NB), 0.5 * (MA + MB)
            a_mix = np.concatenate([aA, aA + dsh])
            d_mix = np.concatenate([0.5 * dA, 0.5 * dA])
            S_ex = np.sum(d_mix * a_mix ** 2)
            A1_ex = np.sum(d_mix * a_mix)
            row = {}
            for spec in variants:
                cl = V.make_closure(spec)
                P = cl.pieces(g, Nm, Mm)
                S = np.sum(V.radius_moments(P, 2))
                A1 = np.sum(V.radius_moments(P, 1))
                Ng, Mg, *_ = V.remap(g, Nm, Mm, lambda x: x + 1e-8, cl)
                Mg_ex = np.sum(d_mix * MK.mass(a_mix + 1e-8))
                row[spec] = dict(S=S / S_ex - 1, A1=A1 / A1_ex - 1, dMg=(Mg.sum() - Mm.sum()) /
                                 (Mg_ex - np.sum(d_mix * MK.mass(a_mix))) - 1)
            s5[("static", n, dsh)] = row
            lines.append(f"  static N={n:2d} d={dsh * 1e7:.0f} nm " + " ".join(
                f"{k}:{v['S']:+.3f}/{v['A1']:+.3f}/{v['dMg']:+.3f}" for k, v in row.items()))
        # cycled
        from pycalima.gsd.grid import MassGrid
        af = np.geomspace(MK.A_MIN, MK.A_MAX, 513)
        gf = MassGrid(MK.mass(af), MK.RHO)
        NAf, MAf = binned(gf, aA, dA)
        refA, refB = (NAf.copy(), MAf.copy()), (NAf.copy(), MAf.copy())
        clf = V.make_closure("pl2m")
        for _ in range(200):
            NA_, MA_, *_ = V.remap(gf, *refA, lambda x: x + 5e-9, clf)
            Nmix, Mmix = 0.5 * (NA_ + refB[0]), 0.5 * (MA_ + refB[1])
            refA = refB = (Nmix, Mmix)
        Pf = clf.pieces(gf, *refA)
        S_ref = np.sum(V.radius_moments(Pf, 2))
        # reference re-binned onto the coarse grid via its samples
        a_rs, d_rs = V.samples(gf, *refA, clf)
        Nref_c, Mref_c = binned(g, a_rs, d_rs)
        row = {}
        for spec in variants:
            cl = V.make_closure(spec)
            NA0, MA0 = binned(g, aA, dA)
            sA, sB = (NA0.copy(), MA0.copy()), (NA0.copy(), MA0.copy())
            for _ in range(200):
                NA_, MA_, *_ = V.remap(g, *sA, lambda x: x + 5e-9, cl)
                Nmix, Mmix = 0.5 * (NA_ + sB[0]), 0.5 * (MA_ + sB[1])
                sA = sB = (Nmix, Mmix)
            P = cl.pieces(g, *sA)
            S = np.sum(V.radius_moments(P, 2))
            row[spec] = dict(l1M=l1(sA[1], Mref_c), S=S / S_ref - 1)
        s5[("cycled", n)] = row
        lines.append(f"  cycled N={n:2d} " + " ".join(f"{k}:{v['l1M']:.4f}/{v['S']:+.4f}" for k, v in row.items()))
    res["S5"] = s5
    print("\n".join(lines[-10:]), flush=True)

    # S8: RAMSES-like steps on the real grids
    lines.append("")
    lines.append("S8 RAMSES-like steps: MRN 5-250 nm under constant accretion, Delta a = 5 nm over 10 Myr, steps of "
                 "1e2, 1e3, 1e4 yr; silicate 8 bins (0.45 nm - 1 um, eta = 18.0) and the carbon layout (4 bins "
                 "eta = 2.1 over N_C 20-400, 7 bins eta = 18.8 to 1 um). |dM/M|, L1 of M, W1_M, surface error")
    s8 = {}
    for gname, g in (("silicate8", silicate_grid8()), ("carbon", carbon_grid())):
        a_s, dN_s = mrn_samples(5e-7, 2.5e-5, n=200001)
        for dt_yr in (1e4, 1e3, 1e2):
            steps = int(round(10e6 / dt_yr))
            N0, M0 = binned(g, a_s, dN_s)
            Ne, Me = binned(g, a_s + 5e-7, dN_s)
            S_ex = np.sum(dN_s * (a_s + 5e-7) ** 2)
            row = {}
            for spec in variants:
                cl = V.make_closure(spec)
                N, M = N0.copy(), M0.copy()
                t0 = time.perf_counter()
                for _ in range(steps):
                    N, M, *_ = V.remap(g, N, M, lambda x, d=5e-7 / steps: x + d, cl)
                wN, wM = w1_lna(g, N, M, cl, a_s + 5e-7, dN_s)
                S = np.sum(V.radius_moments(cl.pieces(g, N, M), 2))
                row[spec] = dict(merr=abs(M.sum() / Me.sum() - 1), l1M=l1(M, Me), l1N=l1(N, Ne), w1M=wM, w1N=wN,
                                 S=S / S_ex - 1, wall=time.perf_counter() - t0)
            s8[(gname, dt_yr)] = row
            lines.append(f"  {gname:9s} dt={dt_yr:7.0e} yr ({steps} steps) " + " ".join(
                f"{k}:{v['merr']:.1e}/{v['l1M']:.3f}/{v['w1M']:.3f}/{v['S']:+.3f}" for k, v in row.items()))
            print(lines[-1], flush=True)
    res["S8"] = s8
    save(out, "stress", res, lines)


# ---------------------------------------------------------------------------------------------------------------------
# A0 cases
# ---------------------------------------------------------------------------------------------------------------------

A0_VARIANTS = {"pl2m": ("pl2m", "median"), "tc8": ("tc:8", "median"), "ea": ("ea:1e-2", "median"),
               "ea_tc8": ("ea:1e-2:8", "median"), "aff": ("aff", "median"), "exp": ("exp", "median"),
               "gauss": ("pl2m", "gauss"), "ea_gauss": ("ea:1e-2", "gauss"), "ea6": ("ea:1e-6", "median"),
               "ea6_gauss": ("ea:1e-6", "gauss"), "eaf6": ("eaf:1e-6:-3.5", "median")}
A0_CASES = ("sn_diffuse", "mrn_accretion", "mrn_coagulation", "mrn_shattering", "mrn_sputtering")
A0_BINS = (4, 6, 8, 12)


def _a0_task(kw):
    """One A0 run in a worker process: the variant patched into exchange, sputtering, collisions and the
    observables; the states at each snapshot recorded for the PL2M-closure observables."""
    out = Path(kw["out"])
    os.environ["CALIMA_RESULTS"] = str(out.resolve())
    os.environ["PYCALIMA_GSD_THREADS"] = "1"
    import check_bin_convergence as cbc
    tag, (spec, rule) = kw["tag"], A0_VARIANTS[kw["tag"]]
    cl = V.make_closure(spec)
    rec = []
    own = (lambda grid, N, M: V.samples(grid, N, M, cl))
    orig = cbc.bin_samples

    def bs(grid, N, M):
        rec.append((float(grid.rho), np.asarray(N, float).copy(), np.asarray(M, float).copy()))
        return own(grid, N, M) if spec != "pl2m" else orig(grid, N, M)

    cbc.bin_samples = bs
    try:
        with V.patched(cl if spec != "pl2m" else None, rule=rule):
            path = cbc.run_one(kw["case"], "pl2m", kw["n_bins"], 3.5e-8, 4.5e-8, 1e4, 1, tag="_" + tag)
    finally:
        cbc.bin_samples = orig
    with open(str(path).replace(".npz", "_states.pkl"), "wb") as fh:
        pickle.dump(rec, fh)
    return path


def cmd_a0(args):
    out = Path(args.out) / "a0"
    (out / "gsd_bin_convergence").mkdir(parents=True, exist_ok=True)
    import shutil
    src = A0_REF / "mie_table.npz"
    dst = out / "gsd_bin_convergence" / "mie_table.npz"
    if not dst.exists():
        shutil.copy(src, dst)
    tags = args.variants.split(",") if args.variants else list(A0_VARIANTS)
    cases = args.cases.split(",") if args.cases else list(A0_CASES)
    tasks = [dict(out=str(out), tag=t, case=c, n_bins=nb) for t in tags for c in cases for nb in A0_BINS]
    done = {p.name for p in (out / "gsd_bin_convergence").glob("*.npz")}
    tasks = [t for t in tasks if f"{t['case']}_pl2m_N{t['n_bins']:02d}_amin3.50A_{t['tag']}.npz" not in done]
    print(f"{len(tasks)} A0 runs", flush=True)
    if args.workers <= 1:
        for t in tasks:
            _a0_task(t)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for p in ex.map(_a0_task, tasks):
                print("done", Path(p).name, flush=True)
    a0_report(args)


def a0_report(args):
    import check_bin_convergence as cbc
    out = Path(args.out)
    d = out / "a0" / "gsd_bin_convergence"
    refs = {}
    for case in A0_CASES:
        f = A0_REF / f"{case}_exact_ref_amin3.50A.npz"
        if not f.exists():
            f = A0_REF / f"{case}_pl2m_ref_amin3.50A.npz"
        refs[case] = cbc._load(f)
    rows, lines = {}, []
    mie = None
    for f in sorted(d.glob("*_amin3.50A_*.npz")):
        meta, t, names, v = cbc._load(f)
        tag = f.stem.split("amin3.50A_")[1]
        case, nb = meta["case"], meta["n_bins"]
        mr, tr, nr, vr = refs[case]
        err = np.array([[cbc.errors(dict(zip(names, vi)), dict(zip(nr, vri)))[q] for q in nr]
                        for vi, vri in zip(v[1:], vr[1:])])
        worst = err.max(axis=0)
        row = dict(err=float(worst.max()), worst=nr[int(worst.argmax())], per=dict(zip(nr, worst.tolist())),
                   wall=meta["wall"], substeps=meta["coll_substeps"], cons=max(meta["conservation"].values()))
        # observables with the PL2M closure on the same states (what s-tables of the PL2M family would see)
        st = str(f).replace(".npz", "_states.pkl")
        if os.path.exists(st) and tag not in ("pl2m", "gauss"):
            if mie is None:
                mie = np.load(d / "mie_table.npz")
                mie = {"a": mie["a"], "graphite": mie["graphite"], "silicate": mie["silicate"]}
            with open(st, "rb") as fh:
                rec = pickle.load(fh)
            mats = cbc.materials(3.5e-8, 4.5e-8)
            from pycalima.gsd.grid import MassGrid
            vals = []
            per_snap = len(rec) // len(t)
            for k in range(len(t)):
                dists = {}
                for j, mat in enumerate(mats):
                    rho, N, M = rec[k * per_snap + j]
                    grid = cbc.make_grid(mat, nb)
                    dists[mat.name] = (mat, *cbc.bin_samples(grid, N, M), 0.0)
                vals.append(cbc.metrics(dists, mie))
            err2 = np.array([[cbc.errors(vi, dict(zip(nr, vri)))[q] for q in nr] for vi, vri in zip(vals[1:], vr[1:])])
            w2 = err2.max(axis=0)
            row["err_pl2m_obs"] = float(w2.max())
            row["worst_pl2m_obs"] = nr[int(w2.argmax())]
        rows[(tag, case, nb)] = row
    # stored PL2M runs for the check
    for case in A0_CASES:
        for nb in A0_BINS:
            f = A0_REF / f"{case}_pl2m_N{nb:02d}_amin3.50A.npz"
            meta, t, names, v = cbc._load(f)
            mr, tr, nr, vr = refs[case]
            err = np.array([[cbc.errors(dict(zip(names, vi)), dict(zip(nr, vri)))[q] for q in nr]
                            for vi, vri in zip(v[1:], vr[1:])])
            worst = err.max(axis=0)
            rows[("stored", case, nb)] = dict(err=float(worst.max()), worst=nr[int(worst.argmax())],
                                             per=dict(zip(nr, worst.tolist())))
    lines.append("A0 cases: worst observable error over the 10/30/100 % snapshots against the stored references "
                 "(exact for accretion and sputtering, 30 bins/decade otherwise); observables from each variant's "
                 "own closure [and, in brackets, from the PL2M closure on the same states]")
    tags = ["stored"] + [t for t in A0_VARIANTS if any(k[0] == t for k in rows)]
    for case in A0_CASES:
        lines.append(f"{case}")
        lines.append(f"  {'variant':9s} " + " ".join(f"{nb:>20d}" for nb in A0_BINS))
        for tag in tags:
            cells = []
            for nb in A0_BINS:
                r = rows.get((tag, case, nb))
                if r is None:
                    cells.append(f"{'-':>20s}")
                    continue
                extra = f"[{r['err_pl2m_obs']:.3f}]" if "err_pl2m_obs" in r else ""
                cells.append(f"{r['err']:.3f}{extra} {r['worst'].split(':')[-1][:7]:>7s}".rjust(20))
            lines.append(f"  {tag:9s} " + " ".join(cells))
    save(out, "a0", rows, lines)


# ---------------------------------------------------------------------------------------------------------------------
# rates
# ---------------------------------------------------------------------------------------------------------------------

def cmd_rates(args):
    import check_rates_convergence as crc
    from pycalima.gsd.grid import MassGrid
    out = Path(args.out)
    d = out / "a0" / "gsd_bin_convergence"
    rates = {}
    res, lines = {}, []

    def totals_variant(grid, N, M, a, g, cl, rule):
        at = lambda r: np.exp(np.interp(np.log(r), np.log(a), np.log(g)))   # noqa: E731
        a_s, dN = V.samples(grid, N, M, cl, step=0.005)
        exact = float(np.sum(dN * at(a_s)))
        m, w, _, _ = V.nodes(grid, N, M, cl, rule, 1)
        f = N > 0
        two = float(np.sum(w[f] * at(grid.radius(m[f]))))
        return exact, two

    # references: the stored high-resolution runs, integrated exactly (PL2M profile weights)
    ref = {}
    for case in A0_CASES:
        meta, pops = crc.load(A0_REF / f"{case}_pl2m_ref_amin3.50A.npz")
        for p, (me, N, M) in pops.items():
            grid = MassGrid(me, crc.rho_of(me, meta["a_min_c"] if p == "graphite" else meta["a_min_si"]))
            key = (p, grid.rho)
            if key not in rates:
                rates[key] = crc.single_grain(crc.POPS[p], grid.a_edges[0], 5e-4)
            a, qs = rates[key]
            for cell, qd in qs.items():
                for q, g in qd.items():
                    ref[(case, cell, q)] = ref.get((case, cell, q), 0.0) + crc.totals(grid, N, M, a, g)["exact"]
    for f in sorted(d.glob("*_amin3.50A_*.npz")):
        tag = f.stem.split("amin3.50A_")[1]
        spec, rule = A0_VARIANTS[tag]
        cl = V.make_closure(spec)
        meta, pops = crc.load(f)
        case, nb = meta["case"], meta["n_bins"]
        acc = {}
        for p, (me, N, M) in pops.items():
            grid = MassGrid(me, crc.rho_of(me, meta["a_min_c"] if p == "graphite" else meta["a_min_si"]))
            key = (p, grid.rho)
            if key not in rates:
                rates[key] = crc.single_grain(crc.POPS[p], grid.a_edges[0], 5e-4)
            a, qs = rates[key]
            for cell, qd in qs.items():
                for q, g in qd.items():
                    e, t2 = totals_variant(grid, N, M, a, g, cl, rule)
                    x = acc.setdefault((cell, q), [0.0, 0.0])
                    x[0] += e
                    x[1] += t2
        for (cell, q), (e, t2) in acc.items():
            r = ref[(case, cell, q)]
            res[(tag, case, nb, cell, q)] = (abs(e / r - 1), abs(t2 / r - 1))
    lines.append("Grain rates on the A0 final states of each variant (PE heating Gamma, recombination cooling Lambda, "
                 "surface S), worst over the cells (WNM, CNM, PDR, dense) and over Gamma/Lambda/S: |run/ref - 1| "
                 "with the variant's own exact in-bin integral / its two nodes")
    tags = [t for t in A0_VARIANTS if any(k[0] == t for k in res)]
    for case in A0_CASES:
        lines.append(case)
        for tag in tags:
            cells = []
            for nb in A0_BINS:
                ks = [k for k in res if k[0] == tag and k[1] == case and k[2] == nb]
                if not ks:
                    cells.append(f"{'-':>13s}")
                    continue
                e = max(res[k][0] for k in ks)
                t2 = max(res[k][1] for k in ks)
                cells.append(f"{e:.3f}/{t2:.3f}".rjust(13))
            lines.append(f"  {tag:9s} " + " ".join(cells))
    save(out, "rates", res, lines)


# ---------------------------------------------------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------------------------------------------------

def local_moments(spec, N, r, uR, ks):
    """<u^k> per cell (one bin per cell) for local closures; spec in pl2m, tc:L, aff, exp."""
    from pycalima.gsd.pl2m import beta_max
    n = N.size
    U = np.full(n, uR)
    if spec == "pl2m":
        eta = uR ** 3
        bm = beta_max(eta)
        lam, sp = V._pl2m_nb(N, r, U, np.full(n, -3 * bm + 2), np.full(n, 3 * bm + 2))
        nn = N[:, None].copy()
        u1 = np.where(sp, r ** (1 / 3), 1.0)[:, None]
        u2 = np.where(sp, r ** (1 / 3), uR)[:, None]
        lam, sp = lam[:, None], sp[:, None]
    elif spec.startswith("tc"):
        lam, u1, u2, sp = V._tc_nb(N, r, U, np.minimum(float(spec.split(":")[1]), 3.0 * beta_max(uR ** 3)) + 0 * U)
        nn, lam, u1, u2, sp = N[:, None].copy(), lam[:, None], u1[:, None], u2[:, None], sp[:, None]
    elif spec == "aff":
        nn, lam, u1, u2, sp = V._aff_nb(N, r, U, V.N_SUB)
    else:
        nn, lam, u1, u2, sp = V._exp_nb(N, r, U, V.N_SUB)
    A = V._amplitudes(nn, lam, u1, u2, sp)
    return [V._bin_moments(nn, A, lam, u1, u2, sp, float(k)) for k in ks]


def s_tables(spec, eta, n=4001):
    """Per-moment velocity factors <u>, <u^4>/<u^3> (w ~ a) and the two nodes (m1, m2 in m_L, weight p1) as
    functions of the in-bin state s, for a local closure: what RAMSES would tabulate."""
    t = np.linspace(0.0, 1.0, n)
    s = 0.5 * (1.0 - np.cos(np.pi * t))
    r = 1.0 + s * (eta - 1.0)
    uR = eta ** (1.0 / 3.0)
    Nn = np.ones(n)
    m0, m1, m3, m4 = local_moments(spec, Nn, np.clip(r, 1.0, eta), uR, (0, 1, 3, 4))
    from pycalima.gsd.grid import MassGrid
    g1 = MassGrid(np.array([1.0, eta]), 1.0)
    cl = V.make_closure(spec)
    nodes = {}
    for rule in ("median", "gauss"):
        mm = np.zeros((n, 2))
        pp = np.zeros(n)
        for k in range(n):
            m, w, _, _ = V.nodes(g1, np.array([1.0]), np.array([r[k]]), cl, rule, 1)
            mm[k] = m[0]
            pp[k] = w[0, 0]
        nodes[rule] = (mm, pp)
    return s, m1 / m0, m4 / m3, nodes


def cmd_drift(args):
    import check_inbin_drift as D
    from pycalima.gsd.pl2m import int_pow, solve_beta
    out = Path(args.out)
    g = D.bin_grid()
    eta = g.etas[0]
    uR = eta ** (1.0 / 3.0)
    s0 = 0.138
    lam0 = 3.0 * float(solve_beta(1.0 + s0 * (eta - 1.0), eta)) + 2.0
    x = (np.arange(D.N_CELLS) + 0.5) / D.N_CELLS
    dx = 1.0 / D.N_CELLS
    shape = ((x > 0.1) & (x < 0.3)).astype(float)
    ue = np.geomspace(1.0, uR, D.N_SUB + 1)
    n_sub = np.array([int_pow(lam0, ue[k], ue[k + 1]) for k in range(D.N_SUB)])
    m_sub = g.m_edges[0] * np.array([int_pow(lam0 + 3, ue[k], ue[k + 1]) for k in range(D.N_SUB)])
    u_mid = np.array([int_pow(lam0 + 4, ue[k], ue[k + 1]) / int_pow(lam0 + 3, ue[k], ue[k + 1])
                      for k in range(D.N_SUB)])
    w0 = 1.0
    N0, M0 = shape * n_sub.sum(), shape * m_sub.sum()
    # the step of check_inbin_drift (its CFL number is ~1e-3: n_steps = t_end / (c dx))
    w_max = w0 * uR
    t_end = 0.25 / (w_max - w0)
    c = 0.5 * dx / w_max
    n_steps = int(np.ceil(t_end / (c * dx)))
    cf = (t_end / n_steps) / dx
    Nref = shape[None, :] * n_sub[:, None]
    Mref = shape[None, :] * m_sub[:, None]
    wsub = (w0 * u_mid)[:, None]
    for _ in range(n_steps):
        F = Nref * wsub
        Nref = Nref - cf * (F - np.roll(F, 1, axis=1))
        F = Mref * wsub
        Mref = Mref - cf * (F - np.roll(F, 1, axis=1))
    Nr, Mr = Nref.sum(axis=0), Mref.sum(axis=0)
    mL = g.m_edges[0]
    runs = {}
    for spec in ("pl2m", "tc:16", "tc:8", "tc:4", "aff", "exp"):
        s_t, fN, fM, nd = s_tables(spec, eta)
        N, M = N0.copy(), M0.copy()
        for _ in range(n_steps):
            f = N > 1e-300
            sv = np.clip(np.where(f, (M / np.where(f, N, 1.0) / mL - 1.0) / (eta - 1.0), 0.5), 0.0, 1.0)
            wN, wM = w0 * np.interp(sv, s_t, fN), w0 * np.interp(sv, s_t, fM)
            FN, FM = N * wN, M * wM
            N, M = N - cf * (FN - np.roll(FN, 1)), M - cf * (FM - np.roll(FM, 1))
        runs[f"moments, {spec}"] = (N, M)
        for rule in (("median", "gauss") if spec == "pl2m" else ()):
            mm, pp = nd[rule]
            N, M = N0.copy(), M0.copy()
            for _ in range(n_steps):
                f = N > 1e-300
                sv = np.clip(np.where(f, (M / np.where(f, N, 1.0) / mL - 1.0) / (eta - 1.0), 0.5), 0.0, 1.0)
                m1 = np.interp(sv, s_t, mm[:, 0])
                m2 = np.interp(sv, s_t, mm[:, 1])
                p1 = np.interp(sv, s_t, pp)
                v1, v2 = w0 * m1 ** (1.0 / 3.0), w0 * m2 ** (1.0 / 3.0)    # u = (m / m_L)^(1/3)
                FN = N * (p1 * v1 + (1.0 - p1) * v2)
                FM = N * mL * (p1 * m1 * v1 + (1.0 - p1) * m2 * v2)
                N, M = N - cf * (FN - np.roll(FN, 1)), M - cf * (FM - np.roll(FM, 1))
            runs[f"two nodes, {rule}"] = (N, M)
    lines = [f"Drift of one grain bin (20-53 nm, w ~ a; check_inbin_drift setup, {n_steps} upwind steps) against 64 "
             "sub-bins: L1 of N and M (relative to the totals), mean |ds| where the dust is. Velocities from tables "
             "in s (4001 points) of each local closure"]
    sr = D.s_of(Nr, Mr, g)
    mask = Nr > 1e-3 * Nr.max()
    res = {"x": x, "ref": (Nr, Mr), "init": (N0, M0), "runs": runs, "g": g}
    for k, (N, M) in runs.items():
        eN = np.abs(N - Nr).sum() / Nr.sum()
        eM = np.abs(M - Mr).sum() / Mr.sum()
        es = np.nanmean(np.abs(D.s_of(N, M, g)[mask] - sr[mask]))
        lines.append(f"  {k:22s} N {eN:.3f}  M {eM:.3f}  mean |ds| {es:.3f}")
        res[k] = (eN, eM, es)
    save(out, "drift", res, lines)


# ---------------------------------------------------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------------------------------------------------

def _best(f, n=200, rep=5):
    f()
    best = np.inf
    for _ in range(rep):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def cmd_cost(args):
    out = Path(args.out)
    lines, res = [], {}
    from pycalima.gsd.collisions import CoagulationModel, _terms, bin_nodes
    from pycalima.gsd.pl2m import BinProfile, beta_max
    from pycalima.gsd.remap import remap
    a, g = MK.grid_of(8)
    a_s, dN_s = mrn_samples(5e-7, 2.5e-5)
    N, M = binned(g, a_s, dN_s)
    # a state with fronts: MRN with edges inside bins
    img = lambda x: x + 1e-7    # noqa: E731
    Nn, Mn, r, uR, filled = V._state(g, N, M)
    uRa = np.ascontiguousarray(uR, float)
    da = np.diff(g.a_edges)
    bm = beta_max(g.eta)
    lines.append("Cost at 8 bins (MRN 5-250 nm, both cut-offs inside bins), best of 5 x 200 calls, this laptop "
                 "(numba, one thread). 'closure' = the compiled closure alone; 'remap kern' = the compiled remap of its "
                 "pieces (one growth step, 4 sub-intervals per piece); these two are what a Fortran routine would do "
                 "per cell and step. 'remap API' = both through the Python API (dominated by Python overhead); "
                 "'nodes' = two collision nodes from the generic (bisection) node routine of pl2m_variants (the "
                 "production analytic bin_nodes takes the time quoted above); 'terms' = one collision-operator "
                 "evaluation (coagulation, McKinnon kernel). 'pieces' = non-empty pieces over the 8 bins.")
    timings = {}
    cap_lo, cap_hi = np.full(8, -3 * bm + 2), np.full(8, 3 * bm + 2)
    W = {L: V.window(g, L) for L in (4.0, 8.0, 16.0, np.inf)}
    timings["pl2m"] = dict(kernel=_best(lambda: V._pl2m_nb(Nn, r, uRa, cap_lo, cap_hi)),
                           kernel_prod=_best(lambda: BinProfile(g, N, M)),
                           remap_prod=_best(lambda: remap(g, N, M, img)),
                           nodes_prod=_best(lambda: bin_nodes(g, N, M, widths=True)))
    for spec in REMAP_VARIANTS + ["eaf:1e-6:-3.5"]:
        cl = V.make_closure(spec)
        if spec.startswith("eaf"):
            kern = lambda: V._ea_nb(Nn, r, uRa, da, W[np.inf], 1e-6, -3.5, True)   # noqa: E731
        elif spec.startswith("tc"):
            L = float(spec.split(":")[1])
            kern = lambda L=L: V._tc_nb(Nn, r, uRa, W[L])    # noqa: E731
        elif spec.startswith("ea"):
            parts = spec.split(":")
            eps = float(parts[1])
            L = float(parts[2]) if len(parts) > 2 else np.inf
            kern = lambda eps=eps, L=L: V._ea_nb(Nn, r, uRa, da, W[L], eps, 0.0)   # noqa: E731
        elif spec == "aff":
            kern = lambda: V._aff_nb(Nn, r, uRa, V.N_SUB)              # noqa: E731
        elif spec == "exp":
            kern = lambda: V._exp_nb(Nn, r, uRa, V.N_SUB)              # noqa: E731
        else:
            kern = lambda: V._pl2m_nb(Nn, r, uRa, cap_lo, cap_hi)   # noqa: E731
        d = timings.setdefault(spec, {})
        d["kernel"] = _best(kern)
        d["pieces"] = _best(lambda: cl.pieces(g, N, M))
        d["remap"] = _best(lambda: V.remap(g, N, M, img, cl))
        d["nodes"] = _best(lambda: V.nodes(g, N, M, cl, "median", 1), n=100)
        d["nodes_gauss"] = _best(lambda: V.nodes(g, N, M, cl, "gauss", 1), n=100)
        d["kp"] = cl.pieces(g, N, M).kp
        # the compiled remap kernel alone, for the pieces of this closure (one growth step)
        P = cl.pieces(g, N, M)
        tt_ = np.linspace(0.0, 1.0, 5)
        lo_, hi_ = P.aL[:, None] * P.u1, P.aL[:, None] * P.u2
        with np.errstate(divide="ignore", invalid="ignore"):
            rat = np.where(lo_ > 0, hi_ / np.where(lo_ > 0, lo_, 1.0), 1.0)
        nodes_ = lo_[:, :, None] * rat[:, :, None] ** tt_[None, None, :]
        X_ = img(nodes_)
        args_ = (g.a_edges, float(g.rho), P.aL, P.n, P.A, P.lam, P.spike, nodes_, X_)
        d["remap_kernel"] = _best(lambda: V._remap_pieces(*args_))
        d["active_pieces"] = int(np.sum(P.n != 0))
    # collision terms
    model = CoagulationModel(MK.coag_kernel)
    tt = {}
    tt["prod"] = _best(lambda: _terms(g, N, M, model), n=20)
    for v in [("pl2m", "median", 1), ("pl2m", "gauss", 1), ("pl2m", "median", 2), ("pl2m", "median", 4),
              ("pl2m", "median", 8), ("ea:1e-6", "median", 1), ("ea:1e-6", "gauss", 1)]:
        import pycalima.gsd.collisions as Cm
        with V.patched(V.make_closure(v[0]) if v[0] != "pl2m" else None, rule=v[1], ka=v[2]):
            tt[cv_name(v)] = _best(lambda: Cm._terms(g, N, M, model), n=20)
    res["timings"], res["terms"] = timings, tt
    lines.append(f"  production PL2M: BinProfile {timings['pl2m']['kernel_prod'] * 1e6:.1f} us, remap "
                 f"{timings['pl2m']['remap_prod'] * 1e6:.1f} us, bin_nodes {timings['pl2m']['nodes_prod'] * 1e6:.1f} "
                 f"us; compiled PL2M closure {timings['pl2m']['kernel'] * 1e6:.2f} us")
    lines.append(f"  {'variant':14s} {'pieces':>6s} {'closure [us]':>12s} {'x PL2M':>7s} {'remap kern [us]':>15s} "
                 f"{'x PL2M':>7s} {'remap API [us]':>14s} {'nodes [us]':>11s} {'gauss [us]':>11s}")
    k0 = timings["pl2m"]["kernel"]
    r0 = timings["pl2m"]["remap"]
    for spec in REMAP_VARIANTS + ["eaf:1e-6:-3.5"]:
        d = timings[spec]
        rk0 = timings["pl2m"]["remap_kernel"]
        lines.append(f"  {spec:14s} {d['active_pieces']:6d} {d['kernel'] * 1e6:12.2f} {d['kernel'] / k0:7.2f} "
                     f"{d['remap_kernel'] * 1e6:15.2f} {d['remap_kernel'] / rk0:7.2f} {d['remap'] * 1e6:14.1f} "
                     f"{d['nodes'] * 1e6:11.1f} {d['nodes_gauss'] * 1e6:11.1f}")
    lines.append("  collision terms (coagulation, 8 bins): " + ", ".join(
        f"{k} {v * 1e3:.2f} ms" for k, v in tt.items()))
    save(out, "cost", res, lines)


# ---------------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["growth", "collisions", "kernels", "stress", "a0", "a0report", "rates", "drift",
                                    "cost", "figures"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--variants", default="")
    ap.add_argument("--cases", default="")
    args = ap.parse_args(argv)
    if args.cmd == "figures":
        import bench_pl2m_figures as F
        F.main(Path(args.out))
        return
    {"growth": cmd_growth, "collisions": cmd_collisions, "kernels": cmd_kernels, "stress": cmd_stress, "a0": cmd_a0,
     "a0report": a0_report, "rates": cmd_rates, "drift": cmd_drift, "cost": cmd_cost}[args.cmd](args)


if __name__ == "__main__":
    main()
