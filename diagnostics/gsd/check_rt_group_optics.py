"""Per-bin RT group cross-sections in the RAMSES cool step: cheap schemes against the exact EA in-bin average.

How RAMSES does it today:
* ``init_SED_table`` (rt/rt_spectra.f90) tabulates each dust bin's group cross-sections (csa, css, csr) per stellar
  age and metallicity, in the node-shared ``SED_table``;
* ``update_SED_group_props`` luminosity-weights them over the stars every ``sedprops_update`` coarse steps
  (``group_csa_dust`` etc.);
* the cool step copies them into each cell (rtz_cooling_module.f90), and on every substep replaces the IR group's
  column (group 1, 0.1-1 eV) by Planck (csa) and Rosseland (csr) means at the cell's T_rad.

With two moments per bin, a bin's per-gram cross-section depends on the cell's in-bin profile. The exact EA average
costs ~4 us per cell-call (``bench_rt_group_optics.py``) against a 30-80 us cool step. This script scores cheaper
schemes against it:

* one value per bin (PL2M at s = 1/2), the grain at the bin's mean mass, a 3-point polynomial in s;
* **pseudo-bins**: the PL2M per-gram values at n fixed in-bin states (Chebyshev points s_k = (1 - cos(pi k/(n-1)))/2),
  linear in the cell's s. Every pseudo-bin is an ordinary RAMSES dust bin for the SED table, the star polling and the
  IR means; the cell finds its bracket and weight once per cell-call. Adopted 2026-11-18 with n = 5;
* the PL2M profile at the cell's s, integrated exactly: the floor of any scheme that ignores EA's fronts.

Columns: the 8 RTZ groups, photon-weighted over a 5e4 K blackbody, absorption (C_abs) and radiation pressure
(C_rp = C_abs + (1 - g) C_sca); and the IR means, Planck of C_abs and Rosseland of C_rp at T_rad = 10-300 K. Group 1 is
scored through the IR means only, since RAMSES replaces it. Scattering is listed apart: RAMSES's rates never use it
(css is copied into the cell and not read).

States: the A0 EA 8-bin end states (five cases; moved onto the A2 grids at the same N and s), and the stress states of
``check_node_tables`` (front segments cut anywhere in a bin, narrow segments, random). Scores, worst over states:
per bin (bins with >= 1 % of the composition's mass) and per composition total (sum_i M_i kappa_i, what the RT
absorption sees).

Second part, if $CALIMA_SED_DIR points at BPASS v2.2.1 (the RAMSES format): how much the per-gram group values of
each bin depend on the stellar population, i.e. whether one fixed spectrum could replace the polling.

    CALIMA_SED_DIR=/path/to/bpass_v221_cha300 python diagnostics/gsd/check_rt_group_optics.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_node_tables as NT      # noqa: E402
import check_optics_tables as A2    # noqa: E402
from pycalima.gsd import closures as gc   # noqa: E402
from pycalima.gsd.optics import node_u, profile_weights   # noqa: E402

A0_DIR = Path("results/gsd_overview/variants/a0_prod/gsd_bin_convergence")
POP = {"carbon": "graphite", "silicate": "silicate"}      # A2 material -> A0 population
N_FINE = 192                                              # intervals per bin of the exact in-bin integrals
H, C, KB = 6.62607015e-27, 2.99792458e10, 1.380649e-16
EV_UM = 1.23984198                                        # hc [eV um]
GROUP_L0 = np.array([0.1, 1.0, 5.6, 11.2, 13.6, 15.2, 24.59, 54.42])     # RTZ's 8 groups [eV]
GROUP_L1 = np.array([1.0, 5.6, 11.2, 13.6, 15.2, 24.59, 54.42, 500.0])
T_BB = 5e4                                                # K, spectrum of the group weights
T_RAD = (10.0, 30.0, 100.0, 300.0)                        # K, IR means
LAM_G = np.geomspace(EV_UM / GROUP_L1[-1], EV_UM / GROUP_L0[0], 300)    # um
LAM_IR = np.geomspace(2.0, 3000.0, 120)                   # um
N_PSEUDO = (2, 3, 5, 9, 17)
ADOPTED = 5
NG = GROUP_L0.size
# column layout: group g, quantity q (abs, sca, rp) at 3 g + q; then Planck at each T_rad, then Rosseland
COLS = {"groups 2-8 abs": [3 * g for g in range(1, NG)],
        "groups 2-8 rp": [3 * g + 2 for g in range(1, NG)],
        "IR Planck": list(range(3 * NG, 3 * NG + len(T_RAD))),
        "IR Rosseland": list(range(3 * NG + len(T_RAD), 3 * NG + 2 * len(T_RAD))),
        "groups 2-8 sca (unused)": [3 * g + 1 for g in range(1, NG)]}
USED = ("groups 2-8 abs", "groups 2-8 rp", "IR Planck", "IR Rosseland")
# RAMSES's SED table (rt_spectra.f90): ages rebinned at dlgA = 0.02 (~247), SED_nZ = 8, real*8, one copy per node
SED_NA, SED_NZ = 247, 8


def chebyshev_s(n):
    return (1.0 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2.0


def group_weights():
    """[8, lam] photon weights of each group over a T_BB blackbody."""
    E = EV_UM / LAM_G
    x = np.minimum(H * C / (LAM_G * 1e-4) / (KB * T_BB), 700.0)
    nph = LAM_G ** -4 / np.expm1(x) * np.gradient(LAM_G)
    W = np.zeros((NG, LAM_G.size))
    for g in range(NG):
        sel = (E >= GROUP_L0[g]) & (E < GROUP_L1[g])
        W[g, sel] = nph[sel]
    return W / W.sum(1, keepdims=True)


def ir_weights():
    """Planck (B_nu) and Rosseland (dB_nu/dT) weights [T_rad, lam]."""
    nu = C / (LAM_IR * 1e-4)
    dnu = np.abs(np.gradient(nu))
    P, R = [], []
    for T in T_RAD:
        x = np.minimum(H * nu / (KB * T), 700.0)
        B = nu ** 3 / np.expm1(x)
        dB = B * x * np.exp(x) / np.expm1(x)
        P.append(B * dnu / (B * dnu).sum())
        R.append(dB * dnu / (dB * dnu).sum())
    return np.array(P), np.array(R)


WG = group_weights()
WP, WR = ir_weights()


def columns(kg, kir):
    """Single-grain per-gram values kg [n, lam_g, 3], kir [n, lam_ir, 3] -> [n, ncol]."""
    g = np.einsum("gl,nlq->ngq", WG, kg).reshape(kg.shape[0], -1)
    p = np.einsum("tl,nl->nt", WP, kir[..., 0])
    r = 1.0 / np.einsum("tl,nl->nt", WR, 1.0 / kir[..., 2])
    return np.concatenate([g, p, r], axis=1)


def in_bin_s(grid, N, M):
    with np.errstate(all="ignore"):
        s = (M / (N * grid.m_edges[:-1]) - 1.0) / (grid.etas - 1.0)
    return np.clip(np.where(N > 0, s, 0.5), 0.0, 1.0)


def a0_states(grid, pop):
    """The A0 EA 8-bin end states, at the same N and s on the A2 grid."""
    out = []
    for f in sorted(A0_DIR.glob("*_ea_N08_*.npz")):
        z = np.load(f, allow_pickle=True)
        me, N, M = z[f"{pop}:m_edges"], z[f"{pop}:N"], z[f"{pop}:M"]
        with np.errstate(all="ignore"):
            s = np.where(N > 0, (M / (N * me[:-1]) - 1.0) / (me[1:] / me[:-1] - 1.0), 0.0)
        M2 = N * grid.m_edges[:-1] * (1.0 + np.clip(s, 0.0, 1.0) * (grid.etas - 1.0))
        out.append((json.loads(str(z["meta"]))["case"], N.copy(), M2))
    if not out:
        raise SystemExit(f"no A0 EA runs in {A0_DIR}")
    return out


class Tables:
    """Per bin: single-grain columns at N_FINE + 1 radii, and PL2M values at fixed s."""

    def __init__(self, grid, kf):
        self.grid = grid
        self.U = np.stack([node_u(e, N_FINE) for e in grid.etas])
        self.K = np.stack([columns(np.asarray(kf(grid.a_edges[i] * self.U[i], LAM_G)),
                                   np.asarray(kf(grid.a_edges[i] * self.U[i], LAM_IR)))
                           for i in range(grid.n_bins)])                              # [nb, N_FINE + 1, ncol]
        self.fixed = {}

    def at_s(self, sv):
        """PL2M per-gram values at the in-bin states sv, per bin: [nb, len(sv), ncol]."""
        key = tuple(np.round(sv, 12))
        if key not in self.fixed:
            self.fixed[key] = np.stack([profile_weights(e, np.asarray(sv), self.U[i], "pl2m") @ self.K[i]
                                        for i, e in enumerate(self.grid.etas)])
        return self.fixed[key]

    def exact(self, N, M, method):
        W = gc.bin_weights(self.grid, gc.pieces(self.grid, N, M, method), self.U, "mass")
        return np.einsum("in,inc->ic", W, self.K)


def schemes(T, N, M):
    """{scheme: [nb, ncol]} per-gram values of each bin under each cheap scheme."""
    grid = T.grid
    s = in_bin_s(grid, N, M)
    nb, ncol = grid.n_bins, T.K.shape[2]
    out = {"one value per bin (s = 1/2)": T.at_s([0.5])[:, 0]}
    um = (1.0 + s * (grid.etas - 1.0)) ** (1.0 / 3.0)               # radius of the mean-mass grain, in a_L
    out["grain at the mean mass"] = np.array([[np.interp(um[i], T.U[i], T.K[i][:, c]) for c in range(ncol)]
                                              for i in range(nb)])
    sp = np.array([0.0, 0.5, 1.0])
    L = np.stack([np.prod([(s - sp[m]) / (sp[j] - sp[m]) for m in range(3) if m != j], axis=0) for j in range(3)])
    out["3-point polynomial in s"] = np.einsum("ji,ijc->ic", L, T.at_s(sp))
    for n in N_PSEUDO:
        sk = chebyshev_s(n)
        P = T.at_s(sk)
        out[f"{n} pseudo-bins"] = np.array([[np.interp(s[i], sk, P[i][:, c]) for c in range(ncol)]
                                            for i in range(nb)])
    out["PL2M at s, exact"] = T.exact(N, M, "pl2m")
    return out


def score(sets, T):
    """{state set: {scheme: {column group: (worst per bin, worst total, label of the worst total)}}}."""
    res = {}
    for sname, sts in sets.items():
        acc = {}
        for label, N, M in sts:
            filled = N > 1e-30 * N.max()
            big = filled & (M / M.sum() >= 0.01)
            ref = T.exact(N, M, "ea")
            for k, X in schemes(T, N, M).items():
                for cname, idx in COLS.items():
                    eb = float(np.abs(X[big][:, idx] / ref[big][:, idx] - 1.0).max()) if big.any() else 0.0
                    et = float(np.abs((M[filled, None] * X[filled][:, idx]).sum(0)
                                      / (M[filled, None] * ref[filled][:, idx]).sum(0) - 1.0).max())
                    wb, wt, wl = acc.setdefault(k, {}).get(cname, (0.0, 0.0, ""))
                    acc[k][cname] = (max(wb, eb), et if et > wt else wt, label if et > wt else wl)
        res[sname] = acc
    return res


def accuracy_lines():
    rng = np.random.default_rng(5)
    lines = ["Per-gram RT group cross-sections per bin: cheap schemes against the exact EA in-bin average "
             f"(|scheme/EA - 1|, worst over states; per bin with >= 1 % of the mass / composition total). Groups: RTZ's "
             f"8, photon-weighted over a {T_BB:.0e} K blackbody; IR means at T_rad = "
             + ", ".join(f"{t:g}" for t in T_RAD) + " K. Pseudo-bins at Chebyshev s, linear in s."]
    for mat, grid, kf in A2.materials():
        start = len(lines)
        T = Tables(grid, kf)
        sets = {"A0 end states": a0_states(grid, POP[mat]), "stress states": NT.states(grid, rng)}
        for sname, acc in score(sets, T).items():
            lines.append("")
            lines.append(f"{mat}, {sname} ({len(sets[sname])})"
                         + "".join(f"  {c:>23s}" for c in COLS))
            for k, by in acc.items():
                lines.append(f"  {k:28s}" + "".join(f"  {by[c][0]:10.1e} /{by[c][1]:9.1e}  " for c in COLS))
            worst = acc[f"{ADOPTED} pseudo-bins"]
            lines.append(f"  worst total of {ADOPTED} pseudo-bins from: "
                         + "; ".join(f"{c}: {worst[c][2]}" for c in USED))
        print("\n".join(lines[start:]), flush=True)
    return lines


def memory_lines(n_bins_total=16):
    """SED_table dust columns per node (real*8): today csa, css, csr per bin; pseudo-bins csa, csr only."""
    col = SED_NA * SED_NZ * NG * 8
    out = [f"SED_table dust columns ({SED_NA} ages x {SED_NZ} Z x {NG} groups, real*8, one copy per node), "
           f"{n_bins_total} bins: one column {col / 1e3:.0f} kB",
           f"  today (csa, css, csr per bin): {3 * n_bins_total * col / 1e6:.1f} MB"]
    for n in N_PSEUDO:
        out.append(f"  {n:2d} pseudo-bins (csa, csr; css dropped): {2 * n * n_bins_total * col / 1e6:.1f} MB")
    return out


def read_bpass(d):
    from scipy.io import FortranFile
    ages = np.loadtxt(d / "age_bins.dat", skiprows=1)                               # yr
    zs = np.loadtxt(d / "metallicity_bins.dat", skiprows=1)
    f = FortranFile(d / "all_seds.dat", "r")
    nl = f.read_ints(np.int32)[0]
    lam = f.read_reals(np.float64)                                                   # Angstrom
    S = np.array([[f.read_reals(np.float64) for _ in ages] for _ in zs])             # [Z, age, lam]
    assert S.shape[-1] == nl
    return ages, zs, lam, S


def sed_lines(d):
    """Per-gram group abs and rp of each bin (PL2M at s = 0, 1/2, 1) under bursts and constant-SFR mixes."""
    ages, zs, lam_A, S = read_bpass(d)
    spectra = {}
    for zv in (0.002, 0.02):
        iz = int(np.argmin(np.abs(zs - zv)))
        for t in (1e6, 3e6, 1e7, 3e7, 1e8, 1e9):
            j = int(np.argmin(np.abs(np.log10(np.maximum(ages, 1.0)) - np.log10(t))))
            spectra[("burst", f"{t / 1e6:g} Myr, Z = {zv:g}")] = S[iz, j]
        for t in (1e7, 1e8, 1e9):
            sel = ages <= t
            spectra[("constant SFR", f"{t / 1e6:g} Myr, Z = {zv:g}")] = np.trapezoid(S[iz, sel], ages[sel], axis=0)
    ref = ("constant SFR", "100 Myr, Z = 0.02")
    E = EV_UM * 1e4 / lam_A
    lam_um = lam_A * 1e-4
    lam_opt = np.geomspace(EV_UM / GROUP_L1[-1], EV_UM / GROUP_L0[0], 160)
    nph = {k: F * lam_A * np.gradient(lam_A) for k, F in spectra.items()}           # photons per Angstrom bin
    sel = [(E >= GROUP_L0[g]) & (E < GROUP_L1[g]) for g in range(NG)]
    lines = ["", f"Stellar-population dependence ($CALIMA_SED_DIR, BPASS): |kappa_g / kappa_g(ref) - 1| of each bin's "
                 f"per-gram group values, worst over the bins and s = 0, 1/2, 1; reference {ref[0]} over {ref[1]}."]
    for mat, grid, kf in A2.materials():
        worst = {mix: np.zeros((NG, 2)) for mix in ("burst", "constant SFR")}
        for i, e in enumerate(grid.etas):
            u = node_u(e, 32)
            K = np.asarray(kf(grid.a_edges[i] * u, lam_opt))[..., [0, 2]]
            for sv in (0.0, 0.5, 1.0):
                k = np.tensordot(profile_weights(e, [sv], u, "pl2m")[0], K, axes=(0, 0))
                kS = np.stack([np.exp(np.interp(np.log(lam_um), np.log(lam_opt), np.log(k[:, q]))) for q in range(2)], -1)
                vals = {nm: np.array([(kS[s_] * n[s_, None]).sum(0) / n[s_].sum() for s_ in sel])
                        for nm, n in nph.items()}
                for nm, v in vals.items():
                    worst[nm[0]] = np.maximum(worst[nm[0]], np.abs(v / vals[ref] - 1.0))
        for mix, w in worst.items():
            lines.append(f"  {mat:8s} {mix:12s} abs per group " + " ".join(f"{x:5.2f}" for x in w[:, 0])
                         + "  | rp " + " ".join(f"{x:5.2f}" for x in w[:, 1]))
    lines.append("  (groups " + ", ".join(f"{a:g}-{b:g}" for a, b in zip(GROUP_L0, GROUP_L1)) + " eV; group 1 is "
                 "replaced by the IR means in RAMSES)")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/gsd_overview/figs/rt_group_optics.txt")
    args = ap.parse_args(argv)
    lines = accuracy_lines() + [""] + memory_lines()
    d = os.environ.get("CALIMA_SED_DIR")
    if d and Path(d).is_dir():
        lines += sed_lines(Path(d))
    else:
        lines += ["", "Stellar-population dependence skipped: set $CALIMA_SED_DIR to BPASS v2.2.1 (RAMSES format)."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
