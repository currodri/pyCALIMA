#!/usr/bin/env python3
"""Track O2: can the GSD's materials and bins reproduce the Milky Way's (and the Magellanic Clouds') dust? (A3c)

The user's request (2026-11-23): make sure our choices do not rule out the basic constraints the benchmark models
fit. O1 showed the binning alone is harmless; O2 asks whether the GSD's materials, on its grids, can fit the data.

Targets (the user's choice, 2026-11-24):
- Milky Way extinction per H: the Gordon et al. (2023) R_V = 3.1 average curve (``dust_extinction``), normalised
  with Hensley & Draine (2021)'s adopted N_H / E(B-V) = 8.8e21 cm^-2 (A_V / N_H = 3.52e-22 mag cm^2), 0.1-3 um;
- Milky Way emission per H: the Hensley & Draine (2023) Astrodust+PAH model at U = 1 (their model output, fitted to
  Hensley & Draine 2021's compilation: a model, not data), in photometric bands from 3.6 to 850 um;
- LMC and SMC: the shapes of the Gordon et al. (2003, 2024) average curves (A(lambda)/A(V); no normalisation).
Uncertainties (labelled): 5 % on extinction, 10 % on emission. Abundance limits on the dust mass per H mass
(labelled): carbon 2.4e-3 (200 ppm C), silicate 5.5e-3 (32 ppm Si as MgFeSiO4).

Materials: carbon THEMIS-like a-C(:H) (DustEM's CM20 optics, which ours reproduce to 0.9 %) on ``CarbonLayout``'s
grid; astrosilicate (DustEM's aSil) on 8 bins over 0.45 nm - 1 um. Control: graphite (DustEM's Gra) for carbon.
All carbon is CM20 here, as in THEMIS (the molecular bins' PAH band emission comes from pahchem, not tested here).

Method:
1. Basis: every bin at five in-bin states (PL2M profiles, s = 0.05 ... 0.95) is a DustEM grain type with a fixed
   mass per H; one DustEM run gives each one's extinction and emission (Mathis ISRF, U = 1).
2. Fit: the components' masses (non-negative, within the abundance limits) by weighted least squares (SLSQP).
3. Check: each bin's N and M from the fit are rebuilt by EA (the GSD's representation) and run through DustEM again;
   its residuals are what the GSD itself would give.

    python diagnostics/gsd/check_o2_fit.py [--out results/gsd_a3c/o2_fit.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import check_o1_binning as O1
import dustem_tools as D
import numpy as np

WAVE_EXT = np.geomspace(0.1, 3.0, 30)                    # um
BANDS = (3.6, 8.0, 12.0, 24.0, 60.0, 100.0, 160.0, 250.0, 350.0, 500.0, 850.0)
SIG_EXT, SIG_SED = 0.05, 0.10
AV_PER_NH = 3.1 / 8.8e21                                  # mag cm^2
LIMITS = {"carbon": 2.4e-3, "silicate": 5.5e-3}
STATES = (0.05, 0.15, 0.27, 0.39, 0.5, 0.61, 0.73, 0.85, 0.95)
N_PER_BIN = 16
MPROP = 1e-4
MAX_TYPES = 50                                            # grain types per DustEM run
HD23 = Path("reference_data/obs_dust_2026-11-23/hensley_draine_2023_astrodust_pah/astrodust+PAH_MW_RV3.1")


def interp_log(x, y, xi):
    return np.exp(np.interp(np.log(xi), np.log(x), np.log(np.maximum(y, 1e-300))))


def mw_targets():
    from astropy import units as u
    from dust_extinction.parameter_averages import G23

    ext = np.asarray(G23(Rv=3.1)(WAVE_EXT * u.micron)) * AV_PER_NH / 1.0857      # cm^2 / H
    wav = np.loadtxt(HD23 / "wav.dat")
    logu = np.loadtxt(HD23 / "logU.dat")
    irem = np.loadtxt(HD23 / "model_irem.dat")
    row = irem[:, int(np.argmin(np.abs(logu)))]                     # U = 1: lambda I_lambda / N_H [1000 wavelengths]
    sed = 4.0 * np.pi * interp_log(wav, row, np.array(BANDS))                        # DustEM's 4 pi nu I_nu / N_H
    return ext, sed


def shape_target(name):
    from astropy import units as u
    from dust_extinction import averages

    curve = getattr(averages, name)()
    lam = WAVE_EXT[(1.0 / WAVE_EXT >= curve.x_range[0]) & (1.0 / WAVE_EXT <= curve.x_range[1])]
    return lam, np.asarray(curve(lam * u.micron))


def compositions(control=False):
    from pycalima.gsd.carbon import CarbonLayout

    _, sgrid = O1.grids()
    sgrid = type(sgrid)(sgrid.m_edges[:-1], sgrid.rho)                 # without the overflow bin
    cgrid = CarbonLayout(material="graphite" if control else "themis_ac").grid()
    return [("carbon", cgrid, "Gra" if control else "CM20"), ("silicate", sgrid, "aSil")]


def state_profile(grid, i, s, a):
    """a^4 dn/da at radii a of bin i's PL2M profile with one grain at in-bin state s, and that grain's mean mass."""
    from pycalima.gsd.grid import MassGrid

    g1 = MassGrid(grid.m_edges[i:i + 2].copy(), grid.rho)
    mbar = grid.m_edges[i] * (1.0 + s * (g1.etas[0] - 1.0))
    from pycalima.gsd import closures as C

    P = C.pieces(g1, np.array([1.0]), np.array([mbar]), "pl2m")
    out = np.zeros(a.size)
    u = a / P.aL[0]
    for j in range(P.kp):
        if P.n[0, j] > 0 and not P.spike[0, j]:
            sel = (u >= P.u1[0, j]) & (u <= P.u2[0, j])
            out[sel] += P.A[0, j] * u[sel] ** P.lam[0, j] / P.aL[0]
    return out * a ** 4, mbar


def basis(work, comps):
    """{'ext': [n_ext_lam, n_comp], 'sed': [n_bands, n_comp]} per unit mass ratio, the components' (comp, bin, state,
    mean mass), and DustEM's extinction wavelengths."""
    lines, sizes, alias, comp = [], {}, {}, []
    for ci, (name, grid, dtype) in enumerate(comps):
        for i in range(grid.n_bins):
            a = np.geomspace(grid.a_edges[i] * (1 + 1e-6), grid.a_edges[i + 1] * (1 - 1e-6), N_PER_BIN)
            for k, s in enumerate(STATES):
                a4n, mbar = state_profile(grid, i, s, a)
                tag = f"{dtype}{ci}b{i}s{k}"
                sizes[tag] = (a, np.maximum(a4n, 0.0), np.full(a.size, grid.rho))
                alias[tag] = dtype
                lines.append(f"{tag}  {N_PER_BIN}  size  {MPROP:.4e}  {grid.rho:.3e}  {a[0]:.4e}  {a[-1]:.4e}")
                comp.append((name, i, s, mbar))
    D.link_aliases(work, alias)
    ext, sed = [], []
    for k0 in range(0, len(lines), MAX_TYPES):                   # DustEM takes a limited number of types per run
        part = lines[k0:k0 + MAX_TYPES]
        r = D.run(work, "sdist", part, {s.split()[0]: sizes[s.split()[0]] for s in part})
        ext += [interp_log(r["ext_lam"], r["ext_types"][:, j], WAVE_EXT) for j in range(len(part))]
        sed += [interp_log(r["lam"], r["sed_types"][:, j], np.array(BANDS)) for j in range(len(part))]
    return {"ext": np.array(ext).T / MPROP, "sed": np.array(sed).T / MPROP}, comp


#: weight of the smoothness penalty on neighbouring bins' masses (real size distributions do not alternate between
#: empty and filled bins; an isolated bin is carried by EA as a single size), relative to the data's chi^2 per point
SMOOTH = 0.3
C_SCALE = 1e-3                                            # the fit's variables are c / C_SCALE, of order one


def fit(A, T, sig, comp, limits, smooth=SMOOTH):
    """Masses per H mass c_j >= 0 minimising sum ((A c - T) / (sig T))^2 plus a smoothness penalty on the per-bin
    masses of each composition (second differences, weighted relative to the data), with each composition's total
    <= its limit: non-negative least squares with the penalty as extra rows; a limit the solution exceeds is made
    binding by a heavily weighted equality row (an active set of at most two constraints)."""
    from scipy.optimize import nnls

    W = A / (sig * T)[:, None] * C_SCALE
    y = np.ones(T.size) / sig
    keys = sorted({(cc[0], cc[1]) for cc in comp}, key=lambda k: (k[0], k[1]))
    S = np.array([[1.0 if (cc[0], cc[1]) == k else 0.0 for cc in comp] for k in keys])
    rows = []
    for name in {k[0] for k in keys}:
        idx = [n for n, k in enumerate(keys) if k[0] == name]
        for j in range(1, len(idx) - 1):
            r = np.zeros(len(keys))
            r[idx[j - 1]], r[idx[j]], r[idx[j + 1]] = 1.0, -2.0, 1.0
            rows.append(r)
    R = np.array(rows) @ S if rows else np.zeros((0, len(comp)))
    x0, _ = nnls(W, y)
    scale = max(float(np.sqrt(np.mean((S @ x0) ** 2))), 1e-30)
    Rw = np.sqrt(smooth * T.size / len(rows)) * R / scale if rows else R
    active = {}
    for _ in range(4):
        extra_A = [Rw] + [1e4 * np.array([cc[0] == nm for cc in comp], float)[None, :] for nm in active]
        extra_y = [np.zeros(Rw.shape[0])] + [np.array([1e4 * lim / C_SCALE]) for lim in active.values()]
        x, _ = nnls(np.vstack([W, *extra_A]), np.concatenate([y, *extra_y]), maxiter=50 * len(comp))
        over = {nm: lim for nm, lim in (limits or {}).items()
                if nm not in active and sum(x[k] for k, cc in enumerate(comp) if cc[0] == nm) > lim / C_SCALE}
        if not over:
            break
        active.update(over)
    return x * C_SCALE


def gsd_blocks(grid, N, M, n=100):
    """What the GSD carries for bins (N, M), as DustEM grain types: the EA reconstruction's continuous part on n sizes
    over the grid, and each spike (an isolated bin: one size) as a narrow block of its own. [(a, a^4 dn/da, mass)]."""
    from pycalima.gsd import closures as C

    P = C.pieces(grid, N, M, "ea")
    a = np.geomspace(grid.a_edges[0] * (1 + 1e-6), grid.a_edges[-1] * (1 - 1e-6), n)
    cont = np.maximum(O1.rebuild(grid, N, M, a), 0.0)
    blocks, m_spike = [], 0.0
    for i in range(grid.n_bins):
        for j in range(P.kp):
            if P.spike[i, j] and P.n[i, j] > 0:
                a_s = P.aL[i] * P.u1[i, j]
                aa = a_s * np.linspace(0.995, 1.005, 8)
                ms = P.n[i, j] * grid.mass(a_s)
                blocks.append((aa, np.full(aa.size, a_s ** 4), ms))
                m_spike += ms
    return [(a, cont, float(np.sum(M)) - m_spike)] + blocks


def rebuilt_run(work, comps, comp, c):
    """DustEM's extinction and emission of the EA reconstruction of each composition's bins from the fit."""
    lines, sizes, alias = [], {}, {}
    for ci, (name, grid, dtype) in enumerate(comps):
        N, M = np.zeros(grid.n_bins), np.zeros(grid.n_bins)
        for cj, (nm, i, _s, mbar) in zip(c, comp, strict=True):
            if nm == name:
                N[i] += cj / mbar
                M[i] += cj
        if M.sum() <= 0:
            continue
        for b, (a, a4n, mass) in enumerate(gsd_blocks(grid, N, M)):
            if mass <= 0.0 or not np.any(a4n > 0):
                continue
            tag = f"{dtype}R{ci}k{b}"
            sizes[tag] = (a, a4n, np.full(a.size, grid.rho))
            alias[tag] = dtype
            lines.append(f"{tag}  {a.size}  size  {mass:.4e}  {grid.rho:.3e}  {a[0]:.4e}  {a[-1]:.4e}")
    D.link_aliases(work, alias)
    r = D.run(work, "sdist", lines, sizes)
    return interp_log(r["ext_lam"], r["ext"], WAVE_EXT), interp_log(r["lam"], r["sed"], np.array(BANDS))


def fit_states(A, T, sig, comp, comps, limits, smooth=SMOOTH, starts=4, seed=3):
    """What the GSD can carry: one in-bin state s_i per bin (its basis column linear in s between the tabulated
    states) and one mass per bin. For given s the masses follow from ``fit`` on the per-bin columns; the states are
    optimised (L-BFGS-B over [0.05, 0.95], several starts). Returns (c, comp, model): the masses and components in
    ``fit``'s form, one per bin, and the fitted values at the targets."""
    from scipy.optimize import minimize

    keys = sorted({(cc[0], cc[1]) for cc in comp}, key=lambda k: (k[0], k[1]))
    cols = {k: [j for j, cc in enumerate(comp) if (cc[0], cc[1]) == k] for k in keys}
    st = np.array(STATES)
    grids = {name: grid for name, grid, _ in comps}

    def columns(s):
        out, cmp_ = [], []
        for k, sk in zip(keys, s, strict=True):
            j = int(np.clip(np.searchsorted(st, sk) - 1, 0, st.size - 2))
            t = (sk - st[j]) / (st[j + 1] - st[j])
            c0, c1 = cols[k][j], cols[k][j + 1]
            out.append((1.0 - t) * A[:, c0] + t * A[:, c1])
            g = grids[k[0]]
            cmp_.append((k[0], k[1], float(sk), g.m_edges[k[1]] * (1.0 + sk * (g.etas[k[1]] - 1.0))))
        return np.array(out).T, cmp_

    def solve(s):
        Ac, cmp_ = columns(s)
        c = fit(Ac, T, sig, cmp_, limits, smooth)
        r = (Ac @ c - T) / (sig * T)
        return float(r @ r), c, cmp_

    rng = np.random.default_rng(seed)
    best = None
    for n in range(starts):
        s0 = np.full(len(keys), 0.5) if n == 0 else rng.uniform(0.1, 0.9, len(keys))
        res = minimize(lambda s: solve(s)[0], s0, method="L-BFGS-B", bounds=[(st[0], st[-1])] * len(keys),
                       options={"maxiter": 200, "eps": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    _, c, cmp_ = solve(best.x)
    return c, cmp_, columns(best.x)[0] @ c


def report_mw(work, control, lines):
    comps = compositions(control)
    B, comp_all = basis(work, comps)
    ext_t, sed_t = mw_targets()
    A = np.vstack([B["ext"], B["sed"]])
    T = np.concatenate([ext_t, sed_t])
    sig = np.concatenate([np.full(ext_t.size, SIG_EXT), np.full(sed_t.size, SIG_SED)])
    c_mix = fit(A, T, sig, comp_all, LIMITS)
    rm = A @ c_mix / T - 1.0
    c, comp, model = fit_states(A, T, sig, comp_all, comps, LIMITS)
    ext_r, sed_r = rebuilt_run(work, comps, comp, c)
    label = "graphite + astrosilicate (control)" if control else "a-C(:H) + astrosilicate"
    lines.append(f"\n=== Milky Way, {label}")
    for name in LIMITS:
        used = sum(cj for cj, cc in zip(c, comp, strict=True) if cc[0] == name)
        lines.append(f"  {name}: dust mass per H mass {used:.3e} (limit {LIMITS[name]:.1e})")
    rf = model / T - 1.0
    rr = np.concatenate([ext_r, sed_r]) / T - 1.0
    n_e = ext_t.size
    lines.append(f"  chi2 / point: free in-bin mixture {np.mean((rm / sig) ** 2):.2f}, one state per bin (fit) "
                 f"{np.mean((rf / sig) ** 2):.2f}, its GSD rebuild {np.mean((rr / sig) ** 2):.2f}")
    lines.append("  states per bin: " + ", ".join(f"{cc[0][0]}{cc[1]} {cc[2]:.2f}" for cc in comp))
    lines.append("  extinction 0.1-3 um, worst |model/target - 1|: fit "
                 f"{100 * np.max(np.abs(rf[:n_e])):.1f} %, rebuild {100 * np.max(np.abs(rr[:n_e])):.1f} %; at "
                 + ", ".join(f"{w:.3g} um {100 * rr[k]:+.1f}" for k, w in enumerate(WAVE_EXT) if k % 6 == 0))
    lines.append("  emission (rebuild) [%]: " + ", ".join(f"{b:g} um {100 * rr[n_e + k]:+.1f}"
                                                      for k, b in enumerate(BANDS)))
    return comps, B, comp_all


def report_shape(name, comps, B, comp, lines):
    lam, shape = shape_target(name)
    idx = np.isin(WAVE_EXT, lam)
    iv = int(np.argmin(np.abs(WAVE_EXT - 0.55)))
    A = B["ext"][idx] / 1.0                                          # per unit mass; the normalisation is free
    # fit A c = shape * B_V c, i.e. (A - shape B_V) c = 0, with sum c = 1: shape-only least squares
    Bv = B["ext"][iv]
    M = (A - shape[:, None] * Bv[None, :]) / (SIG_EXT * shape[:, None])
    M = M / np.abs(M).max()                                          # scaled for the optimiser
    from scipy.optimize import minimize

    n = len(comp)
    res = minimize(lambda c: 0.5 * np.sum((M @ c) ** 2), np.full(n, 1.0 / n), jac=lambda c: M.T @ (M @ c),
                   bounds=[(0.0, None)] * n, constraints=[{"type": "eq", "fun": lambda c: c.sum() - 1.0}],
                   method="SLSQP", options={"maxiter": 5000, "ftol": 1e-16})
    if not res.success:
        raise RuntimeError(f"{name} shape fit failed: {res.message}")
    c = res.x
    model = (A @ c) / (Bv @ c)
    r = model / shape - 1.0
    carbon = sum(cj for cj, cc in zip(c, comp, strict=True) if cc[0] == "carbon")
    lines.append(f"  {name} shape: worst {100 * np.max(np.abs(r)):.1f} %, chi2/point {np.mean((r / SIG_EXT) ** 2):.2f};"
                 f" carbon share of the dust mass {carbon:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="results/gsd_a3c/dustem")
    ap.add_argument("--out", default="results/gsd_a3c/o2_fit.txt")
    args = ap.parse_args(argv)
    work = Path(args.work)
    lines = ["Track O2: the GSD's materials on its grids fitted to the Milky Way (Gordon+23 extinction per H with",
             "Hensley & Draine 2021's N_H/E(B-V); Hensley & Draine 2023's model emission at U = 1) and to the",
             "LMC/SMC curve shapes (Gordon+03/+24). Errors 5 % (extinction), 10 % (emission); limits on carbon and",
             "silicate dust per H (labelled). 'free in-bin mixture': any mix of in-bin states per bin; 'fit': one",
             "state and one mass per bin (what the GSD carries); 'rebuild': that fit's N and M rebuilt by EA."]
    for control in (False, True):
        comps, B, comp = report_mw(work, control, lines)
        for name in ("G03_LMCAvg", "G03_SMCBar", "G24_SMCAvg"):
            report_shape(name, comps, B, comp, lines)
        print("\n".join(lines[-8:]), flush=True)
    txt = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
