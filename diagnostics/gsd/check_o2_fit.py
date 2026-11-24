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
    ext, sed, ext_f, sed_f = [], [], [], []
    for k0 in range(0, len(lines), MAX_TYPES):                   # DustEM takes a limited number of types per run
        part = lines[k0:k0 + MAX_TYPES]
        r = D.run(work, "sdist", part, {s.split()[0]: sizes[s.split()[0]] for s in part})
        ext += [interp_log(r["ext_lam"], r["ext_types"][:, j], WAVE_EXT) for j in range(len(part))]
        sed += [interp_log(r["lam"], r["sed_types"][:, j], np.array(BANDS)) for j in range(len(part))]
        ext_f.append(r["ext_types"])
        sed_f.append(r["sed_types"])
    return {"ext": np.array(ext).T / MPROP, "sed": np.array(sed).T / MPROP, "lam": r["lam"], "ext_lam": r["ext_lam"],
            "ext_full": np.hstack(ext_f) / MPROP, "sed_full": np.hstack(sed_f) / MPROP}, comp


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
    return interp_log(r["ext_lam"], r["ext"], WAVE_EXT), interp_log(r["lam"], r["sed"], np.array(BANDS)), r


def state_columns(A, comp_all, comps, s):
    """Per-bin columns of A (rows: any quantity) at in-bin states s (one per bin, in sorted (composition, bin)
    order), linear in s between the tabulated states; and the per-bin components in ``fit``'s form."""
    keys = sorted({(cc[0], cc[1]) for cc in comp_all}, key=lambda k: (k[0], k[1]))
    st = np.array(STATES)
    grids = {name: grid for name, grid, _ in comps}
    out, cmp_ = [], []
    for k, sk in zip(keys, s, strict=True):
        cols = [j for j, cc in enumerate(comp_all) if (cc[0], cc[1]) == k]
        j = int(np.clip(np.searchsorted(st, sk) - 1, 0, st.size - 2))
        t = (sk - st[j]) / (st[j + 1] - st[j])
        out.append((1.0 - t) * A[:, cols[j]] + t * A[:, cols[j + 1]])
        g = grids[k[0]]
        cmp_.append((k[0], k[1], float(sk), g.m_edges[k[1]] * (1.0 + sk * (g.etas[k[1]] - 1.0))))
    return np.array(out).T, cmp_


def fit_states(A, T, sig, comp_all, comps, limits, smooth=SMOOTH, starts=4, seed=3):
    """What the GSD can carry: one in-bin state s_i per bin (its basis column linear in s between the tabulated
    states) and one mass per bin. For given s the masses follow from ``fit`` on the per-bin columns; the states are
    optimised (L-BFGS-B over [0.05, 0.95], several starts). Returns (c, comp, s): the masses and components in
    ``fit``'s form, one per bin, and the states."""
    from scipy.optimize import minimize

    n = len({(cc[0], cc[1]) for cc in comp_all})
    st = np.array(STATES)

    def solve(s):
        Ac, cmp_ = state_columns(A, comp_all, comps, s)
        c = fit(Ac, T, sig, cmp_, limits, smooth)
        r = (Ac @ c - T) / (sig * T)
        return float(r @ r), c, cmp_

    rng = np.random.default_rng(seed)
    best = None
    for k in range(starts):
        s0 = np.full(n, 0.5) if k == 0 else rng.uniform(0.1, 0.9, n)
        res = minimize(lambda s: solve(s)[0], s0, method="L-BFGS-B", bounds=[(st[0], st[-1])] * n,
                       options={"maxiter": 200, "eps": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    _, c, cmp_ = solve(best.x)
    return c, cmp_, best.x


def bins_of(comps, comp, c):
    """{composition: (N, M)} per bin from per-bin masses c (dust mass per H mass) and the components' mean masses."""
    out = {}
    for name, grid, _ in comps:
        N, M = np.zeros(grid.n_bins), np.zeros(grid.n_bins)
        for cj, (nm, i, _s, mbar) in zip(c, comp, strict=True):
            if nm == name:
                N[i] += cj / mbar
                M[i] += cj
        out[name] = (N, M)
    return out


def report_mw(work, control, lines):
    """Fit the Milky Way; append the report lines; return the results for the figures."""
    comps = compositions(control)
    B, comp_all = basis(work, comps)
    ext_t, sed_t = mw_targets()
    A = np.vstack([B["ext"], B["sed"]])
    T = np.concatenate([ext_t, sed_t])
    sig = np.concatenate([np.full(ext_t.size, SIG_EXT), np.full(sed_t.size, SIG_SED)])
    c_mix = fit(A, T, sig, comp_all, LIMITS)
    rm = A @ c_mix / T - 1.0
    c, comp, s = fit_states(A, T, sig, comp_all, comps, LIMITS)
    model = state_columns(A, comp_all, comps, s)[0] @ c
    ext_r, sed_r, run = rebuilt_run(work, comps, comp, c)
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
    full_ext = state_columns(B["ext_full"], comp_all, comps, s)[0] @ c
    full_sed = state_columns(B["sed_full"], comp_all, comps, s)[0] @ c
    res = {"label": label, "ext_t": ext_t, "sed_t": sed_t, "fit": model, "rebuild": np.concatenate([ext_r, sed_r]),
           "lam": B["lam"], "ext_lam": B["ext_lam"], "fit_ext_full": full_ext, "fit_sed_full": full_sed,
           "reb_lam": run["lam"], "reb_sed_full": run["sed"], "reb_ext_lam": run["ext_lam"],
           "reb_ext_full": run["ext"]}
    for name, (N, M) in bins_of(comps, comp, c).items():
        grid = next(g for nm, g, _ in comps if nm == name)
        res[f"{name}_N"], res[f"{name}_M"], res[f"{name}_edges"] = N, M, grid.m_edges
        res[f"{name}_rho"] = grid.rho
    return comps, B, comp_all, res


def report_shape(name, comps, B, comp, lines):
    lam, shape = shape_target(name)
    idx = np.isin(WAVE_EXT, lam)
    iv = int(np.argmin(np.abs(WAVE_EXT - 0.55)))
    A = B["ext"][idx]                                                # per unit mass; the normalisation is free
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
    return {"lam": lam, "shape": shape, "model": model}


# --- figures (house style) -------------------------------------------------------------------------------------------

def _style():
    from pycalima.plotting_style import use_calima_style

    use_calima_style()


def _ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in", bottom=True, left=True, top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=2.6)


def hd23_curves():
    """HD23's full emission curve at U = 1 [lambda I_lambda / N_H, erg s^-1 sr^-1 H^-1] and its size distribution
    (a [cm], a^4 dn/da per H for astrodust and PAH [cm^3 H^-1])."""
    wav = np.loadtxt(HD23 / "wav.dat")
    logu = np.loadtxt(HD23 / "logU.dat")
    irem = np.loadtxt(HD23 / "model_irem.dat")[:, int(np.argmin(np.abs(logu)))]
    sd = np.loadtxt(HD23 / "size_distribution.dat")
    a = sd[:, 0] * 1e-4
    dlna = np.log(sd[1, 0] / sd[0, 0])
    return wav, irem, a, a ** 3 * sd[:, 1] / dlna, a ** 3 * sd[:, 2] / dlna


def gsd_curve(N, M, edges, rho, n=400):
    """a [cm] and a^4 dn/da per H [cm^3 H^-1] of the EA reconstruction of bins (N, M) in dust mass per H mass."""
    from pycalima.gsd.grid import MassGrid

    g = MassGrid(np.asarray(edges, float), float(rho))
    a = np.geomspace(g.a_edges[0] * (1 + 1e-6), g.a_edges[-1] * (1 - 1e-6), n)
    return a, np.maximum(O1.rebuild(g, np.asarray(N, float), np.asarray(M, float), a), 0.0) * 1.6735575e-24


def plot(res, shapes, outdir):
    """Three figures: the Milky Way fits (extinction and emission with residuals), the fitted size distributions,
    and the LMC/SMC curve shapes; one column per material set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    cols = {"target": "k", "fit": "#2a78d6", "rebuild": "#eb6834"}
    n_e = WAVE_EXT.size
    wav, irem, a_hd, sd_ad, sd_pah = hd23_curves()

    fig = plt.figure(figsize=(6.0 * len(res), 11.0))
    gs = fig.add_gridspec(5, len(res), height_ratios=[3, 1, 0.55, 3, 1], hspace=0.0, wspace=0.22)
    for j, r in enumerate(res):
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        ax2 = fig.add_subplot(gs[3, j])
        ax2r = fig.add_subplot(gs[4, j], sharex=ax2)
        ax.tick_params(labelbottom=False)
        ax2.tick_params(labelbottom=False)
        ax.plot(WAVE_EXT, r["ext_t"] * 1.0857, color=cols["target"], lw=2.2, label="Gordon+23, $R_V = 3.1$")
        sel = (r["ext_lam"] >= 0.1) & (r["ext_lam"] <= 3.0)
        ax.plot(r["ext_lam"][sel], r["fit_ext_full"][sel] * 1.0857, color=cols["fit"], lw=1.6,
                label="fit: one state per bin")
        sel = (r["reb_ext_lam"] >= 0.1) & (r["reb_ext_lam"] <= 3.0)
        ax.plot(r["reb_ext_lam"][sel], r["reb_ext_full"][sel] * 1.0857, color=cols["rebuild"], lw=1.6, ls="--",
                label="its EA rebuild")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 3.0)
        ax.set_title(r["label"].replace("+", r"$+$"), fontsize=14)
        axr.axhline(0.0, color="0.5", lw=0.8)
        axr.fill_between(WAVE_EXT, -100 * SIG_EXT, 100 * SIG_EXT, color="0.9")
        axr.plot(WAVE_EXT, 100 * (r["fit"][:n_e] / r["ext_t"] - 1.0), color=cols["fit"], lw=1.4)
        axr.plot(WAVE_EXT, 100 * (r["rebuild"][:n_e] / r["ext_t"] - 1.0), color=cols["rebuild"], lw=1.4, ls="--")
        axr.set_ylim(-35, 35)
        axr.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=14)
        axr.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0])
        axr.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        axr.xaxis.set_minor_formatter(NullFormatter())
        ax2.plot(wav, irem, color=cols["target"], lw=2.2, label="Hensley \\& Draine (2023) model, $U = 1$")
        ax2.plot(r["lam"], r["fit_sed_full"] / (4 * np.pi), color=cols["fit"], lw=1.6)
        ax2.plot(r["reb_lam"], r["reb_sed_full"] / (4 * np.pi), color=cols["rebuild"], lw=1.6, ls="--")
        ax2.plot(BANDS, r["sed_t"] / (4 * np.pi), "o", mfc="white", mec=cols["target"], ms=6, label="fitted bands")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(2.0, 1500.0)
        ax2.set_ylim(1e-28, 3e-24)
        ax2r.axhline(0.0, color="0.5", lw=0.8)
        ax2r.fill_between([2.0, 1500.0], -100 * SIG_SED, 100 * SIG_SED, color="0.9")
        ax2r.plot(BANDS, 100 * (r["fit"][n_e:] / r["sed_t"] - 1.0), "o-", color=cols["fit"], ms=4, lw=1.2)
        ax2r.plot(BANDS, 100 * (r["rebuild"][n_e:] / r["sed_t"] - 1.0), "s--", color=cols["rebuild"], ms=4, lw=1.2)
        ax2r.set_ylim(-90, 90)
        ax2r.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=14)
        if j == 0:
            ax.set_ylabel(r"$A_\lambda / N_{\rm H}$ [mag cm$^2$]", fontsize=14)
            axr.set_ylabel(r"$\Delta$ [\%]", fontsize=13)
            ax2.set_ylabel(r"$\lambda I_\lambda / N_{\rm H}$ [erg s$^{-1}$ sr$^{-1}$ H$^{-1}$]", fontsize=14)
            ax2r.set_ylabel(r"$\Delta$ [\%]", fontsize=13)
            ax.legend(frameon=False, fontsize=11)
            ax2.legend(frameon=False, fontsize=11, loc="upper right")
        for a_ in (ax, axr, ax2, ax2r):
            _ticks(a_)
    fig.savefig(outdir / "o2_milky_way.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(res), figsize=(6.0 * len(res), 4.6), sharey=True, gridspec_kw={"wspace": 0.05})
    axes = np.atleast_1d(axes)
    for j, r in enumerate(res):
        ax = axes[j]
        ax.plot(a_hd * 1e4, sd_ad, color="0.6", lw=1.4, label="HD23 astrodust")
        ax.plot(a_hd * 1e4, sd_pah, color="0.6", lw=1.4, ls=":", label="HD23 PAH")
        for name, c in (("carbon", cols["fit"]), ("silicate", cols["rebuild"])):
            a, y = gsd_curve(r[f"{name}_N"], r[f"{name}_M"], r[f"{name}_edges"], r[f"{name}_rho"])
            ax.plot(a * 1e4, y, color=c, lw=1.8, label=f"GSD {name} (EA)")
            edges = (3.0 * np.asarray(r[f"{name}_edges"]) / (4 * np.pi * r[f"{name}_rho"])) ** (1 / 3)
            for e in edges:
                ax.axvline(e * 1e4, color=c, lw=0.4, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-30, 3e-27)
        ax.set_xlabel(r"$a$ [$\mu$m]", fontsize=14)
        ax.set_title(r["label"].replace("+", r"$+$"), fontsize=14)
        if j == 0:
            ax.set_ylabel(r"$a^4\, dn/da / n_{\rm H}$ [cm$^3$ H$^{-1}$]", fontsize=14)
            ax.legend(frameon=False, fontsize=10, loc="lower center")
        _ticks(ax)
    fig.savefig(outdir / "o2_size_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    names = list(shapes[0])
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.2), sharey=True, gridspec_kw={"wspace": 0.05})
    for k, nm in enumerate(names):
        ax = axes[k]
        sh = shapes[0][nm]
        ax.plot(1.0 / sh["lam"], sh["shape"], color=cols["target"], lw=2.2, label=nm.replace("_", " "))
        for j, (r, c) in enumerate(zip(res, (cols["fit"], cols["rebuild"]), strict=False)):
            ax.plot(1.0 / shapes[j][nm]["lam"], shapes[j][nm]["model"], color=c, lw=1.6, ls="-" if j == 0 else "--",
                    label=r["label"].replace("+", r"$+$"))
        ax.set_xlabel(r"$1/\lambda$ [$\mu$m$^{-1}$]", fontsize=14)
        if k == 0:
            ax.set_ylabel(r"$A_\lambda / A_V$", fontsize=14)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        _ticks(ax)
    fig.savefig(outdir / "o2_magellanic_shapes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="results/gsd_a3c/dustem")
    ap.add_argument("--out", default="results/gsd_a3c/o2_fit.txt")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from the saved results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    store = out.with_suffix(".npz")
    if not args.plot_only:
        work = Path(args.work)
        lines = ["Track O2: the GSD's materials on its grids fitted to the Milky Way (Gordon+23 extinction per H with",
                 "Hensley & Draine 2021's N_H/E(B-V); Hensley & Draine 2023's model emission at U = 1) and to the",
                 "LMC/SMC curve shapes (Gordon+03/+24). Errors 5 % (extinction), 10 % (emission); limits on carbon",
                 "and silicate dust per H (labelled). 'free in-bin mixture': any mix of in-bin states per bin; 'fit':",
                 "one state and one mass per bin (what the GSD carries); 'rebuild': that fit's N and M rebuilt by EA."]
        res, shapes = [], []
        for control in (False, True):
            comps, B, comp, r = report_mw(work, control, lines)
            res.append(r)
            shapes.append({nm: report_shape(nm, comps, B, comp, lines)
                           for nm in ("G03_LMCAvg", "G03_SMCBar", "G24_SMCAvg")})
            print("\n".join(lines[-8:]), flush=True)
        txt = "\n".join(lines)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt + "\n")
        flat = {f"{j}:{k}": v for j, r in enumerate(res) for k, v in r.items()}
        flat.update({f"{j}:shape:{nm}:{k}": v for j, sh in enumerate(shapes) for nm, d in sh.items()
                     for k, v in d.items()})
        np.savez(store, **flat)
        print(txt)
    z = np.load(store, allow_pickle=False)
    n = 1 + max(int(k.split(":")[0]) for k in z.files)
    res = [{k.split(":", 1)[1]: (str(z[k]) if z[k].dtype.kind == "U" else z[k]) for k in z.files
            if k.startswith(f"{j}:") and ":shape:" not in k} for j in range(n)]
    shapes = []
    for j in range(n):
        d = {}
        for k in z.files:
            if k.startswith(f"{j}:shape:"):
                _, _, nm, q = k.split(":")
                d.setdefault(nm, {})[q] = z[k]
        shapes.append(d)
    plot(res, shapes, out.parent / "figs")
    print(out.parent / "figs")


if __name__ == "__main__":
    main()
