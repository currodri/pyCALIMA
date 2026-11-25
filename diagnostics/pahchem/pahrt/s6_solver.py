"""PAHrt S6: the per-cell solver (``gsd.pahrt``) on pahchem's kernels (``pahchem.pahrt_kernels``) against the truth.

- **V1, kernels.** At each test environment, the kernels' rates are put together from the cell's group fluxes,
  slopes, gas and ions, and compared with pahchem's own network at that environment (``Truth.network``, the full
  field), process by process. The error is flow-weighted with the truth's steady state F:
  sum |K - T| F / sum T F. The outputs are then compared, full and nested.
- **V2, one molecule.** The solver against S1's saved truth on its 214 nodes, per molecule.

The cell's group inputs come from pahchem's field at each node: photon flux and mean photon energy per group (from
1 eV, pahchem's lowest).

    python diagnostics/pahchem/pahrt/s6_solver.py kernels --workers 3   # the study's molecules -> s6/kernels
    python diagnostics/pahchem/pahrt/s6_solver.py v1
    python diagnostics/pahchem/pahrt/s6_solver.py v2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path("results/pahrt_study/s6")
KDIR = OUT / "kernels"


def tag(base):
    """``base`` with the kernels' directory when it is not the default (``--kdir``), so runs do not overwrite."""
    return base if KDIR == OUT / "kernels" else f"{base}_{Path(KDIR).name}"
#: V1's environments: every S1 phase under every S1 field shape, at the phase's G0 and at 10^2 x it; the power laws
#: (1-13.6 eV, slopes on the bases' nodes) are exactly representable, so their errors are the kernels' alone
V1_FIELDS = ("draine", "draine+old", "kurucz10000", "kurucz20000", "kurucz40000", "powerlaw1.0", "powerlaw3.0")
#: outputs scored, and those compared in ratio (the rest in difference)
SCORED = ("f_m1", "f_0", "f_p1", "f_p2", "z_mean", "heating", "cooling", "absorbed", "carbon", "h2", "h_uptake",
          "e_source", "alpha_Cp", "dnh", "intact", "bare")
IN_RATIO = ("heating", "cooling", "absorbed", "carbon", "h2", "h_uptake", "alpha_Cp")


def groups():
    from pycalima.gsd.charging import AGN_GROUP_L0, AGN_GROUP_L1

    return np.asarray(AGN_GROUP_L0, float), np.asarray(AGN_GROUP_L1, float)


def group_inputs(env, L0, L1):
    """(F_g [cm^-2 s^-1], mean photon energy [eV]) of pahchem's field at ``env`` in each group, from 1 eV."""
    from pycalima.pahchem import population as P

    f = P._unit_field(P.field_shape(env))
    E, n = f.energy_ev, f.photon_flux * env.g0
    F, e = np.zeros(L0.size), np.sqrt(L0 * L1)
    for g, (lo, hi) in enumerate(zip(L0, L1, strict=True)):
        k = (E >= max(lo, P.E_PH_MIN_EV)) & (E <= hi)
        if k.sum() < 2:
            continue
        F[g] = np.trapezoid(n[k], E[k])
        if F[g] > 0:
            e[g] = np.trapezoid(n[k] * E[k], E[k]) / F[g]
    return F, e


def cell_of(phase, L0, L1):
    from pycalima.gsd.pahrt import PAHrtCell

    env = phase.environment()
    F, e = group_inputs(env, L0, L1)
    return PAHrtCell(F, e, phase.T, phase.n_e, phase.n_hi, tuple(phase.ions)), env


def solver_outputs(o, z_top=2):
    """The solver's outputs under ``common.OUTPUTS``' names."""
    out = {"f_m1": o.get("f(-1)", 0.0), "f_0": o.get("f(+0)", 0.0), "f_p1": o.get("f(+1)", 0.0),
           "f_p2": o.get("f(+2)", 0.0), "alpha_Cp": o.get("alpha(C+)", 0.0), "alpha_Hp": o.get("alpha(H+)", 0.0)}
    for k in ("z_mean", "heating", "cooling", "absorbed", "carbon", "h2", "h_uptake", "e_source", "bare",
              "dehydrogenated", "intact", "superh", "dnh"):
        out[k] = o[k]
    return out


#: the bins scored (``--bins``; the largest molecules' bases take ~10 min each)
BINS = (0, 1, 2, 3)


def test_molecules():
    """The middle molecule of each bin's five (bins ``BINS``)."""
    return [ms[len(ms) // 2] for b, ms in C.molecules(5).items() if b in BINS]


# ---------------------------------------------------------------------------------------------------------- kernels
def _build(job):
    from pycalima.pahchem import pahrt_kernels as PK

    mol_name, kdir = job
    mols = {m.name: m for ms in C.molecules(5).values() for m in ms}
    mol = mols[mol_name]
    path = Path(kdir) / f"{PK.slug(mol)}.npz"
    if path.exists():
        return mol_name, 0.0
    t = time.time()
    L0, L1 = groups()
    PK.build_molecule(mol, L0, L1).save(path)
    return mol_name, time.time() - t


def kernels(workers, which, prepare=False, only=None):
    from concurrent.futures import ProcessPoolExecutor

    KDIR.mkdir(parents=True, exist_ok=True)
    mols = test_molecules() if which == "test" else [m for b, ms in C.molecules(5).items() if b in BINS for m in ms]
    if only:
        mols = [m for m in mols if m.name.split()[0] in only]
    if prepare:                                   # pahchem's bases per (molecule, shape) first, in parallel
        from pycalima.pahchem import pahrt_kernels as PK

        L0, L1 = groups()
        for name, a, lo, hi, sec in PK.prepare_bases(mols, L0, L1, workers=workers):
            print(f"prepared {name} alpha {a:g} [{lo:g}, {hi:g}] eV: {sec:.0f} s", flush=True)
        workers = min(workers, len(mols))
    jobs = [(m.name, str(KDIR)) for m in sorted(mols, key=lambda m: -m.nc)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for name, s in ex.map(_build, jobs):
                print(f"{name}: {s:.0f} s", flush=True)
    else:
        for job in jobs:
            name, s = _build(job)
            print(f"{name}: {s:.0f} s", flush=True)


#: the multiphoton proxy scored (``--proxy``, ``gsd.pahrt.MoleculeSolver.PROXIES``)
PROXY = "self"


def load_solver(mol):
    from pycalima.gsd.pahrt import MoleculeSolver
    from pycalima.pahchem import pahrt_kernels as PK

    return MoleculeSolver(PK.MoleculeKernels.load(KDIR / f"{PK.slug(mol)}.npz"), PROXY)


# ---------------------------------------------------------------------------------------------------------------- V1
def v1_environments():
    from dataclasses import replace

    out = []
    for name, ph in C.PHASES.items():
        for f in V1_FIELDS:
            for x in (1.0, 100.0):
                if ph.g0 * x <= 1e6:                     # pahchem's k_H tables end at G0 1e6
                    out.append((f"{name} {f} x{x:g}", replace(ph, field=f, g0=ph.g0 * x)))
    return out


def truth_rates(net, ms):
    """{proc key: rate[Z, N]} of pahchem's network on the solver's states."""
    iz = [list(ms.z).index(z) for z in net.charges]
    out = {}
    for p in net.processes:
        r = out.setdefault((p.name, int(p.dz), int(p.dnh), int(p.h2_released)), np.zeros((ms.z.size, ms.nh.size)))
        r[iz, :len(net.nh_values)] += p.rate
    return out


def v1():
    L0, L1 = groups()
    envs = v1_environments()
    rep = []
    res = {}
    for mol in test_molecules():
        ms = load_solver(mol)
        truth = C.Truth(mol)
        errs = {}
        outs = []
        for label, ph in envs:
            cell, env = cell_of(ph, L0, L1)
            net, W = truth.network(env)
            Ft = np.zeros((ms.z.size, ms.nh.size))
            iz = [list(ms.z).index(z) for z in net.charges]
            Ft[iz, :len(net.nh_values)] = net.steady_state_closed()
            to = C.outputs(net, W, mol.nh, truth.ion_rates(env), env)
            Kp, _ = ms.photon(cell)
            Kc, Ki, _ = ms.collisions(cell)
            K = Kp + Kc + sum((n * k for n, k in Ki.values()), np.zeros_like(Kp))
            T = truth_rates(net, ms)
            for p, key in enumerate(ms.mk.procs):
                tr = T.get(key, np.zeros_like(K[p]))
                den = float((tr * Ft).sum())
                num = float((np.abs(K[p] - tr) * Ft).sum())
                if den > 0 or num > 0:
                    errs.setdefault(key[0], []).append((label, num / den if den > 0 else np.inf, den))
            _, of = ms.solve(cell, "full")
            _, on = ms.solve(cell, "nested")
            outs.append((label, to, solver_outputs(of), solver_outputs(on)))
        res[mol.name] = (errs, outs, mol.nc)
        rep.append(f"\n=== {mol.name} (N_C {mol.nc}) ===")
        rep.append("process: flow-weighted relative error of the kernels' rates, median / max over the "
                   f"{len(envs)} environments (only where the process flows)")
        for name, v in sorted(errs.items(), key=lambda kv: -max(e for _, e, d in kv[1] if d > 0) if any(
                d > 0 for _, _, d in kv[1]) else 0):
            e = np.array([x for _, x, d in v if d > 0])
            if e.size:
                worst = max((x, lab) for lab, x, d in v if d > 0)
                rep.append(f"  {name:48s} {np.median(e):9.2e} {e.max():9.2e}  worst: {worst[1]}")
        rep.append("output: solver / truth (ratio outputs) or solver - truth; full | nested; median |dev| and worst")
        for k in SCORED:
            for m, col in (("full", 2), ("nested", 3)):
                d = np.array([dev(o[1][k], o[col][k], k, mol.nc) for o in outs])
                i = int(np.nanargmax(np.abs(d)))
                rep.append(f"  {k:10s} {m:6s} median {np.nanmedian(np.abs(d)):9.3g}  worst {d[i]:+9.3g} "
                           f"({outs[i][0]})")
    text = "\n".join(rep)
    (OUT / f"{tag('v1_kernels')}.txt").write_text(text)
    print(text)
    np.save(OUT / f"{tag('v1_results')}.npy", dict(res), allow_pickle=True)
    fig_v1(res, envs)


#: per-molecule rates below these are negligible (one event per molecule in 10 Gyr; carbon: N_C atoms in 10 Gyr):
#: both sides are floored there before the ratio, so that ratios of negligible rates do not dominate the scores
TEN_GYR_S = 3.156e17


#: the net electron source per molecule below which it is round-off around zero (ion-free cells); 1e-12 s^-1 per
#: molecule at x_PAH ~ 1e-7 is 1e-19 per H, far below cosmic-ray ionisation
E_SOURCE_FLOOR = 1e-12


def floor_of(k, nc):
    if k == "e_source":
        return E_SOURCE_FLOOR
    if k == "carbon":
        return nc / TEN_GYR_S
    if k in ("h2", "h_uptake"):
        return 1.0 / TEN_GYR_S
    return 0.0


def dev(t, s, k, nc=0):
    """log10(s / t) for ratio outputs (floored, ``floor_of``; 0 where both vanish; the electron source in magnitude),
    s - t otherwise."""
    if k == "e_source":
        t, s = abs(t), abs(s)
    if k in IN_RATIO or k == "e_source":
        f = max(floor_of(k, nc), 1e-300)
        if t <= f and s <= f:
            return 0.0
        return float(np.log10(max(s, f) / max(t, f)))
    return float(s - t)


def fig_v1(res, envs):
    import matplotlib.pyplot as plt
    import plots

    plots.setup()
    mols = list(res)
    fig, axes = plt.subplots(len(mols), 2, figsize=(13, 3.2 * len(mols)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    labels = [lab for lab, _ in envs]
    for r, name in enumerate(mols):
        errs, outs, nc = res[name]
        procs = [p for p, v in errs.items() if any(d > 0 for _, _, d in v)]
        M = np.full((len(procs), len(labels)), np.nan)
        for i, p in enumerate(procs):
            for lab, e, d in errs[p]:
                if d > 0:
                    M[i, labels.index(lab)] = np.log10(max(e, 1e-8))
        ax = axes[r, 0]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=-6, vmax=0)
        ax.set_yticks(range(len(procs)), [p[:34] for p in procs], fontsize=6)
        ax.set_xticks(range(0, len(labels), 2), [labels[i].replace(" x1", "") for i in range(0, len(labels), 2)],
                      rotation=90, fontsize=5)
        ax.set_title(f"{name}: kernels vs pahchem, log10 flow-weighted rate error", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax = axes[r, 1]
        D = np.array([[dev(o[1][k], o[2][k], k, nc) for o in outs] for k in SCORED])
        im = ax.imshow(np.clip(D, -1, 1), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(len(SCORED)), SCORED, fontsize=6)
        ax.set_xticks([])
        ax.set_xlabel("the same environments", fontsize=7)
        ax.set_title("full solver vs truth: log10 ratio (rates) or difference", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag('v1_kernels')}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- V2
def v2(method="full"):
    import s1_drivers as S1

    L0, L1 = groups()
    nodes = S1.design()
    rep, res = [], {}
    for mol in [m for ms in C.molecules(5).values() for m in ms]:
        try:
            ms = load_solver(mol)
        except FileNotFoundError:
            continue
        d = np.load(S1.OUT / f"{S1.safe(mol.name)}.npz")
        truth = d["res"]
        D = np.full((len(nodes), len(SCORED)), np.nan)
        for i, (_, _, _, ph) in enumerate(nodes):
            if not np.isfinite(truth[i]).all():
                continue
            cell, _ = cell_of(ph, L0, L1)
            try:
                _, o = ms.solve(cell, method)
            except Exception as exc:                       # one node, not the run
                print(f"{mol.name} node {i}: {exc}")
                continue
            so = solver_outputs(o)
            t = dict(zip(C.OUTPUTS, truth[i], strict=True))
            D[i] = [dev(t[k], so[k], k, mol.nc) for k in SCORED]
        res[mol.name] = D
        rep.append(f"{mol.name:16s} " + " ".join(f"{k}:{np.nanmedian(np.abs(D[:, j])):.2g}/{np.nanmax(np.abs(D[:, j])):.2g}"
                                                 for j, k in enumerate(SCORED)))
    text = (f"median/max |log10 ratio| (rates) or |difference| over S1's nodes, {method} solver vs truth\n"
            + "\n".join(rep))
    name = tag(method if PROXY == "self" else f"{method}_{PROXY}")
    (OUT / f"v2_molecules_{name}.txt").write_text(text)
    print(text)
    np.save(OUT / f"v2_results_{name}.npy", res, allow_pickle=True)
    fig_v2(res, nodes, name)


def fig_v2(res, nodes, method):
    """Median and worst |deviation| per molecule and output, and the worst per S1 input swept."""
    import matplotlib.pyplot as plt
    import plots

    plots.setup()
    names = list(res)
    inputs = list(dict.fromkeys(n[1] for n in nodes))
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.35 * len(names) + 2.5))
    for ax, stat, title in ((axes[0], np.nanmedian, "median"), (axes[1], np.nanmax, "worst")):
        M = np.array([[stat(np.abs(res[n][:, j])) for j in range(len(SCORED))] for n in names])
        im = ax.imshow(np.log10(np.maximum(M, 1e-4)), aspect="auto", cmap="magma_r", vmin=-4, vmax=1)
        ax.set_xticks(range(len(SCORED)), SCORED, rotation=70, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(f"{title} $|$dev$|$ over S1's nodes (log10)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    M = np.array([[np.nanmax(np.abs(np.concatenate([res[n][i:i + 1, :] for n in names
                                                     for i in range(len(nodes)) if nodes[i][1] == inp])[:, j]))
                   for j in range(len(SCORED))] for inp in inputs])
    im = axes[2].imshow(np.log10(np.maximum(M, 1e-4)), aspect="auto", cmap="magma_r", vmin=-4, vmax=1)
    axes[2].set_xticks(range(len(SCORED)), SCORED, rotation=70, fontsize=7)
    axes[2].set_yticks(range(len(inputs)), inputs, fontsize=8)
    axes[2].set_title("worst $|$dev$|$ per input swept (all molecules)", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.04)
    fig.suptitle(f"PAHrt V2: {method} solver vs pahchem; log10 ratio for rates (and $|$e source$|$), difference for "
                 "shares and charge", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"v2_molecules_{method}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- sweeps (V2 figure)
def sweeps(phase="PDR", inputs=("g0", "n_hi")):
    """dnh and carbon loss along S1's one-at-a-time sweeps of ``phase``: pahchem against the solver with each
    multiphoton proxy, per bin 1-2 molecule (where a stripping transition moves, and by how much)."""
    import matplotlib.pyplot as plt
    import plots
    import s1_drivers as S1

    from pycalima.gsd.pahrt import MoleculeSolver

    L0, L1 = groups()
    nodes = S1.design()
    mols = [m for b, ms in C.molecules(5).items() if b in BINS for m in ms]
    proxies = ("self", "power")
    plots.setup()
    fig, axes = plt.subplots(2 * len(inputs), len(mols), figsize=(2.6 * len(mols), 2.3 * 2 * len(inputs)),
                             sharex="col", squeeze=False)
    io = C.OUTPUTS.index
    colours = {"self": "C0", "power": "C3"}
    for c, mol in enumerate(mols):
        try:
            base = load_solver(mol)
        except FileNotFoundError:
            continue
        sol = {p: MoleculeSolver(base.mk, p) for p in proxies}
        truth = np.load(S1.OUT / f"{S1.safe(mol.name)}.npz")["res"]
        floor = mol.nc / TEN_GYR_S
        for r0, inp in enumerate(inputs):
            idx = [i for i, n in enumerate(nodes) if n[0] == phase and n[1] == inp]
            x = np.array([nodes[i][2] for i in idx], float)
            o = {p: [solver_outputs(sol[p].solve(cell_of(nodes[i][3], L0, L1)[0])[1]) for i in idx] for p in proxies}
            ax_d, ax_c = axes[2 * r0, c], axes[2 * r0 + 1, c]
            ax_d.plot(x, truth[idx, io("dnh")], "k-", lw=2, label="pahchem")
            ax_c.plot(x, np.maximum(truth[idx, io("carbon")], floor), "k-", lw=2)
            for p in proxies:
                ax_d.plot(x, [v["dnh"] for v in o[p]], "--", color=colours[p], lw=1.4, label=p)
                ax_c.plot(x, [max(v["carbon"], floor) for v in o[p]], "--", color=colours[p], lw=1.4)
            plots.style(ax_d, logx=True)
            plots.style(ax_c, logx=True, logy=True)
            if c == 0:
                ax_d.set_ylabel(r"$\Delta N_{\rm H}$")
                ax_c.set_ylabel(r"C lost [s$^{-1}$]")
            ax_c.set_xlabel({"g0": r"$G_0$", "n_hi": r"$n({\rm H\,I})$ [cm$^{-3}$]"}[inp])
        axes[0, c].set_title(mol.name.split()[0], fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(rf"PAHrt: the {phase} phase's sweeps; carbon floored at $N_{{\rm C}}$ per 10 Gyr", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag(f'v2_sweeps_{phase}')}.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- V5
def v5():
    """What a lean per-cell solver would cost per molecule: non-zero kernel entries (photon per group and slope;
    multiphoton ones, whose kappa varies over the G0 nodes by > 1 %, keep their G0 table), collision entries per
    kind, the full network's bandwidth (N_H-major ordering) and the flops of fold + banded solve; and the Python time
    of one solve for reference."""
    import time as _t

    L0, L1 = groups()
    rep = ["molecule        states  photon nz (lin/mp)  coll nz  band  flops: fold  solve   memory [kB]  python [ms]"]
    rows = []
    for mol in [m for b, ms in C.molecules(5).items() if b in BINS for m in ms]:
        try:
            ms = load_solver(mol)
        except FileNotFoundError:
            continue
        mk = ms.mk
        nz, nn = len(mk.charges), len(mk.nh_values)
        pk = mk.photon_k[mk.valid]                       # [G, A, N0, P, Z, N]
        nzmask = (pk != 0).any(axis=2)                   # [G, A, P, Z, N]
        hi, lo = pk.max(axis=2), pk.min(axis=2)
        mp = nzmask & (hi > 1.01 * np.where(lo > 0, lo, hi))
        n_lin, n_mp = int((nzmask & ~mp).sum()), int(mp.sum())
        n_coll = int((mk.coll_k != 0).any(axis=1).sum())
        # bandwidth of the generator in N_H-major order (state = n * nz + z)
        offs = [abs(dnh * nz + dz) for (_, dz, dnh, _) in mk.procs]
        band = max(offs)
        n_states = nz * nn
        # fold: per slope pair one multiply-add per linear entry (2 slopes blended), multiphoton entries also a
        # log-log interpolation (~10 flops); collisions: 2 (T interpolation) + 1 (density) per entry
        fold = 2 * 2 * n_lin / mk.alphas.size * 2 + 12 * n_mp / mk.alphas.size * 2 + 3 * n_coll
        solve = n_states * band ** 2
        mem = 8 * (n_lin + n_mp * mk.g0.size + n_coll * mk.lt.size) / 1e3
        cell, _ = cell_of(C.PHASES["PDR"], L0, L1)
        t = _t.perf_counter()
        for _ in range(3):
            ms.solve(cell)
        py = (_t.perf_counter() - t) / 3 * 1e3
        rows.append((mol.name, n_states, n_lin, n_mp, n_coll, band, fold, solve, mem, py))
        rep.append(f"{mol.name:16s} {n_states:6d}  {n_lin:8d}/{n_mp:<8d} {n_coll:8d}  {band:4d}  {fold:10.3g}  "
                   f"{solve:8.3g}  {mem:10.0f}  {py:10.1f}")
    text = "\n".join(rep)
    (OUT / f"{tag('v5_cost')}.txt").write_text(text)
    print(text)
    return rows


def _q(R, c):
    """median / p90 / max of column c."""
    return f"{np.median(R[:, c]):.3g}/{np.percentile(R[:, c], 90):.3g}/{R[:, c].max():.3g}"


def v5b():
    """The per-cell work a lean solver would do on S1's nodes, per molecule (median / 90 % / max over the nodes):
    linear kernel entries in the live states (one multiply-add each, at the run's slope), multiphoton entries past
    their linear range ("self": each group's G0 is its own flux, so an entry is linear below the G0 where its kappa
    leaves its first node's by 1 %), collision entries, and a banded solve on the live states (charges with
    f(Z) > 1e-10 and N_H levels with p > 1e-12 of the full solution, the set a nested predictor would give)."""
    import s1_drivers as S1

    L0, L1 = groups()
    nodes = S1.design()
    rep = ["per cell (median / p90 / max over S1's nodes): live states, flops of fold and solve, total"]
    out = {}
    for mol in [m for b, ms in C.molecules(5).items() if b in BINS for m in ms]:
        try:
            ms = load_solver(mol)
        except FileNotFoundError:
            continue
        mk = ms.mk
        nz, nn = len(mk.charges), len(mk.nh_values)
        pk = mk.photon_k                                            # [G, A, N0, P, Z, N]
        k0 = pk[:, :, :1]
        dev = np.abs(pk - k0) > 0.01 * np.abs(k0) + 1e-300
        # the first G0 node at which each entry leaves its linear value (inf: never)
        first = np.where(dev.any(axis=2), mk.g0[np.argmax(dev, axis=2)], np.inf)     # [G, A, P, Z, N]
        nonzero = (pk != 0).any(axis=2)
        coll_nz = (mk.coll_k != 0).any(axis=(0, 1))                  # [P, Z, N]
        offs = np.array([dnh * 1 for (_, _, dnh, _) in mk.procs])
        dzs = np.array([dz for (_, dz, _, _) in mk.procs])
        rows = []
        for _, _, _, ph in nodes:
            cell, _ = cell_of(ph, L0, L1)
            try:
                F, _ = ms.solve(cell)
            except Exception:                                         # one node, not the run
                continue
            fz, fn = F.sum(axis=1), F.sum(axis=0)
            zl = np.nonzero(fz > 1e-10)[0]
            nl = np.nonzero(fn > 1e-12)[0]
            z0, z1, n0, n1 = zl.min(), zl.max(), nl.min(), nl.max()
            live = np.zeros((nz, nn), bool)
            live[z0:z1 + 1, n0:n1 + 1] = True
            nze, nne = z1 - z0 + 1, n1 - n0 + 1
            lin = mp = 0
            for g in np.nonzero(mk.valid & (cell.F > 0))[0]:
                a = int(np.argmin(np.abs(mk.e_mean[g] - cell.egy[g])))
                g0s = cell.F[g] / mk.f_unit[g, a]
                ent = nonzero[g, a] & live[None]
                act = ent & (first[g, a] <= g0s)
                mp += int(act.sum())
                lin += int((ent & ~act).sum())
            coll = int((coll_nz & live[None]).sum()) * (len(mk.coll_kinds))
            band = max(abs(int(dnh) * nze + int(dz)) for dz, dnh in zip(dzs, offs, strict=True))
            solve = nze * nne * band ** 2
            fold = 2 * lin + 25 * mp + 3 * coll
            rows.append((nze * nne, fold, solve, fold + solve))
        R = np.array(rows, float)
        out[mol.name] = R
        rep.append(f"{mol.name:16s} states {_q(R, 0)}  fold {_q(R, 1)}  solve {_q(R, 2)}  total {_q(R, 3)}")
    text = "\n".join(rep)
    (OUT / f"{tag('v5b_cell_work')}.txt").write_text(text)
    print(text)
    return out


def main(argv=None):
    global BINS, KDIR, PROXY
    os.environ.setdefault("CALIMA_WORKERS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("kernels", "v1", "v2", "v5", "v5b", "sweeps"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--molecules", choices=("test", "all"), default="test")
    ap.add_argument("--only", nargs="+", default=None, help="molecule formulas (e.g. C89H27) to build, of --bins")
    ap.add_argument("--bins", type=int, nargs="+", default=list(BINS))
    ap.add_argument("--method", choices=("full", "nested"), default="full")
    ap.add_argument("--kdir", type=Path, default=None, help="the kernels' directory (default s6/kernels)")
    ap.add_argument("--proxy", choices=("self", "power", "rate"), default="self")
    ap.add_argument("--prepare", action="store_true", help="build pahchem's bases per (molecule, shape) in parallel "
                    "first (for the large molecules)")
    a = ap.parse_args(argv)
    BINS = tuple(a.bins)
    PROXY = a.proxy
    if a.kdir is not None:
        KDIR = a.kdir
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "kernels":
        kernels(a.workers, a.molecules, a.prepare, a.only)
    elif a.what == "v1":
        v1()
    elif a.what == "v5":
        v5()
    elif a.what == "sweeps":
        sweeps()
    elif a.what == "v5b":
        v5b()
    else:
        v2(a.method)


if __name__ == "__main__":
    main()
