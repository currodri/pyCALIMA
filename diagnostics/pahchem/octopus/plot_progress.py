#!/usr/bin/env python3
"""Progress and Malloci+07 comparison plots for the Octopus PAH runs (``make_run.py``, ``queue.py``).

Reads a local mirror of megatron's ``~/pahspec`` (``--sync`` refreshes it with rsync, without the restart files) and
builds every spectrum from the kicks' ``td.general/multipoles`` itself, as ``oct-propagation_spectrum`` does
(polynomial damping, sine transform; equal to Octopus's ``cross_section_tensor`` to 1e-8). A stopped or running
run is transformed up to the time all three kicks have reached, so its resolution is that of the shorter
propagation, ~2 pi hbar / T. It is set against Malloci's spectrum broadened to the same T (``broaden``), as well as
against Malloci's own.

Figures (``--out``):
- ``progress.png``: steps done per run, the time step's wall-clock cost and an ETA; energy drift per kick;
- ``spectra_<molecule>.png``: sigma(E) per C and the cumulative oscillator strength to 30 eV per charge, against
  Malloci (the ``malloci2007.INCONSISTENT`` files marked) and Li & Draine (``photo.sigma_li_draine``);
- ``convergence.png``: grid spacing at a common T, and the method tests (time step, kick, box, pseudopotentials,
  T, anion box) as band-integral ratios;
- ``band_rates.png``: Mathis-ISRF absorption rates per C in 1-5 and 5-13.6 eV.
A summary table (``compare_malloci.summary`` per run) goes to stdout.

    plot_progress.py [--sync megatron] [--root ~/Documents/RAMSES_dev/pahspec/results/runs]
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_malloci import summary  # noqa: E402

from pycalima.pahchem import fields, photo  # noqa: E402
from pycalima.pahchem import malloci2007 as M  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

HA_EV, BOHR_A, C_AU = 27.211386245988, 0.529177210903, 137.035999084
E_GRID = np.round(np.arange(0.0, 30.0 + 1e-9, 0.05), 10)
MIN_T = 2.0               # hbar/eV: below this (resolution > ~3 eV) a partial spectrum is not shown
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]     # validated (dataviz validate_palette.js, light)
STYLES = ["-", "--", "-.", ":"]
RUN_RE = re.compile(r"^(?:malloci_|conv_)?([a-z]+)_(-?\d)(?:_(.*))?$")
ROOT = Path.home() / "Documents/RAMSES_dev/pahspec/results/runs"
OUT = Path(__file__).resolve().parents[3] / "results/plots/pahchem/octopus"
RSYNC = ["rsync", "-az", "--prune-empty-dirs", "--exclude=restart/", "--include=*/", "--include=multipoles",
         "--include=multipoles.*", "--include=out*.log", "--include=inp", "--include=cross_section_tensor",
         "--include=run.log", "--include=jobs.txt", "--include=queue.log", "--exclude=*"]


# ------------------------------------------------------------------ reading runs
def settings(run):
    """The td input of a run: spacing [A], T and dt [hbar/eV], kick [1/A], radius and absorber [A], pseudopotentials."""
    txt = (run / "td_x" / "inp").read_text()
    num = lambda key: float(re.search(rf"{key}\s*=\s*([-\d.eE+]+)", txt).group(1))  # noqa: E731
    return dict(h=num("Spacing"), T=num("TDPropagationTime"), dt=num("TDTimeStep"), kick=num("TDDeltaStrength"),
                radius=num("Radius"), absorb=num("ABWidth"), charge=int(num("ExcessCharge")),
                pseudo=re.search(r"PseudopotentialSet\s*=\s*(\S+)", txt).group(1))


def step_log(td):
    """(iteration, energy [eV], wall-clock seconds per step, mtime of the newest log) from a kick's out*.log files."""
    rows, mtime = {}, None
    for log in sorted(td.glob("out*.log"), key=lambda p: p.stat().st_mtime):
        mtime = log.stat().st_mtime
        for line in log.read_text(errors="replace").splitlines():
            f = line.split()
            if len(f) == 5 and f[0].isdigit():
                try:
                    rows[int(f[0])] = (float(f[2]), float(f[4]))
                except ValueError:
                    pass
    if not rows:
        return None
    it = np.array(sorted(rows))
    e, s = np.array([rows[i] for i in it]).T
    return it, e, s, mtime


def load_multipoles(path):
    """Time [hbar/eV], dipole [A] (spins summed; 3 components) and the kick [1/bohr] of one kick's multipoles."""
    lines = path.read_text().splitlines()
    head = {ln.split()[1]: ln.split()[-1] for ln in lines if ln.startswith("# ") and len(ln.split()) > 2}
    nspin, kick = int(head["nspin"]), float(lines[[ln.startswith("# kick strength") for ln in lines].index(True)]
                                            .split()[-1])
    d = np.loadtxt([ln for ln in lines if not ln.startswith("#")], ndmin=2)
    dip = sum(d[:, 3 + 4 * s:6 + 4 * s] for s in range(nspin))
    return d[:, 1], dip, kick


def damping(t, T):
    """Octopus's PropagationSpectrumDampMode = polynomial."""
    x = t / T
    return 1 - 3 * x ** 2 + 2 * x ** 3


def octopus_sigma(run, t_max=None, damp=damping):
    """(E [eV], (1/3) Tr sigma [cm^2], T used [hbar/eV]) from the three kicks, up to the time all have reached
    (or ``t_max``), with the window ``damp(t, T)``; None if any kick has no multipoles yet."""
    kicks = []
    for j, axis in enumerate("xyz"):
        p = run / f"td_{axis}" / "td.general" / "multipoles"
        if not p.exists():
            p = run / "spectrum" / f"multipoles.{j + 1}"
        if not p.exists():
            return None
        kicks.append(load_multipoles(p))
    T = min(t[-1] for t, _, _ in kicks)
    T = T if t_max is None else min(T, t_max)
    if T <= 0:
        return None
    w = E_GRID / HA_EV
    tr = 0.0
    for j, (t, dip, kick) in enumerate(kicks):
        k = t <= T + 1e-9
        ta, dd = t[k] * HA_EV, (dip[k, j] - dip[0, j]) / BOHR_A
        im_alpha = -np.trapezoid(dd * damp(t[k], T) * np.sin(np.outer(w, ta)), ta, axis=1) / kick
        tr = tr + 4 * np.pi * w * im_alpha / C_AU * BOHR_A ** 2
    return E_GRID, tr / 3 * 1e-16, T


def broaden(E, s, T):
    """sigma(E) as a propagation of length T (polynomial damping) would give it.

    sigma ~ E Im alpha, and Im alpha is odd in E; a damping D(|t|) of the causal response convolves Im alpha with
    K(E) = (1/pi) int_0^T D(t) cos(E t) dt. Checked against a T = 10 run truncated to 5 and 2 hbar/eV: 3 % peak to
    peak, the integral to 0.05 %."""
    dE = E[1] - E[0]
    g = np.where(E > 0, s / np.where(E > 0, E, 1.0), 0.0)
    g_full = np.concatenate([-g[:0:-1], g])
    lags = np.arange(-(len(g_full) - 1), len(g_full)) * dE
    t = np.linspace(0.0, T, 2001)
    K = np.trapezoid(damping(t, T) * np.cos(np.outer(lags, t)), t, axis=1) / np.pi
    conv = np.convolve(g_full, K, mode="full")[len(g_full) - 1:2 * len(g_full) - 1] * dE
    return E * conv[len(E) - 1:]


# ------------------------------------------------------------------ run bookkeeping
def queue_state(root):
    """{run name: (order in jobs.txt, started by queue.log)}"""
    order, started = {}, set()
    jobs = root / "jobs.txt"
    if jobs.exists():
        for ln in jobs.read_text().splitlines():
            if ln.strip() and not ln.startswith("#"):
                cmd = ln.split()[1:]
                d = Path(cmd[-1]) if len(cmd) > 1 else Path(cmd[0]).parent
                order.setdefault(d.name, len(order) + 1)
    log = root / "queue.log"
    if log.exists():
        started = {Path(m).name for m in re.findall(r" start (\S+)", log.read_text())}
    return order, started


def runs(root):
    out = {}
    for d in sorted(p for p in root.iterdir() if (p / "td_x" / "inp").exists()):
        m = RUN_RE.match(d.name)
        if m:
            out[d.name] = dict(dir=d, mol=m.group(1), tag=m.group(3) or "", **settings(d))
            out[d.name]["charge"] = int(m.group(2))
    return out


def tex(s):
    return s.replace("_", r"\_") if plt.rcParams["text.usetex"] else s


def ang():
    return r"\AA" if plt.rcParams["text.usetex"] else "Å"


def once(ax, label):
    """``label`` the first time it is used on ``ax``, None after (one legend entry per marker kind)."""
    return None if label in ax.get_legend_handles_labels()[1] else label


def drop_top_tick(ax):
    """No major tick label at the top edge, where a stacked panel's bottom label sits (hspace = 0)."""
    top = ax.get_ylim()[1]
    ax.set_yticks([t for t in ax.get_yticks() if ax.get_ylim()[0] <= t < 0.96 * top])
    ax.set_ylim(top=top)


def plain(s):
    return s.replace("\n", " ").replace(r"\AA", "A").replace("$", "")


def style(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in", top=True, right=True)


def run_label(r, T_done):
    lab = rf"$h = {r['h']:.2f}$ {ang()}, $T = {r['T']:g}$"
    if r["tag"] and not r["tag"].startswith("h"):
        lab += f" ({tex(r['tag'])})"
    if "box" in r["tag"]:
        lab += f" (edge {r['radius'] - r['absorb']:g} {ang()})"
    if T_done < r["T"] - 1e-6:
        lab += rf", partial: $T = {T_done:.1f}$ of ${r['T']:g}$"
    return lab


# ------------------------------------------------------------------ figures
def fig_progress(R, root, out):
    order, started = queue_state(root)
    names = sorted(R, key=lambda n: (order.get(n, 0), n))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.32 * len(names) + 2.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    report = []
    for i, n in enumerate(names):
        r = R[n]
        total = int(round(r["T"] / r["dt"]))
        done_ok = (r["dir"] / "run.log").exists() and (r["dir"] / "run.log").read_text().rstrip().endswith("RUN_OK")
        logs = [step_log(r["dir"] / f"td_{a}") for a in "xyz"]
        it = [lg[0][-1] if lg else 0 for lg in logs]
        frac = np.array(it) / total
        rate = [np.median(lg[2][-100:]) for lg in logs if lg and len(lg[2]) > 5]
        if done_ok:
            state = "done" if min(it) >= total else f"done, stopped at T = {min(it) * r['dt']:.1f}"
            color = "0.35"
        elif n in started:
            state, color = ("running" if any(it) else "ground state"), COLORS[0]
        elif any(it):
            state, color = "stopped", COLORS[1]
        else:
            state, color = f"queued \\#{order[n]}" if n in order and plt.rcParams["text.usetex"] else \
                (f"queued #{order[n]}" if n in order else "not queued"), "0.7"
        ax.barh(i, frac.mean(), color=color, height=0.6)
        ax.plot([frac.min(), frac.max()], [i, i], color="k", lw=1)
        txt = f"{min(it)}/{total}  {state}"
        if rate and not done_ok:
            s = max(rate)
            txt += f", {s:.2f} s/step"
            if state == "running":
                mt = max(lg[3] for lg in logs if lg)
                eta = mt + (total - min(it)) * s
                txt += ", ETA " + time.strftime("%a %H:%M", time.localtime(eta))
        ax.text(1.02, i, txt, va="center", fontsize=9, transform=ax.get_yaxis_transform())
        report.append(f"  {n:32s} {txt}".replace("\\#", "#"))
        lg = logs[0]
        # x kick; Octopus first updates the logged energy at step 10 (steps 1-9 repeat the pre-kick value)
        if lg is not None and (lg[0] >= 10).sum() > 2:
            k, w = len(ax2.get_lines()), lg[0] >= 10
            ax2.plot(lg[0][w] * r["dt"], (lg[1][w] - lg[1][w][0]) * 1e3, color=COLORS[k % 4],
                     ls=STYLES[k // 4 % 4], lw=1.2, label=tex(n))
    ax.set_yticks(range(len(names)), [tex(n) for n in names], fontsize=9)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of time steps done (bar: mean of the 3 kicks; line: range)", fontsize=12)
    ax2.set_xlabel(r"$t$ [$\hbar$/eV]", fontsize=14)
    ax2.set_ylabel(r"$E_{\rm tot}(t) - E_{\rm tot}$(step 10) [meV], $x$ kick", fontsize=14)
    ax2.legend(frameon=False, fontsize=7, ncol=2)
    for a in (ax, ax2):
        style(a)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    fig.suptitle(f"Octopus runs, mirror of {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest(root)))}",
                 fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.75)
    fig.savefig(out / "progress.png", dpi=200)
    plt.close(fig)
    return report


def newest(root):
    return max(p.stat().st_mtime for p in root.rglob("out.log"))


def production(R, mol, z):
    """Runs of (mol, Z) at the production settings (0.25 A, standard box and pseudopotentials), any T."""
    return [n for n, r in R.items() if r["mol"] == mol and r["charge"] == z and not n.startswith("conv_")
            and abs(r["h"] - 0.25) < 1e-6 and "box" not in r["tag"] and r["pseudo"] == "standard"]


def fig_spectra(R, mol, S, served, out, table):
    nc, nh = S[mol]["nc"], S[mol]["nh"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True, gridspec_kw=dict(hspace=0, wspace=0.22))
    E = E_GRID[1:]
    for row, z in enumerate((-1, 0, 1)):
        ax, axf = axes[row]
        n_val = M.valence_electrons(nc, nh, z)
        ref = S[mol]["sigma"].get(z)
        if ref is not None:
            ax.plot(ref[0], ref[1] / nc / M.MB_CM2, "k-", lw=1.6, label="Malloci+07 ($T = 20$)")
            fM = np.array([M.oscillator_strength(*ref, 0, e) for e in ref[0]])
            axf.plot(ref[0], fM, "k-", lw=1.6, label=f"Malloci+07: $f_{{30}} = {fM[-1]:.1f}$")
            axf.axhspan(0.95 * fM[-1], 1.05 * fM[-1], color="0.85", lw=0, label=r"pass: $\pm 5$ per cent")
        bad = served.get((mol, z))
        if bad is not None:
            ax.plot(bad[0], bad[1] / nc / M.MB_CM2, color="0.45", ls=(0, (1, 1)), lw=1.2,
                    label="Malloci served file, excluded (" + tex(M.INCONSISTENT[(mol, z)].split(",")[0]) + ")")
            fb = M.oscillator_strength(*bad, 0, 30)
            axf.axhline(fb, color="0.45", ls=(0, (1, 1)), lw=1.2, label=f"excluded served file: {fb:.1f}")
        if ref is None:
            axf.axhspan(0.62 * n_val, 0.64 * n_val, color="0.85", lw=0,
                        label=rf"pass: share 0.62--0.64 of {n_val} e$^-$")
        ld = photo.sigma_li_draine(nc, z, E)
        ax.plot(E, ld / nc / M.MB_CM2, color="0.6", lw=1.2,
                label="Li \\& Draine 2001 (" + ("ionised" if z > 0 else "neutral") + ")"
                if plt.rcParams["text.usetex"] else "Li & Draine 2001")
        names = production(R, mol, z)
        if not names:
            ax.text(0.97, 0.9, "no Octopus run yet", ha="right", va="top", transform=ax.transAxes, fontsize=11)
        k_shown = 0
        for n in sorted(names, key=lambda n: R[n]["T"]):
            res = octopus_sigma(R[n]["dir"])
            if res is None or res[2] < MIN_T:
                T_now = 0 if res is None else res[2]
                ax.plot([], [], " ", label=f"{tex(n)}: $T = {T_now:.1f}$ so far, not shown")
                continue
            Eo, so, T = res
            c, ls = COLORS[k_shown % 4], STYLES[k_shown % 4]
            ax.plot(Eo, so / nc / M.MB_CM2, color=c, ls=ls, lw=1.4, label="Octopus " + run_label(R[n], T))
            fo = np.array([M.oscillator_strength(Eo, so, 0, e) for e in Eo])
            lab = f"Octopus: $f_{{30}} = {fo[-1]:.1f}$"
            if ref is not None and T < 19.9:
                b = broaden(*ref, T)
                ax.plot(ref[0], b / nc / M.MB_CM2, color=c, ls=(0, (1, 1.5)), lw=1.4,
                        label=f"Malloci+07 broadened to $T = {T:.1f}$")
            if ref is not None:
                lab += f" ({fo[-1] / M.oscillator_strength(*ref, 0, 30):.3f}" + r"$\times$)"
            else:
                lab += f" (share {fo[-1] / n_val:.3f})"
            axf.plot(Eo, fo, color=c, ls=ls, lw=1.4, label=lab)
            k_shown += 1
            table.append(summary_rows(mol, z, n, Eo, so, T, R[n], ref, n_val))
        ax.set_ylabel(rf"$\sigma/N_{{\rm C}}$ [Mb], $Z = {z:+d}$", fontsize=14)
        axf.set_ylabel(r"$f(<E)$", fontsize=14)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        axf.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        axf.set_ylim(0, axf.get_ylim()[1] * 1.2)
        if row:
            drop_top_tick(ax)
            drop_top_tick(axf)
    for a in axes.flat:
        style(a)
        a.set_xlim(0, 30)
    for a in axes[-1]:
        a.set_xlabel(r"$E$ [eV]", fontsize=14)
    fig.suptitle(f"{mol} (C$_{{{nc}}}$H$_{{{nh}}}$): Octopus 16.4 against Malloci+07 and Li \\& Draine"
                 if plt.rcParams["text.usetex"] else f"{mol}: Octopus 16.4 against Malloci+07 and Li & Draine",
                 fontsize=14, y=0.93)
    fig.savefig(out / f"spectra_{mol}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def summary_rows(mol, z, n, E, s, T, r, ref, n_val):
    o = summary(E, s, n_val)
    lines = [f"{mol} Z = {z:+d}, {n} (h {r['h']} A, T {T:.2f} of {r['T']:g} hbar/eV)"]
    refs = []
    if ref is not None:
        refs.append(("Malloci", summary(*ref, n_val)))
        if T < 19.9:
            refs.append((f"Malloci@T={T:.1f}", summary(ref[0], broaden(*ref, T), n_val)))
    for key, v in o.items():
        cells = [f"{v:10.4g}"]
        for _, t in refs:
            cells.append(f"{t[key]:10.4g}" + ("" if key.startswith("peak") else f" ({v / t[key]:.3f})"))
        lines.append(f"  {key:16s} " + "   ".join(cells))
    lines.insert(1, f"  {'':16s} {'run':>10s}   " + "   ".join(f"{nm:>18s}" for nm, _ in refs))
    return "\n".join(lines)


BANDS = (("f30", r"$f_{30}$"), ("f 1-5", "1--5 eV"), ("f 5-13.6", "5--13.6 eV"), ("f 13.6-30", "13.6--30 eV"))


def fig_convergence(R, S, out, report):
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.8, 1.4], wspace=0.32)
    ax, axh, axm = (fig.add_subplot(g) for g in gs)
    nc, nh = 24, 12
    n0 = M.valence_electrons(nc, nh, 0)
    # (a, b) grid spacing at the T all three reached (coronene_0, 0.30 A, was stopped at T = 6.7 of 20)
    grid = [n for n in ("coronene_0", "coronene_0_h0.25", "coronene_0_h0.2") if n in R]
    T0 = min(octopus_sigma(R[n]["dir"])[2] for n in grid)
    ref = S["coronene"]["sigma"][0]
    b = broaden(*ref, T0)
    ax.plot(ref[0], b / nc / M.MB_CM2, "k-", lw=1.6, label=f"Malloci+07 (0.30 {ang()}) broadened to $T = {T0:.1f}$")
    tM = summary(ref[0], b, n0)
    hs = []
    for k, n in enumerate(grid):
        E, s, T = octopus_sigma(R[n]["dir"], T0)
        ax.plot(E, s / nc / M.MB_CM2, color=COLORS[k], ls=STYLES[k], lw=1.3, label=rf"$h = {R[n]['h']:.2f}$ {ang()}")
        hs.append((R[n]["h"], summary(E, s, n0)))
    for j, (key, lab) in enumerate(BANDS):
        h = [x for x, _ in hs]
        axh.plot(h, [o[key] / tM[key] for _, o in hs], marker="osD^"[j], ms=8, color=COLORS[j], ls=STYLES[j],
                 label=lab)
    for h, o in hs:
        report.append(f"  grid h = {h:.2f} A vs Malloci, T = {T0:.2f}: " +
                      ", ".join(f"{key} {o[key] / tM[key]:.3f}" for key, _ in BANDS))
    axh.axhspan(0.95, 1.05, color="0.88", lw=0)
    axh.axhline(1, color="k", lw=0.8)
    axh.set_xlabel(rf"grid spacing $h$ [{ang()}]", fontsize=14)
    axh.set_ylabel(f"Octopus / Malloci+07, both at $T = {T0:.1f}$", fontsize=14)
    axh.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$E$ [eV]", fontsize=14)
    ax.set_ylabel(r"$\sigma/N_{\rm C}$ [Mb], coronene $Z = 0$", fontsize=14)
    ax.legend(frameon=False, fontsize=9)
    # (c) method tests: (label, test run, reference run)
    tests = [("time step\n0.001", "conv_coronene_0_T5_dt0.001", "conv_coronene_0_T5_ref"),
             ("kick\n0.005/" + ang(), "conv_coronene_0_T5_kick0.005", "conv_coronene_0_T5_ref"),
             (f"box\n6+2 {ang()}", "conv_coronene_0_T5_box6ab2", "conv_coronene_0_T5_ref"),
             ("HGH\npseudo", "conv_coronene_0_T5_hgh", "conv_coronene_0_T5_ref"),
             ("$T = 20$\nvs 10", "coronene_0_h0.25_T20", "coronene_0_h0.25"),
             (f"anion box\n7+2 vs 4+1 {ang()}", "coronene_-1_h0.25_box7ab2", "coronene_-1_h0.25")]
    def sigma_of(name, t_max=None):
        if name in R and (res := octopus_sigma(R[name]["dir"], t_max)) is not None:
            return res
        if name == "conv_coronene_0_T5_ref" and "coronene_0_h0.25" in R:
            # the T = 5 reference has coronene_0_h0.25's settings, so that run cut at T = 5 is the same propagation
            return octopus_sigma(R["coronene_0_h0.25"]["dir"], 5.0 if t_max is None else min(5.0, t_max))
        return None

    for i, (lab, test, refn) in enumerate(tests):
        rt, rr = sigma_of(test), sigma_of(refn)
        if rt is None or rr is None or min(rt[2], rr[2]) < MIN_T:
            Tn = 0 if rt is None else rt[2]
            axm.text(i, 1.0, f"pending\n($T = {Tn:.1f}$)", ha="center", va="center", fontsize=8, rotation=90)
            report.append(f"  {plain(lab):28s} pending (T = {Tn:.2f})")
            continue
        T = min(rt[2], rr[2])
        rt, rr = sigma_of(test, T), sigma_of(refn, T)
        z = R[test]["charge"]
        nv = M.valence_electrons(nc, nh, z)
        o, t = summary(*rt[:2], nv), summary(*rr[:2], nv)
        for j, (key, blab) in enumerate(BANDS):
            axm.plot(i + 0.12 * (j - 1.5), o[key] / t[key], marker="osD^"[j], ms=8, color=COLORS[j], ls="none",
                     label=blab if i == 0 or not axm.get_legend_handles_labels()[1] else None)
        report.append(f"  {plain(lab):28s} T = {T:.2f}: " +
                      ", ".join(f"{key} {o[key] / t[key]:.3f}" for key, _ in BANDS))
    axm.axhspan(0.95, 1.05, color="0.88", lw=0, label=r"$\pm 5$ per cent")
    axm.axhline(1, color="k", lw=0.8)
    axm.set_xticks(range(len(tests)), [t[0] for t in tests], fontsize=8.5)
    axm.set_xlim(-0.6, len(tests) - 0.4)
    axm.set_ylabel("test / reference, same $T$", fontsize=14)
    axm.legend(frameon=False, fontsize=8, loc="lower left")
    for a in (ax, axh, axm):
        style(a)
    axm.tick_params(axis="x", which="minor", bottom=False, top=False)
    lo, hi = axm.get_ylim()
    axm.set_ylim(min(lo, 0.9), max(hi, 1.1))
    fig.suptitle(f"Octopus convergence, coronene (grid at $T = {T0:.1f}$; method tests at the $T$ both reached)",
                 fontsize=13)
    fig.savefig(out / "convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_band_rates(R, S, served, out):
    ma = fields.mathis1983(1.0, e_min=0.5)
    E = E_GRID[1:]

    def rate(E, s, lo, hi):
        w = (E >= lo) & (E <= hi)
        return ma.photon_rate(s[w], E[w])

    cases = [(m, z) for m in ("coronene", "pyrene", "ovalene") for z in (-1, 0, 1)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(hspace=0))
    for ax, (lo, hi) in zip(axes, ((1, 5), (5, 13.6))):
        for i, (m, z) in enumerate(cases):
            nc = S[m]["nc"]
            ref = S[m]["sigma"].get(z)
            first = i == 0
            ld_lab = "Li \\& Draine 2001" if plt.rcParams["text.usetex"] else "Li & Draine 2001"
            ax.plot(i - 0.2, rate(E, photo.sigma_li_draine(nc, z, E), lo, hi) / nc / 1e-9, "^", color="0.55", ms=8,
                    label=once(ax, ld_lab))
            if ref is not None:
                ax.plot(i, rate(*ref, lo, hi) / nc / 1e-9, "ks", ms=8, label=once(ax, "Malloci+07 ($T = 20$)"))
            elif (m, z) in served:
                ax.plot(i, rate(*served[(m, z)], lo, hi) / nc / 1e-9, "x", color="0.3", ms=8,
                        label=once(ax, "Malloci served file, excluded"))
            names = [n for n in production(R, m, z)]
            best = None
            for n in names:
                res = octopus_sigma(R[n]["dir"])
                if res is not None and res[2] >= MIN_T and (best is None or res[2] > best[2]):
                    best = res
            if best is not None:
                Eo, so, T = best
                partial = T < 19.9 and not any((R[n]["dir"] / "run.log").exists() and
                                               (R[n]["dir"] / "run.log").read_text().rstrip().endswith("RUN_OK")
                                               and abs(R[n]["T"] - T) < 1e-6 for n in names)
                ax.plot(i + 0.2, rate(Eo, so, lo, hi) / nc / 1e-9, "o", ms=9, color=COLORS[0],
                        mfc="white" if partial else COLORS[0], label=once(ax, "Octopus (open: partial run)"))
                ax.annotate(f"$T = {T:.1f}$", (i + 0.2, rate(Eo, so, lo, hi) / nc / 1e-9), xytext=(6, 0),
                            textcoords="offset points", fontsize=8, va="center")
                if ref is not None and T < 19.9:
                    ax.plot(i + 0.1, rate(ref[0], broaden(*ref, T), lo, hi) / nc / 1e-9, "s", ms=7, mfc="white",
                            mec="k", label=once(ax, "Malloci+07 at the run's $T$"))
        ax.set_ylabel(rf"$k_{{\rm abs}}/N_{{\rm C}}$ [$10^{{-9}}$ s$^{{-1}}$], {lo}--{hi} eV", fontsize=14)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        style(ax)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")
    drop_top_tick(axes[1])
    axes[-1].set_xticks(range(len(cases)), [f"{m}\n$Z = {z:+d}$" for m, z in cases], fontsize=9)
    axes[0].set_title("Absorption rate per C in the Mathis+83 ISRF ($G_0$ scale 1)", fontsize=13)
    fig.savefig(out / "band_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="local mirror of megatron's ~/pahspec")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sync", metavar="HOST", default=None, help="rsync HOST:pahspec/ into --root first")
    a = ap.parse_args()
    if a.sync:
        a.root.mkdir(parents=True, exist_ok=True)
        subprocess.run(RSYNC + [f"{a.sync}:pahspec/", f"{a.root}/"], check=True)
    a.out.mkdir(parents=True, exist_ok=True)
    use_calima_style()
    R = runs(a.root)
    S = M.spectra()
    S_all = M.spectra(include_inconsistent=True)
    served = {key: S_all[key[0]]["sigma"][key[1]] for key in M.INCONSISTENT}
    print("progress:")
    print("\n".join(fig_progress(R, a.root, a.out)))
    table = []
    for mol in sorted({r["mol"] for r in R.values()}):
        fig_spectra(R, mol, S, served, a.out, table)
    conv = []
    fig_convergence(R, S, a.out, conv)
    fig_band_rates(R, S, served, a.out)
    print("\nconvergence tests (test / reference at a common T):")
    print("\n".join(conv))
    print("\nagainst Malloci (ratios in parentheses):")
    print("\n\n".join(table))
    print(f"\nfigures in {a.out}")


if __name__ == "__main__":
    main()
