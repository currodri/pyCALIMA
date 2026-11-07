#!/usr/bin/env python3
"""PAH charge and the electron fraction in dark clouds against Tielens (2021) and Flower & Pineau des Forets
(2026) (validation V15).

Targets (Tielens 2021, Molecular Astrophysics, §10.5.3, and FPdF 2026 §3, as read in
calima-pdr/docs/literature/pah_gas/tielens2021_notes.md and supplied2_notes.md):
- eq. 10.77, x_e ~ 2e-7 (zeta/3e-17)^1/2 (1e4/n(H2))^1/2, "insensitive to which carrier dominates";
- Fig. 10.22 (x_PAH 1e-7, metals 1e-6): PAH- take over the negative charge from n ~ 3e3, and the PAHs lower
  x_e by ~10x; §10.5.3: PAH- dominate when x_e < 2e-7 (x_PAH/1e-7);
- FPdF 2026 (n_H = 1e4, T = 10 K, zeta = 5e-17 and 3e-16, x_PAH = 1e-8, 1e-7, 1e-6, attachment fixed at
  1e-6): PAH- out-neutralise the electrons only when x(PAH-)/x_e >= (m_p/m_e)^1/2 ~ 43, reached only for
  x_PAH >~ 1e-6. The paper's curves are not tabulated in the repo docs, so only this criterion is compared.

Model: a minimal dark-cloud balance, steady state, no photons (G0 = 0), the gas fully molecular.
- Cosmic rays ionise H2 at zeta per H2; every ionisation ends as one molecular ion M+ (H3+ -> HCO+ with CO),
  lumped as HCO+ (29 amu, IP(HCO) 8.14 eV; registered in chemistry.IONS for this script only, which has no
  molecular ions). M+ is lost to dissociative recombination alpha_dr (``--alpha-dr``: "od", 1e-6 (T/300)^-1/2,
  the default; "hco+", 2.4e-7 (T/300)^-0.69; "h3+", 6.8e-8 (T/300)^-0.52, FPdF 2026's H3+), to charge transfer
  with the metal (Mg, 2.9e-9), and to the PAHs.
- One metal, Mg (``--x-metal`` per H2, 1e-6 as Fig. 10.22): Mg+ recombines radiatively at 2.8e-12 (T/300)^-0.68
  (UMIST-type) and on the PAHs.
- PAHs: the v4.00 molecules with N_C 50-100 and an anion (349, the calibrated IP shifts), equal shares of
  x_PAH (per H nucleus), each in pahchem's dark chain: attachment (statistical s_e, ``capture_coefficient``),
  recombination of the cation, mutual neutralisation with M+ and Mg+ (0.58 DS87), M+ and Mg+ + PAH0 (0.9
  Langevin, IP(X) > IP(PAH)), and associative detachment of PAH- by atomic H at 7.8e-10 with
  ``--associative-detachment all`` (``population.anion_associative_detachment``; since 2026-10-26 pahchem's default
  leaves the parent anions without it, as Demarais+12 measured it on dehydrogenated anions only). n(H) is
  zeta / R_gr with R_gr = 3e-17 (T/100)^1/2 (H2 formation on grains against CR destruction; 3.2 cm^-3 at 3e-17,
  10 K), or 0 (``--n-h 0``: no associative detachment, as the classical balances).
- n is n(H2); x_e = n_e / n(H2) as in eq. 10.77; n_H = 2 n(H2).
The charge fractions are pahchem's: they are recomputed with ``population.charge_fractions`` over the whole
library, and with ``Population.charge_matrix`` / ``gas_rates`` (G0 = 0 and 1e-6) for three molecules, for each
``Params(detachment=...)``.

Default since 2026-10-26 (no associative detachment of these parents): the "without AD" numbers below.
Fig. 10.22 (x_PAH 1e-7): n(PAH-) = n_e at n(H2) = 1.3e2 (5.5e2 at 5e-8) against ~3e3; PAHs lower x_e at 1e4
63x; FPdF's x(PAH-)/x_e = 66 at x_PAH 1e-6 (zeta 5e-17), 6.0 at 1e-7. Both sit on the attachment, the full
classical capture for these EAs (s_e = 1), whose capture-to-anion share is not measured above N_C = 14.

Result (2026-10-25, with associative detachment, i.e. ``--associative-detachment all``; T = 10 K, alpha_dr "od",
x_metal 1e-6 per H2, n(H) = zeta/R_gr; ~100 s, 1 core):
- G0 = 0 works in ``Population.charge_matrix`` / ``gas_rates`` (a divide-by-zero warning in a log where an up-rate
  is 0, harmless); f equals this chain's and ``charge_fractions``' to 2e-16 for jochims, statistical and valence
  (the dark chain has no photodetachment); f(Z > 1) = 0; G0 = 1e-6 (Kurucz 40 kK) moves f by <= 3e-6;
  gas_rates' alpha_mn and electron_source match the chain's ion losses to 1e-6.
- pahchem's attachment at 10 K is 1.3-2.3e-6 (median 1.65e-6) against FPdF's recommended 1e-6.
- eq. 10.77: without PAHs x_e is 0.46-4.9x the equation (metal-dominated, slope -0.28); with x_PAH 5e-8-6e-7 it is
  0.005-0.18x at n(H2) 1e3-1e5, with slope -0.7 to -1.2 and zeta exponent +0.7 to +1.1 (fails: PAH- neutralise
  the ions and x_e ~ zeta / (k_att x_PAH n)).
- Fig. 10.22 (x_PAH 1e-7): n(PAH-) = n_e at n(H2) = 8.3e3 (2.6e3 at x_PAH 3e-7) against ~3e3; without
  associative detachment (--n-h 0) at 1.3e2-1.0e3 (alpha_dr od-h3+). PAHs lower x_e at 1e4 by 14-19x
  (x_metal 1e-6) or 1.8-2x (1e-8); without AD 53-63x and 5.6-7.5x (against ~10x).
- FPdF 2026 (n_H 1e4): x(PAH-)/x_e >= 43 needs x_PAH ~ 1e-6 without AD (66 at zeta 5e-17, 26 at 3e-16; 6 and 2 at
  1e-7), as they find; with AD (n(H) 5-32 cm^-3) it stays <= 12 even at 3e-6: AD by H sets f(PAH-) ~
  k_att n_e / (7.8e-10 n(H)) <= 0.15.
    check_dense_cloud_charge.py [--alpha-dr od|hco+|h3+] [--x-metal 1e-6] [--n-h -1] [--associative-detachment all]
                                [--temperature 10] [--out DIR]
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from pycalima import _paths
from pycalima.pahchem import chemistry, population
from pycalima.pahchem.baseline import Environment, Params
from pycalima.plotting_style import use_calima_style

IP_SHIFTS = {-1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64}
DETACHMENT = ("jochims", "statistical", "valence")
#: molecular ion M+: (mass [amu], IP of the neutral [eV]); chemistry.IONS has only atomic ions (2026-10-24)
MOLECULAR_ION = ("HCO+", (29.018, 8.14))
METAL = "Mg+"
#: dissociative recombination of M+ [cm^3 s^-1] at T
ALPHA_DR = {"od": lambda t: 1e-6 * (t / 300.0) ** -0.5,
            "hco+": lambda t: 2.4e-7 * (t / 300.0) ** -0.69,
            "h3+": lambda t: 6.8e-8 * (t / 300.0) ** -0.52}
K_CT_METAL = 2.9e-9                                   # HCO+ + Mg -> Mg+ + HCO [cm^3 s^-1]
N_H2 = np.logspace(2.0, 6.0, 25)
ZETAS = (1e-17, 3e-17, 1e-16)
X_PAH = (5e-8, 1e-7, 3e-7, 6e-7)
REPRESENTATIVE = ("C54H18 (37)", "C66H20 (115)", "C96H24 (108)")


def alpha_rr(t):
    """Radiative recombination of Mg+ [cm^3 s^-1]."""
    return 2.8e-12 * (t / 300.0) ** -0.68


def sci(v):
    """LaTeX a x 10^b of v (math mode content)."""
    b = int(np.floor(np.log10(v)))
    a = v / 10.0**b
    return rf"10^{{{b}}}" if abs(a - 1.0) < 1e-9 else rf"{a:.0f}\times10^{{{b}}}"


def eq_10_77(zeta, n_h2):
    return 2e-7 * np.sqrt(zeta / 3e-17) * np.sqrt(1e4 / n_h2)


def library():
    lib = population.shift_thresholds(population.library_v400(), IP_SHIFTS)
    return [m for m in lib if 50 <= m.nc <= 100 and -1 in m.charges]


class DarkChain:
    """Rate coefficients of pahchem's dark chain -1 <-> 0 <-> +1 for every molecule, at T [cm^3 s^-1]."""

    def __init__(self, lib, t, params=Params()):
        self.t = t
        self.ad = population.anion_associative_detachment(params)       # PAH- + H -> PAH-H + e
        self.att = np.array([population.capture_coefficient(m, 0, t, params) for m in lib])
        self.rec = np.array([population.capture_coefficient(m, 1, t, params) for m in lib])
        self.mn, self.ct = {}, {}
        for ion in (MOLECULAR_ION[0], METAL):
            self.mn[ion] = np.array([chemistry.mutual_neutralisation(m.nc, m.nh, -1, ion, t, params.mn_efficiency)
                                     for m in lib])
            self.ct[ion] = np.array([chemistry.ion_neutral_rate(m.nc, m.nh, ion, params.ion_neutral_efficiency)
                                     if chemistry.IONS[ion][1] > m.ionisation_ev[0] else 0.0 for m in lib])

    def fractions(self, ne, ions, n_h):
        """(f-, f0, f+)[s] at n_e, {ion: n} and atomic n(H) [cm^-3]."""
        up_m = self.ad * n_h + sum(n * self.mn[i] for i, n in ions.items())
        up_0 = sum(n * self.ct[i] for i, n in ions.items())
        r1, r2 = up_m / (ne * self.att), up_0 / (ne * self.rec)
        fm = 1.0 / (1.0 + r1 + r1 * r2)
        return fm, r1 * fm, r1 * r2 * fm


def solve(chain, n_h2, zeta, x_pah, x_metal, alpha_dr, n_h, guesses=None):
    """Steady state: dict(ne, M+, Mg+, f-, f0, f+ [s], ...) for each distinct root from the starting guesses."""
    t = chain.t
    w = x_pah * 2.0 * n_h2 / chain.att.size                 # n of each molecule [cm^-3]
    m_tot = x_metal * n_h2
    prod = zeta * n_h2
    a_dr, a_rr = alpha_dr(t), alpha_rr(t)
    mi = MOLECULAR_ION[0]

    def unpack(u):
        ne, nm = np.exp(u[0]), np.exp(u[1])
        nmg = m_tot / (1.0 + np.exp(-u[2])) if m_tot > 0 else 0.0     # logistic: 0 < Mg+ < m_tot
        mg0 = m_tot / (1.0 + np.exp(u[2])) if m_tot > 0 else 0.0      # neutral Mg, without the cancellation
        return ne, nm, nmg, mg0

    def balance(u):
        ne, nm, nmg, mg0 = unpack(u)
        fm, f0, fp = chain.fractions(ne, {mi: nm, METAL: nmg}, n_h)
        loss_m = nm * (a_dr * ne + K_CT_METAL * mg0 + w * np.sum(fm * chain.mn[mi] + f0 * chain.ct[mi]))
        gain_mg = K_CT_METAL * nm * mg0
        loss_mg = nmg * (a_rr * ne + w * np.sum(fm * chain.mn[METAL] + f0 * chain.ct[METAL]))
        neg, pos = ne + w * fm.sum(), nm + nmg + w * fp.sum()
        r = [(prod - loss_m) / (prod + loss_m), (neg - pos) / (neg + pos)]
        r.append((gain_mg - loss_mg) / (gain_mg + loss_mg) if m_tot > 0 else u[2])
        return np.array(r), (ne, nm, nmg, fm, f0, fp, loss_m, loss_mg)

    if guesses is None:
        ne_mol = np.sqrt(prod / a_dr)
        ne_met = min(np.sqrt(prod / a_rr), m_tot) if m_tot > 0 else ne_mol
        guesses = [(ne_mol, ne_mol, 0.0), (ne_met, prod / (a_dr * ne_met + K_CT_METAL * m_tot), 3.0),
                   (ne_mol * 1e-2, prod / (a_dr * ne_mol * 1e-2), 0.0)]
    out = []

    def attempt(ne0, nm0, s0):
        with np.errstate(all="ignore"):                  # the solver's trial steps may overflow
            sol = optimize.root(lambda u: balance(u)[0], [np.log(ne0), np.log(nm0), s0], method="hybr",
                                options={"xtol": 1e-13})
            r, (ne, nm, nmg, fm, f0, fp, loss_m, loss_mg) = balance(sol.x)
        if not np.all(np.isfinite(r)) or np.max(np.abs(r)) > 1e-8:
            return
        if any(abs(np.log(ne / o["ne"])) < 1e-5 for o in out):
            return
        # share of the positive ions (M+ and Mg+) neutralised by PAH- (mutual neutralisation), PAH0 and electrons
        by_mn = nm * w * np.sum(fm * chain.mn[mi]) + nmg * w * np.sum(fm * chain.mn[METAL])
        by_ct = nm * w * np.sum(f0 * chain.ct[mi]) + nmg * w * np.sum(f0 * chain.ct[METAL])
        by_e = nm * a_dr * ne + nmg * a_rr * ne
        tot = by_mn + by_ct + by_e
        out.append(dict(ne=ne, nm=nm, nmg=nmg, fm=fm, f0=f0, fp=fp, x_e=ne / n_h2, w=w,
                        neg_ratio=w * fm.sum() / ne, by_mn=by_mn / tot, by_ct=by_ct / tot, by_e=by_e / tot,
                        u=sol.x.copy()))

    for g in guesses:
        attempt(*g)
    if not out:                                          # a wider net of starting points
        for ne0 in np.geomspace(1e-13, 1e-4, 10) * n_h2:
            for s0 in (-3.0, 3.0):
                attempt(ne0, prod / (a_dr * ne0 + K_CT_METAL * m_tot), s0)
    return out


def n_h_atomic(zeta, t, arg):
    return zeta / (3e-17 * np.sqrt(t / 100.0)) if arg < 0 else arg


def sweep(chain, zeta, x_pah, args):
    """[solution] along N_H2 (continuation from the previous point), with a flag if any point has two roots."""
    res, prev, multi = [], None, 0
    for n in N_H2:
        sols = solve(chain, n, zeta, x_pah, args.x_metal, ALPHA_DR[args.alpha_dr], n_h_atomic(zeta, chain.t, args.n_h))
        if prev is not None:
            ne, nm, s = np.exp(prev[0]) * (n / res[-1]["n"]) ** -0.5, np.exp(prev[1]), prev[2]
            more = solve(chain, n, zeta, x_pah, args.x_metal, ALPHA_DR[args.alpha_dr],
                         n_h_atomic(zeta, chain.t, args.n_h), guesses=[(ne, nm, s)])
            sols += [m for m in more if all(abs(np.log(m["ne"] / o["ne"])) > 1e-5 for o in sols)]
        if not sols:
            raise RuntimeError(f"no steady state at n(H2) = {n:g}, zeta = {zeta:g}, x_PAH = {x_pah:g}")
        multi += len(sols) > 1
        s = min(sols, key=lambda o: abs(np.log(o["ne"] / res[-1]["ne"])) if res else o["ne"])
        s["n"], s["n_roots"] = n, len(sols)
        res.append(s)
        prev = s["u"]
    return res, multi


def at(res, n):
    return res[int(np.argmin(np.abs(np.log(N_H2 / n))))]


def crossover(res):
    """n(H2) where n(PAH-) = n_e (log interpolation), or None."""
    r = np.log([o["neg_ratio"] for o in res])
    i = np.nonzero(np.diff(np.sign(r)))[0]
    if i.size == 0:
        return None
    i = i[0]
    return float(np.exp(np.interp(0.0, [r[i], r[i + 1]], np.log(N_H2[i:i + 2]))))


def cross_check(lib, chain, point, args):
    """max |f(this chain) - f(pahchem)| over the library (charge_fractions) and over REPRESENTATIVE
    (Population.charge_matrix, gas_rates), for each detachment; and G0 = 1e-6 against G0 = 0."""
    n_h2, zeta, x_pah = point
    s = solve(chain, n_h2, zeta, x_pah, args.x_metal, ALPHA_DR[args.alpha_dr], n_h_atomic(zeta, chain.t, args.n_h))[0]
    n_h = n_h_atomic(zeta, chain.t, args.n_h)
    ions = ((MOLECULAR_ION[0], s["nm"]), (METAL, s["nmg"]))
    env = Environment("dark cloud", 0.0, n_h, s["ne"], chain.t, 40000, ions=ions)
    mine = np.stack([s["fm"], s["f0"], s["fp"]], axis=1)
    rep = [i for i, m in enumerate(lib) if m.name in REPRESENTATIVE]
    out = {}
    for d in DETACHMENT:
        p = Params(detachment=d, associative_detachment=args.associative_detachment)
        cf = np.array([[population.charge_fractions(m, env, p)[z] for z in (-1, 0, 1)] for m in lib])
        sub = [lib[i] for i in rep]
        pop = population.Population(sub, {"all": (1.0, 1e4)}, params=p)
        pop.prepare(population.field_shape(env), workers=1)
        with np.errstate(divide="ignore"):
            fm = pop.charge_matrix(env)
            x = np.full(len(sub), s["w"] / (2.0 * n_h2))
            gr = pop.gas_rates(env, x)
            tiny = pop.charge_matrix(Environment("dark cloud, G0 1e-6", 1e-6, n_h, s["ne"], chain.t, 40000, ions=ions))
        zi = [pop.zs.index(z) for z in (-1, 0, 1)]
        # gas_rates' alpha (per H nucleus) against this chain's loss of each ion on the same molecules
        a_mn = sum(gr["alpha_mn"][i] * n for i, n in ions) * 2.0 * n_h2
        own = sum(n * s["w"] * np.sum(s["fm"][rep] * chain.mn[i][rep]) for i, n in ions)
        up_ions = sum(n * s["w"] * np.sum(s["fm"][rep] * chain.mn[i][rep] + s["f0"][rep] * chain.ct[i][rep])
                      for i, n in ions)
        out[d] = dict(lib=float(np.max(np.abs(cf - mine))), pop=float(np.max(np.abs(fm[:, zi] - mine[rep]))),
                      higher=float(fm[:, [k for k, z in enumerate(pop.zs) if z > 1]].max(initial=0.0)),
                      g0_tiny=float(np.max(np.abs(tiny - fm))), alpha_mn=a_mn / own,
                      e_source=gr["electron_source"] * 2.0 * n_h2 / -up_ions)
    return s, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alpha-dr", default="od", choices=tuple(ALPHA_DR))
    ap.add_argument("--x-metal", type=float, default=1e-6, help="total metal (Mg) per H2")
    ap.add_argument("--n-h", type=float, default=-1.0,
                    help="atomic n(H) [cm^-3] for associative detachment; < 0: zeta / R_gr (default); 0: off")
    ap.add_argument("--associative-detachment", default=Params().associative_detachment, choices=("dehydrogenated", "all"),
                    help="'all': PAH- + H detaches every anion; 'dehydrogenated' (pahchem's default): none of these parents")
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    chemistry.IONS.setdefault(*MOLECULAR_ION)
    t0 = time.time()
    lib = library()
    chain = DarkChain(lib, args.temperature, Params(associative_detachment=args.associative_detachment))
    t = args.temperature
    print(f"{len(lib)} v4.00 molecules, N_C 50-100 with an anion; T = {t:g} K, alpha_dr '{args.alpha_dr}' = "
          f"{ALPHA_DR[args.alpha_dr](t):.2e}, alpha_rr(Mg+) = {alpha_rr(t):.2e}, x_metal = {args.x_metal:g} per H2, "
          f"n(H) = {'zeta/R_gr' if args.n_h < 0 else args.n_h}")
    print(f"pahchem's dark-chain coefficients [cm^3 s^-1], median (range) over the library: attachment "
          f"{np.median(chain.att):.2e} ({chain.att.min():.1e}-{chain.att.max():.1e}; FPdF 2026: 1e-6), "
          f"recombination {np.median(chain.rec):.2e}, mutual neutralisation HCO+ {np.median(chain.mn['HCO+']):.2e}, "
          f"Mg+ {np.median(chain.mn['Mg+']):.2e}; HCO+ + PAH0 {np.median(chain.ct['HCO+']):.2e}, "
          f"Mg+ + PAH0 {np.median(chain.ct['Mg+']):.2e} (open for {np.mean(chain.ct['Mg+'] > 0):.0%})")
    print(f"eq. 10.77 implies zeta / (x_e^2 n) = {3e-17 / (2e-7**2 * 1e4):.2e} cm^3 s^-1 (an effective recombination)")

    runs, multi = {}, 0
    for zeta in ZETAS:
        for x in (0.0, *X_PAH):
            runs[(zeta, x)], m = sweep(chain, zeta, x, args)
            multi += m
    print(f"points with more than one steady state: {multi}")

    print("\nx_e = n_e / n(H2) against eq. 10.77 (ratio model / eq. 10.77), and the slopes d ln x_e / d ln n over "
          "1e3-1e5 (eq.: -0.5)")
    sel = [1e2, 1e3, 3e3, 1e4, 1e5, 1e6]
    idx = [int(np.argmin(np.abs(N_H2 - v))) for v in sel]
    print(f"{'zeta':>7} {'x_PAH':>7} | " + " ".join(f"{v:>15.0e}" for v in sel) + " | slope")
    for (zeta, x), res in runs.items():
        xe = np.array([o["x_e"] for o in res])
        i3, i5 = int(np.argmin(np.abs(N_H2 - 1e3))), int(np.argmin(np.abs(N_H2 - 1e5)))
        slope = np.log(xe[i5] / xe[i3]) / np.log(N_H2[i5] / N_H2[i3])
        cells = " ".join(f"{xe[i]:.2e} ({xe[i] / eq_10_77(zeta, N_H2[i]):4.2f})" for i in idx)
        print(f"{zeta:7.0e} {x:7.0e} | {cells} | {slope:+.2f}")
    print("zeta exponent of x_e at n(H2) = 1e4 (eq.: +0.5): " + ", ".join(
        f"x_PAH {x:g}: {np.log(at(runs[(1e-16, x)], 1e4)['x_e'] / at(runs[(1e-17, x)], 1e4)['x_e']) / np.log(10.0):+.2f}"
        for x in (0.0, *X_PAH)))

    print("\nFig. 10.22 analogue (zeta = 3e-17): f(PAH-), f(PAH0), f(PAH+), n(PAH-)/n_e, and the share of ion "
          "neutralisations by PAH- / PAH0 / electrons")
    for x in X_PAH:
        res = runs[(3e-17, x)]
        nx = crossover(res)
        print(f" x_PAH {x:g}: n(PAH-) = n_e at n(H2) = {nx:.2e}" if nx else f" x_PAH {x:g}: no crossover in 1e2-1e6",
              f"(Fig. 10.22: ~3e3); x_e reduction by the PAHs at 1e4 and 1e5: "
              f"{at(runs[(3e-17, 0.0)], 1e4)['x_e'] / at(res, 1e4)['x_e']:.1f}x, "
              f"{at(runs[(3e-17, 0.0)], 1e5)['x_e'] / at(res, 1e5)['x_e']:.1f}x (~10x)")
        for i in idx:
            o = res[i]
            print(f"   n(H2) {N_H2[i]:7.0e}: x_e {o['x_e']:.2e}  f(-, 0, +) {o['fm'].mean():.3f} {o['f0'].mean():.3f} "
                  f"{o['fp'].mean():.1e}  n(PAH-)/n_e {o['neg_ratio']:8.2f}  by PAH-/PAH0/e {o['by_mn']:.2f}/"
                  f"{o['by_ct']:.2f}/{o['by_e']:.2f}  Mg+/M+ {o['nmg'] / o['nm']:.1e}")
        # Tielens §10.5.3: PAH- dominate when x_e < 2e-7 (x_PAH/1e-7), x_PAH taken per H2 here (2 x per H)
        if nx:
            xe_x = float(np.exp(np.interp(np.log(nx), np.log(N_H2), np.log([o["x_e"] for o in res]))))
            print(f"   at the crossover x_e = {xe_x:.2e}; §10.5.3 criterion 2e-7 (x_PAH/1e-7) = {2e-7 * 2 * x / 1e-7:.2e} "
                  f"(x_PAH per H2) or {2e-7 * x / 1e-7:.2e} (per H)")

    print("\nFlower & Pineau des Forets 2026: n_H = 1e4 (n(H2) = 5e3), T = 10 K; x(PAH-)/x_e >= 43 only for x_PAH >~ 1e-6")
    for n_h_arg in dict.fromkeys((args.n_h, 0.0)):
        for zeta in (5e-17, 3e-16):
            cells = []
            for x in (1e-8, 1e-7, 1e-6, 3e-6):
                o = solve(chain, 5e3, zeta, x, args.x_metal, ALPHA_DR[args.alpha_dr], n_h_atomic(zeta, t, n_h_arg))[0]
                cells.append(f"{x:g}: x_e(H) {o['ne'] / 1e4:.1e}, x(PAH-)/x_e {o['neg_ratio']:5.2f}, by PAH- {o['by_mn']:.2f}")
            print(f" n(H) {n_h_atomic(zeta, t, n_h_arg):.2f}, zeta {zeta:g}, x_PAH " + "; ".join(cells))

    print("\nSensitivity (zeta = 3e-17, x_PAH = 1e-7; n(H2) = 1e4): x_e / eq. 10.77 without and with PAHs, f(PAH-), the "
          "x_e reduction by the PAHs, and the n(H2) where n(PAH-) = n_e (Fig. 10.22: ~3e3)")
    for n_h_arg in (-1.0, 0.0):
        for a in ALPHA_DR:
            for xm in (1e-6, 1e-8):
                opt = argparse.Namespace(x_metal=xm, alpha_dr=a, n_h=n_h_arg)
                r0, r1 = sweep(chain, 3e-17, 0.0, opt)[0], sweep(chain, 3e-17, 1e-7, opt)[0]
                o0, o1 = at(r0, 1e4), at(r1, 1e4)
                nx = crossover(r1)
                print(f" n(H) {'zeta/R_gr' if n_h_arg < 0 else '0':9s} alpha_dr {a:5s} x_metal {xm:g}: "
                      f"{o0['x_e'] / eq_10_77(3e-17, 1e4):5.2f}, {o1['x_e'] / eq_10_77(3e-17, 1e4):6.3f}; "
                      f"f(PAH-) {o1['fm'].mean():.2f}; reduction {o0['x_e'] / o1['x_e']:5.1f}x; "
                      + (f"crossover {nx:.1e}" if nx else "no crossover in 1e2-1e6"))

    print("\npahchem cross-check (max |delta f|; alpha_mn and electron_source of gas_rates over this chain's)")
    for point in ((1e3, 3e-17, 1e-7), (1e4, 3e-17, 1e-7), (1e5, 1e-16, 6e-7)):
        s, cc = cross_check(lib, chain, point, args)
        for d, v in cc.items():
            print(f" n(H2) {point[0]:g}, zeta {point[1]:g}, x_PAH {point[2]:g}, {d:11s}: charge_fractions {v['lib']:.1e}, "
                  f"charge_matrix {v['pop']:.1e}, f(Z > 1) {v['higher']:.1e}, G0 1e-6 vs 0 {v['g0_tiny']:.1e}, "
                  f"alpha_mn ratio {v['alpha_mn']:.6f}, electron_source ratio {v['e_source']:.6f}")

    use_calima_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.0), sharex=True)
    colours = {1e-17: "#2a78d6", 3e-17: "#1baf7a", 1e-16: "#eb6834"}
    ax = axes[0]
    for zeta, c in colours.items():
        z_lab = rf"$\zeta = {sci(zeta)}$ s$^{{-1}}$"
        ax.loglog(N_H2, eq_10_77(zeta, N_H2), color=c, ls=":", lw=1.4)
        ax.loglog(N_H2, [o["x_e"] for o in runs[(zeta, 0.0)]], color=c, ls="--", lw=1.2)
        ax.loglog(N_H2, [o["x_e"] for o in runs[(zeta, 1e-7)]], color=c, lw=1.8, label=z_lab)
        ax.loglog(N_H2, [o["x_e"] for o in runs[(zeta, 6e-7)]], color=c, lw=1.0, ls="-.")
    ax.plot([], [], "k:", label="Tielens (2021) eq. 10.77")
    ax.plot([], [], "k--", label="no PAHs")
    ax.plot([], [], "k-", label=r"$x_{\rm PAH} = 10^{-7}$")
    ax.plot([], [], "k-.", label=r"$x_{\rm PAH} = 6\times10^{-7}$")
    ax.set_ylabel(r"$x_e = n_e/n({\rm H_2})$", fontsize=14)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax = axes[1]
    for x, ls in ((1e-7, "-"), (6e-7, "-.")):
        res = runs[(3e-17, x)]
        ax.loglog(N_H2, [o["ne"] for o in res], color="k", ls=ls, lw=1.6, label=rf"$n_e$, $x_{{\rm PAH}} = {sci(x)}$")
        ax.loglog(N_H2, [o["w"] * o["fm"].sum() for o in res], color="#eb6834", ls=ls, lw=1.6, label=r"$n({\rm PAH^-})$")
        ax.loglog(N_H2, [o["nm"] for o in res], color="#2a78d6", ls=ls, lw=1.2, label=r"$n({\rm M^+})$")
        ax.loglog(N_H2, [o["nmg"] for o in res], color="#1baf7a", ls=ls, lw=1.2, label=r"$n({\rm Mg^+})$")
    ax.axvline(3e3, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel(r"$n({\rm H_2})$ [cm$^{-3}$]", fontsize=14)
    ax.set_ylabel(r"$n$ [cm$^{-3}$]", fontsize=14)
    ax.text(0.04, 0.93, r"$\zeta = 3\times10^{-17}$ s$^{-1}$; Fig. 10.22 crossover (dotted)", transform=ax.transAxes,
            fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.subplots_adjust(hspace=0)
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    name = f"pahchem_dense_cloud_charge_{args.alpha_dr.replace('+', 'p')}{'_noad' if args.n_h == 0 else ''}.png"
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
