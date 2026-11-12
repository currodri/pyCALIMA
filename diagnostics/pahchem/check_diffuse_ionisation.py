#!/usr/bin/env python3
"""Diffuse-gas ionisation balance with pahchem's PAHs: W08's C I/C II (V3), Lepp+88's metal ion ratios (V13) and
WD01b's electron fraction (V4).

A small gas side is written here; the PAH side is ``Population.gas_rates`` (alpha_mn + alpha_ct per H for each
ion, and the charge fractions f, whose mean charge enters charge neutrality). Nothing is tuned. PAHs: the v4.00
molecules with N_C 32-40 at x_PAH (as check_wolfire08_cplus.py, 72 molecules, IP shifts -1: 0.37, 0: 0.6,
+1/+2: 0.64), and for V4 also the whole library weighted to WD01's b_C = 6e-5 graphite grains of the same carbon
count (as check_wd01b_alpha.py). Field: the Draine (1978) shape with G0 in Habing units, and the old stars kept at
optical_isrf = G0 / 1.7 (solar circle at the Draine field; attenuated with the UV), so that one set of bases serves
every cell. Every result for detachment = jochims, statistical and valence.

Gas rates (all cited where defined below): photoionisation in the Draine ISRF and its dust shielding
exp(-gamma A_V) from Heays+17 (UMIST RATE22 entries); radiative recombination of Mg+, S+, Si+, Ca+, Fe+ from
Pequignot & Aldrovandi 1986 (the source Lepp+88 used); C+ + e from W08 (1.8e-11 T2^-0.83, Badnell RR + DR);
H+ case B 3.5e-12 (T/300)^-0.75 (Liszt 2001, as WD01b); C+, S+, Si+, H+ charge exchange with neutral metals from
Prasad & Huntress 1980 (UMIST RATE22). Ca+ is not in ``chemistry.IONS``: it is added at run time here
(40.078 u, IP(Ca) 6.113 eV), so pahchem treats it like the other ions (mutual neutralisation, and charge transfer
only on the few molecules with IP < 6.11 eV); pahchem has no X+ + PAH -> XH+ channel, which Lepp+88 assume for Ca+.

- **V3** (W08, ApJ 680, 384, arXiv:0803.0138, App. B). pahchem's effective Phi: the Phi at which W08's three-state
  balance (Table 2 rates) gives pahchem's alpha(C+), at CNM conditions (G0 = 1.7, n = 30, x(C+) = 1.4e-4,
  x(H+) = 1.6e-4, x_e = 3e-4 as in eq. B15) over T = 50-300 K. The sight lines of W08's Table 1 with A_V > 0.25
  get <G0>/n from their H2 (eqs. B7-B8 with R = 3.5e-17, I = 4.7e-11 and the factor 1/2 of eq. B12); eqs. B15 + B17
  (both terms) then give the Phi each line implies, with W08's C I and, for the four lines in common, Burgh+10's
  (Liszt 2011 Table 1; 4-21x lower on three of them). T = 80 K for the lines (W08 does not list one; T^-1/4).
  pahchem's own N(C I)/N(C II) at each <G0>/n (``direct``) is (alpha_PAH(C+) n + k_rr n_e) / (2.1e-10 <G0>).
- **V13** (Lepp, Dalgarno, van Dishoeck & Black 1988, ApJ 329, 418, Table 2; the validation table cites ApJ 324,
  553, which has no such table). N(X+)/N(X) for C, Mg, S, Ca, Fe toward zeta Per (model F: n = 325, T = 30 K,
  I_UV = 3.5) and zeta Oph (model G: n = 350, T = 30 K, I_UV = 3.5), here a uniform two-sided slab at the central
  values (their models have density gradients), A_V from W08's Table 1 (0.79, 0.71), f_H2 from it (0.59, 0.63),
  zeta = 1e-16 s^-1 on atomic H, gas-phase C = 1.4e-4 and assumed cold-cloud metal abundances (they enter only x_e
  and the charge exchange). PAH photorates attenuated as exp(-1.8 A_V) (W08). Reported: the ratios against x_PAH,
  their fall relative to x_PAH = 0 (which divides out the gas rates) against Lepp+88's, and the x_PAH fitted to the
  observed ratios (log chi^2 over the five elements, range half-widths as errors) for pahchem and for Lepp+88's own
  models (which gives back their 1e-7 and 6e-7).
- **V4** (WD01b, ApJ 563, 842, Sect. 7). x_e = x_e0 + x_H with x_e0 = 2e-4 fixed (metals), and
  xi (1 - x_H) = (alpha_B x_e + alpha_g) x_H n (their eq. 23) at n = 30, T = 100 K, G = 1.13 (Mathis ISRF; pahchem
  uses the Draine shape at G0 = 1.13). WD01b's xi (cosmic rays 3e-17 plus X-rays from Wolfire+95 App. A, which
  depend on x_e) is not given as a number: it is taken from their grain-free x_e = 8.0e-4 (xi = 1.15e-16 s^-1) and
  held fixed. The WD01b fit (Table 2, psi = G sqrt(T)/n_e) must then give back their 4.5e-4; pahchem replaces
  alpha_g by alpha_PAH(H+). PAHs pass the ions' charge to electrons: charge neutrality includes their mean charge,
  and ``electron_source`` = -sum alpha n(X+) is checked at the solution.

Result (2026-10-25; N_C 32-40 at x_PAH = 6e-7 unless stated; Jochims / statistical / valence):
- V3: effective Phi 0.37-0.39 / 0.32-0.35 / 0.32-0.35 over T = 50-300 K (f(PAH-) 0.11 / 0.08), inside W08's
  0.2-0.4. The 28 W08 lines (A_V > 0.25, no 23 Ori, pi Sco) imply Phi = 0.26 (16-84 %: 0.09-0.54) by eqs. B15 + B17
  with their H2-derived <G0>/n; pahchem's own C I/C II at that <G0>/n gives obs/model 0.89 / 1.01 / 1.01 (median,
  16-84 % 0.42-3.4). On the four Burgh+10 lines the implied Phi falls from 0.50, 0.51, 0.57, 0.18 (W08 C I) to
  0.00, 0.06, 0.15, 0.19 (Burgh C I), and obs/pahchem from 4.5, 2.3, 2.2, 0.57 to 0.22, 0.34, 0.49, 0.59;
- V13: fitted to the observed ratios, x_PAH = 1.2e-7 / 1.4e-7 / 1.4e-7 (zeta Per; Lepp+88 1e-7) and
  5.6e-7 / 7.0e-7 / 7.0e-7 (zeta Oph; 6e-7); the same fit of Lepp+88's own models gives 4.8e-8 and 6.7e-7.
  pahchem's fall of each ratio with x_PAH follows Lepp+88's to within 0.1 at 1e-7 (C falls less, 0.73 against
  0.63 for zeta Per; Mg more, 0.43 against 0.54); at x_PAH = 0 S+/S and Fe+/Fe are 1.4-2x theirs (gas rates);
- V4: x_e = 2.91e-4 / 3.06e-4 / 3.07e-4 at n = 30, T = 100 K, against 3.12e-4 from WD01b's published alpha_g(H+)
  in the same set-up (alpha(H+) 4.0e-14 / 3.4e-14 / 3.35e-14 against 3.16e-14) and 4.5e-4 in WD01b's text, which
  the arXiv v2 coefficients give back (4.48e-4; their alpha_g(H+) is 2.7x smaller). electron_source equals minus
  the ions neutralised to 1e-12. The whole-library (b_C = 6e-5) case is not run: its ~1800 missing bases did not
  finish in 10 h on 2 workers.

The numbers are kept in results/pahchem_diffuse_ionisation.json (``--plot-only`` redraws from it). 77 s and
0.5 GB with the N_C 32-40 bases cached (one field shape for every cell).

Rerun on 2026-11-02 with the current defaults: section 4 of the gas-coupling doc, where ``check_default_changes.py``
adds the changes since 2026-10-25 one at a time; ``--nc-range 50 100`` puts W08's carbon
in N_C 50-100 (V3 Phi 0.36, V13 within x2.4 of Lepp+88), which the size-logistic attachment needs.

    check_diffuse_ionisation.py [--workers 2] [--full-library] [--nc-range LO HI] [--plot-only] [--out DIR]
"""
import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parents[1] / "dust_charge"))
import check_grain_size_distribution as gsd  # noqa: E402

from pycalima import _paths  # noqa: E402
from pycalima.pahchem import chemistry, population  # noqa: E402
from pycalima.pahchem import group_tables as G  # noqa: E402
from pycalima.pahchem.baseline import Environment, Params  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

chemistry.IONS.setdefault("Ca+", (40.078, 6.113))        # not in pahchem's list; see the module docstring

DETACHMENTS = ("jochims", "statistical", "valence")
STYLE = {"jochims": ("#eb6834", ":"), "statistical": ("#1baf7a", "--"), "valence": ("#2a78d6", "-")}   # as the other V figures
IP_SHIFTS = population.CALIBRATED_SHIFTS
BASIS_CACHE = _paths.get_model_data_dir() / "pahchem_population_basis"
OLD_STARS_PER_G0 = 1.0 / 1.7          # old-star light per unit G0: solar circle at the Draine field
G_DRAINE = 1.7                        # Draine (1978) field in Habing units (W08)
X_PAH_W08 = 6e-7

# -- gas side ------------------------------------------------------------------------------------
#: photoionisation in the Draine ISRF [s^-1] and dust shielding gamma, k = k0 exp(-gamma A_V): Heays, Bosman &
#: van Dishoeck 2017 (A&A 602, A105) as entered in UMIST RATE22 (McElroy+ 2013 format)
PHOTO = {"C": (3.50e-10, 3.76), "S": (1.12e-9, 3.52), "Si": (4.45e-9, 2.61), "Mg": (6.59e-11, 2.43),
         "Fe": (4.70e-10, 2.62), "Ca": (3.50e-10, 2.34)}
ELEMENTS = tuple(PHOTO)
#: radiative recombination alpha (T/300)^beta [cm^3 s^-1]: Pequignot & Aldrovandi 1986 (A&A 161, 169) via UMIST
#: RATE22; H+ case B (Liszt 2001, as WD01b eq. 23)
RR = {"S+": (5.49e-12, -0.59), "Si+": (4.26e-12, -0.62), "Mg+": (2.78e-12, -0.68), "Fe+": (2.55e-12, -0.69),
      "Ca+": (2.64e-12, -0.68), "H+": (3.5e-12, -0.75)}
#: Y+ + X -> Y + X+ [cm^3 s^-1]: Prasad & Huntress 1980 (ApJS 43, 1) via UMIST RATE22
CHARGE_EXCHANGE = (("C+", "Mg", 1.1e-9), ("C+", "Fe", 2.6e-9), ("C+", "Si", 2.1e-9), ("S+", "Fe", 1.8e-10),
                   ("Si+", "Fe", 1.9e-9), ("H+", "Fe", 7.4e-9), ("H+", "Mg", 1.1e-9), ("H+", "S", 1.3e-9),
                   ("H+", "Si", 9.9e-10))
IONS = tuple(e + "+" for e in ELEMENTS) + ("H+",)


def alpha_rr(ion, T):
    """Radiative (+ dielectronic) recombination [cm^3 s^-1]; C+: W08 eq. 3 (Badnell RR + DR, 20-300 K to 8 %)."""
    if ion == "C+":
        return 1.8e-11 * (T / 100.0) ** -0.83
    a, b = RR[ion]
    return a * (T / 300.0) ** b


# -- PAH side ------------------------------------------------------------------------------------
def snap(g0):
    """(g0, optical_isrf) with optical_isrf / g0 == OLD_STARS_PER_G0 to the bit, so that every cell shares one field
    shape (and one set of bases); g0 moves by a few ulp at most."""
    g = float(g0)
    for _ in range(64):
        o = g * OLD_STARS_PER_G0
        for cand in (o, np.nextafter(o, np.inf), np.nextafter(o, -np.inf)):
            if cand / g == OLD_STARS_PER_G0:
                return g, float(cand)
        g = float(np.nextafter(g, np.inf))
    raise RuntimeError(f"no exact old-star ratio near G0 = {g0}")


def shape():
    g, o = snap(G_DRAINE)
    return population.field_shape(Environment("x", g, 1.0, 1.0, 100.0, None, optical_isrf=o))


def library_v400():
    return population.shift_thresholds(
        population.library_v400(min_nc=21, cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index"), IP_SHIFTS)


def radius_cm(nc):
    return 1e-7 * (np.asarray(nc, dtype=float) / 468.0) ** (1.0 / 3.0)


def wd01_weighted(full):
    """(molecules, abundance per H): the whole v4.00 library in N_C bins 20-40-70-120-270, each carrying the WD01
    (R_V = 3.1, b_C = 6e-5) graphite grains of the same carbon count, N_C = 468 (a / 10 A)^3 (check_wd01b_alpha.py)."""
    edges = (20, 40, 70, 120, 270)
    a_fine = np.geomspace(gsd.A_MIN, radius_cm(edges[-1]), 400)
    x = np.zeros(len(full))
    for lo, hi in pairwise(edges):
        w = G.nc_range_weights(full, lo, hi)
        a = a_fine[(a_fine >= radius_cm(lo)) & (a_fine <= radius_cm(hi))]
        x += float(np.trapezoid([gsd.graphite_dn_da(v) for v in a], a)) * w / w.sum()
    keep = x > 0
    return [m for m, k in zip(full, keep) if k], x[keep]


class PAHs:
    """One library at one detachment: alpha_PAH(X) per H and the PAHs' mean charge per H at a cell."""

    def __init__(self, lib, x, detachment, workers):
        self.pop = population.Population(lib, {"all": (1.0, 1e4)}, params=Params(detachment=detachment),
                                         cache_dir=BASIS_CACHE)
        self.shape = shape()
        self.pop.prepare(self.shape, workers=workers)
        self.x, self.z = np.asarray(x, dtype=float), np.array(self.pop.zs, dtype=float)

    def env(self, g0, n_h_atomic, n_e, T, ions):
        g, o = snap(g0)
        env = Environment("diffuse", g, n_h_atomic, n_e, T, None, ions=tuple(ions.items()), optical_isrf=o)
        if population.field_shape(env) != self.shape:      # never let gas_rates build bases on every core
            raise RuntimeError(f"field shape {population.field_shape(env)} is not the prepared {self.shape}")
        return env

    def rates(self, g0, n_h_atomic, n_e, T, ions, scale=1.0):
        """({ion: alpha_mn + alpha_ct [cm^3 s^-1 per H]}, sum x <Z> per H, gas_rates dict) at x * scale."""
        x = self.x * scale
        r = self.pop.gas_rates(self.env(g0, n_h_atomic, n_e, T, ions), x)
        alpha = {i: r["alpha_mn"][i] + r["alpha_ct"][i] for i in ions}
        return alpha, float(x @ (r["f"] @ self.z)), r


# -- one cell of gas -----------------------------------------------------------------------------
def cell(n, T, photo_rates, abundance, zeta, n_h_atomic, pahs=None, g0_pah=0.0, scale=0.0, tol=1e-7):
    """Steady ions [cm^-3], n_e, alpha_PAH and the PAHs' charge per H at one point.

    n(X+)/n(X) = (Gamma_X + sum k_CE n(Y+)) / (alpha_rr n_e + alpha_PAH n + sum k_CE' n(Z)); n(H+) from zeta on
    atomic H; n_e from charge neutrality including the PAHs' mean charge. Iterated with the PAH charge balance,
    which depends on n_e and the ions.
    """
    ions = {e + "+": 0.9 * abundance[e] * n for e in ELEMENTS}
    ions["H+"] = 1e-5 * n
    ne, alpha, q = sum(ions.values()), dict.fromkeys(IONS, 0.0), 0.0

    def balance(ne_, ions_):
        out = {}
        for e in ELEMENTS:
            ion = e + "+"
            src = photo_rates[e] + sum(k * ions_[y] for y, xx, k in CHARGE_EXCHANGE if xx == e)
            loss = alpha_rr(ion, T) * ne_ + alpha[ion] * n + sum(
                k * max(abundance[xx] * n - ions_[xx + "+"], 0.0) for y, xx, k in CHARGE_EXCHANGE if y == ion)
            out[ion] = abundance[e] * n * src / (src + loss)
        loss_h = alpha_rr("H+", T) * ne_ + alpha["H+"] * n + sum(
            k * max(abundance[xx] * n - ions_[xx + "+"], 0.0) for y, xx, k in CHARGE_EXCHANGE if y == "H+")
        out["H+"] = zeta * n_h_atomic / loss_h
        return out

    for _ in range(200):
        if pahs is not None and scale > 0:
            alpha, q, _r = pahs.rates(g0_pah, n_h_atomic, ne, T, ions, scale)

        def g(lne, ions_=ions):
            return np.log(max(sum(balance(np.exp(lne), ions_).values()) + q * n, 1e-300)) - lne

        ne_new = float(np.exp(brentq(g, np.log(1e-9 * n), np.log(1e-2 * n), xtol=1e-12)))
        new = balance(ne_new, ions)
        done = abs(ne_new / ne - 1.0) < tol and all(abs(new[i] / ions[i] - 1.0) < tol for i in IONS if ions[i] > 0)
        ions = {i: np.sqrt(new[i] * ions[i]) if ions[i] > 0 else new[i] for i in IONS}
        ne = ne_new
        if done:
            break
    else:
        raise RuntimeError("cell did not converge")
    return ions, ne, alpha, q


# == V3: W08 C I / C II =========================================================================
#: W08 Table 2 at Phi = 1, T = 100 K [cm^3 s^-1]: (X+ + PAH-, X+ + PAH0)
W08_ION = {"C+": (2.4e-7, 8.8e-9), "H+": (8.3e-7, 3.1e-8)}
V3_CNM = {"G0": 1.7, "n": 30.0, "xC": 1.4e-4, "xH": 1.6e-4}   # x_e = 3e-4, as W08 eq. B15
V3_T = (50.0, 70.0, 100.0, 150.0, 200.0, 300.0)
T_LINES = 80.0
R_W08, I_H2, N0_H2 = 3.5e-17, 4.7e-11, 1e14               # W08 Sect. 4.2 and eq. B3, I from Abgrall+92

#: W08 Table 1, A_V > 0.25: star, log N(H I), log N(H2), log N(C I), log N(C II), A_V. N(C II) is 1.6e-4 N_H unless
#: observed. 23 Ori and pi Sco are W08's outliers (time dependence) and are not in the statistics.
W08_TABLE1 = """kappa Cas 21.21 20.27 15.55 17.50 0.98
40 Per 21.05 20.46 15.25 17.43 0.85
o Per 20.91 20.61 15.68 17.41 0.79
zeta Per 20.81 20.67 15.52 17.35 0.79
HD 24534 20.72 20.92 16.30 17.53 1.10
xi Per 21.07 20.53 15.12 17.51 0.98
HD 34078 21.43 20.81 16.24 17.80 2.00
23 Ori 20.74 18.30 14.99 16.94 0.28
lambda Ori 20.79 19.11 14.32 17.00 0.32
139 Tau 20.90 19.74 14.23 17.16 0.46
HD 112244 21.08 20.14 14.69 17.37 0.74
1 Sco 21.19 19.23 14.00 17.40 0.80
pi Sco 20.72 19.32 13.03 16.95 0.28
delta Sco 21.14 19.41 14.25 17.36 0.74
beta1 Sco 21.10 19.83 14.42 17.37 0.70
omega1 Sco 21.18 20.05 14.35 17.44 0.88
nu Sco 21.14 19.89 14.56 17.39 0.78
sigma Sco 21.34 19.79 14.38 17.56 1.15
rho Oph 21.55 20.57 15.52 17.83 2.13
chi Oph 21.14 20.63 15.33 17.55 1.15
22 Sco 20.95 18.74 13.95 17.16 0.46
zeta Oph 20.72 20.65 15.51 17.36 0.71
HD 154368 21.00 21.16 16.22 17.79 1.96
gamma Ara 20.71 19.24 13.87 16.94 0.26
kappa Aql 20.90 20.31 14.37 17.28 0.61
HD 185418 21.11 20.76 15.57 17.59 1.24
HD 192639 21.29 20.69 15.37 17.67 1.55
HD 206267 21.31 20.86 15.30 17.74 1.74
HD 210839 21.16 20.84 14.98 17.65 1.42
sigma Cas 20.88 20.23 14.74 17.24 0.55"""
W08_OUTLIERS = ("23 Ori", "pi Sco")
#: log N(C I) of Burgh+10 for the lines in common with W08 (Liszt 2011, A&A 527, A45, Table 1)
BURGH10_CI = {"HD 24534": 14.98, "HD 185418": 14.74, "HD 192639": 14.73, "HD 210839": 15.00}


def sight_lines():
    out = []
    for row in W08_TABLE1.splitlines():
        p = row.split()
        name, (hi, h2, ci, cii, av) = " ".join(p[:-5]), map(float, p[-5:])
        n_hi, n_h2 = 10**hi, 10**h2
        x_h = 0.25 * n_hi / (((N0_H2 + n_h2) ** 0.25 - N0_H2**0.25) * N0_H2**0.75)          # eq. B8
        out.append({"name": name, "av": av, "g_over_n": 0.5 * x_h * R_W08 / I_H2,        # <G0>/n: eqs. B7, B12
                    "ratio": 10 ** (ci - cii), "ratio_burgh": 10 ** (BURGH10_CI[name] - cii) if name in BURGH10_CI
                    else None, "outlier": name in W08_OUTLIERS})
    return out


def w08_ratio(phi, g_over_n, T):
    """N(C I)/N(C II) of eqs. B15 + B17 (both C+ sinks) at <G0>/n."""
    t2 = T / 100.0
    u = 1.0 / g_over_n
    return (4.1e-3 * phi * t2**-0.25 * u) ** 2 + (1.8e-5 * phi + 2.6e-5 * t2**-0.83) * u


def implied_phi(ratio, g_over_n, T):
    """Phi of eqs. B15 + B17 for an observed ratio; 0 if electrons alone already give more C I."""
    t2, u = T / 100.0, 1.0 / g_over_n
    a, b, c = (4.1e-3 * t2**-0.25 * u) ** 2, 1.8e-5 * u, 2.6e-5 * t2**-0.83 * u - ratio
    if c >= 0:
        return 0.0
    return float((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a))


def w08_alpha(phi, T, G0, n, xC, xH, x=X_PAH_W08, ion="C+"):
    """alpha(ion) per H of W08's Table 2 rates in a three-state balance (check_wolfire08_cplus.py, with T)."""
    ne, t2 = (xC + xH) * n, T / 100.0
    att, det = 1.3e-6 * phi * ne, 1.2e-8 * G0
    mn = phi * t2**-0.5 * (W08_ION["C+"][0] * xC * n + W08_ION["H+"][0] * xH * n)
    up = 4.6e-9 * G0 + phi * (W08_ION["C+"][1] * xC * n + W08_ION["H+"][1] * xH * n)
    rm, rp = att / (det + mn), up / (3.5e-5 * phi * t2**-0.5 * ne)
    f0 = 1.0 / (1.0 + rm + rp)
    k_mn, k_ct = W08_ION[ion]
    return x * phi * (k_mn * t2**-0.5 * rm * f0 + k_ct * f0)


def v3(pahs):
    c = V3_CNM
    out = {"T": list(V3_T), "lines": sight_lines(), "w08_alpha_phi04": [w08_alpha(0.4, T, **c) for T in V3_T]}
    gn_grid = np.geomspace(2e-4, 0.3, 24)
    for det, p in pahs.items():
        a, phi, fneg = [], [], []
        for T in V3_T:
            ions = {"C+": c["xC"] * c["n"], "H+": c["xH"] * c["n"]}
            al, _, r = p.rates(c["G0"], c["n"], (c["xC"] + c["xH"]) * c["n"], T, ions)
            a.append(al["C+"])
            phi.append(brentq(lambda ph: w08_alpha(ph, T, **c) - al["C+"], 1e-3, 100.0))
            fneg.append(float(p.x @ r["f"][:, 0] / p.x.sum()))
        direct = []
        for gn in gn_grid:                                   # pahchem's own C I/C II at <G0>/n, G0 = 1.7
            n = c["G0"] / gn
            ions = {"C+": c["xC"] * n, "H+": c["xH"] * n}
            al, _, _ = p.rates(c["G0"], n, 3e-4 * n, T_LINES, ions)
            direct.append((al["C+"] * n + alpha_rr("C+", T_LINES) * 3e-4 * n) / (2.1e-10 * c["G0"]))
        out[det] = {"alpha_C+": a, "phi_eff": phi, "f_anion": fneg, "direct_g_over_n": gn_grid.tolist(),
                    "direct_ratio": direct}
    return out


# == V13: Lepp, Dalgarno, van Dishoeck & Black 1988 ==============================================
#: Lepp+88 Table 2, N(X+)/N(X): x_PAH -> (C, Mg, S, Ca, Fe); observed: (best, lo, hi)
LD88 = {
    "zeta Per": {"model": "F", "n": 325.0, "T": 30.0, "I_UV": 3.5, "A_V": 0.79, "f_H2": 10**-0.23,
                 "table": {1e-11: (130, 229, 457, 556, 624), 1e-8: (123, 211, 434, 509, 572),
                           3e-8: (111, 182, 393, 436, 493), 5e-8: (100, 160, 360, 381, 432),
                           1e-7: (81.6, 123, 296, 290, 330), 1e-6: (19.6, 24.1, 71.7, 58.7, 64.3)},
                 "observed": ((91, 63, 135), (290, 66, 1700), (490, 350, 710), (290, 180, 440), (860, 91, np.inf)),
                 "inferred": 1e-7},
    "zeta Oph": {"model": "G", "n": 350.0, "T": 30.0, "I_UV": 3.5, "A_V": 0.71, "f_H2": 10**-0.20,
                 "table": {1e-11: (196, 277, 683, 849, 814), 1e-8: (183, 249, 639, 760, 729),
                           1e-7: (115, 131, 398, 393, 386), 5e-7: (43.6, 42.1, 148, 129, 126),
                           6.2e-7: (39.1, 36.1, 131, 114, 110), 1e-6: (25.0, 23.0, 83.8, 72.8, 69.2),
                           1e-5: (3.8, 2.8, 11.0, 10.3, 8.9)},
                 "observed": ((29, 13, 260), (48, 17, 67), (80, 52, 124), (108, 98, 118), (162, 40, 280)),
                 "inferred": 6e-7},
}
LD88_ELEMENTS = ("C", "Mg", "S", "Ca", "Fe")
#: gas-phase abundances per H: C 1.4e-4 (Sofia+04); S, Si, Mg, Fe, Ca: assumed cold-cloud values (about Savage &
#: Sembach 1996's zeta Oph depletions on Asplund+09 solar); they only set x_e and the charge exchange
ABUNDANCE = {"C": 1.4e-4, "S": 1.3e-5, "Si": 1.6e-6, "Mg": 2.1e-6, "Fe": 1.7e-7, "Ca": 4e-10}
ZETA_H = 1e-16
X_GRID = (0.0, 1e-8, 3e-8, 1e-7, 3e-7, 6e-7, 1e-6, 3e-6, 1e-5)
N_ZONES = 10


def slab(cloud, pahs, scale):
    """Column ratios N(X+)/N(X) of a uniform two-sided slab (Gauss-Legendre in depth over its half), and N(e)/N_H."""
    c = LD88[cloud]
    u, w = np.polynomial.legendre.leggauss(N_ZONES)
    a1 = 0.25 * c["A_V"] * (u + 1.0)                     # depth from one face, 0 .. A_V / 2
    a2 = c["A_V"] - a1
    n, T = c["n"], c["T"]
    col_ion, col_neu, col_e = dict.fromkeys(ELEMENTS, 0.0), dict.fromkeys(ELEMENTS, 0.0), 0.0
    for d1, d2, wt in zip(a1, a2, w):
        rates = {e: PHOTO[e][0] * c["I_UV"] * 0.5 * (np.exp(-PHOTO[e][1] * d1) + np.exp(-PHOTO[e][1] * d2))
                 for e in ELEMENTS}
        g0 = G_DRAINE * c["I_UV"] * 0.5 * (np.exp(-1.8 * d1) + np.exp(-1.8 * d2))
        ions, ne, _, _ = cell(n, T, rates, ABUNDANCE, ZETA_H, n * (1 - c["f_H2"]), pahs, g0, scale)
        for e in ELEMENTS:
            col_ion[e] += wt * ions[e + "+"]
            col_neu[e] += wt * (ABUNDANCE[e] * n - ions[e + "+"])
        col_e += wt * ne
    return {e: col_ion[e] / col_neu[e] for e in ELEMENTS}, col_e / (2.0 * n)


def fit_x(xs, ratios, observed):
    """x_PAH minimising sum over elements of (log model - log observed)^2 / sigma^2 (sigma: half the log range;
    an open upper end counts as a lower limit), on a log-x grid interpolated between the x > 0 models."""
    xs, R = np.asarray(xs, float), np.log10(np.asarray(ratios, float))
    keep = xs > 0
    lx = np.log10(xs[keep])
    fine = np.linspace(lx.min(), lx.max(), 400)
    chi = np.zeros_like(fine)
    for k, (best, lo, hi) in enumerate(observed):
        m = np.interp(fine, lx, R[keep, k])
        if np.isfinite(hi):
            sig = 0.5 * (np.log10(hi) - np.log10(lo))
            chi += ((m - np.log10(best)) / sig) ** 2
        else:
            sig = np.log10(best) - np.log10(lo)
            chi += np.where(m < np.log10(best), ((m - np.log10(best)) / sig) ** 2, 0.0)
    i = int(np.argmin(chi))
    ok = fine[chi <= chi[i] + 1.0]
    return {"x_best": 10 ** fine[i], "x_lo": 10 ** ok.min(), "x_hi": 10 ** ok.max(), "chi2": float(chi[i])}


def v13(pahs):
    out = {"x": list(X_GRID)}
    for cloud, c in LD88.items():
        xs = sorted(c["table"])
        ld = [c["table"][x] for x in xs]
        res = {"ld88_x": xs, "ld88": ld, "ld88_fit": fit_x(xs, ld, c["observed"])}
        for det, p in pahs.items():
            rows, xe = [], []
            for x in X_GRID:
                r, e = slab(cloud, p, x / p.x.sum())
                rows.append([r[k] for k in LD88_ELEMENTS])
                xe.append(e)
            res[det] = {"ratios": rows, "x_e": xe, "fit": fit_x(X_GRID, rows, c["observed"])}
            print(f"  V13 {cloud} {det}: done", flush=True)
        out[cloud] = res
    return out


# == V4: WD01b Sect. 7 ===========================================================================
V4_ENV = {"n": 30.0, "T": 100.0, "G": 1.13, "xe0": 2e-4}
XE_NOGRAIN = 8.0e-4
#: WD01b Table 2, H+, as printed in arXiv:astro-ph/0105237v2 (2001 May 21). pyCALIMA's table
#: (check_grain_size_distribution.TABLE_2_COEFFS: 12.25, 8.074e-6, 1.378, ...) is the published one and gives an
#: alpha_g(H+) 2.7x larger at the Sect. 7 point; the Sect. 7 numbers were checked here against the preprint only.
WD01B_H_ARXIV_V2 = (3.436, 1.909e-6, 1.496, 1.790e3, 9.749e-4, 0.5612, 7.245e-5)
V4_N = (3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0)


def xi_wd01b():
    """xi [s^-1] that gives WD01b's grain-free x_e = 8.0e-4 at n = 30, T = 100 K (their eq. 23 with alpha_g = 0)."""
    n, T, xh = V4_ENV["n"], V4_ENV["T"], XE_NOGRAIN - V4_ENV["xe0"]
    return alpha_rr("H+", T) * XE_NOGRAIN * xh * n / (1.0 - xh)


def solve_xe(n, T, xi, alpha_g=None, pahs=None, g0=V4_ENV["G"]):
    """(x_e, x_H, alpha(H+) per H, PAH charge per H, electron-source check) of WD01b eq. 23 at x_e0 = 2e-4."""
    xe0 = V4_ENV["xe0"]
    state = {}

    def resid(lxh):
        xh = np.exp(lxh)
        q, a = 0.0, 0.0
        for _ in range(100):                             # the PAHs' charge enters n_e; iterate it
            xe = xe0 + xh + q
            if alpha_g is not None:
                a = alpha_g(xe * n)
            elif pahs is not None:
                al, q_new, r = pahs.rates(g0, n, xe * n, T, {"C+": xe0 * n, "H+": xh * n})
                a = al["H+"]
                state.update(r=r, al=al)
                if abs(q_new - q) < 1e-6 * xe:
                    q = q_new
                    break
                q = q_new
            else:
                break
        state.update(xe=xe0 + xh + q, a=a, q=q)
        return np.log(xi * (1.0 - xh)) - np.log((alpha_rr("H+", T) * (xe0 + xh + q) + a) * xh * n)

    lxh = brentq(resid, np.log(1e-7), np.log(0.5), xtol=1e-10)
    resid(lxh)
    out = {"x_e": state["xe"], "x_H": float(np.exp(lxh)), "alpha_H": state["a"], "q": state["q"]}
    if pahs is not None:
        r, al, xh = state["r"], state["al"], np.exp(lxh)
        sink = al["C+"] * xe0 * n + al["H+"] * xh * n            # ions neutralised per H per s
        out.update(alpha_C=al["C+"], electron_source=r["electron_source"], ion_sink=sink)
    return out


def v4(pahs, pahs_full):
    n0, T = V4_ENV["n"], V4_ENV["T"]
    xi = xi_wd01b()

    def wd(ne):
        return gsd.get_analytical_recomb_coefficient("H+", V4_ENV["G"], ne, T)

    def wd_arxiv(ne):
        c0, c1, c2, c3, c4, c5, c6 = WD01B_H_ARXIV_V2
        psi = V4_ENV["G"] * np.sqrt(T) / ne
        return 1e-14 * c0 / (1.0 + c1 * psi**c2 * (1.0 + c3 * T**c4 * psi ** (-c5 - c6 * np.log(T))))

    out = {"xi": xi, "n": list(V4_N), "no_grains": [solve_xe(n, T, xi)["x_e"] for n in V4_N],
           "wd01b": [solve_xe(n, T, xi, alpha_g=wd) for n in V4_N],
           "wd01b_arxiv": [solve_xe(n, T, xi, alpha_g=wd_arxiv) for n in V4_N]}
    for det, p in pahs.items():
        out[det] = [solve_xe(n, T, xi, pahs=p) for n in V4_N]
        out[det + "_G1.7"] = solve_xe(n0, T, xi, pahs=p, g0=1.7)
    for det, p in pahs_full.items():
        out["full_" + det] = solve_xe(n0, T, xi, pahs=p)
    return out


# -- report and plots ----------------------------------------------------------------------------
def report(d):
    v = d["V3"]
    print("\n== V3: W08 C I/C II ==")
    print("pahchem's effective Phi (W08 three-state balance giving pahchem's alpha(C+)), CNM G0 1.7, n 30, x_e 3e-4")
    print("   T [K] " + "".join(f"{det:>26s}" for det in DETACHMENTS) + "   W08 alpha(Phi=0.4)")
    for k, T in enumerate(v["T"]):
        cells = "".join(f"   Phi {v[det]['phi_eff'][k]:5.2f} a {v[det]['alpha_C+'][k]:.2e} f- {v[det]['f_anion'][k]:.2f}"
                        for det in DETACHMENTS)
        print(f"  {T:6.0f}{cells}   {v['w08_alpha_phi04'][k]:.2e}")
    lines = [s for s in v["lines"] if not s["outlier"]]
    phis = np.array([implied_phi(s["ratio"], s["g_over_n"], T_LINES) for s in lines])
    print(f"\nW08 Table 1, A_V > 0.25 without 23 Ori and pi Sco ({len(lines)} lines), T = {T_LINES:g} K, R = 3.5e-17:")
    print(f"  implied Phi (W08 C I): median {np.median(phis):.2f}, 16-84 % {np.percentile(phis, 16):.2f}-"
          f"{np.percentile(phis, 84):.2f}; <G0>/n {min(s['g_over_n'] for s in lines):.1e}-"
          f"{max(s['g_over_n'] for s in lines):.1e}")
    print("  line          A_V  <G0>/n   C I/C II  Phi(W08)  Phi(Burgh+10)  " +
          "  ".join(f"obs/pahchem({det[:4]})" for det in DETACHMENTS))
    for s in v["lines"]:
        pb = implied_phi(s["ratio_burgh"], s["g_over_n"], T_LINES) if s["ratio_burgh"] else None
        pred = [np.exp(np.interp(np.log(s["g_over_n"]), np.log(v[det]["direct_g_over_n"]),
                                 np.log(v[det]["direct_ratio"]))) for det in DETACHMENTS]
        s["pred"] = pred
        print(f"  {s['name']:12s} {s['av']:4.2f}  {s['g_over_n']:.2e}  {s['ratio']:.2e}  "
              f"{implied_phi(s['ratio'], s['g_over_n'], T_LINES):7.2f}  " +
              (f"{pb:12.2f}" if pb is not None else " " * 12) + "  " +
              "  ".join(f"{s['ratio'] / p:18.2f}" for p in pred) + ("   (outlier)" if s["outlier"] else ""))
    for det in DETACHMENTS:
        k = int(np.argmin(np.abs(np.array(v["T"]) - T_LINES)))
        lr = np.log10([s["ratio"] / s["pred"][DETACHMENTS.index(det)] for s in lines])
        print(f"  {det}: Phi_eff(T=70-100) {v[det]['phi_eff'][k]:.2f}-{v[det]['phi_eff'][k + 1]:.2f}; "
              f"obs/pahchem direct: median {10 ** np.median(lr):.2f}, 16-84 % {10 ** np.percentile(lr, 16):.2f}-"
              f"{10 ** np.percentile(lr, 84):.2f}")
    pb = [implied_phi(s["ratio_burgh"], s["g_over_n"], T_LINES) for s in v["lines"] if s["ratio_burgh"]]
    pw = [implied_phi(s["ratio"], s["g_over_n"], T_LINES) for s in v["lines"] if s["ratio_burgh"]]
    print(f"  Burgh+10 lines: Phi with W08 C I {np.round(pw, 2)}, with Burgh+10 C I {np.round(pb, 2)}")
    for det in DETACHMENTS:
        k = DETACHMENTS.index(det)
        print(f"  {det}: obs/pahchem direct on the Burgh+10 lines, W08 C I " +
              str(np.round([s["ratio"] / s["pred"][k] for s in v["lines"] if s["ratio_burgh"]], 2)) + ", Burgh+10 C I " +
              str(np.round([s["ratio_burgh"] / s["pred"][k] for s in v["lines"] if s["ratio_burgh"]], 2)))

    v = d["V13"]
    print("\n== V13: Lepp+88 Table 2, N(X+)/N(X) ==")
    for cloud, c in LD88.items():
        r = v[cloud]
        print(f"\n{cloud} (Lepp+88 model {c['model']}: n {c['n']:g}, T {c['T']:g}, I_UV {c['I_UV']:g}; "
              f"A_V {c['A_V']}, f_H2 {c['f_H2']:.2f})")
        print("  observed     " + "  ".join(f"{e:>5s} {b:g} ({lo:g}-{hi:g})" for e, (b, lo, hi) in
                                         zip(LD88_ELEMENTS, c["observed"])))
        for x, row in zip(r["ld88_x"], r["ld88"]):
            print(f"  Lepp+88 x {x:7.1e}: " + "  ".join(f"{e} {val:6.1f}" for e, val in zip(LD88_ELEMENTS, row)))
        f = r["ld88_fit"]
        print(f"  Lepp+88 fitted here: x {f['x_best']:.1e} ({f['x_lo']:.1e}-{f['x_hi']:.1e}), chi2 {f['chi2']:.1f}; "
              f"they infer {c['inferred']:.0e}")
        ld0 = np.array(r["ld88"][0])
        for det in DETACHMENTS:
            R = np.array(r[det]["ratios"])
            print(f"  pahchem {det}:")
            for x, row, xe in zip(v["x"], R, r[det]["x_e"]):
                print(f"    x {x:7.1e}: " + "  ".join(f"{e} {val:7.1f}" for e, val in zip(LD88_ELEMENTS, row)) +
                      f"   x_e {xe:.2e}   fall vs x=0: " + " ".join(f"{val:.2f}" for val in row / R[0]))
            f = r[det]["fit"]
            print(f"    fitted x_PAH {f['x_best']:.1e} ({f['x_lo']:.1e}-{f['x_hi']:.1e}), chi2 {f['chi2']:.1f}")
            for xl, row in zip(r["ld88_x"][1:], r["ld88"][1:]):
                fall_ld = np.array(row) / ld0
                fall_p = np.array([np.exp(np.interp(np.log(xl), np.log(v["x"][1:]), np.log(R[1:, k] / R[0, k])))
                                   for k in range(5)])
                print(f"    x {xl:.1e}: fall Lepp+88 {np.round(fall_ld, 2)}  pahchem {np.round(fall_p, 2)}")

    v = d["V4"]
    print("\n== V4: WD01b Sect. 7, x_e at T = 100 K ==")
    print(f"  xi from the grain-free x_e = 8.0e-4 at n = 30: {v['xi']:.3e} s^-1")
    print("    n   no grains   WD01b fit (published, arXiv v2)  " +
          "  ".join(f"pahchem {det:11s}" for det in DETACHMENTS))
    for k, n in enumerate(v["n"]):
        print(f"  {n:5.0f}  {v['no_grains'][k]:.2e}   {v['wd01b'][k]['x_e']:.2e}  {v['wd01b_arxiv'][k]['x_e']:.2e}  " +
              "  ".join(f"{v[det][k]['x_e']:.2e} ({v[det][k]['alpha_H']:.1e})" for det in DETACHMENTS))
    k = list(v["n"]).index(30.0)
    print(f"  at n = 30: WD01b fit alpha(H+) {v['wd01b'][k]['alpha_H']:.2e} (published table), "
          f"{v['wd01b_arxiv'][k]['alpha_H']:.2e} (arXiv v2 table)")
    for det in DETACHMENTS:
        s = v[det][k]
        print(f"  pahchem N_C {d.get('nc_range', '32-40')} {det}: x_e {s['x_e']:.2e}, alpha(H+) {s['alpha_H']:.2e}, alpha(C+) "
              f"{s['alpha_C']:.2e}, PAH charge {s['q']:.1e} per H; electron_source {s['electron_source']:.3e} vs "
              f"-ion sink {-s['ion_sink']:.3e}; at G0 = 1.7: x_e {v[det + '_G1.7']['x_e']:.2e}")
        if "full_" + det in v:
            s = v["full_" + det]
            print(f"  pahchem whole library (WD01 b_C = 6e-5) {det}: x_e {s['x_e']:.2e}, alpha(H+) {s['alpha_H']:.2e}, "
                  f"alpha(C+) {s['alpha_C']:.2e}, PAH charge {s['q']:.1e}; electron_source {s['electron_source']:.3e}"
                  f" vs -ion sink {-s['ion_sink']:.3e}")


def _axes(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in", top=True, right=True)


def plot(d, out):
    use_calima_style()
    # V3
    v = d["V3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for det in DETACHMENTS:
        col, ls = STYLE[det]
        ax.plot(v["T"], v[det]["phi_eff"], color=col, ls=ls, lw=1.8, label=f"pahchem, {det}")
    ax.axhspan(0.2, 0.4, color="0.85", zorder=0, label=r"W08 fit, $\Phi_{\rm PAH}$ 0.2--0.4")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"$T$ [K]", fontsize=14)
    ax.set_ylabel(r"effective $\Phi_{\rm PAH}$", fontsize=14)
    ax.legend(frameon=False, fontsize=9)
    _axes(ax)
    ax = axes[1]
    gn = np.geomspace(2e-3, 0.3, 100)
    ax.loglog(gn, w08_ratio(0.4, gn, T_LINES), "k-", lw=1.2, label=r"W08 eqs. B15+B17, $\Phi = 0.4$")
    k = int(np.argmin(np.abs(np.array(v["T"]) - T_LINES)))
    for det in DETACHMENTS:
        col, ls = STYLE[det]
        ax.loglog(v[det]["direct_g_over_n"], v[det]["direct_ratio"], color=col, ls=ls, lw=1.8,
                  label=f"pahchem direct, {det}")
    for s in v["lines"]:
        mk = "o" if not s["outlier"] else "x"
        ax.plot(s["g_over_n"], s["ratio"], mk, color="0.25", ms=4.5, mfc="0.25" if not s["outlier"] else None)
        if s["ratio_burgh"]:
            ax.plot(s["g_over_n"], s["ratio_burgh"], "s", color="#b8336a", ms=5, mfc="none")
            ax.plot([s["g_over_n"]] * 2, [s["ratio"], s["ratio_burgh"]], color="#b8336a", lw=0.8)
    ax.plot([], [], "o", color="0.25", ms=4.5, label=r"W08 Table 1, $A_V > 0.25$")
    ax.plot([], [], "s", color="#b8336a", mfc="none", ms=5, label="Burgh+10 C I")
    ax.set_xlabel(r"$\langle G_0\rangle/n$ [cm$^3$], from H$_2$", fontsize=14)
    ax.set_ylabel(r"$N({\rm C\,I})/N({\rm C\,II})$", fontsize=14)
    ax.set_xlim(gn[0], gn[-1])
    ax.legend(frameon=False, fontsize=8)
    _axes(ax)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out / "pahchem_v3_w08_cicii.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # V13
    v = d["V13"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True)
    xs = np.array(v["x"])
    for r_, cloud in enumerate(LD88):
        c = LD88[cloud]
        for k, el in enumerate(LD88_ELEMENTS):
            ax = axes[r_, k]
            best, lo, hi = c["observed"][k]
            ax.axhspan(lo, hi if np.isfinite(hi) else 1e5, color="0.88", zorder=0)
            ax.axhline(best, color="0.5", lw=0.8)
            ax.plot(v[cloud]["ld88_x"][1:], np.array(v[cloud]["ld88"])[1:, k], "ko-", ms=4, lw=1.0,
                    label="Lepp+88 model " + c["model"])
            ax.axvline(c["inferred"], color="k", lw=0.6, ls="--")
            for det in DETACHMENTS:
                col, ls = STYLE[det]
                ax.plot(xs[1:], np.array(v[cloud][det]["ratios"])[1:, k], color=col, ls=ls, lw=1.8,
                        label=f"pahchem, {det}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1.0, 3e3)
            if r_ == 0:
                ax.set_title(el, fontsize=14)
            if k == 0:
                ax.set_ylabel(("$\\zeta$ Per" if cloud == "zeta Per" else "$\\zeta$ Oph") +
                              r"  $N({\rm X^+})/N({\rm X})$", fontsize=14)
            if r_ == 1:
                ax.set_xlabel(r"$x_{\rm PAH}$", fontsize=14)
            _axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.subplots_adjust(hspace=0, wspace=0.25)
    fig.savefig(out / "pahchem_v13_lepp88.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # V4
    v = d["V4"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.loglog(v["n"], v["no_grains"], "k--", lw=1.2, label="no grains")
    ax.loglog(v["n"], [s["x_e"] for s in v["wd01b"]], "k-", lw=1.4, label=r"WD01b fit $\alpha_g$, published")
    ax.loglog(v["n"], [s["x_e"] for s in v["wd01b_arxiv"]], "k-.", lw=1.2, label=r"WD01b fit $\alpha_g$, arXiv v2")
    for det in DETACHMENTS:
        col, ls = STYLE[det]
        ax.loglog(v["n"], [s["x_e"] for s in v[det]], color=col, ls=ls, lw=1.8,
                  label=r"pahchem $N_{\rm C}$ 32--40, " + det)
        if "full_" + det in v:
            ax.plot(30.0, v["full_" + det]["x_e"], "D", color=col, ms=6, mfc="none")
    ax.plot([30, 30], [8.0e-4, 4.5e-4], "s", color="#b8336a", ms=6, label=r"WD01b Sect. 7, $n = 30$")
    if any("full_" + det in v for det in DETACHMENTS):
        ax.plot([], [], "D", color="0.3", mfc="none", ms=6, label="pahchem, whole library")
    ax.set_xlabel(r"$n_{\rm H}$ [cm$^{-3}$]", fontsize=14)
    ax.set_ylabel(r"$x_e$ ($T = 100$ K)", fontsize=14)
    ticks = (2e-4, 3e-4, 5e-4, 1e-3, 2e-3)
    ax.set_yticks(ticks, [rf"${t * 1e4:g}\times10^{{-4}}$" for t in ticks])
    ax.legend(frameon=False, fontsize=8)
    _axes(ax)
    fig.savefig(out / "pahchem_v4_wd01b_xe.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "pahchem_v3_w08_cicii.png", out / "pahchem_v13_lepp88.png", out / "pahchem_v4_wd01b_xe.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--full-library", action="store_true",
                    help="add V4's whole-library (b_C = 6e-5) run: ~1800 new bases per field shape, >10 h on 2 workers")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--nc-range", type=int, nargs=2, default=(32, 40), metavar=("LO", "HI"),
                    help="the molecules (with an anion) that carry the PAHs: W08's N_C 32-40 by default, at x_PAH = 6e-7; "
                         "another range takes W08's carbon, 6e-7 x 35 C per H, shared equally by carbon (a suffix on the outputs)")
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    lo, hi = args.nc_range
    tag = "" if (lo, hi) == (32, 40) else f"_nc{lo}-{hi}"
    js = _paths.get_results_dir() / f"pahchem_diffuse_ionisation{tag}.json"
    if not args.plot_only:
        full = library_v400()
        lib = [m for m in full if lo <= m.nc <= hi and -1 in m.charges]
        if tag:
            x = X_PAH_W08 * 35.0 / np.array([m.nc for m in lib], dtype=float) / len(lib)
        else:
            x = np.full(len(lib), X_PAH_W08 / len(lib))
        pahs = {det: PAHs(lib, x, det, args.workers) for det in DETACHMENTS}
        pahs_full = {}
        if args.full_library:
            lib_f, x_f = wd01_weighted(full)
            print(f"whole library: {len(lib_f)} molecules, {x_f.sum():.2e} per H", flush=True)
            pahs_full = {det: PAHs(lib_f, x_f, det, args.workers) for det in DETACHMENTS}
        print(f"N_C {lo}-{hi}: {len(lib)} molecules at x_PAH = {x.sum():.3g} per H ({(x * [m.nc for m in lib]).sum():.3g} C per H)",
              flush=True)
        d = {"V3": v3(pahs)}
        print("V3 done", flush=True)
        d["V4"] = v4(pahs, pahs_full)
        print("V4 done", flush=True)
        d["V13"] = v13(pahs)
        d["nc_range"] = f"{lo}-{hi}"
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(d, default=float))
    d = json.loads(js.read_text())
    report(d)
    out = (args.out or _paths.get_plots_dir("pahchem")) / (tag.lstrip("_") or ".")
    out.mkdir(parents=True, exist_ok=True)
    plot(d, out)


if __name__ == "__main__":
    main()
