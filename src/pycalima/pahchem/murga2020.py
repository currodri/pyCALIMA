"""Murga, Kirsanova, Vasyunin & Pavlyuchenkov (2020) per-molecule PAH evolution.

Method card: ``method_cards/murga2020.toml`` ([network], [hydrogen],
[dissociation], [environment]). One parent (N_C0, N_H0) is evolved at fixed
charge Z in the Fig. 2 environment of one A_V point, with real-valued N_C(t)
and N_H(t) (M20 eqs. 1-3):

    dN_C/dt = -2 R_C2H2
    dN_H/dt = -R_H - 2 R_H2 - 2 R_C2H2 + H
    N_C2H2  = (1/2) sum ``|dN_C|`` over steps with N_H > 1          (eq. 3)

- **Photo-rates.** R_frag come from ``pahchem.shiva`` (DL01 or Tielens
  T(E), LD01 C_abs, WD01a Y_ion, stationary p(E) with continuous cooling;
  M20 eqs. 4-6), with the M20 Table 1 parameters. For X_H = N_H/N_C <=
  X_H0 = N_H0/N_C0 the normal set applies (H 4.3 eV, H2 3.52 eV, C2H2 4.6
  eV); above it the super-H set applies (H 1.4 eV for Z <= 0 and 1.55 eV
  for Z > 0; C2H2 2.0 eV; no H2 photo-channel). They are tabulated on an
  (N_C, N_H) grid once per (Z, chi) and interpolated in log R.
- **H addition** (Sect. 2.2, Fig. 1):

  - neutrals: n_H x 0.5 (N_edge k_edge + N_core k_core), with k_edge =
    0.5 (k_1 + k_4) and k_core = k_3a1 from Goumans (2011) Table 3. The
    edge/core split is fixed by the parent (N_edge = N_H0), and the ratio is
    kept as N_C falls;
  - ions: n_H x {1.4e-10 N_H/N_C (normal), 0.5 (1.4e-10 N_H/N_C + 5e-11)
    (dehydrogenated), 1e-12 (super-H)}, plus n_H2 x 5e-13 (eq. 11).
- **Eley-Rideal** abstraction on super-H states: 8.7e-13 sqrt(T/100) n_H
  (eq. 10).

The ODE is stiff (H exchange can reach 1e4 yr^-1 while carbon is lost over
1e4 yr), so it is integrated with an implicit adaptive solver (LSODA/Radau).
M20 use explicit Euler with dt = 1 yr; ``euler`` reproduces that for slow
cases, and the test suite checks that the two agree there.

Choices where the card says "not stated" are fields of ``Options``, each with
a documented default and the sensitivity measured in check_murga2020.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from pycalima import _paths
from pycalima.pahchem import shiva

YR = 3.15576e7

#: M20 Sect. 3: the 20 parents, in the order of Fig. 4
PARENTS = ["C16H10", "C24H12", "C30H14", "C32H14", "C40H16", "C42H24", "C48H12", "C48H18", "C54H18",
           "C66H20", "C78H22", "C90H24", "C96H24", "C96H26", "C110H26", "C112H26", "C128H28", "C130H28",
           "C144H30", "C150H30"]

#: M20 Fig. 1 / Sect. 2.2
K_ION_NORMAL, K_ION_DEHYDRO, K_ION_SUPERH, K_H2_ION = 1.4e-10, 5.0e-11, 1.0e-12, 5.0e-13
K_ER = 8.7e-13                                   # x sqrt(T/100) (eq. 10)


def parse(formula):
    nc, nh = re.match(r"C(\d+)H(\d+)", formula).groups()
    return int(nc), int(nh)


@dataclass(frozen=True)
class Options:
    """The choices M20 leave open (card), with the defaults used here.

    - ``temperature``: "dl01" (as M20 state) or "tielens" (as Murga & Wiebe
      2022 Fig. 1 behaves); the T_m of k_diss only.
    - ``p_reading``: "time" (the time-averaged p(E), Pavlyuchenkov+12's
      ergodic distribution) or "post" (p(E) just after an absorption). Fig.
      4 decides it: with "post" the normal C24H12 loses C2H2 at ~3 yr^-1 in
      the Horsehead at A_V = 0.1, against ~1e-3 molecules per 1e6 yr in Fig.
      4 (check_murga2020.py).
    - ``compete``: True puts all channels of the regime in the denominator:
      the SHIVA definition, Murga+2019 eq. B2 ("k_diss = k_H + k_H2 +
      k_C2H2"; the outcomes' probabilities "sum up to unity"), which M20
      cite for the scheme ("according to the scheme presented in the works
      of murga16a, murga19") and imply for the super-H set ("we do not
      consider this channel [H2] as a competitor in the dissociation").
      False (default) reads eq. 4 per fragment against k_IR alone. Fig. 4
      cannot choose: competition brings the 140 partial-destruction points
      from 2.5 to 1.2 dex rms (median +2.4 -> +0.1), but leaves the Orion
      A_V = 0.1 neutrals with N_C >= 110 0.6-1.3 dex short of full
      destruction within 1e6 yr (C150H30 sits at the maximum N_H and yields
      3.9 of 75 C2H2, where Figshare C2H2 production ends at 1e4 yr;
      check_murga2020.py). False keeps the full-destruction gates.
    - ``one_minus_yion``: False (default) is eq. 6 as printed, Y_diss times
      every 5-13.6 eV absorption; True counts only non-ionising ones (M19
      eq. B2's [1 - Y_ion]; M20: the photon energy "either is spent to
      ionization or converts to the internal energy"; ``rate_table``). With
      competition it moves Fig. 4 by < 0.1 dex rms and costs a further 0.2
      dex on the Orion A_V = 0.1 large neutrals.
    - ``atoms_per_molecule_event``: 2 counts one H2 / C2H2 event as 2 atoms
      in eq. 1 (R = events s^-1); 4 applies the text's "rates for molecules
      are multiplied by 2" on top of eq. 1's factor 2.
    - ``low_nh``: what C2H2 loss does at N_H <= 1, where eq. 3 counts no
      C2H2: "c2" (literal eq. 1: N_C still falls, as C2, uncounted) or
      "none" (no carbon loss).
    - ``nh_max``: "nc_plus_nh0": the M22 'maximum' state, every peripheral C
      with two H and every inner C with one (N_H <= N_C + N_edge); addition
      stops there.
    - ``er_atoms``: H removed per Eley-Rideal event (1).
    - ``normal_band``: ``|N_H - N_H,normal|`` below which an ion takes the
      'normal' addition rate (with real-valued N_H, exact normality never
      occurs).
    - ``goumans_t_range``: the Goumans fits are valid 40-2000 K, and Goumans
      warns against their use below 40 K (the low-T fit has gamma < 0 and
      diverges: at 15 K it gives ~4e-14 cm^3 s^-1 per site). ``"clamp_low"``
      (default) holds the 40 K value below 40 K and extrapolates the high-T
      fit above 2000 K; ``"clamp"`` holds both ends; ``"extrapolate"`` uses
      the formula everywhere.
    - ``nc_min``: below it the molecule is taken as destroyed at once, the
      remaining carbon counted as C2H2 if N_H > 1.
    - ``superh_offset``: H above the normal N_H before the super-H rate set
      applies (0: X_H > X_H0 literally). The super-H rates are ~1e12 x the
      normal ones, so with real-valued N_H any trickle of H addition above
      X_H0 opens the 2.0 eV C2H2 channel; M20 do not say how N_H crosses the
      threshold, and the A_V = 2 results hinge on it.
    - ``h2_addition``: "all" applies the ion H2 addition (n_H2 x 5e-13, eq.
      11) at every N_H; "dehydro" only below the normal state.
    """

    temperature: str = "dl01"
    p_reading: str = "time"
    compete: bool = False
    one_minus_yion: bool = False
    atoms_per_molecule_event: float = 2.0
    low_nh: str = "c2"
    nh_max: str = "nc_plus_nh0"
    er_atoms: float = 1.0
    normal_band: float = 0.5
    goumans_t_range: str = "clamp_low"
    nc_min: float = 6.0
    superh_offset: float = 0.0
    h2_addition: str = "all"


DEFAULT = Options()


# ---------------------------------------------------------------------------
# Environment and H addition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Environment:
    av: float
    chi: float
    t_gas: float
    n_h: float                 # atomic H [cm^-3]
    n_h2: float                # H2 [cm^-3]


def environment(site, av, root=None):
    """The Fig. 2 point of ``site`` (murga_data.SITES_2020) nearest ``av``.

    n(H) = x(H) n_tot and n(H2) = x(H2) n_tot, with n_tot and the x as each
    file prints them (the Orion and Horsehead files normalise x differently;
    murga_data.read_profile_2020).
    """
    from pycalima.pahchem.murga_data import read_profile_2020

    p = read_profile_2020(site, root)
    i = int(np.argmin(np.abs(p["av"] - av)))
    return Environment(float(p["av"][i]), float(p["chi"][i]), float(p["t_gas"][i]),
                       float(p["x_h"][i] * p["n_tot"][i]), float(p["x_h2"][i] * p["n_tot"][i]))


@lru_cache(maxsize=None)
def _goumans():
    return pd.read_csv(_paths.get_external_data_path("pahchem/goumans2011_table3.csv"), comment="#")


def goumans_rate(site, T, t_range="clamp_low"):
    """Goumans (2011) Table 3, H addition per site [cm^3 site^-1 s^-1]."""
    t = _goumans()
    r = t[(t.isotope == "H") & (t.site.astype(str) == site)]
    if t_range == "clamp":
        T = min(max(T, 40.0), 2000.0)
    elif t_range == "clamp_low":
        T = max(T, 40.0)
    row = r[r.range == ("high" if T > r.t_c_K.iloc[0] else "low")].iloc[0]
    return float(row.alpha_cm3_site_s * (T / 300.0) ** row.beta * np.exp(-row.gamma_K / T))


def h_addition_neutral(nc, nc0, nh0, T, opts=DEFAULT):
    """0.5 (N_edge k_edge + N_core k_core) [cm^3 s^-1] (Sect. 2.2)."""
    k_edge = 0.5 * (goumans_rate("1", T, opts.goumans_t_range) + goumans_rate("4", T, opts.goumans_t_range))
    k_core = goumans_rate("3a1", T, opts.goumans_t_range)
    edge = nc * nh0 / nc0
    return 0.5 * (edge * k_edge + (nc - edge) * k_core)


def h_addition_ion(nc, nh, nh_normal, opts=DEFAULT):
    """Fig. 1 ion rates [cm^3 s^-1]: normal, dehydrogenated, super-H."""
    if abs(nh - nh_normal) <= opts.normal_band:
        return K_ION_NORMAL * nh / nc
    if nh < nh_normal:
        return 0.5 * (K_ION_NORMAL * nh / nc + K_ION_DEHYDRO)
    return K_ION_SUPERH


# ---------------------------------------------------------------------------
# Photo-rate tables
# ---------------------------------------------------------------------------

CHANNEL_SETS = {False: ("H", "H2", "C2H2"),                       # X_H <= X_H0
                True: ("H_superh", None, "C2H2_superh")}           # X_H > X_H0


def _nh_grid(nh_top):
    g = list(range(0, 11)) + list(range(12, 41, 2)) + list(range(44, int(nh_top) + 5, 4))
    return np.array(sorted(set(v for v in g if v <= nh_top + 4)), float)


@lru_cache(maxsize=None)
def rate_table(charge, chi, nc_max, nh_top, temperature="dl01", de_ev=0.1, e_max_ev=50.0, one_minus_yion=False):
    """log10 R [s^-1] on (N_C, N_H): array [nc, nh, superh(2), channel(3), variant(4)].

    variant: 0 post/independent, 1 post/compete, 2 time/independent,
    3 time/compete. Channels H, H2, C2H2 (H2 is absent in the super-H set:
    -300). Points with N_H > N_C + nh_top/... beyond the evaluated range take
    the last evaluated N_H of their N_C.

    ``one_minus_yion``: the rate counts only the absorptions that do not
    ionise, int_5^13.6 C_abs (1 - Y_ion) F dE (Murga+2019 eq. B2's
    [1 - Y_ion]; M20 Sect. 2.1: "the photon energy either is spent to
    ionization or converts to the internal energy"); False multiplies the
    yield by all absorptions (M20 eq. 6 read alone).
    """
    field = shiva.mmp83(chi)
    nc_grid = np.unique(np.concatenate([np.arange(4.0, nc_max + 1.0, 4.0), [float(nc_max)]]))
    nh_grid = _nh_grid(nh_top)
    out = np.full((nc_grid.size, nh_grid.size, 2, 3, 4), -300.0)
    hs = "H_superh_neg" if charge <= 0 else "H_superh_pos"
    for i, nc in enumerate(nc_grid):
        last = None
        for j, nh in enumerate(nh_grid):
            if nh > 1.7 * nc + 2 and last is not None:
                out[i, j] = last
                continue
            m = shiva.ShivaMolecule(int(nc), int(nh), charge, de_ev=de_ev, e_max_ev=e_max_ev, temperature=temperature)
            p_time, p_post = m.distribution(field)
            if one_minus_yion:
                E_ph, h = m.heating(field, ionising_photons_heat=False)       # C_abs (1 - Y_ion) F
                keep = E_ph >= shiva.E_DESTRUCTION_MIN
                r_abs = float(np.trapezoid(h[keep], E_ph[keep]))
            else:
                r_abs = m.absorption_rate(field)
            for s, names in enumerate((("H", "H2", "C2H2"), (hs, None, "C2H2_superh"))):
                # computed at every N_H (the lack of H atoms at N_H < 1, 2 is applied in the ODE,
                # not here, so that log R interpolates smoothly)
                ks = [m.k_diss(n) if n is not None else np.zeros_like(m.energy_ev) for n in names]
                tot = sum(ks)
                for c, k in enumerate(ks):
                    y_ind = np.where(k > 0, k / (k + m.k_ir + 1e-300), 0.0)
                    y_cmp = np.where(k > 0, k / (tot + m.k_ir + 1e-300), 0.0)
                    vals = (p_post @ y_ind, p_post @ y_cmp, p_time @ y_ind, p_time @ y_cmp)
                    out[i, j, s, c] = np.log10(np.maximum(np.array(vals) * r_abs, 1e-300))
            last = out[i, j].copy()
    return nc_grid, nh_grid, out


VARIANTS = {("post", False): 0, ("post", True): 1, ("time", False): 2, ("time", True): 3}
TABLE_NC_MAX = 150
TABLE_NH_TOP = 180.0           # C150H30 at N_H <= N_C + N_H0


class Rates:
    """Photo-rates of one parent at fixed Z and chi, bilinear in log R on the table grid."""

    def __init__(self, nc0, nh0, charge, chi, opts=DEFAULT):
        self.nc0, self.nh0, self.charge, self.opts = nc0, nh0, charge, opts
        # one table per (Z, chi, T(E)) serves every parent: the grid spans the largest (C150H30)
        nc_max = max(TABLE_NC_MAX, int(nc0))
        nh_top = max(TABLE_NH_TOP, float(nc0 + nh0))
        nc_grid, nh_grid, table = rate_table(charge, float(chi), nc_max, nh_top, opts.temperature,
                                             one_minus_yion=opts.one_minus_yion)
        self.nc_grid, self.nh_grid = nc_grid, nh_grid
        self.table = table[..., VARIANTS[(opts.p_reading, opts.compete)]]          # [nc, nh, regime, channel]

    def both(self, nc, nh):
        """log10 R [s^-1], shape (2 regimes, 3 channels), at real-valued (N_C, N_H)."""
        g1, g2 = self.nc_grid, self.nh_grid
        x = min(max(nc, g1[0]), g1[-1])
        y = min(max(nh, g2[0]), g2[-1])
        i = min(max(int(np.searchsorted(g1, x)) - 1, 0), g1.size - 2)
        j = min(max(int(np.searchsorted(g2, y)) - 1, 0), g2.size - 2)
        u = (x - g1[i]) / (g1[i + 1] - g1[i])
        v = (y - g2[j]) / (g2[j + 1] - g2[j])
        T = self.table
        return ((1 - u) * (1 - v) * T[i, j] + u * (1 - v) * T[i + 1, j] + (1 - u) * v * T[i, j + 1]
                + u * v * T[i + 1, j + 1])

    def __call__(self, nc, nh, superh_weight=None):
        """(R_H, R_H2, R_C2H2) [s^-1]; the super-H set above X_H0 (``superh_weight`` blends)."""
        lg = 10.0 ** self.both(nc, nh)
        if superh_weight is None:
            superh_weight = float(nh / nc > self.nh0 / self.nc0)
        r = (1.0 - superh_weight) * lg[0] + superh_weight * lg[1]
        return float(r[0]), float(r[1]), float(r[2])


# ---------------------------------------------------------------------------
# The trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    t_yr: np.ndarray
    nc: np.ndarray
    nh: np.ndarray
    n_c2h2: np.ndarray
    destroyed_at_yr: float | None

    @property
    def total_c2h2(self):
        return float(self.n_c2h2[-1])


def _ramp(x):
    return min(max(x + 0.5, 0.0), 1.0)


def _rhs(rates, env, nc0, nh0, charge, opts):
    """dy/dt for y = (N_C, N_H, N_C2H2), times in s.

    The switches of the model (normal / super-H rate sets at X_H0, the ion
    'normal' rate, eq. 3's N_H > 1) are linear ramps of width
    ``opts.normal_band`` in N_H (and 1 at N_H = 1): a regularisation that
    lets the stiff solver step across them; M20's Euler smears them too.
    """
    w = opts.normal_band
    k_edge = 0.5 * (goumans_rate("1", env.t_gas, opts.goumans_t_range)
                    + goumans_rate("4", env.t_gas, opts.goumans_t_range))
    k_core = goumans_rate("3a1", env.t_gas, opts.goumans_t_range)
    edge_frac = nh0 / nc0
    er_rate = K_ER * np.sqrt(env.t_gas / 100.0) * env.n_h * opts.er_atoms
    a = opts.atoms_per_molecule_event

    def f(t, y):
        nc = max(y[0], opts.nc_min)
        nh = max(y[1], 0.0)
        nh_normal = edge_frac * nc
        s = min(max((nh - nh_normal - opts.superh_offset) / w, 0.0), 1.0)    # 0 normal/dehydro, 1 super-H
        r_h, r_h2, r_c = rates(nc, nh, s)
        if charge == 0:
            add = env.n_h * 0.5 * (edge_frac * nc * k_edge + (1.0 - edge_frac) * nc * k_core)
        else:
            near = _ramp(0.5 - abs(nh - nh_normal) / w)          # 1 at the normal state, 0 beyond +-w
            dehydro = 0.5 * (K_ION_NORMAL * nh / nc + K_ION_DEHYDRO)
            s_add = min(max((nh - nh_normal) / w, 0.0), 1.0)
            k_ion = near * K_ION_NORMAL * nh / nc + (1.0 - near) * ((1.0 - s_add) * dehydro + s_add * K_ION_SUPERH)
            h2 = 1.0 if opts.h2_addition == "all" else min(max((nh_normal - nh) / w, 0.0), 1.0)
            add = env.n_h * k_ion + env.n_h2 * K_H2_ION * h2
        if opts.nh_max == "nc_plus_nh0":
            add *= min(max(nc + edge_frac * nc - nh, 0.0), 1.0)
        counted = min(max(nh - 1.0, 0.0), 1.0)                  # eq. 3: N_H > 1
        c_on = 1.0 if opts.low_nh == "c2" else counted
        dnc = -a * r_c * c_on
        h_loss = (r_h * min(nh, 1.0) + a * r_h2 * counted + a * r_c * counted + s * er_rate * min(nh, 1.0))
        return [dnc, add - h_loss, 0.5 * a * r_c * counted]

    return f


def evolve(formula, env, charge, opts=DEFAULT, t_end_yr=1.0e6, rates=None, method="LSODA", rtol=1e-6):
    """Integrate one parent; returns a ``Trajectory`` (times in yr)."""
    nc0, nh0 = parse(formula)
    rates = Rates(nc0, nh0, charge, env.chi, opts) if rates is None else rates
    f = _rhs(rates, env, nc0, nh0, charge, opts)
    g = lambda t, y: f(t, y)

    def gone(t, y):
        return y[0] - opts.nc_min

    gone.terminal, gone.direction = True, -1
    sol = solve_ivp(g, (0.0, t_end_yr * YR), [float(nc0), float(nh0), 0.0], method=method, rtol=rtol,
                    atol=[1e-6, 1e-6, 1e-9], events=gone, dense_output=False)
    t, (nc, nh, n_c2) = sol.t / YR, sol.y
    destroyed = None
    if sol.t_events[0].size:
        destroyed = float(sol.t_events[0][0] / YR)
        n_c2 = n_c2.copy()
        if nh[-1] > 1.0:                          # the remaining skeleton, as C2H2 (eq. 3: 1/2 dN_C)
            n_c2[-1] += 0.5 * max(nc[-1], 0.0)
    return Trajectory(t, nc, nh, n_c2, destroyed)


def euler(formula, env, charge, opts=DEFAULT, t_end_yr=1.0e4, dt_yr=1.0, rates=None):
    """M20's explicit Euler (eq. 2) with dt_yr, for comparison on slow cases."""
    nc0, nh0 = parse(formula)
    rates = Rates(nc0, nh0, charge, env.chi, opts) if rates is None else rates
    f = _rhs(rates, env, nc0, nh0, charge, opts)
    y, dt = np.array([float(nc0), float(nh0), 0.0]), dt_yr * YR
    for _ in range(int(round(t_end_yr / dt_yr))):
        y = y + dt * np.asarray(f(0.0, y))
        y[1] = max(y[1], 0.0)
        if y[0] <= opts.nc_min:
            break
    return y
