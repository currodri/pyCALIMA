"""One-zone collapse along a prescribed track with evolving dust.

The dust state is, per population (material), the mass per H nucleus of each
size bin, x_k = rho_k / n_H, which compression leaves unchanged. Each step in
ln n_H along the ``tracks.Track``:

1. gas: T, n(H), n(H2), n(He) from the track; dt = t_col d ln n_H;
2. dust temperature of every bin (``dust_temperature``), with the continuum
   escape fraction beta = min(1, tau^-2), tau = sum_k n_k pi a_k^2 <Q>(a_k, T_d,k)
   lambda_J (dust only: the track's own gas opacity is not re-derived) and
   lambda_J = c_s (pi / G rho)^1/2, c_s the isothermal sound speed (Omukai+05, Eq. 10);
3. accretion (``accretion.accrete`` and ``shift_remap``), if enabled;
4. coagulation within each population (``coagulation.advance``), if enabled,
   with the kernel ``coag_kernel(T, nH, mu, rho_gas, material, grid)``;
5. diagnostics: Lambda_gd, Gamma_comp = p / t_col, R = Lambda_gd / Gamma_comp,
   C = Lambda_gd t_col / e_th (collapse time over the dust cooling time of the
   gas thermal energy), S (geometric cross-section per gram of dust), D
   (dust-to-gas mass ratio).

The fragmentation criterion (``critical_dust_to_gas``) is C >= 1 for some
density between 1e8 cm^-3 and the density where H2 is 90 % molecular: dust
cooling must win while H2 formation heating holds T near 1e3 K, which is what
triggers the sharp temperature drop (Schneider et al. 2012a, Sect. 5). With
T_d << T this reproduces their S D_cr = 1.4e-3 cm^2/g (T/1e3 K)^-1/2
(n_H/1e12)^-1/2 (Eq. 16), which their one-zone models obey (their Table 1 with
D_cr = 2.6-6.3e-9). Their Eq. 15, R >= 1, would need 2.5x less dust: their full
models need the dust to cool the gas faster than it collapses, not only to
match p dV. Because T is prescribed, C and R are diagnostics: the dust does not
feed back on T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pycalima.collapse import gas_grain as gg
from pycalima.collapse.accretion import accrete, shift_remap
from pycalima.collapse.coagulation import advance
from pycalima.collapse.dust_temperature import dust_temperature, escape_fraction
from pycalima.collapse.grid import SizeGrid
from pycalima.collapse.materials import get_material
from pycalima.collapse.optics import planck_table
from scipy.optimize import brentq

from pycalima.collapse.tracks import G_CGS, MH_CGS, Y_HE, Track, omukai_f

KB_CGS = 1.380649e-16
MSUN_G = 1.98847e33
GAMMA_WINDOW = 0.25      # dex in n_H for the slope of ln p along the solution


@dataclass
class Population:
    material: str
    grid: SizeGrid
    x: np.ndarray                  # mass per H nucleus in each bin [g]
    table: object = None           # optics.PlanckTable

    def __post_init__(self):
        if self.table is None:
            self.table = planck_table(self.material)


def lognormal_population(material, mass_per_H, a_peak, sigma_ln=0.5, a_min=1e-7, a_max=1e-3, n_bins=80):
    """A lognormal in radius (dn/d ln a ∝ exp(-ln^2(a/a_peak) / 2 sigma^2)) holding mass_per_H [g]."""
    rho_s = get_material(material).rho_s
    g = SizeGrid.from_radii(a_min, a_max, n_bins, rho_s)
    ae = g.a_edges
    a = np.sqrt(ae[1:] * ae[:-1])
    w = np.exp(-0.5 * (np.log(a / a_peak) / sigma_ln) ** 2) * a ** 3 * np.log(ae[1:] / ae[:-1])
    return Population(material, g, mass_per_H * w / w.sum())


@dataclass
class History:
    nH: list = field(default_factory=list)
    t: list = field(default_factory=list)
    T: list = field(default_factory=list)
    p: list = field(default_factory=list)
    mu: list = field(default_factory=list)
    Lambda_gd: list = field(default_factory=list)
    Gamma_comp: list = field(default_factory=list)
    e_th: list = field(default_factory=list)
    t_col: list = field(default_factory=list)
    beta: list = field(default_factory=list)
    S: list = field(default_factory=list)
    D: list = field(default_factory=list)
    y_gas: list = field(default_factory=list)
    Td: list = field(default_factory=list)          # per population: area-weighted mean T_d [K]
    n_grains: list = field(default_factory=list)    # grains per H nucleus, all populations

    @property
    def R(self):
        return np.asarray(self.Lambda_gd) / np.asarray(self.Gamma_comp)

    @property
    def C(self):
        return np.asarray(self.Lambda_gd) * np.asarray(self.t_col) / np.asarray(self.e_th)

    def arrays(self):
        out = {k: np.asarray(v) for k, v in self.__dict__.items() if k != "y_gas"}
        out["R"], out["C"] = self.R, self.C
        return out


def _state(pops, nH):
    """Per population: (number density, mean radius, mean area) of each bin."""
    out = []
    for p in pops:
        rho = p.x * nH
        beta = p.grid.slopes(rho)
        out.append((p.grid.number_density(rho, beta), p.grid.mean_radius(beta), p.grid.mean_area(beta)))
    return out


def _gamma_trailing(hist):
    """d ln p / d ln n_H of the solution over the last GAMMA_WINDOW dex (1 if too short)."""
    ln = np.log10(hist.nH)
    w = ln >= ln[-1] - GAMMA_WINDOW
    if w.sum() < 3:
        return 1.0
    return float(np.polyfit(ln[w], np.log10(np.asarray(hist.p)[w]), 1)[0])


def run_collapse(track, populations, y_gas=None, n0=1e2, n_end=1e16, dlnn=0.05, T_rad=2.725,
                 accretion=False, coagulation=False, coag_kernel=None, alpha_T=1.0, sticking=1.0,
                 energy=False, q_cool=2.0, accretion_focus=None):
    """Integrate the collapse; returns (History, populations at the end).

    y_gas: gas-phase nuclei per H of the growth elements (``accretion.GAS_ELEMENTS``),
    required with ``accretion=True``. accretion_focus: ``accretion.accrete``'s ``focus``.

    energy: evolve T with du/dt = p / (n_H t_col) - lambda_other - lambda_gd per H
    nucleus (u = c_v k T), lambda_other the track's implied non-dust cooling
    (``Track.implied_cooling``), scaled as (T / T_track)^q_cool where it is net
    cooling and held fixed where it is net heating (H2 formation); t_col from
    the solution's own gamma. Without dust this reproduces the track. Each step
    is backward Euler in T (a scalar root), stable when dust cooling is faster
    than the collapse. Otherwise T = T_track(n_H).
    """
    pops = [Population(p.material, p.grid, p.x.copy(), p.table) for p in populations]
    y = dict(y_gas) if y_gas is not None else None
    hist = History()
    t_edges, n_edges = track.steps(n0, n_end, dlnn)
    beta_esc = 1.0
    T_sol = float(track.T(n_edges[0]))
    t_now = 0.0
    for k in range(n_edges.size):
        nH = n_edges[k]
        T = T_sol if energy else float(track.T(nH))
        nH_, nH2, nHe = track.species(nH)
        F = gg.collision_flux_factor(T, float(nH_), float(nH2), float(nHe))
        rho_gas = float(track.rho(nH))
        st = _state(pops, nH)
        # dust temperatures, with the escape fraction iterated once on the new T_d
        for _ in range(2):
            Td = [dust_temperature(p.table, s[1], T, F, T_rad, beta=beta_esc, alpha_T=alpha_T)
                  for p, s in zip(pops, st)]
            mu = float(track.mu(nH))
            lam_J = math.sqrt(KB_CGS * T / (mu * MH_CGS)) * math.sqrt(math.pi / (G_CGS * rho_gas))
            tau = lam_J * sum(np.sum(s[0] * s[2] * p.table(s[1], td)) for p, s, td in zip(pops, st, Td))
            beta_esc = float(escape_fraction(tau)) if tau > 0 else 1.0
        Lgd = sum(gg.cooling_rate(s[0], s[1], T, td, F, alpha_T) for s, td in zip(st, Td))
        dust_mass = sum(np.sum(p.x) for p in pops) * nH
        area = sum(np.sum(s[0] * s[2]) for s in st)
        n_tot = float(nH_ + nH2 + nHe)
        hist.nH.append(nH)
        hist.t.append(t_now if energy else t_edges[k])
        hist.T.append(T)
        hist.p.append(n_tot * KB_CGS * T)
        hist.mu.append(float(track.mu(nH)))
        hist.Lambda_gd.append(Lgd)
        if energy:
            f = 0.0 if math.log10(nH) < track.log_n_frag else float(omukai_f(_gamma_trailing(hist)))
            t_col = math.sqrt(3.0 * math.pi / (32.0 * G_CGS * rho_gas)) / math.sqrt(1.0 - f)
        else:
            t_col = float(track.t_col(nH))
        hist.Gamma_comp.append(n_tot * KB_CGS * T / t_col)
        hist.e_th.append((1.5 * float(nH_ + nHe) + 2.5 * float(nH2)) * KB_CGS * T)
        hist.t_col.append(t_col)
        hist.beta.append(beta_esc)
        hist.S.append(area / dust_mass if dust_mass > 0 else 0.0)
        hist.D.append(dust_mass / rho_gas)
        hist.y_gas.append(dict(y) if y is not None else None)
        hist.Td.append([float(np.sum(s[0] * s[2] * td) / np.sum(s[0] * s[2])) if np.sum(s[0]) > 0 else math.nan
                        for s, td in zip(st, Td)])
        hist.n_grains.append(sum(float(np.sum(s[0])) for s in st) / nH)
        if k == n_edges.size - 1:
            break
        # processes over the step to the next density, at mid-step conditions
        n_mid = math.sqrt(n_edges[k] * n_edges[k + 1])
        step = math.log(n_edges[k + 1] / n_edges[k])
        dt = t_col * step if energy else t_edges[k + 1] - t_edges[k]
        if energy:
            T_sol = _energy_step(track, pops, st, Td, T, nH, n_edges[k + 1], step, t_col, T_rad,
                                 alpha_T, q_cool)
        t_now += dt
        T_mid = math.sqrt(T * T_sol) if energy else float(track.T(n_mid))
        if accretion:
            work = [dict(material=p.material, grid=p.grid, rho=p.x * n_mid) for p in pops]
            y, da = accrete(work, T_mid, n_mid, y, dt, alpha=sticking, focus=accretion_focus)
            for p, d in zip(pops, da):
                if d > 0.0:
                    rho = p.x * n_mid
                    lam = 3.0 * p.grid.slopes(rho) + 2.0
                    rho_new, _ = shift_remap(p.grid.a_edges, rho, lam, p.grid.rho_s, d)
                    p.x = rho_new / n_mid
        if coagulation:
            mu_mid = float(track.mu(n_mid))
            for p in pops:
                K = coag_kernel(T_mid, n_mid, mu_mid, float(track.rho(n_mid)), p.material, p.grid)
                rho, _, _ = advance(p.grid, p.x * n_mid, dt, K, weights="geometric")
                p.x = rho / n_mid
    return hist, pops


def _energy_step(track, pops, st, Td, T, n0, n1, step, t_col, T_rad, alpha_T, q_cool):
    """Backward-Euler gas temperature at n1 from T at n0 (per H nucleus):
    c1 T' - c0 T = step [ (n_tot / n_H) T' - (lambda_o(T') + lambda_gd(T')) t_col / k ],
    with the dust temperatures of the step start."""
    c0, _ = track.specific_heat(n0)
    c1, _ = track.specific_heat(n1)
    nH_, nH2, nHe = (float(v) for v in track.species(n1))
    ntot_per_H = (nH_ + nH2 + nHe) / n1
    T_tr = float(track.T(n1))
    lam_tr = float(track.implied_cooling(n1))
    v1 = gg.collision_flux_factor(1.0, nH_, nH2, nHe)        # F ∝ T^1/2
    area = [s[0] * math.pi * s[1] ** 2 for s in st]

    def lam_gd(Tp):
        F = v1 * math.sqrt(Tp)
        return sum(float(np.sum(A * alpha_T * F * 2.0 * KB_CGS * (Tp - td))) for A, td in zip(area, Td)) / n1

    def lam_o(Tp):
        return lam_tr * (Tp / T_tr) ** q_cool if lam_tr > 0.0 else lam_tr

    def g(Tp):
        return c1 * Tp - c0 * T - step * (ntot_per_H * Tp - (lam_o(Tp) + lam_gd(Tp)) * t_col / KB_CGS)

    lo, hi = max(0.5 * T_rad, 1.0), 20.0 * max(T, T_tr)
    if g(lo) >= 0.0:
        return lo
    while g(hi) < 0.0:
        hi *= 2.0
    return brentq(g, lo, hi, rtol=1e-10)


def fragmentation(hist, gamma_low=0.8, gamma_high=0.97, M_J_max=0.8, window=0.1):
    """Schneider & Omukai (2010) / Chiaki et al. (2015) fragmentation of the
    solution: gamma = d ln p / d ln rho drops below 0.8 (rapid cooling) and then
    rises above 0.97 (cooling inefficient again) with the Jeans mass there below
    0.8 Msun. Returns the list of (n_H at gamma < 0.8, n_H at gamma > 0.97, M_J [Msun]).
    gamma is fitted over +-window dex, widened to at least +-1.6 steps so that
    every fit has three points."""
    a = hist.arrays()
    ln = np.log10(a["nH"])
    window = max(window, 1.6 * float(np.median(np.diff(ln))))
    lp = np.log10(a["p"])
    gam = np.array([np.polyfit(ln[w], lp[w], 1)[0] if w.sum() >= 3 else 1.0
                    for w in (np.abs(ln - x) <= window for x in ln)])
    rho = (1.0 + 4.0 * Y_HE) * MH_CGS * a["nH"]
    cs = np.sqrt(KB_CGS * a["T"] / (a["mu"] * MH_CGS))
    M_J = math.pi / 6.0 * rho * (cs * np.sqrt(math.pi / (G_CGS * rho))) ** 3 / MSUN_G
    out, i = [], 0
    while i < len(gam):
        if gam[i] < gamma_low:
            j = i + np.argmax(gam[i:] > gamma_high) if np.any(gam[i:] > gamma_high) else None
            if j is None:
                break
            out.append((a["nH"][i], a["nH"][j], M_J[j]))
            i = j
        i += 1
    return [o for o in out if o[2] < M_J_max]


def brownian_kernel(T, nH, mu, rho_gas, material, grid):
    """K_ij = pi (a_i + a_j)^2 (8 k T / pi m_red)^1/2: thermal (Brownian) coagulation
    with perfect sticking, as in Hirashita & Omukai (2009)."""
    def kernel(mean_m, mean_a):
        m_red = mean_m[:, None] * mean_m[None, :] / (mean_m[:, None] + mean_m[None, :])
        return math.pi * (mean_a[:, None] + mean_a[None, :]) ** 2 * np.sqrt(8.0 * KB_CGS * T / (math.pi * m_red))
    return kernel


def turbulent_kernel(mach=1.0, sigma_neutral=2.0e-15, v_stick_01=None):
    """Coagulation kernel factory: K_ij = pi (a_i + a_j)^2 <dv>_ij.

    The rms relative velocity adds Brownian motion ((3 k T / m_red)^1/2) and
    turbulence (``velocities.ormel_cuzzi_dv``, the full Ormel & Cuzzi 2007 form)
    in quadrature; turbulence is driven at the Jeans length L_J with v_g =
    mach c_s (c_s adiabatic, gamma = 5/3), tau_L = L_J / v_g, Re = v_g L_J / nu
    with the neutral kinematic viscosity nu = v_th / (3 n_H sigma_neutral),
    Epstein drag. <dv> is the mean speed of the collisions that stick
    (``velocities.mean_collision_speed``, Guillet et al. 2020 Eq. 18): perfect
    sticking by default, or below V = v_stick_01 (a_hat / 0.1 um)^-5/6 with the
    reduced radius a_hat (Chokshi et al. 1993; Guillet+20 take 0.4 km/s).
    mach = 1 is the Guillet+20 / Marchand+21 choice."""
    from pycalima.collapse import velocities as vel

    def factory(T, nH, mu, rho_gas, material, grid):
        cs = vel.sound_speed(T, mu)
        v_g = mach * cs
        L = vel.jeans_length(T, rho_gas, mu)
        tau_L = L / v_g
        nu = vel.thermal_speed(T, mu) / (3.0 * nH * sigma_neutral)
        tau_eta = tau_L / math.sqrt(max(v_g * L / nu, 1.0))
        rho_s = grid.rho_s

        def kernel(mean_m, mean_a):
            t_s = vel.epstein_stopping_time(mean_a, rho_s, T, rho_gas, mu)
            dV = np.sqrt(vel.brownian_rms(T, mean_m) ** 2
                         + vel.ormel_cuzzi_dv(t_s, v_g, tau_L, tau_eta) ** 2)
            V = np.inf
            if v_stick_01 is not None:
                ahat = mean_a[:, None] * mean_a[None, :] / (mean_a[:, None] + mean_a[None, :])
                V = v_stick_01 * (ahat / 1e-5) ** (-5.0 / 6.0)
            return math.pi * (mean_a[:, None] + mean_a[None, :]) ** 2 * vel.mean_collision_speed(dV, V)
        return kernel
    return factory


def critical_dust_to_gas(make_populations, track, n_min=1e8, n_max=None, criterion="C",
                         lo=1e-12, hi=1e-4, rtol=0.01, scan_dex=None, **run_kw):
    """Smallest initial dust-to-gas ratio D0 for which the criterion reaches 1
    for some n_H in [n_min, n_max] (default n_max: H2 90 % molecular, or the
    peak of f_H2 on a track that never gets there).

    criterion: ``"C"`` (cooling time vs collapse time, the default), ``"R"``, or
    ``"frag"`` (``fragmentation`` of the energy-equation solution, any density;
    implies ``energy=True`` and runs to n_end, default 1e16; it converges for
    d ln n <= 0.1, e.g. [Z_cr] = -4.91, -4.95, -4.97 at 0.1, 0.05, 0.025 for
    N07M20n0, while 0.2 is too coarse for the energy step).
    make_populations(D0) -> (populations, y_gas). Bisection in log D0.
    track: a ``Track``, or a function of D0 returning one (e.g. ``Track.grackle``
    at the trial metallicity, when D0 is Z / Z_sun).
    scan_dex: if given, first step up from ``lo`` by this many dex to the first D0
    that passes, then bisect below it, instead of requiring ``hi`` to pass. For
    ``"frag"`` with much dust (e.g. [Z] > -3.5 for the N07 CCSN dust) the gas
    cools to the end of the run, gamma never returns above 0.97, and nothing is
    found: the criterion is then not monotonic in D0.
    """
    run_kw = dict(run_kw)
    if criterion == "frag":
        run_kw["energy"] = True
        run_kw.setdefault("n_end", 1e16)

    def peak(D0):
        pops, y = make_populations(D0)
        tr = track if isinstance(track, Track) else track(D0)
        kw = dict(run_kw)
        if criterion != "frag":
            n_hi = tr.n_molecular(0.9) if n_max is None else n_max
            if math.isinf(n_hi):       # never 90 % molecular (Grackle's dust-free gas): its peak f_H2
                n_hi = 10.0 ** tr.log_n[int(np.argmax(tr.log_fH2))]
            kw.setdefault("n_end", 1.5 * n_hi)     # the criterion does not look further
        h, _ = run_collapse(tr, pops, y_gas=y, **kw)
        if criterion == "frag":
            return 2.0 if fragmentation(h) else 0.0
        a = h.arrays()
        sel = (a["nH"] >= n_min) & (a["nH"] <= n_hi)
        return a[criterion][sel].max()

    if scan_dex is not None:
        steps = np.arange(math.log10(lo), math.log10(hi) + 1e-9, scan_dex)
        first = next((x for x in steps if peak(10.0 ** x) >= 1.0), None)
        if first is None:
            return math.inf
        if first == steps[0]:
            return lo
        lo, hi = 10.0 ** (first - scan_dex), 10.0 ** first
    elif peak(hi) < 1.0:
        return math.inf
    elif peak(lo) >= 1.0:
        return lo
    llo, lhi = math.log(lo), math.log(hi)
    while lhi - llo > rtol:
        mid = 0.5 * (llo + lhi)
        (llo, lhi) = (llo, mid) if peak(math.exp(mid)) >= 1.0 else (mid, lhi)
    return math.exp(lhi)
