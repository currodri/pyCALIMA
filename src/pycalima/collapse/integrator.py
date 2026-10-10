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
from pycalima.collapse.tracks import G_CGS, MH_CGS, Y_HE

KB_CGS = 1.380649e-16


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
    Lambda_gd: list = field(default_factory=list)
    Gamma_comp: list = field(default_factory=list)
    e_th: list = field(default_factory=list)
    t_col: list = field(default_factory=list)
    beta: list = field(default_factory=list)
    S: list = field(default_factory=list)
    D: list = field(default_factory=list)
    y_gas: list = field(default_factory=list)

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


def run_collapse(track, populations, y_gas=None, n0=1e2, n_end=1e16, dlnn=0.05, T_rad=2.725,
                 accretion=False, coagulation=False, coag_kernel=None, alpha_T=1.0, sticking=1.0):
    """Integrate the collapse; returns (History, populations at the end).

    y_gas: gas-phase nuclei per H of the growth elements (``accretion.GAS_ELEMENTS``),
    required with ``accretion=True``.
    """
    pops = [Population(p.material, p.grid, p.x.copy(), p.table) for p in populations]
    y = dict(y_gas) if y_gas is not None else None
    hist = History()
    t_edges, n_edges = track.steps(n0, n_end, dlnn)
    beta_esc = 1.0
    for k in range(n_edges.size):
        nH = n_edges[k]
        T = float(track.T(nH))
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
        hist.nH.append(nH)
        hist.t.append(t_edges[k])
        hist.T.append(T)
        hist.Lambda_gd.append(Lgd)
        hist.Gamma_comp.append(float(track.compression_heating(nH)))
        hist.e_th.append(float(track.thermal_energy(nH)))
        hist.t_col.append(float(track.t_col(nH)))
        hist.beta.append(beta_esc)
        hist.S.append(area / dust_mass if dust_mass > 0 else 0.0)
        hist.D.append(dust_mass / rho_gas)
        hist.y_gas.append(dict(y) if y is not None else None)
        if k == n_edges.size - 1:
            break
        # processes over the step to the next density, at mid-step conditions
        n_mid = math.sqrt(n_edges[k] * n_edges[k + 1])
        dt = t_edges[k + 1] - t_edges[k]
        T_mid = float(track.T(n_mid))
        if accretion:
            work = [dict(material=p.material, grid=p.grid, rho=p.x * n_mid) for p in pops]
            y, da = accrete(work, T_mid, n_mid, y, dt, alpha=sticking)
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


def brownian_kernel(T, nH, mu, rho_gas, material, grid):
    """K_ij = pi (a_i + a_j)^2 (8 k T / pi m_red)^1/2: thermal (Brownian) coagulation
    with perfect sticking, as in Hirashita & Omukai (2009)."""
    def kernel(mean_m, mean_a):
        m_red = mean_m[:, None] * mean_m[None, :] / (mean_m[:, None] + mean_m[None, :])
        return math.pi * (mean_a[:, None] + mean_a[None, :]) ** 2 * np.sqrt(8.0 * KB_CGS * T / (math.pi * m_red))
    return kernel


def critical_dust_to_gas(make_populations, track, n_min=1e8, n_max=None, criterion="C",
                         lo=1e-12, hi=1e-4, rtol=0.01, **run_kw):
    """Smallest initial dust-to-gas ratio D0 for which the criterion reaches 1
    for some n_H in [n_min, n_max] (default n_max: H2 90 % molecular).

    criterion: ``"C"`` (cooling time vs collapse time, the default) or ``"R"``.
    make_populations(D0) -> (populations, y_gas). Bisection in log D0.
    """
    n_max = track.n_molecular(0.9) if n_max is None else n_max

    run_kw = dict(run_kw)
    run_kw.setdefault("n_end", 1.5 * n_max)    # the criterion does not look further

    def peak(D0):
        pops, y = make_populations(D0)
        h, _ = run_collapse(track, pops, y_gas=y, **run_kw)
        a = h.arrays()
        sel = (a["nH"] >= n_min) & (a["nH"] <= n_max)
        return a[criterion][sel].max()

    if peak(hi) < 1.0:
        return math.inf
    if peak(lo) >= 1.0:
        return lo
    llo, lhi = math.log(lo), math.log(hi)
    while lhi - llo > rtol:
        mid = 0.5 * (llo + lhi)
        (llo, lhi) = (llo, mid) if peak(math.exp(mid)) >= 1.0 else (mid, lhi)
    return math.exp(lhi)
