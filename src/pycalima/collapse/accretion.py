"""Grain growth by accretion of gas-phase key species (Chiaki et al. 2015, Eq. 9).

A grain of species i grows at

    da/dt = alpha (4 pi / 3) a0_ij^3 (k T / 2 pi m_j)^1/2 n_j,

with the key species j the one that gives the slowest growth (the shortest
collision time controls the kinetics, Chiaki+15 Sect. 2.2.5), sticking
alpha = 1 by default. The rate does not depend on a, so in a step every grain
of a species grows by the same Delta a: the size distribution shifts rigidly in
radius. Two consequences make the step exact:

* the mass added is a cubic in Delta a with the distribution's moments,
  dM = (4 pi / 3) rho_s sum_k N_k [3 Da <a^2>_k + 3 Da^2 <a>_k + Da^3],
  so the depletion of the gas-phase elements, shared between species (Fe in Fe,
  Fe3O4 and FeS; Mg in the silicates and MgO; Si in Si, the silicates and
  SiO2), is a small set of ODEs in the Delta a_i, integrated with RK4;
* the shift of each bin's in-bin power law onto the fixed grid is an analytic
  integral (``shift_remap``): number is conserved exactly, the mass gained is
  exactly dM, and there is no CFL limit.

Gas-phase reservoirs are per element, in nuclei per H nucleus (``GAS_ELEMENTS``);
oxygen (from H2O) is taken as unlimited. Every key species of an element draws on
that element's whole gas-phase reservoir (e.g. both Si and SiO on gas-phase Si):
the molecular partition of Chiaki+15's network is not followed.
"""

from __future__ import annotations

import math
import re

import numpy as np

from pycalima.collapse.grid import bin_means, pint
from pycalima.collapse.materials import AMU_CGS, get_material

KB_CGS = 1.380649e-16
GAS_ELEMENTS = ("C", "Mg", "Al", "Si", "S", "Fe")
KEY_ELEMENT = {"Si": "Si", "SiO": "Si", "SiO2": "Si", "Fe": "Fe", "Mg": "Mg", "C": "C", "Al": "Al", "S": "S"}
ATOMIC_AMU = {"C": 12.011, "O": 15.999, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "S": 32.06, "Fe": 55.845}

# Asplund, Grevesse, Sauval & Scott (2009, ARA&A 47, 481), Table 1, photospheric:
# log eps = log10(N_X / N_H) + 12
ASPLUND09_LOG_EPS = {"C": 8.43, "O": 8.69, "Mg": 7.60, "Al": 6.45, "Si": 7.51, "S": 7.12, "Fe": 7.50}


def solar_abundances(Z_over_Zsun=1.0):
    """Nuclei per H nucleus of the growth elements, solar pattern scaled by Z / Z_sun."""
    return {e: Z_over_Zsun * 10.0 ** (ASPLUND09_LOG_EPS[e] - 12.0) for e in GAS_ELEMENTS}


def formula_atoms(formula):
    """{'Mg': 2, 'Si': 1, 'O': 4} for 'Mg2SiO4'."""
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def formula_mass(formula):
    return sum(ATOMIC_AMU[e] * n for e, n in formula_atoms(formula).items()) * AMU_CGS


# Key species taken to be absent from the gas: Si atoms are depleted onto SiO
# (and C atoms onto CO, handled in the reservoir), so Si grains do not grow
# (Chiaki et al. 2015, Fig. 7). ``growth_rate(..., absent=())`` lifts it.
ABSENT_KEYS = ("Si",)


def growth_rate(material, T, n_gas, alpha=1.0, absent=ABSENT_KEYS):
    """da/dt [cm s^-1] of a material: the slowest over its key species.

    n_gas: gas-phase number density [cm^-3] of each element in ``GAS_ELEMENTS``.
    absent: key species with no gas-phase abundance (default: Si atoms, in SiO).
    """
    rates = []
    for k in get_material(material).keys:
        n = 0.0 if k.gas in absent else max(n_gas[KEY_ELEMENT[k.gas]], 0.0)
        rates.append(alpha * 4.0 / 3.0 * math.pi * k.a0 ** 3
                     * math.sqrt(KB_CGS * T / (2.0 * math.pi * k.m_gas * AMU_CGS)) * n)
    return min(rates)


def mass_gain(rho_s, N, a1, a2, da):
    """dM per volume for a rigid shift da of bins with number N and moments <a>, <a^2>."""
    return 4.0 / 3.0 * math.pi * rho_s * np.sum(N * (3.0 * da * a2 + 3.0 * da ** 2 * a1 + da ** 3))


def shift_remap(a_edges, rho, lam, rho_s, da):
    """Shift every grain by da in radius and redistribute onto the same bins.

    a_edges: radius edges [cm]; rho: mass density per bin; lam: in-bin slopes of
    dn/da ∝ a^lam (= 3 beta + 2 for dn/dm ∝ m^beta). Returns (rho_new, overflow mass).

    In the bin's own scale-free variable x = u / r0 (u the radius before the
    shift), the mass that lands in [x0, x1] is
        rho_k [I(l+3) + 3 d I(l+2) + 3 d^2 I(l+1) + d^3 I(l)] / I_bin(l+3),
    with I(p) = int_{x0}^{x1} x^p dx and d = da / r0, so no large powers of r0 appear.
    """
    nb = rho.size
    out = np.zeros(nb)
    overflow = 0.0
    for k in range(nb):
        if rho[k] <= 0.0:
            continue
        r0, r1, l = a_edges[k], a_edges[k + 1], lam[k]
        x_top = r1 / r0
        d = da / r0
        norm = pint(x_top, l + 3.0)

        def piece(x0, x1):
            I = lambda p: pint(x1, p) - pint(x0, p)
            return rho[k] * (I(l + 3.0) + 3.0 * d * I(l + 2.0) + 3.0 * d * d * I(l + 1.0)
                             + d ** 3 * I(l)) / norm

        lo, hi = r0 + da, r1 + da
        j = max(int(np.searchsorted(a_edges, lo, side="right")) - 1, 0)
        while j < nb and a_edges[j] < hi:
            x0, x1 = max(lo, a_edges[j]), min(hi, a_edges[j + 1])
            if x1 > x0:
                out[j] += piece((x0 - da) / r0, (x1 - da) / r0)
            j += 1
        if hi > a_edges[-1]:
            overflow += piece((max(lo, a_edges[-1]) - da) / r0, x_top)
    return out, overflow


def accrete(populations, T, nH, y_gas, dt, alpha=1.0, n_sub=16):
    """Grow every population over dt, depleting the shared gas-phase elements.

    populations: list of dicts with keys ``material``, ``grid`` (SizeGrid), ``rho``.
    y_gas: gas-phase nuclei per H of each element (updated copy returned).
    Returns (y_gas_new, list of da per population). The rho arrays are updated
    in place by the caller with ``shift_remap``; here only da is decided.
    """
    mats = [get_material(p["material"]) for p in populations]
    form = [formula_atoms(m.formula) for m in mats]
    mform = [formula_mass(m.formula) for m in mats]
    npop = len(populations)
    # the mass a shift da adds is (4 pi / 3) rho_s [3 da S2 + 3 da^2 S1 + da^3 S0]
    # with S_k = sum_bins N <a^k>: three scalars per population
    S = np.zeros((npop, 3))
    for i, p in enumerate(populations):
        g, rho = p["grid"], p["rho"]
        beta = g.slopes(rho)
        N = g.number_density(rho, beta)
        a2 = g.a_edges[:-1] ** 2 * bin_means(np.ones(g.n_bins), g.eta, beta, 2.0 / 3.0)
        S[i] = (N.sum(), np.sum(N * g.mean_radius(beta)), np.sum(N * a2))
    c43 = np.array([4.0 / 3.0 * math.pi * p["grid"].rho_s for p in populations])
    # nuclei of each element consumed per gram of each population's solid, per H
    els = list(y_gas)
    nu = np.array([[form[i].get(e, 0) / mform[i] / nH for e in els] for i in range(npop)])
    y0 = np.array([y_gas[e] for e in els])
    # growth rate coefficients per key species: da/dt = coef * n_key, key element index
    vth = math.sqrt(KB_CGS * T / (2.0 * math.pi * AMU_CGS))
    keys = []
    for m in mats:
        keys.append([(alpha * 4.0 / 3.0 * math.pi * k.a0 ** 3 * vth / math.sqrt(k.m_gas), els.index(KEY_ELEMENT[k.gas]),
                      k.gas in ABSENT_KEYS) for k in m.keys])
    alive = S[:, 0] > 0.0

    def gas_vec(da):
        dM = c43 * (3.0 * da * S[:, 2] + 3.0 * da ** 2 * S[:, 1] + da ** 3 * S[:, 0])
        return y0 - dM @ nu

    def gas(da):
        return dict(zip(els, gas_vec(da)))

    def rhs(da):
        n = np.maximum(gas_vec(da), 0.0) * nH
        out = np.zeros(npop)
        for i in range(npop):
            if alive[i]:
                out[i] = min(0.0 if absent else c * n[j] for c, j, absent in keys[i])
        return out

    da = np.zeros(len(populations))
    h = dt / n_sub
    for _ in range(n_sub):
        k1 = rhs(da)
        k2 = rhs(da + 0.5 * h * k1)
        k3 = rhs(da + 0.5 * h * k2)
        k4 = rhs(da + h * k3)
        da = da + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # never consume more of an element than there is: scale back the growth of
    # every species that uses an exhausted element until it is exactly used up
    y = gas(da)
    for e in els:
        if y[e] < 0.0:
            users = [i for i in range(len(populations)) if form[i].get(e, 0) > 0 and da[i] > 0]
            lo, hi = 0.0, 1.0
            for _ in range(60):
                s = 0.5 * (lo + hi)
                trial = da.copy()
                trial[users] *= s
                (lo, hi) = (s, hi) if gas(trial)[e] >= 0.0 else (lo, s)
            da[users] *= lo
            y = gas(da)
    return {e: max(v, 0.0) for e, v in y.items()}, da
