"""Initial dust and gas-phase metals of a cloud enriched by one Pop III supernova.

The cloud has total metallicity Z (gas + dust; Z_sun = 0.02 and X_H = 0.75, the
convention of Chiaki et al. 2015, so that [Z_cr] compares with theirs), all from
the unmixed ejecta of one core-collapse supernova of progenitor mass M, the N07
models (Nozawa et al. 2003, 2007):

* dust: Nozawa+03 yields per species (``collapse_ics/nozawa2003_ccsn_unmixed_yields.txt``,
  their Fig. 13a) with the C20 size distributions (``nozawa2003_c20_unmixed_sizes.txt``,
  their Fig. 8a, the only CCSN model they give sizes for);
* the dust-to-metal mass ratio f_dep,0 and the element abundances (C, O, Mg, Si,
  Fe) of the same models from Chiaki+15 (``chiaki2015_n07_models.txt``, their
  Table 3 and Fig. 2): M_metal = M_dust / f_dep,0, A_j = Z M_j / (mu_j X_H M_metal);
* Al and S, absent from Chiaki+15 Fig. 2, from the Al/Mg and S/Si ratios of
  Kobayashi's Z = 0 supernova yields (``galaxysam/external_yields/SN2SAGBYIELD.DAT``).

Kobayashi's yields are not used for the main elements: their explosion models
differ from N07's, and Nozawa's dust would then hold up to 2.3x more Si than the
ejecta (20 Msun). Each element's gas-phase share is its abundance less what the
dust locks up; any shortfall (negative gas) is set to zero and reported.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from pycalima import _paths
from scipy.optimize import brentq

from pycalima.collapse.accretion import ATOMIC_AMU, GAS_ELEMENTS, formula_atoms, formula_mass, shift_remap
from pycalima.collapse.grid import SizeGrid
from pycalima.collapse.integrator import Population
from pycalima.collapse.materials import AMU_CGS, get_material
from pycalima.collapse.tracks import MH_CGS, Y_HE

Z_SUN = 0.02          # Chiaki+15
RHO_PER_H = (1.0 + 4.0 * Y_HE) * MH_CGS     # = m_H / X_H with X_H = 0.75


@lru_cache(maxsize=None)
def nozawa2003_yields():
    """{species: array of dust masses [Msun] at progenitor masses}, and the masses."""
    path = _paths.get_external_data_path("collapse_ics", "nozawa2003_ccsn_unmixed_yields.txt")
    with open(path) as fh:
        names = [ln for ln in fh if ln.startswith("# columns")][0].split("of")[-1].split()
    d = np.loadtxt(path)
    return d[:, 0], {n: d[:, i + 1] for i, n in enumerate(names)}


@lru_cache(maxsize=None)
def nozawa2003_sizes():
    """(log10 r [cm], {species: log10 dN/dr (relative), nan where none})."""
    path = _paths.get_external_data_path("collapse_ics", "nozawa2003_c20_unmixed_sizes.txt")
    with open(path) as fh:
        names = [ln for ln in fh if ln.startswith("# columns")][0].split("of")[-1].split()
    d = np.loadtxt(path)
    return d[:, 0], {n: d[:, i + 1] for i, n in enumerate(names)}


@lru_cache(maxsize=None)
def chiaki2015_models():
    """{progenitor mass: dict(log_A={el: log10 A_j at Z_sun}, f_dep, f_dep_C, S_C, f_dep_Sil,
    S_Sil, D0, Dcr_ng, Zcr_ng)} from Chiaki+15 (see the data file for units)."""
    d = np.loadtxt(_paths.get_external_data_path("collapse_ics", "chiaki2015_n07_models.txt"))
    out = {}
    for row in d:
        out[float(row[0])] = dict(log_A=dict(zip(("C", "O", "Mg", "Si", "Fe"), row[1:6])),
                                  f_dep_C=row[6], S_C=row[7] * 1e4, f_dep_Sil=row[8], S_Sil=row[9] * 1e4,
                                  f_dep=row[10], D0=row[11] * 1e-3, Dcr_ng=row[12] * 1e-8, Zcr_ng=row[13])
    return out


@lru_cache(maxsize=None)
def kobayashi_z0_yields(progenitor_mass):
    """({element: ejected mass [Msun]}, total metal mass [Msun]) for a Z = 0 CCSN."""
    from pycalima.galaxysam import yield_models as ym
    data_dir = Path(ym.__file__).parent / "yield_files" / "yield_files"
    block = ym._load_kobayashi_raw_sn_block(None, 0.0)
    labels = ym._load_kobayashi_isotope_labels(data_dir, 0.0)
    masses = block["masses"]
    k = int(np.argmin(np.abs(masses - progenitor_mass)))
    if abs(masses[k] - progenitor_mass) > 1e-6:
        raise ValueError(f"no Kobayashi Z = 0 model at {progenitor_mass} Msun; have {masses[masses >= 10]}")
    elements = ["C", "O"] + [e for e in GAS_ELEMENTS if e != "C"]
    el = ym._aggregate_kobayashi_raw_elemental_yields(block, labels, elements)[k]
    iso = np.asarray(block["isotope_yields"])[:, k]
    metals = float(np.sum(iso[~labels["element"].isin(["H", "He"]).to_numpy()]))
    return dict(zip(elements, el)), metals


@lru_cache(maxsize=None)
def nozawa2007_destruction():
    """{(progenitor mass, n_amb): {species: eps_dest}} from Nozawa+07 Table 3 (unmixed)."""
    path = _paths.get_external_data_path("collapse_ics", "nozawa2007_destruction_unmixed.txt")
    names = ["Fe", "Si", "FeS", "MgSiO3", "SiO2", "Al2O3", "Mg2SiO4", "MgO", "C"]
    return {(float(r[0]), float(r[1])): dict(zip(names, r[2:])) for r in np.loadtxt(path)}


def erode(population, survival):
    """Thin every grain of a population by the same Delta a (thermal sputtering
    erodes at a rate independent of size) so that the fraction ``survival`` of
    its mass remains; grains smaller than Delta a are destroyed. Returns
    (new mass per bin, Delta a)."""
    g, x = population.grid, population.x
    if survival >= 1.0:
        return x.copy(), 0.0
    if survival <= 0.0:
        return np.zeros_like(x), np.inf
    lam = 3.0 * g.slopes(x) + 2.0
    left = lambda da: shift_remap(g.a_edges, x, lam, g.rho_s, -da)[0].sum() / x.sum()
    da = brentq(lambda d: left(d) - survival, 0.0, g.a_edges[-1], xtol=1e-14, rtol=1e-10)
    return shift_remap(g.a_edges, x, lam, g.rho_s, -da)[0], da


def _distribution_on_grid(grid, species):
    """Mass fraction per bin of a species' Nozawa+03 size distribution."""
    lr, logf = nozawa2003_sizes()
    lf = logf[species]
    ok = np.isfinite(lf)
    r = 10.0 ** lr[ok]
    f = 10.0 ** lf[ok]
    w = np.zeros(grid.n_bins)
    ae = grid.a_edges
    rf = np.geomspace(r[0], r[-1], 4000)
    ff = 10.0 ** np.interp(np.log10(rf), np.log10(r), np.log10(f))
    mass = rf ** 3 * ff                         # ∝ dM/dr
    for k in range(grid.n_bins):
        sel = (rf >= ae[k]) & (rf < ae[k + 1])
        if sel.sum() > 1:
            w[k] = np.trapezoid(mass[sel], rf[sel])
        elif sel.sum() == 1:
            w[k] = mass[sel][0] * (ae[k + 1] - ae[k])
    return w / w.sum() if w.sum() > 0 else w


def nozawa_ccsn(progenitor_mass, Z_over_Zsun, a_min=1e-8, a_max=1e-3, bins_per_decade=10,
                species=None, carbon_in_co=True, n_amb=None):
    """Populations and gas-phase metals of a cloud enriched by one N07 CCSN.

    n_amb: ambient density [cm^-3] of the reverse-shock model (0.1, 1 or 10;
    None: unprocessed): each species keeps the mass fraction 1 - eps_dest of
    Nozawa+07 Table 3, by uniform erosion (``erode``); the metals of the
    destroyed grains return to the gas. The metallicity Z counts gas + dust,
    as before. This matches the surviving masses (f_dep of Chiaki+15 Table 3)
    but not the size distributions: the reverse shock destroys small grains
    preferentially, so for N07M20n1 uniform erosion gives S_C = 4.1e4 cm^2/g
    where their Fig. 6b (``nozawa2007_m20_n1_sizes.txt``) and Chiaki+15 give 1.8e4
    and 2.0e4.
    carbon_in_co: only carbon in excess of oxygen is free to grow carbon grains,
    the rest being locked in CO at the densities of interest (these ejecta are
    O-rich, so then none is; Chiaki+15 follow CO with their chemistry).
    Returns (populations, y_gas, info) with info: f_dep, the dust and metal
    masses per H, and any element shortfall (dust holding more than the abundance).
    """
    masses, ylds = nozawa2003_yields()
    k = int(np.argmin(np.abs(masses - progenitor_mass)))
    if abs(masses[k] - progenitor_mass) > 1e-6:
        raise ValueError(f"Nozawa+03 CCSN yields exist for {masses} Msun")
    model = chiaki2015_models()[float(masses[k])]
    M_dust = sum(float(v[k]) for v in ylds.values())
    M_met = M_dust / model["f_dep"]
    metal_per_H = Z_over_Zsun * Z_SUN * RHO_PER_H        # g of metals per H nucleus
    y_tot = {e: Z_over_Zsun * 10.0 ** model["log_A"][e] for e in model["log_A"]}
    kob, _ = kobayashi_z0_yields(float(masses[k]))
    y_tot["Al"] = y_tot["Mg"] * (kob["Al"] / ATOMIC_AMU["Al"]) / (kob["Mg"] / ATOMIC_AMU["Mg"])
    y_tot["S"] = y_tot["Si"] * (kob["S"] / ATOMIC_AMU["S"]) / (kob["Si"] / ATOMIC_AMU["Si"])
    n_bins = int(round(np.log10(a_max / a_min) * bins_per_decade))
    pops, locked = [], {e: 0.0 for e in GAS_ELEMENTS}
    dust_per_H = 0.0
    for sp in (species or ylds):
        Md = float(ylds[sp][k])
        if Md <= 0.0:
            continue
        mat = get_material(sp)
        g = SizeGrid.from_radii(a_min, a_max, n_bins, mat.rho_s)
        x_tot = metal_per_H * Md / M_met
        pop = Population(sp, g, x_tot * _distribution_on_grid(g, sp))
        if n_amb is not None:
            eps = nozawa2007_destruction()[(float(masses[k]), float(n_amb))][sp]
            x_new, _ = erode(pop, 1.0 - eps)
            if x_new.sum() <= 0.0:
                continue
            pop = Population(sp, g, x_new, pop.table)
            x_tot = x_new.sum()
        pops.append(pop)
        dust_per_H += x_tot
        nf = x_tot / formula_mass(mat.formula)           # formula units per H
        for e, nu in formula_atoms(mat.formula).items():
            if e in locked:
                locked[e] += nu * nf
    y_gas, shortfall = {}, {}
    for e in GAS_ELEMENTS:
        y_gas[e] = max(y_tot[e] - locked[e], 0.0)
        if locked[e] > y_tot[e]:
            shortfall[e] = locked[e] / y_tot[e] - 1.0
    if carbon_in_co:
        y_gas["C"] = max(y_gas["C"] - y_tot["O"], 0.0)
    info = dict(f_dep=dust_per_H / metal_per_H, dust_per_H=dust_per_H, metal_per_H=metal_per_H,
                shortfall=shortfall, model=model, y_tot=y_tot)
    return pops, y_gas, info
