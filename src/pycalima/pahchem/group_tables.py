"""PAH charge, photoelectric heating and ion neutralisation for RAMSES-RT photon groups, per PAH bin.

The PAH counterpart of the grain tables of ``models.dust_charge.rt_group_charging`` (WDB06rt): the same groups,
the same blackbody in-group shapes and the same UV background, with pahchem's physics (stage P,
docs/physics/pahchem-gas-coupling.md) in place of WDB06's.

A bin is a set of PAHdb molecules with weights (``nc_range_weights``, or any other: the bin scheme of CALIMA is
open), each with the charges -1, 0, +1, +2 (``CHARGES``). Photoionisation and photodetachment are linear in the
photons, so per state z and group g the table holds, averaged over the bin's molecules and over the group with a
blackbody photon spectrum n_E ~ E^2 / (exp(E/kT_s) - 1) for each T_s of ``TS_GRID`` (as WDB06rt):

    K_g(z) = < sigma Y >_g            [cm^2 per PAH]       photoionisation (detachment) out of z
    H_g(z) = < sigma Y <E_k> >_g      [cm^2 eV per PAH]    kinetic energy given to the gas
    A_g(z) = < sigma >_g              [cm^2 per PAH]       absorption

with sigma, Y and <E_k> of ``population`` (``photo.sigma_li_draine``, ``population._yield``,
``population.electron_energy``) on a grid from the lowest group edge: photons below 5 eV detach anions too (the
population's grid starts at 1 eV). The same three are given for the UV background (``K_bg`` etc.,
rates per PAH in the Mathis ISRF at G0 = 1 Habing, 1-13.6 eV, as WDB06rt). Collisions are tabulated in T:

    alpha(z, T)  electron capture by z = 0, +1, +2 [cm^3 s^-1] (``population.capture_coefficient``)
    L(z, T)      alpha times the mean energy of the captured electron [cm^3 s^-1 eV] (``population.capture_energy``)
    k_mn(i, T)   mutual neutralisation of the anion by ion i [cm^3 s^-1]
    k_ct(i)      ion-neutral reaction of the neutral with ion i [cm^3 s^-1] (zero where IP(X) < IP(PAH))

(``population.ion_up_rate``; ions: those of ``rt_group_charging.RECOMB_IONS``), and associative detachment of the
anion by H, k_ad n_H (``population.anion_associative_detachment``: zero unless ``Params.associative_detachment``
is "all").

Per cell (``PAHGroupTable.solve``, the algorithm RAMSES would implement) the four states are in detailed balance,
f(z+1)/f(z) = up(z) / down(z+1) with

    up(z)   = sum_g F_g K_g(z) + G0_bg K_bg(z) + [z = -1] (k_ad n_H + sum_i n_i k_mn(i)) + [z = 0] sum_i n_i k_ct(i)
    down(z) = n_e alpha(z),

F_g = c N_g the group photon flux. The rates are averaged over the bin before the balance is solved: one chain per
bin, not one per molecule (``check_pah_group_tables.py`` measures what that costs). Photoionisation out of +2 leaves
the chain (``overflow``): it adds electrons and heat, but the +3 state is not followed. +2 is the top state of every
molecule (``_chain``), also of the few with a +3 state in PAHdb, so it ionises only above ``population.Y_UNITY_EV``
as ``Population``'s top states do.

What it gives up against ``Population``: the in-group shape (blackbody), the averaging above, the dehydrogenated
states (only the parents' thresholds), the multiphoton k_H, and the IR emission.

    python -m pycalima.pahchem.group_tables --nc 20 50 --label small
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.pahchem import chemistry, photo
from pycalima.pahchem import population as pop
from pycalima.pahchem.baseline import Params

FORMAT_VERSION = 1
CHARGES = (-1, 0, 1, 2)
LT = np.round(np.arange(1.0, 8.0 + 1e-9, 0.05), 6)              # log10 T [K]
ION_NAMES = tuple(f"{el}+" for el, _, _ in rg.RECOMB_IONS if f"{el}+" in chemistry.IONS)
N_E_PER_GROUP = rg.N_E_PER_GROUP
BG_GRID = np.geomspace(1.0, rg.MATHIS_BREAKS[-1], 400)          # the UV background's nodes (as WDB06rt) [eV]
M_C, M_H = 12.011 * chemistry.AMU_G, 1.008 * chemistry.AMU_G


def complete(mol):
    """True if the molecule has every state of ``CHARGES``."""
    return all(z in mol.charges for z in CHARGES)


def nc_range_weights(library, nc_min, nc_max):
    """Equal weights for the complete molecules (``complete``) with nc_min <= N_C < nc_max, zero for the rest."""
    return np.array([1.0 if (nc_min <= m.nc < nc_max and complete(m)) else 0.0 for m in library])


def _group_nodes(L0, L1, n_E):
    """n_E geometric nodes per group. A group with the top state's step at ``population.Y_UNITY_EV`` inside it
    gets nodes just below and at the step, so its trapezoid does not smear it (4 % for +2 in 13.6-24.6 eV)."""
    out = []
    for l0, l1 in zip(L0, L1):
        e = pop.Y_UNITY_EV
        if not l0 < e < l1:
            out.append(np.geomspace(l0, l1, n_E))
            continue
        k = int(np.clip(round(n_E * np.log(e / l0) / np.log(l1 / l0)), 2, n_E - 2))
        out.append(np.concatenate([np.geomspace(l0, e * (1.0 - 1e-9), k), np.geomspace(e, l1, n_E - k)]))
    return out


def _mathis_flux():
    """Mathis ISRF photon flux on BG_GRID at G0 = 1 Habing (6-13.6 eV energy flux) [cm^-2 s^-1 eV^-1]."""
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    E = BG_GRID
    u = np.array([Mathis83_radiation_field(e) for e in E])
    band = E >= 6.0
    u *= rg.HABING_FLUX / (rg.C_CGS * np.trapezoid(u[band], E[band]))
    return rg.C_CGS * u / (E * chemistry.EV_ERG)


def _chain(mol):
    """``mol`` with the charge states of ``CHARGES`` only, so that +2 is the top state for every molecule: the few
    with a +3 state would otherwise ionise out of +2 from ~10 eV (``population._yield``) and the rest only above
    ``population.Y_UNITY_EV``, two definitions of the overflow in one bin average."""
    from dataclasses import replace

    if max(mol.charges) <= CHARGES[-1]:
        return mol
    keep = [z for z in mol.charges if z <= CHARGES[-1]]
    return replace(mol, species={z: mol.species[z] for z in keep}, uids={z: mol.uids[z] for z in keep if z in mol.uids},
                   ionisation_ev={z: v for z, v in mol.ionisation_ev.items() if z < CHARGES[-1]}, _cache={})


def molecule_kernels(job):
    """Everything one molecule gives the table: per state, (A, K, H) on the group nodes and the background rates,
    and the collisional tables on ``lT``. ``job`` = (mol, params, partition, L0, L1, n_E, lT)."""
    mol, params, partition, L0, L1, n_E, lT = job
    mol = _chain(mol)
    nodes = _group_nodes(L0, L1, n_E)
    E = np.concatenate(nodes + [BG_GRID])
    nz, ng = len(CHARGES), len(nodes)
    A, K, H = (np.zeros((nz, E.size)) for _ in range(3))
    for i, z in enumerate(CHARGES):
        sigma = photo.sigma_uv(mol.nc, z, E, params.uv_cross_section) * params.sigma_scale
        y = pop._yield(mol, z, params, energy_ev=E)
        A[i], K[i] = sigma, sigma * y
        H[i] = K[i] * pop.electron_energy(mol, z, params, partition, energy_ev=E)
    cut = ng * n_E
    shape = (nz, ng, n_E)
    Fb = _mathis_flux()
    bg = [np.trapezoid(X[:, cut:] * Fb, BG_GRID, axis=1) for X in (A, K, H)]
    T = 10.0 ** np.asarray(lT, dtype=float)
    alpha = np.array([[pop.capture_coefficient(mol, z, t, params) for t in T] for z in CHARGES[1:]])
    e_cap = np.array([[pop.capture_energy(mol, z, t) for t in T] for z in CHARGES[1:]])
    k_mn = np.zeros((len(ION_NAMES), T.size))
    k_ct = np.zeros(len(ION_NAMES))
    for j, ion in enumerate(ION_NAMES):
        k_mn[j] = [pop.ion_up_rate(mol, -1, SimpleNamespace(temperature_k=t, ions=((ion, 1.0),)), params) for t in T]
        k_ct[j] = pop.ion_up_rate(mol, 0, SimpleNamespace(temperature_k=T[0], ions=((ion, 1.0),)), params)
    return dict(A=A[:, :cut].reshape(shape), K=K[:, :cut].reshape(shape), H=H[:, :cut].reshape(shape),
                A_bg=bg[0], K_bg=bg[1], H_bg=bg[2], alpha=alpha, L=alpha * e_cap, k_mn=k_mn, k_ct=k_ct,
                k_ad=pop.anion_associative_detachment(params), nc=mol.nc, nh=mol.nh)


def combine(kernels, weights, L0, L1, n_E=N_E_PER_GROUP, Ts=rg.TS_GRID, lT=LT, label="", meta=""):
    """The bin table from per-molecule ``molecule_kernels`` and their weights: kernels averaged over the
    molecules, then over each group for every T_s."""
    w = np.asarray(weights, dtype=float)
    if w.size != len(kernels) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("one non-negative weight per molecule, not all zero")
    w = w / w.sum()
    avg = {k: np.tensordot(w, np.array([m[k] for m in kernels]), axes=1)
           for k in ("A", "K", "H", "A_bg", "K_bg", "H_bg", "alpha", "L", "k_mn", "k_ct", "k_ad", "nc", "nh")}
    nodes = _group_nodes(L0, L1, n_E)
    nz, ng, nTs = len(CHARGES), len(nodes), np.size(Ts)
    egy = np.zeros((ng, nTs))
    out = {k: np.zeros((nz, ng, nTs)) for k in ("A", "K", "H")}
    for g, E in enumerate(nodes):
        for it, t in enumerate(np.atleast_1d(Ts)):
            wE = rg._photon_weights(E, t)
            norm = np.trapezoid(wE, E)
            egy[g, it] = np.trapezoid(wE * E, E) / norm
            for k in out:
                out[k][:, g, it] = np.trapezoid(wE * avg[k][:, g], E, axis=1) / norm
    return dict(version=FORMAT_VERSION, label=label, meta=meta, n_mol=int((w > 0).sum()),
                nc=float(avg["nc"]), nh=float(avg["nh"]), L0=np.asarray(L0, float), L1=np.asarray(L1, float),
                log10_Ts=np.log10(np.atleast_1d(Ts)), egy=egy, **out,
                A_bg=avg["A_bg"], K_bg=avg["K_bg"], H_bg=avg["H_bg"], lT=np.asarray(lT, float),
                alpha=avg["alpha"], L=avg["L"], ions=ION_NAMES, k_mn=avg["k_mn"], k_ct=avg["k_ct"],
                k_ad=float(avg["k_ad"]))


def build_bin_table(library, weights, params=Params(), partition=photo.PHOTOELECTRON_PARTITION,
                    group_L0=rg.DEFAULT_GROUP_L0, group_L1=rg.DEFAULT_GROUP_L1, n_E=N_E_PER_GROUP,
                    Ts=rg.TS_GRID, lT=LT, workers=1, label=""):
    """The table of one bin: the molecules of ``library`` with weight > 0 (each must be ``complete``)."""
    w = np.asarray(weights, dtype=float)
    mols = [m for m, x in zip(library, w) if x > 0]
    if not mols:
        raise ValueError("no molecule in the bin")
    if not all(complete(m) for m in mols):
        raise ValueError(f"every molecule of a bin needs the charges {CHARGES}")
    jobs = [(m, params, partition, group_L0, group_L1, n_E, lT) for m in mols]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            kernels = list(ex.map(molecule_kernels, jobs, chunksize=4))
    else:
        kernels = [molecule_kernels(j) for j in jobs]
    meta = (f"detachment={params.detachment} yield_model={params.yield_model} yield_cation={params.yield_cation} "
            f"capture_model={params.capture_model} sigma_scale={params.sigma_scale} "
            f"mn_efficiency={params.mn_efficiency} ion_neutral_efficiency={params.ion_neutral_efficiency} "
            f"metal_adducts={params.metal_adducts} associative_detachment={params.associative_detachment} "
            f"attachment_fraction={params.attachment_fraction} uv_cross_section={photo.uv_model_key(params.uv_cross_section)} "
            f"partition={partition}")
    return combine(kernels, w[w > 0], group_L0, group_L1, n_E, Ts, lT, label, meta)


# --------------------------------------------------------------------------- file
#   comment lines starting with '#' (the parameters on the second)
#   version nCharges nGroups nTs nT nIons n_mol
#   N_C N_H (weighted means) mass per PAH [g] k_ad [cm^3 s^-1]
#   charges / L0 / L1 [eV] / log10 Ts / log10 T
#   egy(1:nTs) for each group [eV]
#   A, then K, then H: (1:nTs) for each charge, then each group     [cm^2], [cm^2], [cm^2 eV]
#   A_bg, K_bg, H_bg (1:nCharges)  [s^-1, s^-1, eV s^-1 per PAH]; A_bg is absorptions per second
#   alpha(1:nT) for charges 0, +1, +2 [cm^3 s^-1];  L(1:nT) for the same [cm^3 s^-1 eV]
#   ion names;  k_mn(1:nT) for each ion [cm^3 s^-1];  k_ct(1:nIons) [cm^3 s^-1]
def _line(v):
    return " ".join(f"{x:.12e}" for x in np.atleast_1d(v)) + "\n"


def write_bin_table(t, path):
    path = Path(path)
    nz, ng, nTs = t["K"].shape
    with open(path, "w") as f:
        f.write(f"# pyCALIMA pahchem PAH charge table for RT photon groups, format {t['version']}: bin {t['label']}, "
                f"{t['n_mol']} PAHdb molecules; see pycalima.pahchem.group_tables\n")
        f.write(f"# {t['meta']}\n")
        f.write(f"{t['version']} {nz} {ng} {nTs} {t['lT'].size} {len(t['ions'])} {t['n_mol']}\n")
        f.write(_line([t["nc"], t["nh"], t["nc"] * M_C + t["nh"] * M_H, t["k_ad"]]))
        f.write(_line(CHARGES))
        for key in ("L0", "L1", "log10_Ts", "lT"):
            f.write(_line(t[key]))
        for g in range(ng):
            f.write(_line(t["egy"][g]))
        for key in ("A", "K", "H"):
            for g in range(ng):
                for i in range(nz):
                    f.write(_line(t[key][i, g]))
        for key in ("A_bg", "K_bg", "H_bg"):
            f.write(_line(t[key]))
        for key in ("alpha", "L"):
            for row in t[key]:
                f.write(_line(row))
        f.write(" ".join(t["ions"]) + "\n")
        for row in t["k_mn"]:
            f.write(_line(row))
        f.write(_line(t["k_ct"]))
    return path


def read_bin_table(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    meta = lines[1][2:] if len(lines) > 1 and lines[1].startswith("#") else ""
    it = iter(ln for ln in lines if ln.strip() and not ln.startswith("#"))
    version, nz, ng, _, _, ni, n_mol = (int(x) for x in next(it).split())
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: format {version}, expected {FORMAT_VERSION} (re-export the table)")
    vec = lambda: np.array(next(it).split(), dtype=float)
    nc, nh, _, k_ad = vec()
    t = dict(version=version, meta=meta, n_mol=n_mol, nc=nc, nh=nh, k_ad=k_ad)
    if tuple(int(z) for z in vec()) != CHARGES:
        raise ValueError(f"{path}: charges differ from {CHARGES}")
    for key in ("L0", "L1", "log10_Ts", "lT"):
        t[key] = vec()
    t["egy"] = np.array([vec() for _ in range(ng)])
    for key in ("A", "K", "H"):
        t[key] = np.array([[vec() for _ in range(nz)] for _ in range(ng)]).transpose(1, 0, 2)
    for key in ("A_bg", "K_bg", "H_bg"):
        t[key] = vec()
    for key in ("alpha", "L"):
        t[key] = np.array([vec() for _ in range(nz - 1)])
    t["ions"] = tuple(next(it).split())
    t["k_mn"] = np.array([vec() for _ in range(ni)])
    t["k_ct"] = vec()
    return t


# --------------------------------------------------------------------------- per cell
class PAHGroupTable:
    """One bin's table with the per-cell charge balance RAMSES would perform."""

    def __init__(self, tab, egy=None):
        self.t = tab
        self.set_group_energies(tab["egy"][:, tab["egy"].shape[1] // 2] if egy is None else egy)
        self._ln = {k: np.log(np.maximum(tab[k], 1e-300)) for k in ("alpha", "L", "k_mn")}

    def set_group_energies(self, egy):
        """Per-group kernels at mean energies ``egy`` [eV]: linear in egy between the two T_s shapes that bracket
        it (``rt_group_charging.GroupChargingTable.set_group_energies``)."""
        t = self.t
        self.Kg, self.Hg, self.Ag = (np.zeros(t["K"].shape[:2]) for _ in range(3))
        for g in range(t["L0"].size):
            e = t["egy"][g]
            eg = min(max(float(egy[g]), e[0]), e[-1])
            j = int(np.clip(np.searchsorted(e, eg) - 1, 0, e.size - 2))
            w = 0.0 if e[j + 1] == e[j] else (eg - e[j]) / (e[j + 1] - e[j])
            for out, key in ((self.Kg, "K"), (self.Hg, "H"), (self.Ag, "A")):
                out[:, g] = (1 - w) * t[key][:, g, j] + w * t[key][:, g, j + 1]

    def _in_T(self, key, T):
        """exp of the table's ln ``key`` at T, linear in log10 T, clamped to the grid."""
        lt = self.t["lT"]
        x = min(max(np.log10(T), lt[0]), lt[-1])
        i = min(int((x - lt[0]) / (lt[1] - lt[0])), lt.size - 2)
        v = (x - lt[i]) / (lt[i + 1] - lt[i])
        return np.exp((1 - v) * self._ln[key][..., i] + v * self._ln[key][..., i + 1])

    def solve(self, flux, T, n_e, n_h=0.0, ions=(), G0_bg=0.0):
        """The cell's PAH charge balance and what it gives the gas, per PAH.

        flux[g]: group photon fluxes c N_g [cm^-2 s^-1]; T [K]; n_e, n_h (atomic H) [cm^-3]; ions: ((name, n), ...)
        of ``ion names``; G0_bg: UV background [Habing]. Returns f(Z) over CHARGES; gamma, lam [erg s^-1]
        photoelectric heating and recombination cooling; electron_source and overflow [s^-1] (as
        ``Population.gas_rates``); alpha_mn, alpha_ct, per ion [cm^3 s^-1]; carbon_accretion [cm^3 s^-1].
        """
        t = self.t
        F = np.asarray(flux, dtype=float)
        photo_up = self.Kg @ F + G0_bg * t["K_bg"]
        heat = self.Hg @ F + G0_bg * t["H_bg"]
        n_ion = np.zeros(len(t["ions"]))
        for name, n in ions:
            n_ion[t["ions"].index(name)] += n
        k_mn = self._in_T("k_mn", T)
        up = photo_up[:-1].copy()
        up[0] += t["k_ad"] * n_h + n_ion @ k_mn
        up[1] += n_ion @ t["k_ct"]
        alpha = self._in_T("alpha", T)
        down = n_e * alpha
        lnr = np.log(np.maximum(up, 1e-300)) - np.log(np.maximum(down, 1e-300))
        lnf = np.concatenate([[0.0], np.cumsum(lnr)])
        f = np.exp(lnf - lnf.max())
        f /= f.sum()
        lam = n_e * f[1:] @ self._in_T("L", T)
        e_src = f[:-1] @ photo_up[:-1] + f[0] * t["k_ad"] * n_h - n_e * f[1:] @ alpha
        mn = {name: float(f[0] * k_mn[j]) for j, name in enumerate(t["ions"])}
        ct = {name: float(f[1] * t["k_ct"][j]) for j, name in enumerate(t["ions"])}
        return dict(f=f, gamma=float(f @ heat * chemistry.EV_ERG), lam=float(lam * chemistry.EV_ERG),
                    electron_source=float(e_src), overflow=float(f[-1] * photo_up[-1]),
                    alpha_mn=mn, alpha_ct=ct, carbon_accretion=0.4 * ct.get("C+", 0.0))


def group_fluxes(field, L0, L1):
    """(F_g [cm^-2 s^-1], egy_g [eV]) of a ``fields.RadiationField`` projected on the groups (zero outside its
    energies): what RAMSES-RT would carry for it."""
    F, egy = np.zeros(len(L0)), np.zeros(len(L0))
    for g, (l0, l1) in enumerate(zip(L0, L1)):
        E = np.geomspace(l0, l1, 2001)
        n = field.flux_at(E)
        F[g] = np.trapezoid(n, E)
        egy[g] = np.trapezoid(n * E, E) / F[g] if F[g] > 0 else np.sqrt(l0 * l1)
    return F, egy


def main(argv=None):
    import argparse

    from pycalima._paths import get_model_data_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nc", type=float, nargs=2, required=True, metavar=("MIN", "MAX"), help="N_C range [MIN, MAX)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--detachment", default=Params().detachment)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default model_data/pahchem_group_tables)")
    a = ap.parse_args(argv)
    lib = pop.shift_thresholds(pop.library_v400(), {-1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64})
    w = nc_range_weights(lib, *a.nc)
    tab = build_bin_table(lib, w, params=Params(detachment=a.detachment), workers=a.workers, label=a.label)
    out = a.out or Path(get_model_data_dir()) / "pahchem_group_tables"
    out.mkdir(parents=True, exist_ok=True)
    print(write_bin_table(tab, out / f"pah_group_table_{a.label}.dat"), f"({tab['n_mol']} molecules, <N_C> {tab['nc']:.1f})")


if __name__ == "__main__":
    main()
