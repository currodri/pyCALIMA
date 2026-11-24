"""The molecular bins' tables (carbon below N_C,max): carbon loss from pahchem's equilibrium hydrogen tables.

RAMSES tabulates the molecular bins' state rather than tracking it (decided 2026-11-15): per bin,
``pahchem.hydrogen_tables`` solves every molecule's joint (Z, N_H) chain in steady state on the pahchem-nd v1 axes
(log10 G0 [Habing], log10 n_H, log10 T, log10 x_e) and averages over the bin's molecules. Its ``<label>_carbon.dat``
holds log10 of the carbon atoms lost per molecule per second / 2 (the pahchem unit); ``MolecularTables.carbon_loss``
returns the carbon atoms lost per molecule per second, which the operator applies as erosion of the molecular bins
(a shift; what erodes below the grid's N_C 20 goes to the gas).

Two table sets: the cold production tables (to 1e4 K) and, above ``t_join``, the hot-gas tables (``--hot-gas``:
thermal electron and ion passages), the user's choice of 2026-11-23. Outside the axes each table is clamped to its
edge.

The charge, photoelectric heating, recombination cooling and photon absorption of the molecular bins come from
pahchem's group tables (``pahchem.group_tables``, one per bin, on the cell's photon groups; built for the GSD by
``diagnostics/gsd/build_molecular_group_tables.py``): ``MolecularTables.charge`` solves each bin's four-state balance
(``PAHGroupTable.solve``) and gives per molecule f(Z), Gamma, Lambda and the absorption rate R_abs with its split over
the groups (the photo-release term's input, ``release``). The tables hold the charge to 1e5 K (no hot-gas collisions;
``group_tables.T_VALID_MAX_K``); above it the charge is the tables' clamped value (labelled).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: the Habing energy density in 6-13.6 eV [erg cm^-3], G0 = 1
U_HABING = 5.33e-14
EV = 1.602176634e-12
C_CGS = 2.99792458e10
AMU = 1.66053906660e-24


def habing_g0(photons, e_lo=6.0, e_hi=13.6):
    """G0 [Habing] of RT photon groups (``charging.PhotonGroups``): their energy density in [e_lo, e_hi] eV (groups
    counted by the share of their width inside), over the Habing value."""
    if photons is None:
        return 0.0
    L0, L1 = np.asarray(photons.L0, float), np.asarray(photons.L1, float)
    share = np.clip((np.minimum(L1, e_hi) - np.maximum(L0, e_lo)) / (L1 - L0), 0.0, 1.0)
    return float(np.sum(share * np.asarray(photons.N, float) * np.asarray(photons.egy, float)) * EV / U_HABING)


def pah_ions(ions, n_e):
    """The cell's ions [(n [cm^-3], charge, mass [g])] (``CellInputs.ions``; None: H+ at n_e) as the group tables'
    ((name, n), ...): singly charged ions named by the nearest element mass of ``rt_group_charging.RECOMB_IONS``."""
    from pycalima.models.dust_charge.rt_group_charging import RECOMB_IONS

    if ions is None:
        return (("H+", float(n_e)),)
    masses = np.array([m for _, m, _ in RECOMB_IONS])
    out = []
    for n, z, m in ions:
        if round(float(z)) == 1:
            out.append((f"{RECOMB_IONS[int(np.argmin(np.abs(masses - m / AMU)))][0]}+", float(n)))
    return tuple(out)


@dataclass
class MolecularState:
    """Per molecular bin (lowest first): f over ``CHARGES`` (-1, 0, +1, +2), its mean and width; per molecule the
    photoelectric heating ``gamma`` and recombination cooling ``lam`` [erg s^-1], the absorption rate ``r_abs``
    [s^-1] and its split ``absorbed`` [bins, k] over the photon energies ``energy`` [k] [eV] (the cell's groups at
    their mean energies, then the UV background's share in the same groups at the Mathis field's); ``valid``: T
    within the tables' charge range."""

    f: np.ndarray
    mean: np.ndarray
    width: np.ndarray
    gamma: np.ndarray
    lam: np.ndarray
    r_abs: np.ndarray
    absorbed: np.ndarray
    energy: np.ndarray
    valid: bool


class MolecularTables:
    """The molecular bins' tables, lowest bin first: ``cold`` and optionally ``hot`` lists of ``<label>_carbon.dat``
    paths (one per bin, the same bins in both), the hot ones used above ``t_join`` [K]; ``groups`` (optional) the
    bins' ``pahchem.group_tables`` files (``<label>_group.dat``) for ``charge``."""

    def __init__(self, cold, hot=None, t_join=1e4, groups=None):
        from pycalima.solvers.table_io import build_pah_photolysis_interpolator_nd

        self.cold = [build_pah_photolysis_interpolator_nd(p)[0] for p in cold]
        self.hot = None if hot is None else [build_pah_photolysis_interpolator_nd(p)[0] for p in hot]
        if self.hot is not None and len(self.hot) != len(self.cold):
            raise ValueError("the cold and hot tables must cover the same bins")
        self.t_join = float(t_join)
        self.groups = None
        if groups is not None:
            from pycalima.pahchem.group_tables import PAHGroupTable, read_bin_table

            self.groups = [g if isinstance(g, PAHGroupTable) else PAHGroupTable(read_bin_table(g)) for g in groups]
            if len(self.groups) != len(self.cold):
                raise ValueError("one group table per molecular bin")
        self._bg = None

    @classmethod
    def from_directories(cls, cold_dir, hot_dir=None, t_join=1e4, group_dir=None):
        """The ``bin_nc<lo>-<hi>_carbon.dat`` (and ``_group.dat``) files of each directory, in order of N_C."""

        def files(d, kind):
            fs = list(Path(d).glob(f"bin_nc*_{kind}.dat"))
            return sorted(fs, key=lambda f: float(f.name.split("bin_nc")[1].split("-")[0]))

        return cls(files(cold_dir, "carbon"), None if hot_dir is None else files(hot_dir, "carbon"), t_join,
                   None if group_dir is None else files(group_dir, "group"))

    @property
    def n_bins(self):
        return len(self.cold)

    def carbon_loss(self, g0, n_h, T, n_e):
        """Carbon atoms lost per molecule per second [s^-1], per molecular bin."""
        tabs = self.hot if (self.hot is not None and T > self.t_join) else self.cold
        x = (np.log10(max(g0, 1e-30)), np.log10(n_h), np.log10(T), np.log10(max(n_e / n_h, 1e-30)))
        return np.array([2.0 * 10.0 ** f(*x) for f in tabs])

    def _background(self):
        """(F_g / G0 [cm^-2 s^-1], mean energy [eV]) of the Mathis field (``group_tables``' UV background) in the
        tables' groups."""
        if self._bg is None:
            from pycalima.pahchem.group_tables import BG_GRID, _mathis_flux

            t = self.groups[0].t
            fb = _mathis_flux()
            F, e = np.zeros(t["L0"].size), np.sqrt(t["L0"] * t["L1"])
            for g, (l0, l1) in enumerate(zip(t["L0"], t["L1"], strict=True)):
                E = np.geomspace(max(l0, BG_GRID[0]), min(l1, BG_GRID[-1]), 400) if l0 < BG_GRID[-1] else None
                if E is None or E[0] >= E[-1]:
                    continue
                n = np.interp(np.log(E), np.log(BG_GRID), fb)
                F[g] = np.trapezoid(n, E)
                e[g] = np.trapezoid(n * E, E) / F[g]
            self._bg = (F, e)
        return self._bg

    def charge(self, photons, T, n_e, ions=None, G0_bg=0.0) -> MolecularState:
        """Each bin's charge balance in the cell (``PAHGroupTable.solve``; atomic-H associative detachment off, the
        group tables' default) and its absorption: R_abs = sum_z f(z) [A_g(z) F_g + G0_bg A_bg(z)], F_g = c N_g. The
        background's absorptions are split over the groups by the group-averaged cross-sections times the Mathis
        field's group fluxes (an approximation; their total is the table's exact A_bg)."""
        if self.groups is None:
            raise ValueError("no group tables")
        t0 = self.groups[0].t
        if photons is None:
            F, egy = np.zeros(t0["L0"].size), np.sqrt(t0["L0"] * t0["L1"])
        else:
            if not (np.allclose(photons.L0, t0["L0"]) and np.allclose(photons.L1, t0["L1"])):
                raise ValueError("the group tables' photon groups differ from the cell's")
            F, egy = C_CGS * np.asarray(photons.N, float), np.asarray(photons.egy, float)
        Fb, eb = self._background()
        pi = pah_ions(ions, n_e)
        from pycalima.pahchem.group_tables import CHARGES

        q = np.asarray(CHARGES, float)
        rows = []
        for tab in self.groups:
            tab.set_group_energies(egy)
            r = tab.solve(F, T, n_e, 0.0, pi, G0_bg)
            f = r["f"]
            ag = f @ tab.Ag                                       # group-averaged cross-section [cm^2] per group
            a_rt = ag * F
            bg_tot = G0_bg * float(f @ tab.t["A_bg"])
            share = ag * Fb
            a_bg = bg_tot * share / share.sum() if share.sum() > 0 else np.zeros_like(share)
            mu = float(f @ q)
            rows.append((f, mu, float(np.sqrt(max(f @ q ** 2 - mu ** 2, 0.0))), r["gamma"], r["lam"],
                         float(a_rt.sum() + bg_tot), np.concatenate([a_rt, a_bg]), r["valid"]))
        f, mu, sg, gam, lam, rab, absd, ok = zip(*rows, strict=True)
        return MolecularState(np.array(f), np.array(mu), np.array(sg), np.array(gam), np.array(lam), np.array(rab),
                              np.array(absd), np.concatenate([egy, eb]), bool(all(ok)))
