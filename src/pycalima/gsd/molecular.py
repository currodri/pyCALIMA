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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: the Habing energy density in 6-13.6 eV [erg cm^-3], G0 = 1
U_HABING = 5.33e-14
EV = 1.602176634e-12


def habing_g0(photons, e_lo=6.0, e_hi=13.6):
    """G0 [Habing] of RT photon groups (``charging.PhotonGroups``): their energy density in [e_lo, e_hi] eV (groups
    counted by the share of their width inside), over the Habing value."""
    if photons is None:
        return 0.0
    L0, L1 = np.asarray(photons.L0, float), np.asarray(photons.L1, float)
    share = np.clip((np.minimum(L1, e_hi) - np.maximum(L0, e_lo)) / (L1 - L0), 0.0, 1.0)
    return float(np.sum(share * np.asarray(photons.N, float) * np.asarray(photons.egy, float)) * EV / U_HABING)


class MolecularTables:
    """The carbon-loss tables of the molecular bins, lowest bin first: ``cold`` and optionally ``hot`` lists of
    ``<label>_carbon.dat`` paths (one per bin, the same bins in both), the hot ones used above ``t_join`` [K]."""

    def __init__(self, cold, hot=None, t_join=1e4):
        from pycalima.solvers.table_io import build_pah_photolysis_interpolator_nd

        self.cold = [build_pah_photolysis_interpolator_nd(p)[0] for p in cold]
        self.hot = None if hot is None else [build_pah_photolysis_interpolator_nd(p)[0] for p in hot]
        if self.hot is not None and len(self.hot) != len(self.cold):
            raise ValueError("the cold and hot tables must cover the same bins")
        self.t_join = float(t_join)

    @classmethod
    def from_directories(cls, cold_dir, hot_dir=None, t_join=1e4):
        """The ``bin_nc<lo>-<hi>_carbon.dat`` files of each directory, in order of N_C."""

        def files(d):
            fs = list(Path(d).glob("bin_nc*_carbon.dat"))
            return sorted(fs, key=lambda f: float(f.name.split("bin_nc")[1].split("-")[0]))

        return cls(files(cold_dir), None if hot_dir is None else files(hot_dir), t_join)

    @property
    def n_bins(self):
        return len(self.cold)

    def carbon_loss(self, g0, n_h, T, n_e):
        """Carbon atoms lost per molecule per second [s^-1], per molecular bin."""
        tabs = self.hot if (self.hot is not None and T > self.t_join) else self.cold
        x = (np.log10(max(g0, 1e-30)), np.log10(n_h), np.log10(T), np.log10(max(n_e / n_h, 1e-30)))
        return np.array([2.0 * 10.0 ** f(*x) for f in tabs])
