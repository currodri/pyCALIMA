"""A PAH species: composition, charge and harmonic vibrational modes.

A ``PAHSpecies`` is the molecular input that every statistical quantity in
``pahchem`` is computed from. It carries its own frequencies, so each
(N_C, N_H, Z) state can have its own modes instead of inheriting the parent's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EINSTEIN_A_PER_KM_MOL = 1.2512e-7  # A [s^-1] = this x nu^2 [cm^-2] x I [km/mol]
_STATES_NAME = re.compile(r"^C(\d+)(?:H(\d+))?_(-?\d+)$")


@dataclass(frozen=True, eq=False)
class PAHSpecies:
    """One PAH state and its harmonic modes.

    Attributes
    ----------
    nc, nh, charge : int
        Carbon atoms, hydrogen atoms and charge Z.
    freqs_cm : ndarray
        Harmonic frequencies [cm^-1], already scaled, one per mode.
    einstein_a : ndarray or None
        Einstein A coefficient of each fundamental [s^-1].
    source : str
        Where the modes come from (file, database version, level of theory).
    """

    nc: int
    nh: int
    charge: int
    freqs_cm: np.ndarray = field(repr=False)
    einstein_a: np.ndarray | None = field(default=None, repr=False)
    source: str = ""

    def __post_init__(self):
        n_expected = 3 * (self.nc + self.nh) - 6
        if len(self.freqs_cm) != n_expected:
            raise ValueError(f"{self.formula}: {len(self.freqs_cm)} modes, "
                             f"expected 3N-6 = {n_expected}")

    @property
    def formula(self) -> str:
        return f"C{self.nc}" + (f"H{self.nh}" if self.nh else "")

    @property
    def n_modes(self) -> int:
        return len(self.freqs_cm)

    @property
    def zpe_cm(self) -> float:
        """Zero-point energy [cm^-1]."""
        return 0.5 * float(np.sum(self.freqs_cm))

    def stripped(self, nh: int) -> "PAHSpecies":
        """An approximate mode set for the same skeleton with ``nh`` H atoms.

        Removes, per missing H, one C-H stretch (2900-3200 cm^-1), in-plane
        bend (1050-1350) and out-of-plane bend (700-950), the least intense
        first, and rescales the Einstein A of the modes left in each window to
        nh / N_H of the window's total. In PAHdb a few symmetric combinations
        carry nearly all of a window's intensity (95 % of the coronene C-H
        stretch in 2 of 12 modes), so removing the most intense modes, as
        before 2026-10-20, took about half the IR cooling away with the first
        H (pahchem-audit, C1).

        This keeps 3N-6 modes but not the skeleton's softening on
        dehydrogenation: against the PAHdb C54 modes, ln rho(E) is 6-8 lower at
        5-20 eV, while d ln rho/dE (which sets k(E)) is closer to C54 than the
        parent's is. A placeholder until DFT modes of each state exist
        (pahchem-data-requests, T1).
        """
        n = self.nh - int(nh)
        if n < 0:
            raise ValueError("stripped() removes hydrogens only")
        f = np.asarray(self.freqs_cm, dtype=float)
        A = np.ones_like(f) if self.einstein_a is None else np.asarray(self.einstein_a, dtype=float).copy()
        keep = np.ones(f.size, bool)
        for lo, hi in ((2900.0, 3200.0), (1050.0, 1350.0), (700.0, 950.0)):
            window = (f >= lo) & (f < hi)
            cand = np.nonzero(keep & window)[0]
            if cand.size < n:
                raise ValueError(f"{self.formula}: fewer than {n} modes in {lo}-{hi} cm^-1")
            total = A[cand].sum()
            keep[cand[np.argsort(A[cand])[:n]]] = False
            left = keep & window
            if A[left].sum() > 0.0:
                A[left] *= total * int(nh) / self.nh / A[left].sum()
        return PAHSpecies(self.nc, int(nh), self.charge, f[keep],
                          None if self.einstein_a is None else A[keep],
                          f"{self.source}, C-H modes stripped to N_H = {nh}")

    @classmethod
    def from_states_file(cls, path) -> "PAHSpecies":
        """Load a PAHdb transitions file named ``C<nc>[H<nh>]_<Z>.dat``.

        These are the files ``models.PAH_photophysics.pah_db_lookup``
        extracts into ``model_data/PAH_states/``.

        The FREQUENCY column is used as written: PAHdb stores frequencies
        already scaled, and the SCALE column only records the factor applied
        (coronene's duo C-H out-of-plane bend is 861.4 cm^-1 there, the
        matrix-isolation band 857). The legacy ``pah_mol_data.load_pah_modes``
        multiplies by SCALE again, which puts every mode 2.1 % low (C-H
        stretches 4.0 %); it is not used here. Einstein A = 1.2512e-7 nu^2 I
        [s^-1] for nu in cm^-1 and I in km/mol.
        """
        path = Path(path)
        m = _STATES_NAME.match(path.stem)
        if not m:
            raise ValueError(f"{path.name}: expected C<nc>[H<nh>]_<Z>.dat")
        nc, nh, charge = int(m[1]), int(m[2] or 0), int(m[3])
        freq, intensity = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                tokens = line.split()
                if not tokens or line.lstrip()[0] in "\\|":
                    continue
                freq.append(float(tokens[1]))
                intensity.append(float(tokens[2]))
        if not freq:
            raise ValueError(f"{path.name}: no transitions")
        freq = np.array(freq)
        return cls(nc=nc, nh=nh, charge=charge, freqs_cm=freq,
                   einstein_a=_EINSTEIN_A_PER_KM_MOL * freq**2 * np.array(intensity),
                   source=f"PAHdb transitions file {path.name}")
