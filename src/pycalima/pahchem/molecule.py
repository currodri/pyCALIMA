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

from pycalima.models.PAH_photophysics.pah_mol_data import load_pah_modes

_HC_EV_CM = 1.23984193e-4          # h c [eV cm]
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

    @classmethod
    def from_states_file(cls, path) -> "PAHSpecies":
        """Load a PAHdb transitions file named ``C<nc>[H<nh>]_<Z>.dat``.

        These are the files ``models.PAH_photophysics.pah_db_lookup``
        extracts into ``model_data/PAH_states/``.
        """
        path = Path(path)
        m = _STATES_NAME.match(path.stem)
        if not m:
            raise ValueError(f"{path.name}: expected C<nc>[H<nh>]_<Z>.dat")
        nc, nh, charge = int(m[1]), int(m[2] or 0), int(m[3])
        freq_ev, einstein_a = load_pah_modes(str(path))
        return cls(nc=nc, nh=nh, charge=charge, freqs_cm=freq_ev / _HC_EV_CM,
                   einstein_a=einstein_a, source=f"PAHdb transitions file {path.name}")
