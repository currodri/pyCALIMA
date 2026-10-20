"""Cosmic-ray spectra at a PAH: primary protons, nuclei and electrons, attenuated through H2.

These are sourced inputs to ``impact``, not predictions.

- **Interstellar spectra.** Padovani et al. (2018, A&A 614, A111) Table 1:
  j(E) = C E^a / (E + E0)^b in particles eV^-1 s^-1 cm^-2 sr^-1, with E the
  kinetic energy in eV. Protons come in model L (a = 0.1, the Voyager 1
  extrapolation) and H (a = -0.8); the paper calls these lower and upper
  bounds. Electrons use a = -1.3.
- **Nuclei.** The PDG (2022) Table 30.1 abundances at 10.6 GeV/nucleon,
  relative to H at the same kinetic energy per nucleon. The proton shape per
  nucleon is assumed for every nucleus (PDG: "the power law indices for each
  element may differ slightly"). Each charge group is represented by one
  (Z, A).
- **Attenuation.** Continuous slowing down through an H2 column N:
  j(E, N) = j(E0) L(E0) / L(E) with R(E0) = R(E) + N (Padovani et al. 2009,
  eqs. 21-25), where L is the loss per H2 molecule from NIST PSTAR, ASTAR and
  ESTAR for H2 gas. A nucleus of charge z and A nucleons loses, at a given
  speed, (z*)^2 times the proton loss (effective charge as in ``impact``).
  Pion production and the secondary e+/- of Padovani et al. (2018) are left
  out. They matter above ~1e24-1e25 cm^-2, beyond this module's use.

Kinetic energies are per particle in eV, and columns are N(H2) in cm^-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pycalima import _paths
from pycalima.pahchem import impact

M_H2_G = 2.01588 / 6.02214076e23


@dataclass(frozen=True)
class Padovani:
    """j(E) = C E^a / (E + E0)^b [particles eV^-1 s^-1 cm^-2 sr^-1], E in eV."""

    C: float
    e0_ev: float
    a: float
    b: float

    def __call__(self, e_ev) -> np.ndarray:
        E = np.asarray(e_ev, dtype=float)
        return self.C * E**self.a / (E + self.e0_ev) ** self.b


#: Padovani+18 Table 1 (b = a + (b - a))
PROTONS_L = Padovani(2.4e15, 650e6, 0.1, 2.8)
PROTONS_H = Padovani(2.4e15, 650e6, -0.8, 1.9)
ELECTRONS = Padovani(2.1e18, 710e6, -1.3, 1.9)
PROTON_MODELS = {"L": PROTONS_L, "H": PROTONS_H}

#: PDG (2022) Table 30.1: (name, Z, A, abundance relative to O at 10.6 GeV/nucleon);
#: one representative (Z, A) per charge group
PDG2022_GROUPS = (("He", 2, 4.0, 34.0), ("Li-B", 4, 9.0, 0.40), ("C-O", 7, 14.0, 2.20),
                  ("F-Ne", 10, 20.0, 0.30), ("Na-Mg", 12, 24.0, 0.22), ("Al-Si", 14, 28.0, 0.19),
                  ("P-S", 16, 32.0, 0.03), ("Cl-Ar", 18, 38.0, 0.01), ("K-Ca", 20, 40.0, 0.02),
                  ("Sc-Mn", 24, 52.0, 0.05), ("Fe-Ni", 26, 56.0, 0.12))
PDG2022_H = 550.0


def nuclei() -> list[tuple[impact.Projectile, float]]:
    """[(projectile, abundance relative to H at the same energy per nucleon)] (PDG 2022)."""
    out = []
    for name, z, a, f in PDG2022_GROUPS:
        proj = impact.ALPHA if z == 2 else impact.nucleus(name, z, a)
        out.append((proj, f / PDG2022_H))
    return out


def nucleus_spectrum(proton_spectrum, proj: impact.Projectile, abundance: float):
    """j_k(T) per unit kinetic energy of the nucleus: a_k j_p(T/A) / A."""
    A = proj.nucleons / impact.PROTON.nucleons
    return lambda t_ev: abundance * proton_spectrum(np.asarray(t_ev, dtype=float) / A) / A


# --- energy loss in H2 and CSDA attenuation ---------------------------------

@lru_cache(maxsize=None)
def _star_h2(prog: str):
    d = np.loadtxt(_paths.get_external_data_path(f"pahchem/nist_{prog}_hydrogen.csv"), delimiter=",", comments="#")
    return d[:, 0] * 1e6, d[:, 3] * 1e6 * M_H2_G        # T [eV], total loss per H2 [eV cm^2]


def loss_h2(proj: impact.Projectile, t_ev) -> np.ndarray:
    """Energy loss per H2 molecule [eV cm^2] (NIST, total). Electrons use ESTAR.
    Protons and alphas use PSTAR and ASTAR. Heavier nuclei use PSTAR at the same
    speed times (z*)^2. Power-law extrapolation outside the tables."""
    t = np.asarray(t_ev, dtype=float)
    if proj.is_electron:
        T, L = _star_h2("estar")
        tt, scale = t, 1.0
    elif proj.z == 2:
        T, L = _star_h2("astar")
        tt, scale = t * (impact.ALPHA.nucleons / proj.nucleons), 1.0
    else:
        T, L = _star_h2("pstar")
        tt = t / proj.nucleons * impact.PROTON.nucleons
        scale = 1.0 if proj.z == 1 else impact.effective_charge(proj, impact.kinematics(proj, t)[0]) ** 2
    lt, lT, lL = np.log(tt), np.log(T), np.log(L)
    lo = lL[0] + (lt - lT[0]) * (lL[1] - lL[0]) / (lT[1] - lT[0])
    hi = lL[-1] + (lt - lT[-1]) * (lL[-1] - lL[-2]) / (lT[-1] - lT[-2])
    return scale * np.exp(np.where(lt < lT[0], lo, np.where(lt > lT[-1], hi, np.interp(lt, lT, lL))))


@dataclass(frozen=True)
class CSDA:
    """Range R(T) = int_0^T dT'/L(T') [H2 cm^-2] of one projectile on a log grid."""

    t: np.ndarray
    loss: np.ndarray
    range_: np.ndarray

    @classmethod
    def build(cls, proj: impact.Projectile, t_min: float = 1.0e3, t_max: float = 1.0e13, n: int = 3000) -> "CSDA":
        t = np.geomspace(t_min, t_max, n)
        L = loss_h2(proj, t)
        y = t / L
        R = np.concatenate([[y[0]], y[0] + np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(np.log(t)))])
        return cls(t, L, R)

    def attenuated(self, j_is, t_ev, n_h2: float) -> np.ndarray:
        """j(T, N) for the interstellar spectrum j_is incident at N = 0."""
        t = np.asarray(t_ev, dtype=float)
        if n_h2 <= 0.0:
            return j_is(t)
        r = np.interp(np.log(t), np.log(self.t), self.range_) + n_h2
        ok = r <= self.range_[-1]
        t0 = np.exp(np.interp(np.log(r[ok]), np.log(self.range_), np.log(self.t)))
        out = np.zeros_like(t)
        L = lambda x: np.exp(np.interp(np.log(x), np.log(self.t), np.log(self.loss)))  # noqa: E731
        out[ok] = j_is(t0) * L(t0) / L(t[ok])
        return out


def projectile_spectra(protons: str = "L", n_h2: float = 0.0, electrons: bool = True,
                       heavy: bool = True) -> list[tuple[impact.Projectile, object]]:
    """[(projectile, j(T) at column n_h2)] for protons (model ``L`` or ``H``),
    PDG nuclei and primary electrons."""
    jp = PROTON_MODELS[protons]
    items = [(impact.PROTON, jp)]
    if heavy:
        items += [(proj, nucleus_spectrum(jp, proj, a)) for proj, a in nuclei()]
    if electrons:
        items.append((impact.ELECTRON, ELECTRONS))
    if n_h2 <= 0.0:
        return items
    out = []
    for proj, j in items:
        csda = CSDA.build(proj)
        out.append((proj, (lambda c, jj: (lambda t: c.attenuated(jj, t, n_h2)))(csda, j)))
    return out
