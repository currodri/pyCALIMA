"""Radiation fields for pahchem, with one G0 convention.

A ``RadiationField`` is the 4 pi-integrated photon flux per unit photon
energy, F(E) = 4 pi J_E / E [photons cm^-2 s^-1 eV^-1], on an energy grid in
eV. Every rate in pahchem is an integral of sigma(E) F(E).

G0 is always the Habing normalisation: G0 = int_6^13.6 E F(E) dE / 1.6e-3 erg
cm^-2 s^-1. With it, the Draine (1978) field has G0 = 1.67 and the Mathis et
al. (1983) field 1.13. (pyCALIMA's other modules use at least five G0
definitions; see docs/physics/radiation.md. This module does not.) A field with
no flux in that band (X-rays only) is scaled by its whole energy flux in the same
unit instead (``RadiationField.intensity``).

Below 5 eV the interstellar field is old-star light, which G0 does not measure
and which does not scale with it: Draine's (1978) fit is a far-UV fit, and
Meudon PDR adds a separate, unscaled stellar component to its FUV field (as
Cloudy's ``table Draine`` stops at 5 eV). ``old_stars`` is that component, the
three diluted blackbodies of Mathis, Mezger & Panagia (1983) below their
5.04 eV break; ``draine1978(old_stars=...)`` and ``mathis1983(fixed_optical=True)``
add it unscaled to a G0-scaled far UV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pycalima import _paths

HABING_FLUX = 1.6e-3            # erg cm^-2 s^-1 between 6 and 13.6 eV
EV_ERG = 1.602176634e-12
HC_EV_NM = 1239.84193
LYMAN_EV = 13.6
#: lowest energy of Draine's (1978) far-UV fit as used here [eV]; below it the fit does not apply
DRAINE_E_MIN_EV = 5.0
#: Mathis, Mezger & Panagia (1983): power laws above this energy, diluted blackbodies (old stars) below [eV]
MMP83_BREAK_EV = 5.04


@dataclass(frozen=True, eq=False)
class RadiationField:
    """Photon flux F(E) [photons cm^-2 s^-1 eV^-1] on ``energy_ev``."""

    energy_ev: np.ndarray = field(repr=False)
    photon_flux: np.ndarray = field(repr=False)
    name: str = ""

    def __post_init__(self):
        E = np.asarray(self.energy_ev, dtype=float)
        if E.ndim != 1 or E.size < 2 or np.any(np.diff(E) <= 0):
            raise ValueError("energy_ev must be a strictly increasing 1-D grid")
        if np.shape(self.photon_flux) != E.shape or np.any(np.asarray(self.photon_flux) < 0):
            raise ValueError("photon_flux must be non-negative, one value per energy")

    def g0(self) -> float:
        """Habing G0 of this field."""
        m = (self.energy_ev > 6.0) & (self.energy_ev < LYMAN_EV)
        E = np.concatenate([[6.0], self.energy_ev[m], [LYMAN_EV]])      # whole band, also on a coarse grid
        return float(np.trapezoid(E * self.flux_at(E), E) * EV_ERG / HABING_FLUX)

    def intensity(self) -> float:
        """The scale ``scaled`` sets: the Habing G0, or for a field with no flux at 6-13.6 eV (X-rays only, the far and
        extreme UV absorbed on the way) its whole energy flux in the same unit, int E F dE / 1.6e-3 erg cm^-2 s^-1."""
        g = self.g0()
        if g > 0.0:
            return g
        return float(np.trapezoid(self.energy_ev * self.photon_flux, self.energy_ev) * EV_ERG / HABING_FLUX)

    def scaled(self, g0: float) -> "RadiationField":
        """The same spectral shape with Habing G0 = ``g0`` (``intensity`` = g0 for a field with no far UV)."""
        return RadiationField(self.energy_ev, self.photon_flux * (g0 / self.intensity()),
                              f"{self.name} (G0={g0:g})")

    def flux_at(self, energy_ev) -> np.ndarray:
        """F at arbitrary energies: log-linear interpolation, zero outside."""
        E = np.asarray(energy_ev, dtype=float)
        with np.errstate(divide="ignore"):
            logF = np.log(self.photon_flux)
        out = np.exp(np.interp(E, self.energy_ev, logF, left=-np.inf, right=-np.inf))
        return np.where(np.isfinite(out), out, 0.0)

    def photon_rate(self, sigma_cm2, energy_ev=None) -> float:
        """int sigma(E) F(E) dE [s^-1], for sigma given on ``energy_ev``
        (default: this field's grid)."""
        E = self.energy_ev if energy_ev is None else np.asarray(energy_ev, dtype=float)
        return float(np.trapezoid(np.asarray(sigma_cm2) * self.flux_at(E), E))


#: flat-F_nu equivalent of Habing G0 = 1, 1.6e-3 erg cm^-2 s^-1 over 6-13.6 eV [erg cm^-2 s^-1 Hz^-1] (Peeters+24 App. A)
HABING_FNU = 8.7e-19
H_EV_S = 4.135667696e-15


def fnu_g0_ratio(field, lam_angstrom=(1027.0, 1040.0), smooth_ev=0.15):
    """F_nu at ``lam_angstrom`` (mean over the lines) of ``field`` per unit Habing G0, over ``HABING_FNU``.

    A G0 defined as F_nu(lambda) / 8.7e-19, as Peeters+24 measure it from the
    O I 1.129 / 1.317 um fluorescence pumped at 1027 / 1040 A, equals this
    ratio times the Habing G0 of the same field. F_nu is the median within
    +-``smooth_ev`` of each line, to smooth over model-atmosphere line structure.
    """
    out = []
    for lam in lam_angstrom:
        E = HC_EV_NM * 10.0 / lam
        win = (field.energy_ev > E - smooth_ev) & (field.energy_ev < E + smooth_ev)
        fnu = E * EV_ERG * np.median(field.photon_flux[win]) * H_EV_S
        out.append(fnu / HABING_FNU)
    return float(np.mean(out) / field.g0())


def _grid(e_min, e_max, n):
    return np.linspace(e_min, e_max, n)


def draine1978(g0=None, e_min=5.0, e_max=LYMAN_EV, n=861, old_stars=0.0) -> RadiationField:
    """Draine (1978) ISRF; G0 = 1.67 unless rescaled.

    The fit is evaluated from ``DRAINE_E_MIN_EV`` (5 eV) only; below it the field is zero, plus ``old_stars``
    times the solar-circle old-star light (``old_stars_flux``), which is not scaled by ``g0``. Its 5 eV edge is
    sharp (a node 1e-6 eV below it).
    """
    from pycalima.models.tools.radiation_fields import Draine_1978_isrf

    if e_min < DRAINE_E_MIN_EV:
        n_lo = max(round(float(DRAINE_E_MIN_EV - e_min) / 0.01), 2)
        E = np.concatenate([np.linspace(e_min, DRAINE_E_MIN_EV - 1e-6, n_lo), _grid(DRAINE_E_MIN_EV, e_max, n)])
    else:
        E = _grid(e_min, e_max, n)
    F = np.where(E >= DRAINE_E_MIN_EV, Draine_1978_isrf(HC_EV_NM / E) * HC_EV_NM / E**2, 0.0)  # per nm -> per eV
    f = RadiationField(E, np.clip(F, 0.0, None), "Draine 1978")
    if g0 is not None:
        f = f.scaled(g0)
    if old_stars > 0.0:
        add = np.where(E < DRAINE_E_MIN_EV, old_stars * old_stars_flux(E), 0.0)
        f = RadiationField(E, f.photon_flux + add, f"{f.name} + {old_stars:g} x old stars")
    return f


def old_stars_flux(energy_ev) -> np.ndarray:
    """Photon flux [cm^-2 s^-1 eV^-1] of the old-star light at the solar circle: the three diluted blackbodies
    of Mathis, Mezger & Panagia (1983; 7500, 4000, 3000 K), below their ``MMP83_BREAK_EV`` break, zero above."""
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    E = np.atleast_1d(np.asarray(energy_ev, dtype=float))
    c = 2.99792458e10
    u = np.array([Mathis83_radiation_field(e) if 0.0 < e <= MMP83_BREAK_EV else 0.0 for e in E])
    return c * u / (E * EV_ERG)


def mathis1983(g0=None, e_min=0.5, e_max=LYMAN_EV, n=1311, fixed_optical=False) -> RadiationField:
    """Mathis, Mezger & Panagia (1983) ISRF at the solar circle; G0 = 1.13.

    ``g0`` scales the whole spectrum (as the papers reproduced in pahchem do), unless ``fixed_optical``: then
    only the far UV above ``MMP83_BREAK_EV`` is scaled and the old-star light below it stays at its solar-circle
    value.
    """
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    E = _grid(e_min, e_max, n)
    c = 2.99792458e10
    u = np.array([Mathis83_radiation_field(e) for e in E])        # erg cm^-3 eV^-1
    f = RadiationField(E, c * u / (E * EV_ERG), "Mathis 1983")
    if g0 is None:
        return f
    if not fixed_optical:
        return f.scaled(g0)
    uv = E > MMP83_BREAK_EV
    return RadiationField(E, np.where(uv, f.photon_flux * (g0 / f.g0()), f.photon_flux),
                          f"Mathis 1983 (G0={g0:g}, old stars fixed)")


def blackbody(temperature_k, g0, e_min=0.5, e_max=LYMAN_EV, n=1311) -> RadiationField:
    """A blackbody spectrum cut at ``e_max``, scaled to Habing ``g0``."""
    E = _grid(e_min, e_max, n)
    x = E / (8.617333262e-5 * temperature_k)
    shape = E**2 / np.expm1(np.minimum(x, 700.0))                 # photon flux, arbitrary units
    return RadiationField(E, shape, f"blackbody {temperature_k:g} K").scaled(g0)


def powerlaw(alpha, g0, e_min=1.0, e_max=1e4, n=2001) -> RadiationField:
    """F_nu ~ nu^-alpha (photon flux per eV ~ E^-(alpha+1)) from ``e_min`` to ``e_max`` [eV], log-spaced, scaled to ``g0``:
    an X-ray-bright source (AGN-like for alpha ~ 1-1.5). With ``e_min`` above 13.6 eV it is X-rays only, a crude stand-in for
    a shielded field (a sharp cut, not a column's attenuation), and ``g0`` is its energy flux in Habing units (``intensity``)."""
    E = np.unique(np.concatenate([np.geomspace(e_min, e_max, n), [6.0, LYMAN_EV]]))
    E = E[(E >= e_min) & (E <= e_max)]
    return RadiationField(E, E ** -(alpha + 1.0), f"power law alpha = {alpha:g}").scaled(g0)


def kurucz(teff, g0, e_max=LYMAN_EV, e_min=None) -> RadiationField:
    """Kurucz atmosphere ``kp00_<teff>`` (bundled), cut at ``e_max`` (and below ``e_min``), scaled to ``g0``.

    The ``e_min`` cut adds a node at ``e_min`` with the log-interpolated flux, so integrals above it are those of
    the uncut spectrum.
    """
    data = np.loadtxt(_paths.get_external_data_path(f"kp00_{int(teff)}"), skiprows=3)
    E = HC_EV_NM / data[:, 0]
    F = data[:, 1] * HC_EV_NM / E**2 / (E * EV_ERG)               # 4 pi dropped: shape only
    order = np.argsort(E)
    E, F = E[order], F[order]
    keep = (E <= e_max) & (F > 0)
    E, F = E[keep], F[keep]
    E, idx = np.unique(E, return_index=True)
    F = F[idx]
    if e_min is not None and e_min > E[0]:
        f_min = np.exp(np.interp(e_min, E, np.log(F)))
        above = E > e_min
        E, F = np.concatenate([[e_min], E[above]]), np.concatenate([[f_min], F[above]])
    return RadiationField(E, F, f"Kurucz {int(teff)} K").scaled(g0)
