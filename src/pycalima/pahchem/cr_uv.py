"""The ultraviolet field cosmic rays induce in molecular gas, as pahchem's PAHs absorb it.

Cosmic rays and their secondary electrons excite H2. Its Lyman-Werner decays and H's Lyman alpha then emit
R(lambda) = zeta(H2) x(H2) P(lambda) photons per H nucleus (Heays, Bosman & van Dishoeck 2017, A&A 602, A105, eq. 16),
with x(H2) = n(H2) / n_H and zeta(H2) the H2 ionisation rate, primary and secondary. The photons are absorbed close to
where they are made, mostly by dust, so a molecule sees the 4 pi photon flux F = R / sigma_abs, sigma_abs being the
absorption cross section per H nucleus (their eqs. 17-18 with the dust alone).

The generation spectrum comes from a *source*: ``SOURCES`` maps a name to a function that returns (wavelength [nm],
cumulative photons per H2 ionisation below it). Another model is one more entry, which ``field`` then takes by name:
CRISpy's escaping Lyman-Werner lines (Padovani+24), the planned replacement.

- "heays17" (the default, the user's choice of 2026-11-04): Heays+17's P(lambda), Gredel+89's spectrum for para-H2 and
  1 km/s lines, 0.36 photons per ionisation in all. Digitised from their Fig. 32 (``diagnostics/pahchem/
  digitise_heays17_fig32.py``), since RAMSES-RTZ's CR-induced photorates of metals are Heays+17's
  (rtz/cosmic_ray_ionization_module.f90, at zeta(H2) = 1e-16 and scaled linearly).

The dust is the one Heays+17 used, as CRISpy does: Draine's (2003) Milky Way R_V = 3.1 grains (WD01 renormalised;
``draine2003_mw_rv3.1_uv.csv``, from CRISpy's copy of the Princeton table), absorbing C_ext (1 - albedo) per H. Hensley
& Draine (2023) astrodust + PAH absorb 1.2-1.4x less at 100-150 nm, which raised the atoms' rates 1.4-1.6x above
Heays+17's (``check_cr_uv.py``).

Not followed:
- H2's absorption of its own lines to v'' = 0 (those photons re-pump H2 and mostly re-emerge redder);
- H2's absorption of the continuum below 110 nm (Heays+17 find factors of 0.5-5 for species absorbing there);
- photons above 13.6 eV;
- the dust-to-gas ratio (F scales as 1 / sigma_abs: a metallicity input once the dust is one).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pycalima import _paths
from pycalima.pahchem import fields

HC_EV_NM = 1239.841984
#: zeta(H2) [s^-1] of RTZ's normalisation: its CR-induced rates are Heays+17's at this zeta, scaled by zeta(H2) / 1e-16
ZETA_REF = 1.0e-16
#: x(H2) = n(H2) / n_H of Heays+17's dark cloud (x(H) = 1e-4), the default
X_H2_REF = 0.5
#: bump when this module's construction of the field changes (``signature`` also hashes the data it reads)
FIELD_VERSION = 1


@lru_cache(maxsize=None)
def heays17():
    """(wavelength [nm], cumulative photons per H2 ionisation) of Heays+17 Fig. 32, digitised (Lyman alpha a step)."""
    path = _paths.get_external_data_path("pahchem", "heays17_cr_uv_cumulative.csv")
    with open(path) as f:
        rows = [line for line in f if not line.startswith("#")][1:]
    d = np.array([[float(x) for x in r.split(",")] for r in rows])
    return d[:, 0], d[:, 1]


#: generation spectra by name: () -> (wavelength [nm], cumulative photons per H2 ionisation below it)
SOURCES = {"heays17": heays17}


def secondary_factor(x_e):
    """1 + phi_s(x_e): RTZ's secondary ionisation, phi_s = (1 - x_e / 1.2) 0.67 / (1 + x_e / 0.05)
    (rtz/cosmic_ray_ionization_module.f90, secondary_cr_rates), x_e = n_e / n_H."""
    x = np.asarray(x_e, dtype=float)
    return 1.0 + (1.0 - x / 1.2) * 0.670 / (1.0 + x / 0.05)


def zeta_h2(zeta_h, x_e):
    """zeta(H2) [s^-1] as RTZ has it: 2 zeta_H (1 + phi_s(x_e)), zeta_H the primary rate per H (``Environment.zeta_cr``)."""
    return 2.0 * float(zeta_h) * float(secondary_factor(x_e))


def scale(zeta_h, x_e):
    """zeta(H2) / ZETA_REF, RTZ's cosmic_ray_scale_factor: the g0 of an Environment with the "cr_uv" field shape."""
    return zeta_h2(zeta_h, x_e) / ZETA_REF


@lru_cache(maxsize=None)
def _dust_table():
    path = _paths.get_external_data_path("pahchem", "draine2003_mw_rv3.1_uv.csv")
    with open(path) as f:
        rows = [line for line in f if not line.startswith("#")][1:]
    d = np.array([[float(x) for x in r.split(",")] for r in rows])
    return d[:, 0] * 1e3, d[:, 3] * (1.0 - d[:, 1])                       # lam [nm], C_abs = C_ext (1 - albedo) per H


def dust_absorption_per_h(energy_ev) -> np.ndarray:
    """sigma_abs [cm^2 per H] of Draine's (2003) R_V = 3.1 grains at ``energy_ev`` (log-log interpolation; the table
    holds 91.2 nm-1 um, so 1.24-13.6 eV)."""
    lam, s = _dust_table()
    x = HC_EV_NM / np.asarray(energy_ev, dtype=float)
    return np.exp(np.interp(np.log(x), np.log(lam), np.log(s)))


@lru_cache(maxsize=None)
def signature(source: str = "heays17") -> str:
    """A key of everything ``field`` reads for ``source``: FIELD_VERSION, the source's spectrum and the dust table, so
    that the population's cached bases of the "cr_uv" shape change with them."""
    import hashlib

    h = hashlib.sha1(f"cr_uv{FIELD_VERSION}:{source}".encode())
    for a in (*SOURCES[source](), *_dust_table()):
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()[:16]


def photons_per_cell(edges_ev, source: str = "heays17") -> np.ndarray:
    """Photons generated per H2 ionisation in each energy cell between ascending ``edges_ev``: differences of the
    source's cumulative, none beyond its wavelength range."""
    lam, cum = SOURCES[source]()
    c = np.interp(HC_EV_NM / np.asarray(edges_ev, dtype=float), lam, cum, left=0.0, right=float(cum[-1]))
    return c[:-1] - c[1:]


def field(zeta_h2_s: float, x_h2: float = X_H2_REF, source: str = "heays17", energy_ev=None,
          e_max: float = fields.LYMAN_EV) -> fields.RadiationField:
    """The induced 4 pi photon flux [photons cm^-2 s^-1 eV^-1] at zeta(H2) = ``zeta_h2_s`` [s^-1] on ``energy_ev``
    (default 1 eV to ``e_max`` in 0.025 eV steps).

    Each node gets the photons generated in its trapezoid cell (half-way to its neighbours, cut at ``e_max``), so the
    trapezoid integral of sigma F on that grid counts every photon once, Lyman alpha included:
    F = zeta(H2) x(H2) dN / (w sigma_abs)."""
    E = np.arange(1.0, e_max + 1e-9, 0.025) if energy_ev is None else np.asarray(energy_ev, dtype=float)
    edges = np.minimum(np.concatenate([[E[0]], 0.5 * (E[1:] + E[:-1]), [E[-1]]]), e_max)
    w = np.diff(np.concatenate([[E[0]], 0.5 * (E[1:] + E[:-1]), [E[-1]]]))
    dn = photons_per_cell(edges, source)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(w > 0.0, float(zeta_h2_s) * float(x_h2) * dn / (w * dust_absorption_per_h(E)), 0.0)
    return fields.RadiationField(E, np.maximum(F, 0.0), f"cr_uv:{source}")


def environment(name, zeta_h, x_e, n_h, n_e, temperature_k, x_h2=X_H2_REF, source="heays17", **kw):
    """A ``baseline.Environment`` lit only by the cosmic-ray-induced field of ``source`` at the primary rate ``zeta_h``
    [s^-1 per H] and x_e = n_e / n_H (RTZ's secondaries): g0 = ``scale(zeta_h, x_e)``, field shape "cr_uv". ``n_h`` is the
    atomic H density (re-hydrogenation), as everywhere in ``baseline.Environment``. The direct passages need
    ``zeta_cr`` (and ``Params.cosmic_rays``) besides, through ``kw``."""
    from pycalima.pahchem.baseline import Environment

    return Environment(name, scale(zeta_h, x_e), n_h, n_e, temperature_k, None, optical_isrf=0.0, field_cr_uv=source,
                       cr_uv_x_h2=x_h2, **kw)
