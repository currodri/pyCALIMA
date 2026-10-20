"""Cumulative single-stellar-population (SSP) release tables.

For one solar mass formed at metallicity Z, tabulate against age the
cumulative mass returned to the gas by AGB winds, core-collapse SNe and
SNe Ia: gross ejecta per element, total metals, total mass, condensed
carbon and silicate dust, and the number of SNe.

This is the Python counterpart of ``cmp_yield_release.pro`` +
``cmp_typeia_yield_release.pro``, which the old ``galaxysam`` port never
reproduced. A galaxy model gets the release of a stellar population between
ages ``t0`` and ``t1`` exactly (and mass-conservingly) as the difference of
two cumulative values; see :meth:`SSPTable.released`.

The dust species' element masses are *part of* the element ejecta: the
gas-phase ejecta are ``ejecta - dust x composition``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import imf as imf_mod
from .lifetimes import dying_mass
from .snia import W7_EJECTA, W7_TOTAL, PowerLawDTD
from .stellar_dust import OLIVINE, condense
from .stellar_yields import ELEMENTS, load_yield_set

CHANNELS = ("agb", "snii", "snia")
DUST_SPECIES = ("carbon", "silicate")
_IH, _IHE = ELEMENTS.index("H"), ELEMENTS.index("He")
_Z_EPS = 1e-6  # so that a Z = 0 grid point has a finite log coordinate


def default_ages() -> np.ndarray:
    """Age grid [yr]: 0, then 1e5 yr to 2e10 yr at ~0.017 dex."""
    return np.concatenate([[0.0], np.logspace(5.0, np.log10(2e10), 301)])


@dataclass
class SSPTable:
    """Cumulative release per Msun formed; arrays are (nZ, nAge[, ...])."""

    Z: np.ndarray
    ages: np.ndarray
    ejecta: dict        # channel -> (nZ, nAge, nEl) gross element ejecta
    total: dict         # channel -> (nZ, nAge) total ejected mass
    dust: dict          # channel -> (nZ, nAge, nSpecies) condensed dust
    n_sn: dict          # 'snii' / 'snia' -> (nZ, nAge) cumulative SN number
    silicate_composition: dict
    meta: dict = field(default_factory=dict)
    elements: tuple = tuple(ELEMENTS)

    # -- summed views ----------------------------------------------------
    def summed(self, what: str) -> np.ndarray:
        return sum(getattr(self, what)[c] for c in CHANNELS)

    @property
    def metals(self) -> dict:
        return {c: self.total[c] - self.ejecta[c][..., _IH] - self.ejecta[c][..., _IHE]
                for c in CHANNELS}

    # -- interpolation ---------------------------------------------------
    def _interp(self, arr: np.ndarray, Z: float, t: float) -> np.ndarray:
        """Bilinear in (log Z, log age), clamped to the grid."""
        xz = np.log10(self.Z + _Z_EPS)
        x = np.clip(np.log10(Z + _Z_EPS), xz[0], xz[-1])
        iz = int(np.clip(np.searchsorted(xz, x) - 1, 0, len(xz) - 2))
        wz = (x - xz[iz]) / (xz[iz + 1] - xz[iz])
        at_z = (1.0 - wz) * arr[iz] + wz * arr[iz + 1]
        la = np.log10(np.maximum(self.ages, 1.0))
        lt = np.log10(max(t, 1.0))
        if t <= 0.0:
            return at_z[0]
        flat = at_z.reshape(len(self.ages), -1)
        out = np.array([np.interp(lt, la, flat[:, k]) for k in range(flat.shape[1])])
        return out.reshape(at_z.shape[1:])

    def released(self, Z: float, t0: float, t1: float) -> dict:
        """Mass released per Msun formed between ages ``t0`` and ``t1`` [yr].

        Returns a dict with ``ejecta`` (nEl,), ``total``, ``metals``,
        ``dust`` (nSpecies,), ``n_snii`` and ``n_snia``, summed over channels.
        """
        def diff(arr):
            return self._interp(arr, Z, t1) - self._interp(arr, Z, t0)
        out = {
            "ejecta": diff(self.summed("ejecta")),
            "total": float(diff(self.summed("total"))),
            "dust": diff(self.summed("dust")),
            "n_snii": float(diff(self.n_sn["snii"])),
            "n_snia": float(diff(self.n_sn["snia"])),
        }
        out["metals"] = out["total"] - out["ejecta"][_IH] - out["ejecta"][_IHE]
        return out

    # -- persistence -----------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"Z": self.Z, "ages": self.ages}
        for c in CHANNELS:
            arrays[f"ejecta_{c}"] = self.ejecta[c]
            arrays[f"total_{c}"] = self.total[c]
            arrays[f"dust_{c}"] = self.dust[c]
        for c, v in self.n_sn.items():
            arrays[f"n_{c}"] = v
        header = {"elements": list(self.elements), "dust_species": list(DUST_SPECIES),
                  "silicate_composition": self.silicate_composition, "meta": self.meta}
        np.savez_compressed(path, header=json.dumps(header), **arrays)
        return path

    @classmethod
    def load(cls, path) -> "SSPTable":
        with np.load(path) as f:
            header = json.loads(str(f["header"]))
            if header["elements"] != ELEMENTS:
                raise ValueError(f"{path}: element list {header['elements']} != {ELEMENTS}")
            return cls(
                Z=f["Z"], ages=f["ages"],
                ejecta={c: f[f"ejecta_{c}"] for c in CHANNELS},
                total={c: f[f"total_{c}"] for c in CHANNELS},
                dust={c: f[f"dust_{c}"] for c in CHANNELS},
                n_sn={c: f[f"n_{c}"] for c in ("snii", "snia")},
                silicate_composition=header["silicate_composition"],
                meta=header["meta"],
            )


def _cumulative_from_top(m: np.ndarray, integrand: np.ndarray) -> np.ndarray:
    """C(m_k) = ∫_{m_k}^{m_up} integrand dm, along axis 0."""
    full = cumulative_trapezoid(integrand, m, axis=0, initial=0.0)
    return full[-1] - full


def build_ssp_table(yield_set: str = "lc18_karakas", imf: str = "chabrier",
                    dust_preset: str = "dwek98", silicate_composition: dict | None = None,
                    snia: PowerLawDTD | None = None, ages: np.ndarray | None = None,
                    m_low: float = 0.1, m_up: float = 100.0, n_mass: int = 4000,
                    yield_kwargs: dict | None = None) -> SSPTable:
    """Integrate per-star ejecta over the IMF and stellar lifetimes."""
    ys = load_yield_set(yield_set, **(yield_kwargs or {}))
    phi_fn = imf_mod.create_imf(imf, m_low=m_low, m_up=m_up)
    snia = snia or PowerLawDTD()
    ages = default_ages() if ages is None else np.asarray(ages, dtype=float)
    comp = silicate_composition or OLIVINE

    m = np.logspace(np.log10(m_low), np.log10(m_up), n_mass)
    phi = phi_fn(m)
    is_agb = m < ys.mass_separatrix
    nZ, nt, nel = len(ys.Z), len(ages), len(ELEMENTS)
    ejecta = {c: np.zeros((nZ, nt, nel)) for c in CHANNELS}
    total = {c: np.zeros((nZ, nt)) for c in CHANNELS}
    dust = {c: np.zeros((nZ, nt, len(DUST_SPECIES))) for c in CHANNELS}
    n_sn = {c: np.zeros((nZ, nt)) for c in ("snii", "snia")}

    n_ia = snia.cumulative(ages)
    dust_ia = np.array(condense(W7_EJECTA, "snia", dust_preset, comp))

    for iz, Z in enumerate(ys.Z):
        ej, tot = ys.per_star(m, iz)
        m_die = np.clip(dying_mass(ages, Z), m_low, m_up)

        for chan, sel in (("agb", is_agb), ("snii", ~is_agb)):
            w = np.where(sel, phi, 0.0)
            carb, sil = condense(ej, chan, dust_preset, comp)
            per_star = np.column_stack([ej, tot, carb, sil, np.ones_like(m)])
            cum = _cumulative_from_top(m, per_star * w[:, None])
            at_age = np.column_stack([np.interp(m_die, m, cum[:, k])
                                      for k in range(cum.shape[1])])
            ejecta[chan][iz] = at_age[:, :nel]
            total[chan][iz] = at_age[:, nel]
            dust[chan][iz] = at_age[:, nel + 1:nel + 3]
            if chan == "snii":
                n_sn["snii"][iz] = at_age[:, nel + 3]

        ejecta["snia"][iz] = n_ia[:, None] * W7_EJECTA
        total["snia"][iz] = n_ia * W7_TOTAL
        dust["snia"][iz] = n_ia[:, None] * dust_ia
        n_sn["snia"][iz] = n_ia

    meta = {
        "yield_set": ys.name, "yield_notes": ys.notes, "imf": imf,
        "m_low": m_low, "m_up": m_up, "mass_separatrix": ys.mass_separatrix,
        "lifetimes": "raiteri1996", "dust_preset": dust_preset,
        "snia": {"dtd": "power_law", "n_per_msun": snia.n_per_msun, "slope": snia.slope,
                 "t_min_yr": snia.t_min, "t_norm_yr": snia.t_norm, "yields": "W7"},
        "sources": sorted({t.source for t in ys.agb + ys.snii}),
    }
    try:
        from pycalima._provenance import get_provenance
        meta["provenance"] = get_provenance()
    except Exception:  # provenance is informative only
        pass
    return SSPTable(Z=np.asarray(ys.Z, dtype=float), ages=ages, ejecta=ejecta, total=total,
                    dust=dust, n_sn=n_sn, silicate_composition=dict(comp), meta=meta)


def default_table_path(yield_set="lc18_karakas", imf="chabrier", dust_preset="dwek98") -> Path:
    """``model_data/ssp_yields/ssp_<yield_set>_<imf>_<dust_preset>.npz``."""
    from pycalima._paths import get_model_data_dir
    return get_model_data_dir() / "ssp_yields" / f"ssp_{yield_set}_{imf}_{dust_preset}.npz"
