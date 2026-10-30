"""a-C(:H) grain optics from the size-dependent optEC(s)(a) model of Jones (2012).

The complex index m = n - ik of hydrogenated amorphous carbon is tabulated by
Jones (2012, A&A 542, A98, paper III; corrigendum A&A 545, C3) for 14 band gaps
E_g from -0.1 to 2.67 eV (``E_GAPS``) and 7 particle radii from 0.33 to 100 nm
(``RADII_NM``), at 1000 wavelengths from 0.02 um to 1 m (CDS J/A+A/542/A98,
registered dataset ``optical-jones-2012-size``). E_g sets the composition, the
H atomic fraction X_H = E_g / 4.3 eV (paper I), and the density,
rho = 1.3 + 0.4 exp[-(E_g + 0.2)] g/cm^3 (paper II, eq. 26): 1.60 for a-C
(E_g = 0.1 eV) and 1.33 for H-rich a-C:H (E_g = 2.5 eV). The E_g of a column is
that of the bulk material; the tables at small radii carry the size effects
(fewer, smaller aromatic clusters, H-passivated surfaces).

- E_g must be one of ``E_GAPS``. The ReadMe labels columns 7 and 8 as 1.5 and
  1.15 eV; they are 1.25 and 1.5 eV (paper II Table 1; k is monotonic in E_g in
  that order), and are read so here.
- Between tabulated radii, n and ln k are interpolated linearly in ln a; outside
  [0.33, 100] nm the nearest table is used (100 nm is the bulk to within the
  paper's stated accuracy). Jones et al. (2013, App. C) note that the IR bands
  depend on size in a way such interpolation only approximates.
- Below 0.02 um (h nu > 62 eV) the index comes from the Henke atomic scattering
  factors at the same X_H and density, not rescaled to meet the table
  (``splice_ratio``).
- Q_abs, Q_sca and g are Mie values for homogeneous spheres (``efficiencies``).

THEMIS (Jones et al. 2013, A&A 558, A62, App. C) used slightly modified,
unpublished versions of these constants; DustEM's Q_CM20 (``themis_optics``) is
built from those. For a-C (E_g = 0.1 eV) the two agree to ~5% at 10 nm and
differ by factors up to ~1.5 below 1 nm (diagnostics/dust_radiation/
check_jones2012_optics.py).
"""
from __future__ import annotations

import gzip
from functools import lru_cache

import numpy as np

DATASET = "optical-jones-2012-size"
E_GAPS = (-0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.67)
RADII_NM = (0.33, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0)
LAMBDA_MIN_TABLE_UM = 0.02
_FILE_TAG = ("0_33", "0_5", "1", "3", "10", "30", "100")


def hydrogen_fraction(e_gap):
    """H atomic fraction X_H = E_g / 4.3 eV (Jones 2012, paper I)."""
    return np.asarray(e_gap, float) / 4.3


def density(e_gap):
    """Bulk density [g/cm^3]: 1.3 + 0.4 exp[-(E_g + 0.2)] (Jones 2012, paper II)."""
    return 1.3 + 0.4 * np.exp(-(np.asarray(e_gap, float) + 0.2))


@lru_cache(maxsize=1)
def load():
    """(lam_um, n, k): lam (1000,) increasing; n, k (7 radii, 1000, 14 E_g)."""
    from pycalima._datasets import ensure_dataset

    root = ensure_dataset(DATASET)
    lam, n, k = None, [], []
    for tag in _FILE_TAG:
        for name, out in (("n", n), ("k", k)):
            with gzip.open(root / f"{name}{tag}nm.dat.gz", "rt") as fh:
                d = np.loadtxt(fh)
            if lam is None:
                lam = d[:, 0]
            assert d.shape == (1000, 16) and np.array_equal(d[:, 0], lam), f"{name}{tag}nm"
            out.append(d[:, 2:])
    order = np.argsort(lam)
    return lam[order], np.array(n)[:, order], np.array(k)[:, order]


def _column(e_gap):
    j = np.nonzero(np.isclose(E_GAPS, e_gap, atol=1e-9))[0]
    if j.size != 1:
        raise ValueError(f"E_g = {e_gap} eV is not tabulated; choose one of {E_GAPS}")
    return int(j[0])


def _henke_index(e_gap, lam_um):
    x_h = float(hydrogen_fraction(e_gap))
    return np.atleast_1d(_henke_ext().compute_refractive_index({"C": 1.0 - x_h, "H": x_h},
                                                                float(density(e_gap)), lam_um))


@lru_cache(maxsize=1)
def _henke_ext():
    from pycalima.models.tools.henke_extension import HenkeExtension

    return HenkeExtension()


def refractive_index(a_um, lam_um, e_gap):
    """m = n - ik of a-C(:H) with band gap ``e_gap`` [eV], shape (nlam, na)."""
    lam_t, n_t, k_t = load()
    j = _column(e_gap)
    a_nm = np.atleast_1d(np.asarray(a_um, float)) * 1e3
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    if lam_um.max() > lam_t[-1] * (1 + 1e-9):
        raise ValueError(f"wavelength above the tables ({lam_t[-1]} um)")
    la = np.log(np.clip(a_nm, RADII_NM[0], RADII_NM[-1]))
    lr = np.log(RADII_NM)
    i = np.clip(np.searchsorted(lr, la, side="right") - 1, 0, len(RADII_NM) - 2)
    w = (la - lr[i]) / (lr[i + 1] - lr[i])
    m = np.empty((lam_um.size, a_nm.size), complex)
    tab = lam_um >= LAMBDA_MIN_TABLE_UM
    if tab.any():
        ll = np.log(lam_um[tab])
        llt = np.log(lam_t)
        for c in range(a_nm.size):
            nr = [np.interp(ll, llt, n_t[r, :, j]) for r in (i[c], i[c] + 1)]
            kr = [np.interp(ll, llt, np.log(np.maximum(k_t[r, :, j], 1e-300))) for r in (i[c], i[c] + 1)]
            n = (1 - w[c]) * nr[0] + w[c] * nr[1]
            k = np.exp((1 - w[c]) * kr[0] + w[c] * kr[1])
            m[tab, c] = n - 1j * k
    if (~tab).any():
        m[~tab, :] = _henke_index(e_gap, lam_um[~tab])[:, None]
    return m


def efficiencies(a_um, lam_um, e_gap):
    """Mie Q_abs, Q_sca and g of a-C(:H) spheres, each (nlam, na)."""
    import miepython

    a_um = np.atleast_1d(np.asarray(a_um, float))
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    m = refractive_index(a_um, lam_um, e_gap)
    shape = (lam_um.size, a_um.size)
    qabs, qsca, g = np.empty(shape), np.empty(shape), np.empty(shape)
    for c, a in enumerate(a_um):
        qe, qs, _, gg = miepython.efficiencies(m[:, c], 2.0 * a, lam_um)
        qabs[:, c], qsca[:, c], g[:, c] = np.asarray(qe) - np.asarray(qs), qs, gg
    return qabs, qsca, g


def cross_sections(a_um, lam_um, e_gap):
    """C_abs, C_sca and the radiation-pressure C_rp = C_abs + (1 - g) C_sca per
    grain [cm^2], each (nlam, na)."""
    qabs, qsca, g = efficiencies(a_um, lam_um, e_gap)
    area = np.pi * (np.atleast_1d(np.asarray(a_um, float)) * 1e-4) ** 2
    return {"C_abs": qabs * area, "C_sca": qsca * area, "C_rp": (qabs + (1.0 - g) * qsca) * area}


def splice_ratio(a_um, e_gap):
    """Henke-extension over tabulated Q_abs at the 0.02 um splice, per radius."""
    a_um = np.atleast_1d(np.asarray(a_um, float))
    lam = np.array([LAMBDA_MIN_TABLE_UM])
    q_tab = efficiencies(a_um, lam, e_gap)[0][0]
    q_henke = efficiencies(a_um, lam * (1 - 1e-9), e_gap)[0][0]
    return q_henke / q_tab
