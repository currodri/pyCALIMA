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

- The ReadMe labels columns 7 and 8 as 1.5 and 1.15 eV; they are 1.25 and 1.5 eV
  (paper II Table 1; k is monotonic in E_g in that order), and are read so here.
  Between columns n and ln k are linear in E_g.
- Between tabulated radii the steep size dependence of H-poor grains is carried by
  the size-limited band gap, E_eff = max(E_g, E_lim(a)) with E_lim = 0.75 eV (1 nm/a)^1.06
  (``size_limited_gap``), and only the remainder is interpolated in ln a
  (``refractive_index``, ``size_interp="effective_gap"``). Plain ln-a interpolation of
  the 7 radii (``"lna"``) misses the a-C far-IR between 1 and 10 nm by up to x2.4.
  Outside [0.33, 100] nm the nearest radius is used (100 nm is the bulk to within the
  paper's stated accuracy).
- Below 0.02 um (h nu > 62 eV) the index comes from the Henke atomic scattering
  factors at the same X_H and density, not rescaled to meet the table
  (``splice_ratio``).
- Q_abs, Q_sca and g are Mie values for homogeneous spheres (``efficiencies``).
- A particle's hydrogen is the bulk X_H plus the H atoms that passivate its surface
  carbon (paper III, App. A). ``h_per_carbon`` gives H atoms per C atom against radius
  and E_g; ``e_gap_of_h_per_carbon`` inverts it, mapping a tracked H/C to the table
  column of each radius. The surface term matters below ~30 nm (E_g 0.1: H/C = 1.5 at
  0.4 nm, 0.86 at 1.2 nm, against Jones & Ysard 2025's ~1.4 and ~0.8). Paper III's lower
  effective density below 1 nm (a_eff = a^0.8) is not applied.

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
N_AVOGADRO = 6.02214076e23
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


def atom_radius_cm(e_gap):
    """Effective radius of an atom in the solid [cm], (3 A / 4 pi rho N_A)^(1/3) with the
    mean atomic mass A = 12 - 11 X_H (paper III, eqs. A.1, A.7)."""
    x_h = hydrogen_fraction(e_gap)
    return (3.0 * (12.0 - 11.0 * x_h) / (4.0 * np.pi * density(e_gap) * N_AVOGADRO)) ** (1.0 / 3.0)


def surface_fraction(a_cm, e_gap):
    """F_s, the fraction of a particle's atoms in its one-atom-deep surface layer,
    1 - (1 - 2 r_atom / a)^3 (paper III, eq. A.9); 1 below a = 2 r_atom."""
    x = 2.0 * atom_radius_cm(e_gap) / np.asarray(a_cm, float)
    return 1.0 - np.clip(1.0 - x, 0.0, None) ** 3


def surface_hydrogen_fraction(a_cm, e_gap):
    """X_Hs(a), the extra H atoms passivating the surface carbon, per atom of the bulk
    (paper III, eq. A.13): (1 - X_H) F_s m_C / 2, with the mean carbon coordination
    m_C = (3 X_sp2 + 4 X_sp3) / (1 - X_H) of the DG or eRCN composition (paper I).
    Between the networks (no composition) it is linear in E_g between their ends."""
    from pycalima.models.dust_radiation.carbon_heat_capacity import composition

    lo, hi = _NETWORK_GAP_EV
    e = float(e_gap)
    if lo < e < hi:
        w = (e - lo) / (hi - lo)
        return (1 - w) * surface_hydrogen_fraction(a_cm, lo) + w * surface_hydrogen_fraction(a_cm, hi)
    _, x_sp2, x_sp3 = composition(e)
    return surface_fraction(a_cm, e) * 0.5 * (3.0 * x_sp2 + 4.0 * x_sp3)


def h_per_carbon(a_cm, e_gap):
    """H atoms per C atom of an a-C(:H) particle of radius ``a_cm`` with bulk band gap
    ``e_gap``, bulk and surface: (X_H + X_Hs(a)) / (1 - X_H) (paper III, eq. 2)."""
    x_h = float(hydrogen_fraction(e_gap))
    return (x_h + surface_hydrogen_fraction(a_cm, e_gap)) / (1.0 - x_h)


#: the band-gap range of the carbon carriers, a-C (THEMIS) to a-C:H
E_GAP_RANGE = (0.1, 2.5)
#: X_H between the networks (DG up to 0.35, eRCN from 0.38) has no composition
_NETWORK_GAP_EV = (0.35 * 4.3 * (1 - 1e-9), 0.38 * 4.3 * (1 + 1e-9))


def e_gap_of_h_per_carbon(a_cm, h_per_c, e_range=E_GAP_RANGE):
    """The bulk band gap [eV] of a particle of radius ``a_cm`` holding ``h_per_c`` H
    atoms per C atom: ``h_per_carbon`` inverted at fixed radius, clipped to ``e_range``.
    h/C rises with E_g within each network. Between them it is linear in E_g (the
    surface term interpolated, ``surface_hydrogen_fraction``), except below ~1.5 nm, where the eRCN branch starts below the end of the DG branch
    (by 0.5 % at 1.06 nm, 2 % at 0.5 nm: the sp3 fraction jumps between the networks);
    there the DG branch is used to its end and the eRCN branch only above that h/C.
    Returns an array with the broadcast shape of the inputs."""
    a_cm, h_per_c = np.broadcast_arrays(np.asarray(a_cm, float), np.asarray(h_per_c, float))
    lo, hi = _NETWORK_GAP_EV
    e_dg, e_rcn = np.linspace(e_range[0], lo, 120), np.linspace(hi, e_range[1], 120)
    out = np.empty(a_cm.shape)
    for idx in np.ndindex(a_cm.shape):
        h_dg = np.array([h_per_carbon(a_cm[idx], e) for e in e_dg])
        h_rcn = np.array([h_per_carbon(a_cm[idx], e) for e in e_rcn])
        keep = h_rcn > h_dg[-1]
        curve, grid = np.concatenate([h_dg, h_rcn[keep]]), np.concatenate([e_dg, e_rcn[keep]])
        if np.any(np.diff(curve) <= 0.0):
            raise RuntimeError(f"h/C is not increasing in E_g at a = {a_cm[idx]:.3e} cm")
        out[idx] = np.interp(h_per_c[idx], curve, grid)
    return out


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


#: size-limited band gap E_lim(a) = E_LIM_1NM (1 nm / a)^E_LIM_SLOPE [eV]
E_LIM_1NM, E_LIM_SLOPE = 0.75, 1.06


def size_limited_gap(a_um):
    """E_lim(a) [eV]: the smallest band gap a particle of radius ``a_um`` can have, its
    aromatic clusters being limited by its size (paper III, Sect. 2.3, Fig. 2, App. B).
    E_lim = 0.75 eV (1 nm / a)^1.06 is the power law through the gaps that make the
    a-C far-IR match DustEM's Q_CM20 (THEMIS, built from Jones's continuous-in-radius
    model) at 1-5 nm (scatter +-7%). It agrees with the best-fitting bulk gap of the
    tables themselves at 1 nm (0.74) and 3 nm (0.24); paper III's empirical limit,
    1/a - 0.2 eV, gives 0.80 and 0.13 there. Below ~0.5 nm a bulk gap no longer
    describes the tables (diagnostics/dust_radiation/check_jones2012_optics.py)."""
    return E_LIM_1NM / (np.asarray(a_um, float) * 1e3) ** E_LIM_SLOPE


def _column_weights(e_gap):
    """(j, w): E_g lies between columns j and j + 1, a fraction w of the way (w = 0 on j)."""
    e = float(e_gap)
    if not E_GAPS[0] - 1e-9 <= e <= E_GAPS[-1] + 1e-9:
        raise ValueError(f"E_g = {e_gap} eV is outside the tables ({E_GAPS[0]} to {E_GAPS[-1]} eV)")
    j = int(np.clip(np.searchsorted(E_GAPS, e, side="right") - 1, 0, len(E_GAPS) - 2))
    w = (e - E_GAPS[j]) / (E_GAPS[j + 1] - E_GAPS[j])
    return j, (0.0 if abs(w) < 1e-9 else 1.0 if abs(w - 1.0) < 1e-9 else w)


@lru_cache(maxsize=1)
def _log_k():
    return np.log(np.maximum(load()[2], 1e-300))


def _nk_at(r, e_gap):
    """n and ln k of table radius ``r`` at ``e_gap``, linear in E_g between columns."""
    n_t, lk = load()[1], _log_k()
    j, w = _column_weights(e_gap)
    if w == 0.0:
        return n_t[r, :, j], lk[r, :, j]
    if w == 1.0:
        return n_t[r, :, j + 1], lk[r, :, j + 1]
    return (1 - w) * n_t[r, :, j] + w * n_t[r, :, j + 1], (1 - w) * lk[r, :, j] + w * lk[r, :, j + 1]


def _henke_index(e_gap, lam_um):
    x_h = float(hydrogen_fraction(e_gap))
    return np.atleast_1d(_henke_ext().compute_refractive_index({"C": 1.0 - x_h, "H": x_h},
                                                                float(density(e_gap)), lam_um))


@lru_cache(maxsize=1)
def _henke_ext():
    from pycalima.models.tools.henke_extension import HenkeExtension

    return HenkeExtension()


SIZE_INTERP = ("effective_gap", "lna")
#: below this radius [nm] a bulk gap does not describe the tables; "effective_gap" falls back to "lna"
EFFECTIVE_GAP_MIN_NM = 1.0


def refractive_index(a_um, lam_um, e_gap, size_interp="effective_gap"):
    """m = n - ik of a-C(:H) with bulk band gap ``e_gap`` [eV], shape (nlam, na).

    Any E_g from -0.1 to 2.67 eV: n and ln k are linear in E_g between the tabulated
    columns. Between tabulated radii:

    - ``"effective_gap"`` (default): the steep part of the size dependence is the
      size-limited gap, so m(a) = m_bulk(E_eff(a)) + R(a), with E_eff = max(E_g,
      E_lim(a)) (``size_limited_gap``) and the remainder R = m_table - m_bulk(E_eff),
      known at the tabulated radii, interpolated linearly in ln a (n and ln k). It
      reproduces the tables at their radii, equals ``"lna"`` wherever E_g > E_lim
      (e.g. a-C:H at every radius), and, for a-C at 1-10 nm, brings Q_abs to within
      0-18% of DustEM's Q_CM20 in every band, where ``"lna"`` misses the far-IR by up
      to x2.8 at 1.5 nm. Below 1 nm (``EFFECTIVE_GAP_MIN_NM``) a bulk gap no longer
      describes the tables, and the scheme falls back to ``"lna"``.
    - ``"lna"``: the tables themselves, n and ln k linear in ln a.

    Interpolated n and k are only approximately Kramers-Kronig consistent, as paper
    III cautions. Outside [0.33, 100] nm the nearest radius is used.
    """
    if size_interp not in SIZE_INTERP:
        raise ValueError(f"size_interp must be one of {SIZE_INTERP}")
    lam_t = load()[0]
    a_nm = np.atleast_1d(np.asarray(a_um, float)) * 1e3
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    if lam_um.max() > lam_t[-1] * (1 + 1e-9):
        raise ValueError(f"wavelength above the tables ({lam_t[-1]} um)")
    _column_weights(e_gap)
    a_c = np.clip(a_nm, RADII_NM[0], RADII_NM[-1])
    la, lr = np.log(a_c), np.log(RADII_NM)
    i = np.clip(np.searchsorted(lr, la, side="right") - 1, 0, len(RADII_NM) - 2)
    w = (la - lr[i]) / (lr[i + 1] - lr[i])
    bulk = len(RADII_NM) - 1
    e_node = [max(float(e_gap), float(size_limited_gap(r * 1e-3))) for r in RADII_NM]
    m = np.empty((lam_um.size, a_nm.size), complex)
    tab = lam_um >= LAMBDA_MIN_TABLE_UM
    if tab.any():
        ll, llt = np.log(lam_um[tab]), np.log(lam_t)
        for c in range(a_nm.size):
            (n0, k0), (n1, k1) = _nk_at(i[c], e_gap), _nk_at(i[c] + 1, e_gap)
            if size_interp == "lna" or a_c[c] < EFFECTIVE_GAP_MIN_NM:
                n, lk = (1 - w[c]) * n0 + w[c] * n1, (1 - w[c]) * k0 + w[c] * k1
            else:
                nb, kb = _nk_at(bulk, max(float(e_gap), float(size_limited_gap(a_c[c] * 1e-3))))
                nb0, kb0 = _nk_at(bulk, e_node[i[c]])
                nb1, kb1 = _nk_at(bulk, e_node[i[c] + 1])
                n = nb + (1 - w[c]) * (n0 - nb0) + w[c] * (n1 - nb1)
                lk = kb + (1 - w[c]) * (k0 - kb0) + w[c] * (k1 - kb1)
            m[tab, c] = np.interp(ll, llt, n) - 1j * np.exp(np.interp(ll, llt, lk))
    if (~tab).any():
        m[~tab, :] = _henke_index(e_gap, lam_um[~tab])[:, None]
    return m


def efficiencies(a_um, lam_um, e_gap, size_interp="effective_gap"):
    """Mie Q_abs, Q_sca and g of a-C(:H) spheres, each (nlam, na)."""
    import miepython

    a_um = np.atleast_1d(np.asarray(a_um, float))
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    m = refractive_index(a_um, lam_um, e_gap, size_interp)
    shape = (lam_um.size, a_um.size)
    qabs, qsca, g = np.empty(shape), np.empty(shape), np.empty(shape)
    for c, a in enumerate(a_um):
        qe, qs, _, gg = miepython.efficiencies(m[:, c], 2.0 * a, lam_um)
        qabs[:, c], qsca[:, c], g[:, c] = np.asarray(qe) - np.asarray(qs), qs, gg
    return qabs, qsca, g


def cross_sections(a_um, lam_um, e_gap, size_interp="effective_gap"):
    """C_abs, C_sca and the radiation-pressure C_rp = C_abs + (1 - g) C_sca per
    grain [cm^2], each (nlam, na)."""
    qabs, qsca, g = efficiencies(a_um, lam_um, e_gap, size_interp)
    area = np.pi * (np.atleast_1d(np.asarray(a_um, float)) * 1e-4) ** 2
    return {"C_abs": qabs * area, "C_sca": qsca * area, "C_rp": (qabs + (1.0 - g) * qsca) * area}


def splice_ratio(a_um, e_gap):
    """Henke-extension over tabulated Q_abs at the 0.02 um splice, per radius."""
    a_um = np.atleast_1d(np.asarray(a_um, float))
    lam = np.array([LAMBDA_MIN_TABLE_UM])
    q_tab = efficiencies(a_um, lam, e_gap)[0][0]
    q_henke = efficiencies(a_um, lam * (1 - 1e-9), e_gap)[0][0]
    return q_henke / q_tab
