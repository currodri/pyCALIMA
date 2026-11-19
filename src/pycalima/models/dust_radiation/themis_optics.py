"""THEMIS a-C(:H) grain optics: the CM20 carbon grains of Jones et al. (2013).

Q_abs, Q_sca and g come from the DustEM v4.3 tables ``Q_CM20.DAT`` and
``G_CM20.DAT`` (98 radii from 0.3 nm to 5 um, 800 wavelengths from 0.04 um to
10 cm), interpolated linearly in (ln a, ln lambda), on ln Q for the
efficiencies and on g itself. These grains are a-C (E_g = 0.1 eV) up to 20 nm
and, above that, an a-C:H core (E_g = 2.5 eV) inside a 20 nm a-C mantle.

Below 0.04 um (h nu > 31 eV), where the tables stop, the grains are computed with
Mie theory using n, k from the Henke atomic scattering factors. Each layer has
the THEMIS composition: X_H = E_g / 4.3 eV (Jones 2012) and the CM20 bulk
density of 1.6 g/cm^3 (DustEM ``GRAIN_J13.DAT``). The core and mantle are
combined into one sphere by volume-weighting (m - 1); this is exact only when
abs(m - 1) << 1.

The Henke extension is not rescaled to meet the tables at the splice. At
0.04 um it gives Q_abs 0.88-1.58 times the tabulated value, most for 3-30 nm
grains (see ``splice_ratio``). Against lab carbon data (Draine 2003 graphite),
Henke's k is about 15% low from 40 to 200 eV and agrees at 500 eV.

The heat capacity per grain (``heat_capacity``) comes from DustEM's
``C_CM20.DAT`` on the same 98 radii, from 0.1 to 5000 K.

The data are the registered dataset ``dustem-themis-cm20`` (see
``data/registry.toml``); set ``$CALIMA_DUSTEM_DIR`` to an extracted DustEM.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATASET = "dustem-themis-cm20"
LAMBDA_MIN_TABLE_UM = 0.04       # shortest wavelength in the DustEM tables
MANTLE_UM = 0.020                # aromatised a-C mantle of the large grains
EG_MANTLE_EV = 0.1               # a-C, also the whole grain below 20 nm
EG_CORE_EV = 2.5                 # aliphatic-rich a-C:H core
RHO_G_CM3 = 1.6                  # CM20 bulk density (GRAIN_J13.DAT)


@dataclass(frozen=True)
class CM20Table:
    a_um: np.ndarray             # (na,)
    lam_um: np.ndarray           # (nlam,), increasing
    qabs: np.ndarray             # (nlam, na)
    qsca: np.ndarray
    g: np.ndarray


def _numbers(path):
    lines = Path(path).read_text().splitlines()
    return [line for line in lines if line.strip() and not line.startswith("#")]


def _read_size_block(path, n_blocks):
    rows = _numbers(path)
    n_size = int(rows[0].split()[0])
    a = np.array(rows[1].split(), float)
    assert a.size == n_size, f"{path}: {a.size} sizes, header says {n_size}"
    data = np.array([r.split() for r in rows[2:]], float)
    return a, np.split(data, n_blocks)


@lru_cache(maxsize=1)
def load_cm20() -> CM20Table:
    """The DustEM CM20 tables, as arrays on their own (a, lambda) grid."""
    from pycalima._datasets import ensure_dataset

    root = ensure_dataset(DATASET)
    lam = np.array([r.split()[0] for r in _numbers(root / "oprop" / "LAMBDA.DAT")[1:]], float)
    a, (qabs, qsca) = _read_size_block(root / "oprop" / "Q_CM20.DAT", 2)
    a_g, (g,) = _read_size_block(root / "oprop" / "G_CM20.DAT", 1)
    assert np.allclose(a, a_g) and qabs.shape == (lam.size, a.size) == g.shape
    order = np.argsort(lam)
    return CM20Table(a, lam[order], qabs[order], qsca[order], g[order])


def _bilinear(table, a_um, lam_um, values, log):
    """Interpolate ``values`` (nlam, na) linearly in (ln lam, ln a)."""
    from scipy.interpolate import RegularGridInterpolator

    v = np.log(np.maximum(values, 1e-300)) if log else values
    f = RegularGridInterpolator((np.log(table.lam_um), np.log(table.a_um)), v)
    L, A = np.meshgrid(np.log(lam_um), np.log(a_um), indexing="ij")
    out = f(np.stack([L, A], axis=-1))
    return np.exp(out) if log else out


@lru_cache(maxsize=1)
def _henke():
    from pycalima.models.tools.henke_extension import HenkeExtension

    return HenkeExtension()


def _layer_index(e_gap_ev, lam_um):
    """Henke m = n - i k of a-C(:H) with band gap ``e_gap_ev``."""
    x_h = e_gap_ev / 4.3                       # H atomic fraction (Jones 2012)
    comp = {"C": 1.0 - x_h, "H": x_h}
    return np.atleast_1d(_henke().compute_refractive_index(comp, RHO_G_CM3, lam_um))


def henke_index(a_um, lam_um):
    """Volume-weighted Henke index of a CM20 grain, shape (nlam, na)."""
    a_um = np.atleast_1d(np.asarray(a_um, float))
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    m_mantle = _layer_index(EG_MANTLE_EV, lam_um)[:, None]
    m_core = _layer_index(EG_CORE_EV, lam_um)[:, None]
    f_core = np.clip((a_um - MANTLE_UM) / a_um, 0.0, 1.0)[None, :] ** 3
    return 1.0 + f_core * (m_core - 1.0) + (1.0 - f_core) * (m_mantle - 1.0)


def _henke_mie(a_um, lam_um):
    import miepython

    m = henke_index(a_um, lam_um)
    shape = (lam_um.size, a_um.size)
    qabs, qsca, g = np.empty(shape), np.empty(shape), np.empty(shape)
    for i, lam in enumerate(lam_um):
        for j, a in enumerate(a_um):
            qext, qs, _, gg = miepython.efficiencies(m[i, j], 2.0 * a, lam)
            qabs[i, j], qsca[i, j], g[i, j] = qext - qs, qs, gg
    return qabs, qsca, g


def efficiencies(a_um, lam_um):
    """Q_abs, Q_sca and g of CM20 grains, each of shape (nlam, na).

    Radii must lie within the tables (0.3 nm - 5 um); wavelengths below
    ``LAMBDA_MIN_TABLE_UM`` use the Henke extension.
    """
    t = load_cm20()
    a_um = np.atleast_1d(np.asarray(a_um, float))
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    if a_um.min() < t.a_um[0] * (1 - 1e-9) or a_um.max() > t.a_um[-1] * (1 + 1e-9):
        raise ValueError(f"radius outside the CM20 tables [{t.a_um[0]}, {t.a_um[-1]}] um")
    if lam_um.max() > t.lam_um[-1] * (1 + 1e-9):
        raise ValueError(f"wavelength above the CM20 tables ({t.lam_um[-1]} um)")
    a_c = np.clip(a_um, t.a_um[0], t.a_um[-1])
    out = [np.empty((lam_um.size, a_um.size)) for _ in range(3)]
    tab = lam_um >= LAMBDA_MIN_TABLE_UM
    if tab.any():
        lam_t = np.minimum(lam_um[tab], t.lam_um[-1])
        out[0][tab] = _bilinear(t, a_c, lam_t, t.qabs, log=True)
        out[1][tab] = _bilinear(t, a_c, lam_t, t.qsca, log=True)
        out[2][tab] = _bilinear(t, a_c, lam_t, t.g, log=False)
    if (~tab).any():
        for k, v in enumerate(_henke_mie(a_um, lam_um[~tab])):
            out[k][~tab] = v
    return tuple(out)


def cross_sections(a_um, lam_um):
    """C_abs, C_sca and the radiation-pressure C_rp = C_abs + (1 - g) C_sca
    per grain [cm^2], each of shape (nlam, na)."""
    qabs, qsca, g = efficiencies(a_um, lam_um)
    area = np.pi * (np.atleast_1d(np.asarray(a_um, float)) * 1e-4) ** 2
    return {"C_abs": qabs * area, "C_sca": qsca * area, "C_rp": (qabs + (1.0 - g) * qsca) * area}


def splice_ratio(a_um):
    """Henke-extension over tabulated Q_abs at the 0.04 um splice, per radius."""
    t = load_cm20()
    a_um = np.atleast_1d(np.asarray(a_um, float))
    q_tab = _bilinear(t, a_um, np.array([LAMBDA_MIN_TABLE_UM]), t.qabs, log=True)[0]
    q_henke = _henke_mie(a_um, np.array([LAMBDA_MIN_TABLE_UM]))[0][0]
    return q_henke / q_tab


# ---------------------------------------------------------------------------
# Heat capacity (DustEM hcap/C_CM20.DAT)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_heat_capacity():
    """(a_um, log10 T [K], log10 C_V [erg K^-1 cm^-3]) of DustEM's C_CM20.DAT:
    98 radii (0.3 nm - 5 um) by 30 temperatures (0.1 - 5000 K)."""
    from pycalima._datasets import ensure_dataset

    rows = [r.split() for r in _numbers(ensure_dataset(DATASET) / "hcap" / "C_CM20.DAT")]
    n_a, a = int(rows[0][0]), np.array(rows[1], float)
    n_t = int(rows[2][0])
    d = np.array(rows[3:3 + n_t], float)
    assert a.size == n_a and d.shape == (n_t, n_a + 1)
    return a, d[:, 0], d[:, 1:]


def heat_capacity(a_um, T):
    """Heat capacity per grain [erg/K] of CM20 grains of radius ``a_um`` at ``T`` [K],
    shape (nT, na): DustEM's per-volume C_V (log interpolated linearly in log T and
    ln a) times the grain volume, as DustEM uses it.

    THEMIS sums per-atom contributions (Jones et al. 2013, App. B), yet at 5000 K
    the a-C entries (a < 20 nm) exceed the Dulong-Petit limit, 3 k per atom at the
    CM20 density of 1.6 g/cm^3, by 35% (0.3 nm) to 49% (3-18 nm), and the core/mantle
    grains exceed it for their a-C:H core's atoms by ~50%. At 0.3 nm the per-volume
    value is graphite's (C_Gra.DAT, 3 k per atom at 2.24 g/cm^3) to 0.3%. Kept as
    DustEM has it (diagnostics/dust_radiation/check_cm20_heat_capacity.py).
    """
    from scipy.interpolate import RegularGridInterpolator

    a_t, lT, lC = load_heat_capacity()
    a_um = np.atleast_1d(np.asarray(a_um, float))
    T = np.atleast_1d(np.asarray(T, float))
    if a_um.min() < a_t[0] * (1 - 1e-9) or a_um.max() > a_t[-1] * (1 + 1e-9):
        raise ValueError(f"radius outside the CM20 tables [{a_t[0]}, {a_t[-1]}] um")
    if T.min() < 10 ** lT[0] * (1 - 1e-9) or T.max() > 10 ** lT[-1] * (1 + 1e-9):
        raise ValueError(f"temperature outside the CM20 heat-capacity table [{10 ** lT[0]:.3g}, {10 ** lT[-1]:.4g}] K")
    f = RegularGridInterpolator((lT, np.log(a_t)), lC)
    pts = np.stack(np.meshgrid(np.clip(np.log10(T), lT[0], lT[-1]),
                               np.clip(np.log(a_um), np.log(a_t[0]), np.log(a_t[-1])), indexing="ij"), -1)
    volume = 4.0 / 3.0 * np.pi * (a_um * 1e-4) ** 3
    return 10.0 ** f(pts) * volume[None, :]


# ---------------------------------------------------------------------------
# CM20 recomputed: the core/mantle grains by coated-sphere Mie
# ---------------------------------------------------------------------------

def mie_efficiencies(a_um, lam_um):
    """Q_abs, Q_sca and g of CM20 grains with the large grains recomputed, (nlam, na).

    Above ``MANTLE_UM`` the grains are an a-C:H core (E_g = 2.5 eV) in a 20 nm
    a-C mantle (E_g = 0.1 eV), here by the stable coated-sphere Mie of
    ``tools.coated_mie`` with the n, k of Jones (2012) at 100 nm (the bulk, which
    THEMIS uses for these layers; ``jones2012_optics``). Below 0.02 um the
    volume-weighted Henke index of ``henke_index`` and homogeneous Mie are used,
    as for the tables. At and below 20 nm the grains are the DustEM tables
    (``efficiencies``).

    For 30-300 nm this reproduces Q_CM20 to 0.5% (1.5% at 3.4 um). Above ~0.9 um
    Q_CM20 is broken in the far UV (Q_sca up to ~700) and its Q_abs is fixed at
    0.840 over 0.04-2.95 um for a >= 0.45 um; this function is not.
    """
    from pycalima.models.dust_radiation import jones2012_optics
    from pycalima.models.tools import coated_mie

    a_um = np.atleast_1d(np.asarray(a_um, float))
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    out = [np.empty((lam_um.size, a_um.size)) for _ in range(3)]
    small = a_um <= MANTLE_UM * (1 + 1e-12)
    if small.any():
        for k, v in enumerate(efficiencies(a_um[small], lam_um)):
            out[k][:, small] = v
    big = np.nonzero(~small)[0]
    if big.size:
        tab = lam_um >= jones2012_optics.LAMBDA_MIN_TABLE_UM
        lam_t = lam_um[tab]
        if tab.any():
            m_core = jones2012_optics.refractive_index(0.1, lam_t, EG_CORE_EV)[:, 0]
            m_mantle = jones2012_optics.refractive_index(0.1, lam_t, EG_MANTLE_EV)[:, 0]
            for c in big:
                v = coated_mie.efficiencies(a_um[c] - MANTLE_UM, a_um[c], lam_t, m_core, m_mantle)
                for k in range(3):
                    out[k][tab, c] = v[k]
        if (~tab).any():
            for k, v in enumerate(_henke_mie(a_um[big], lam_um[~tab])):
                out[k][np.ix_(~tab, big)] = v
    return tuple(out)
