"""Absorption efficiencies and Planck means of the collapse grain materials.

For each material: the complex refractive index from its optical constants
(``materials.Material.optics``), absorption efficiencies Q_abs(a, lambda)
(``miepython``, or the small-particle expansion where ``|m| x`` < 0.1; see
``qabs_grid``), and the Planck mean

    <Q>(a, T) = int Q_abs(a, lambda) B_lambda(T) dlambda / int B_lambda(T) dlambda,

on a grid of radii and temperatures. A grain of radius a and temperature T_d
emits 4 pi a^2 sigma T_d^4 <Q>(a, T_d), and absorbs as much from blackbody
radiation at T_d.

Optical constants are interpolated in log-log. Beyond the longest tabulated
wavelength, where the far infrared sets <Q> below ~30 K: for dielectrics (k < n
there) n stays constant and k continues as a power law fitted over the last
half decade, clamped between lambda^0 and lambda^-2 (an opacity kappa ∝
lambda^-beta with 1 <= beta <= 3); for metals both continue as fitted power
laws. Below the shortest wavelength they are
held constant, which matters only above ~3000 K. The existing
``models.tools.mie_theory`` loader extrapolates linearly in lambda instead,
which can turn k negative, so it is not used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import miepython
import numpy as np

from pycalima import _paths
from pycalima.collapse.materials import get_material

H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10
KB_CGS = 1.380649e-16

LAMBDA_UM = np.geomspace(0.05, 2.0e4, 400)        # 50 nm - 2 cm
RADII_CM = np.geomspace(1e-7, 1e-2, 61)           # 1 nm - 100 um
T_GRID = np.geomspace(2.0, 3000.0, 121)           # K
EXTRAP_DEX = 0.5


def _read_columns(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
    arr = np.array([r for r in rows if len(r) == 3])
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def load_nk(material):
    """Tabulated (lambda [um], n, k) of a material, from its ``optics`` source."""
    src = get_material(material).optics
    kind, _, name = src.partition(":")
    if kind == "bundled":
        return _read_columns(_paths.get_optical_props_path("stx3141_supp", name))
    if kind == "dataset":            # "dataset:<registry name>/<file in it>"
        from pycalima._datasets import ensure_dataset
        dataset, _, fname = name.partition("/")
        return _read_columns(ensure_dataset(dataset) / fname)
    raise ValueError(f"unknown optics source {src!r} for {material}")


def refractive_index(lam_um, lam_tab, n_tab, k_tab):
    """m = n - i k at ``lam_um``: log-log interpolation, power-law far-IR extrapolation."""
    lam_um = np.asarray(lam_um, dtype=float)
    x, lx = np.log(lam_um), np.log(lam_tab)
    ln, lk = np.log(n_tab), np.log(np.maximum(k_tab, 1e-12))
    n = np.exp(np.interp(x, lx, ln))
    k = np.exp(np.interp(x, lx, lk))
    long = x > lx[-1]
    if long.any():
        tail = lx >= lx[-1] - EXTRAP_DEX * np.log(10.0)
        sn = np.polyfit(lx[tail], ln[tail], 1)[0] if tail.sum() > 1 else 0.0
        sk = np.polyfit(lx[tail], lk[tail], 1)[0] if tail.sum() > 1 else -1.0
        if k_tab[-1] < n_tab[-1]:
            # dielectric: n tends to its static value; k -> lambda^0 ... lambda^-2,
            # i.e. a Rayleigh-regime opacity kappa ∝ lambda^-beta with 1 <= beta <= 3
            sn, sk = 0.0, min(max(sk, -2.0), 0.0)
        dx = x[long] - lx[-1]
        n[long] = np.exp(ln[-1] + sn * dx)
        k[long] = np.exp(lk[-1] + sk * dx)
    return n - 1j * k


SMALL_MX = 0.1    # below |m| x the small-particle expansion replaces Mie


def qabs_small(m, x):
    """Small-particle Q_abs to O(x^4) (Bohren & Huffman 1983, Eq. 5.8): the
    electric dipole plus the next order, which carries the magnetic-dipole
    absorption that dominates for metals (``|m|`` >> 1). m = n - i k."""
    m2 = m * m
    r = (m2 - 1.0) / (m2 + 2.0)
    return 4.0 * x * np.imag(-r * (1.0 + x ** 2 / 15.0 * r * (m2 ** 2 + 27.0 * m2 + 38.0) / (2.0 * m2 + 3.0)))


def qabs_grid(material, radii_cm=RADII_CM, lam_um=LAMBDA_UM):
    """Q_abs[a, lambda] for a material: Mie, except where ``|m| x`` < SMALL_MX.

    There Mie's Q_ext - Q_sca cancels catastrophically when absorption is weak
    (crystalline Si at 1-5 um: Q_abs ~ 1e-13 against Q_sca ~ 1e-7), while the
    small-particle expansion is accurate to < 1e-4.
    """
    m = refractive_index(lam_um, *load_nk(material))
    Q = np.empty((radii_cm.size, lam_um.size))
    for i, a in enumerate(radii_cm):
        x = 2.0 * np.pi * a * 1e4 / lam_um
        small = np.abs(m) * x < SMALL_MX
        Q[i, small] = qabs_small(m[small], x[small])
        if (~small).any():
            qext, qsca, _, _ = miepython.efficiencies(m[~small], 2.0 * a * 1e4, lam_um[~small])
            Q[i, ~small] = qext - qsca
    return np.maximum(Q, 0.0)


def planck_mean(Q, lam_um, T):
    """<Q>[..., T] = int Q B_lambda dlambda / int B_lambda dlambda on the given grid."""
    lam = lam_um[None, :] * 1e-4
    x = np.minimum(H_CGS * C_CGS / (lam * KB_CGS * np.asarray(T, dtype=float)[:, None]), 700.0)
    B = lam ** -5 / np.expm1(x)
    w = B / np.trapezoid(B, lam, axis=1)[:, None]
    return np.trapezoid(Q[..., None, :] * w, lam, axis=-1)


@dataclass(frozen=True)
class PlanckTable:
    """<Q>(a, T) of one material, interpolated linearly in (ln a, ln T) of ln(<Q>/a)."""

    material: str
    radii: np.ndarray
    T: np.ndarray
    lnQ_over_a: np.ndarray       # [a, T]

    @classmethod
    def compute(cls, material, radii_cm=RADII_CM, T=T_GRID, lam_um=LAMBDA_UM):
        Q = planck_mean(qabs_grid(material, radii_cm, lam_um), lam_um, T)
        return cls(material, radii_cm, T, np.log(np.maximum(Q, 1e-300) / radii_cm[:, None]))

    def save(self, path):
        np.savez(path, material=self.material, radii=self.radii, T=self.T, lnQ_over_a=self.lnQ_over_a)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(str(d["material"]), d["radii"], d["T"], d["lnQ_over_a"])

    def __call__(self, a, T):
        """<Q> at radii ``a`` [cm] and temperatures ``T`` [K] (broadcast together)."""
        a, T = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(T, dtype=float))
        la, lt = np.log(self.radii), np.log(self.T)
        fa = np.clip((np.log(a) - la[0]) / (la[1] - la[0]), 0.0, la.size - 1.000001)
        ft = np.clip((np.log(T) - lt[0]) / (lt[1] - lt[0]), 0.0, lt.size - 1.000001)
        i, j = fa.astype(int), ft.astype(int)
        wa, wt = fa - i, ft - j
        g = self.lnQ_over_a
        v = ((1 - wa) * (1 - wt) * g[i, j] + wa * (1 - wt) * g[i + 1, j]
             + (1 - wa) * wt * g[i, j + 1] + wa * wt * g[i + 1, j + 1])
        return a * np.exp(v)


def planck_table_path(material):
    return _paths.get_model_data_dir("collapse") / f"planck_qabs_{material}.npz"


def planck_table(material):
    """The tabulated <Q>(a, T) of a material, computing and caching it if absent."""
    path = planck_table_path(material)
    if path.exists():
        return PlanckTable.load(path)
    tab = PlanckTable.compute(material)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab.save(path)
    return tab
