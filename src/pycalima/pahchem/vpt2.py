"""Second-order vibrational perturbation theory (VPT2) from a quartic force field in normal coordinates.

The input is Gaussian's reduced force constants (cm^-1, dimensionless normal coordinates), as printed by
``freq=anharmonic`` and as published by Mackie+18 for their 20 PAHs:

    V = 1/2 sum_i omega_i q_i^2 + 1/6 sum_ijk phi_ijk q_i q_j q_k + 1/24 sum_ijkl phi_ijkl q_i q_j q_k q_l.

``anharmonic_constants`` gives chi_ij (asymmetric top; Barone 2005, J. Chem. Phys. 122, 014108, eqs. 7-8),
with the terms of Fermi resonances (2 omega_i ~ omega_j, omega_i + omega_j ~ omega_k) removed (deperturbed
VPT2). The Coriolis contribution is omitted: it needs the normal-mode vectors, and for molecules of this size
it is of the order of the rotational constants (< 0.1 cm^-1). Vibrational energies are then

    E(n) = sum_k omega_k (n_k + 1/2) + sum_{k<=l} chi_kl (n_k + 1/2)(n_l + 1/2),

so the fundamental of mode k is nu_k = omega_k + 2 chi_kk + 1/2 sum_{l != k} chi_kl.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ForceField:
    """Harmonic wavenumbers and the cubic and semi-diagonal quartic constants [cm^-1], modes 0..n-1."""
    omega: np.ndarray                    # (n,)
    cubic: np.ndarray                    # (n, n, n), symmetric
    quartic_iijj: np.ndarray             # (n, n), symmetric: phi_iijj (phi_iiii on the diagonal)
    label: str = ""
    resonances: list = field(default_factory=list)


def _rows(path, n):
    out = []
    pat = re.compile(r"^\s*" + r"\s+".join([r"(\d+)"] * n) + r"\s+(-?\d+\.\d+)")
    with open(path) as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                out.append(tuple(int(g) for g in m.groups()[:n]) + (float(m.group(n + 1)),))
    return out


def read_gaussian_qff(prefix):
    """``ForceField`` from the three files ``<prefix>_normal_coordinate_{quadratic,cubic,quartic}_force_constants.txt``
    (Gaussian's reduced values, the FI column; Gaussian numbers the modes 1..n, here 0..n-1)."""
    prefix = str(prefix)
    quad = _rows(f"{prefix}_normal_coordinate_quadratic_force_constants.txt", 2)
    n = max(r[0] for r in quad)
    if len(quad) != n or any(i != j for i, j, _ in quad):
        raise ValueError(f"{prefix}: expected {n} diagonal quadratic constants")
    omega = np.zeros(n)
    for i, _, v in quad:
        omega[i - 1] = v
    cubic = np.zeros((n, n, n))
    for i, j, k, v in _rows(f"{prefix}_normal_coordinate_cubic_force_constants.txt", 3):
        for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            cubic[a - 1, b - 1, c - 1] = v
    q = np.zeros((n, n))
    for i, j, k, l, v in _rows(f"{prefix}_normal_coordinate_quartic_force_constants.txt", 4):
        if i == j and k == l:
            q[i - 1, k - 1] = q[k - 1, i - 1] = v
    return ForceField(omega, cubic, q, Path(prefix).name)


def mackie2018_force_field(name, root=None):
    """``ForceField`` of molecule ``name`` (the file prefix, e.g. ``"123678-hexahydropyrene"``) of the Mackie+18
    supplement (manual dataset ``mackie2018-supplement``)."""
    if root is None:
        from pycalima._datasets import ensure_dataset

        root = ensure_dataset("mackie2018-supplement")
    return read_gaussian_qff(Path(root) / f"{name}_b3lyp-n07d")


def _type1(omega, delta_hi, delta_lo, split_cm):
    """(i, j) with 2 omega_i ~ omega_j: within ``delta_hi`` [cm^-1] above ``split_cm``, ``delta_lo`` below."""
    two = 2.0 * omega[:, None]
    lim = np.where(omega[None, :] > split_cm, delta_hi, delta_lo)
    m = np.abs(two - omega[None, :]) < lim
    np.fill_diagonal(m, False)
    return m


def _type2(omega, delta_hi, delta_lo, split_cm):
    """m[i, j, k]: omega_i + omega_j ~ omega_k, for i != j and k distinct from both."""
    s = omega[:, None, None] + omega[None, :, None]
    lim = np.where(omega[None, None, :] > split_cm, delta_hi, delta_lo)
    m = np.abs(s - omega[None, None, :]) < lim
    n = omega.size
    i, j, k = np.ogrid[:n, :n, :n]
    return m & (i != j) & (k != i) & (k != j)


def anharmonic_constants(ff, delta_hi=200.0, delta_lo=25.0, split_cm=2000.0, min_coupling=0.0):
    """chi [cm^-1] (n x n, symmetric) of deperturbed VPT2; records the removed resonances in ``ff.resonances``.

    A Fermi term is removed when its frequency gap is below ``delta_hi`` (resonant mode above ``split_cm``) or
    ``delta_lo`` (below), Mackie+18's choice (200 and 25 cm^-1 around 2000 cm^-1), and its cubic constant
    exceeds ``min_coupling``.
    """
    w = np.asarray(ff.omega, dtype=float)
    n = w.size
    phi = ff.cubic
    f4 = ff.quartic_iijj
    t1 = _type1(w, delta_hi, delta_lo, split_cm)
    t2 = _type2(w, delta_hi, delta_lo, split_cm)
    phi_iij = phi[np.arange(n), np.arange(n), :]                 # phi_iij [i, j]
    t1 &= np.abs(phi_iij) > min_coupling
    t2 &= np.abs(phi) > min_coupling
    chi = np.zeros((n, n))
    wi, wj = w[:, None], w[None, :]
    # chi_ii: 2/w_j + 1/(2(2w_i + w_j)) - 1/(2(2w_i - w_j)), the last dropped in a type-1 resonance
    with np.errstate(divide="ignore", invalid="ignore"):
        res = -0.5 / (2 * wi - wj)
        g = 2.0 / wj + 0.5 / (2 * wi + wj) + np.where(t1, 0.0, res)
    chi_d = (f4.diagonal() - np.sum(phi_iij ** 2 * g, axis=1)) / 16.0
    # chi_ij
    a = 0.25 * f4 - 0.25 * np.einsum("ik,jk,k->ij", phi_iij, phi_iij, 1.0 / w)
    wk = w[None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        # w_k (w_k^2 - w_i^2 - w_j^2) / Delta_ijk, Delta_ijk = (w_i+w_j+w_k)(w_i-w_j-w_k)(-w_i+w_j-w_k)(-w_i-w_j+w_k),
        # = [1/(w_i+w_j+w_k) - 1/(w_i+w_j-w_k) + 1/(w_i-w_j+w_k) + 1/(-w_i+w_j+w_k)] / 4; the term whose
        # denominator closes (w_i + w_j ~ w_k, w_i + w_k ~ w_j, w_j + w_k ~ w_i) is dropped in a type-2 resonance
        wi3, wj3 = w[:, None, None], w[None, :, None]
        p1 = 1.0 / (wi3 + wj3 + wk)
        p2 = 1.0 / (wi3 + wj3 - wk)
        p3 = 1.0 / (wi3 - wj3 + wk)
        p4 = 1.0 / (-wi3 + wj3 + wk)
        r_ij_k = t2                                              # w_i + w_j ~ w_k
        r_ik_j = np.transpose(t2, (0, 2, 1))                     # w_i + w_k ~ w_j
        r_jk_i = np.transpose(t2, (2, 1, 0))                     # w_j + w_k ~ w_i
        frac = 0.25 * (p1 - np.where(r_ij_k, 0.0, p2) + np.where(r_ik_j, 0.0, p3) + np.where(r_jk_i, 0.0, p4))
    # sign: for w_k >> w_i, w_j, eliminating q_k adiabatically gives -phi_ijk^2 / (2 w_k) (frac -> 1/w_k)
    b = -0.5 * np.einsum("ijk,ijk->ij", phi ** 2, np.nan_to_num(frac, posinf=0.0, neginf=0.0))
    chi = a + b
    np.fill_diagonal(chi, chi_d)
    ff.resonances = ([("1", int(i), int(j)) for i, j in zip(*np.nonzero(t1))]
                     + [("2", int(i), int(j), int(k)) for i, j, k in zip(*np.nonzero(t2)) if i < j])
    return 0.5 * (chi + chi.T)


def fundamentals(omega, chi):
    """nu_k = omega_k + 2 chi_kk + 1/2 sum_{l != k} chi_kl [cm^-1]."""
    return omega + 2 * np.diag(chi) + 0.5 * (chi.sum(axis=1) - np.diag(chi))
