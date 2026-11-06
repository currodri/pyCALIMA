"""Mackie+18 (J. Chem. Phys. 149, 134302): anharmonic IR cascade spectra from VPT2, on pahchem's cascade.

Their method, with one simplification:

- **Constants.** Deperturbed VPT2 chi of their B3LYP/N07D quartic force fields (``vpt2``).
- **Polyads** of the CH-stretch region: the bright CH fundamentals and the two-quantum states (overtones
  2_i, combinations 1_i 1_j) in Fermi resonance with them (``vpt2.anharmonic_constants``' resonance list),
  coupled by phi_iij / 4 (type 1) and phi_ijk / (2 sqrt 2) (type 2). Darling-Dennison couplings are omitted.
- **Temperature.** Their Wang-Landau sampling of the states at internal energy E is replaced by its mean-field
  limit at the microcanonical T(E): each mode thermally occupied (Bose mean nbar, variance nbar(nbar + 1)).
  A fundamental's transition shifts by sum_{l != k} chi_kl nbar_l + 4 chi_kk nbar_k (emission weights n_k),
  with the variance of that sum at fixed total energy (the canonical sum chi_kl^2 nbar_l (nbar_l + 1) less its
  part correlated with the energy fluctuation); a combination or overtone by the sum of its modes' shifts
  (their Sect. II D). The polyad is rediagonalised with the shifted diagonal at each T; each eigenstate takes
  the squared bright coefficients as intensity shares (their convention) and a Gaussian of the
  coefficient-weighted variance. Modes outside the polyads are single Gaussian lines.
- **Cascade.** One UV photon of ``e_uv_ev``, pahchem's harmonic cascade (``energy_me.CascadeModel``, rates
  A rho(E - h nu) / rho(E)) on the VPT2 fundamentals; each photon of bin j takes the lines at T(E_j).
- **Intensities.** The supplement has none: ``intensities`` maps PAHdb v4.00's harmonic intensities of the
  same molecule (B3LYP/6-31G*) onto the modes by frequency rank within the CH stretches and the rest.

``CascadeSpectrum.spectrum(nu)`` is the emitted power per cm^-1 per absorbed photon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycalima.pahchem import dos, emission, energy_me, vpt2
from pycalima.pahchem.molecule import _EINSTEIN_A_PER_KM_MOL

CH_CM = 2500.0
SQRT8 = np.sqrt(8.0)


@dataclass
class Polyad:
    states: list                 # ("1", k) | ("2", i) | ("11", i, j)
    e0: np.ndarray               # deperturbed VPT2 transition energies from the ground state [cm^-1]
    coupling: np.ndarray         # symmetric, zero diagonal [cm^-1]
    bright: np.ndarray           # mode index of each bright row (the fundamentals), -1 elsewhere


def build_polyad(ff, chi, nu, lo_cm=CH_CM):
    """The polyad of the fundamentals above ``lo_cm`` and the two-quantum states resonant with them."""
    bright = [k for k in range(nu.size) if nu[k] > lo_cm]
    states = [("1", k) for k in bright]
    index = {s: n for n, s in enumerate(states)}
    pairs = []
    for r in ff.resonances:
        if r[0] == "1" and r[2] in bright:              # 2 omega_i ~ omega_j
            s, w, k = ("2", r[1]), ff.cubic[r[1], r[1], r[2]] / 4.0, r[2]
        elif r[0] == "2" and r[3] in bright:            # omega_i + omega_j ~ omega_k
            i, j = sorted(r[1:3])
            s, w, k = ("11", i, j), ff.cubic[r[1], r[2], r[3]] / SQRT8, r[3]
        else:
            continue
        if s not in index:
            index[s] = len(states)
            states.append(s)
        pairs.append((index[("1", k)], index[s], w))
    e0 = np.zeros(len(states))
    for n, s in enumerate(states):
        if s[0] == "1":
            e0[n] = nu[s[1]]
        elif s[0] == "2":
            e0[n] = 2 * nu[s[1]] + 2 * chi[s[1], s[1]]
        else:
            e0[n] = nu[s[1]] + nu[s[2]] + chi[s[1], s[2]]
    V = np.zeros((len(states), len(states)))
    for a, b, w in pairs:
        V[a, b] = V[b, a] = w
    return Polyad(states, e0, V, np.array([s[1] if s[0] == "1" else -1 for s in states]))


def _occupations(omega, temperature_k):
    x = np.minimum(omega / (dos.KB_CM * max(temperature_k, 1e-3)), 700.0)
    nbar = 1.0 / np.expm1(x)
    return nbar, nbar * (nbar + 1.0)


def _fixed_energy_variance(rows, omega, var_n):
    """Variance of rows @ n at fixed total energy omega @ n: the canonical variance less the part correlated
    with the energy fluctuation, cov^2 / var(E) (independent Bose modes)."""
    v = (rows**2) @ var_n
    cov = rows @ (omega * var_n)
    return np.maximum(v - cov**2 / max(float((omega**2) @ var_n), 1e-300), 0.0)


def mode_shifts(chi, omega, temperature_k):
    """(mean, variance at fixed energy) of each fundamental's emission shift at T [cm^-1, cm^-2] (see module)."""
    nbar, var_n = _occupations(omega, temperature_k)
    off = chi - np.diag(np.diag(chi))
    mean = off @ nbar + 4.0 * np.diag(chi) * nbar
    return mean, _fixed_energy_variance(off, omega, var_n)


def polyad_lines(poly, chi, omega, temperature_k):
    """(energies, bright shares [state a, bright row], variances) of the polyad's eigenstates at T."""
    mean, var = mode_shifts(chi, omega, temperature_k)
    nbar, var_n = _occupations(omega, temperature_k)
    off = chi - np.diag(np.diag(chi))
    d_mean, d_var = np.zeros(len(poly.states)), np.zeros(len(poly.states))
    for n, s in enumerate(poly.states):
        if s[0] == "1":
            d_mean[n], d_var[n] = mean[s[1]], var[s[1]]
        else:
            modes = [s[1], s[1]] if s[0] == "2" else [s[1], s[2]]
            row = off[modes[0]] + off[modes[1]]
            d_mean[n] = row @ nbar
            d_var[n] = _fixed_energy_variance(row[None, :], omega, var_n)[0]
    e, c = np.linalg.eigh(np.diag(poly.e0 + d_mean) + poly.coupling)
    b = np.where(poly.bright >= 0)[0]
    shares = (c[b, :] ** 2).T                            # [eigenstate, bright row]
    return e, shares, (c**2).T @ d_var


def shift_distribution(rows, omega, temperature_k, step_cm=0.5, lo_cm=-700.0, n=2048, smooth_cm=1.0):
    """Density of the shift ``rows @ n`` [per cm^-1] on a grid, for independent Bose occupations n at T.

    Exact (not mean-field): the characteristic function is prod_i (1 - q_i) / (1 - q_i exp(i t r_i)), q_i =
    exp(-omega_i / kT), inverted by FFT; a Gaussian of ``smooth_cm`` (sigma) smooths the lattice of the
    discrete shifts. Returns (grid, densities [row, grid])."""
    rows = np.atleast_2d(rows)
    q = np.exp(-np.minimum(omega / (dos.KB_CM * max(temperature_k, 1e-3)), 700.0))
    t = 2.0 * np.pi * np.fft.fftfreq(n, d=step_cm)
    grid = lo_cm + step_cm * np.arange(n)
    with np.errstate(divide="ignore"):
        logq = np.log(q)
    phase = 1j * t[None, None, :] * rows[:, :, None]
    log_phi = np.sum(np.log1p(-q)[None, :, None] - np.log1p(-np.exp(logq[None, :, None] + phase)), axis=1)
    phi = np.exp(log_phi - 0.5 * (t * smooth_cm) ** 2 - 1j * t * lo_cm)
    dens = np.real(np.fft.fft(phi, axis=1)) / (n * step_cm)
    dens = np.maximum(dens, 0.0)
    return grid, dens / np.maximum(dens.sum(axis=1, keepdims=True) * step_cm, 1e-300)


def intensities(ff, nu, pahdb_uid=336):
    """km/mol per mode: PAHdb v4.00 harmonic intensities of the same molecule, by frequency rank within the
    CH stretches (> 2500 cm^-1) and within the other modes (needs PAHdb, ``pahdb-theoretical-v4-00``)."""
    from pycalima import _paths
    from pycalima.pahchem import ames_suite

    recs = ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"),
                                _paths.get_model_data_dir() / "pahchem_pahdb_index")
    r = next(x for x in recs if x["uid"] == pahdb_uid)
    f, inten = np.asarray(r["freqs_cm"], float), np.asarray(r["intensity"], float)
    if f.size != nu.size:
        raise ValueError(f"PAHdb UID {pahdb_uid} has {f.size} modes, the force field {nu.size}")
    out = np.zeros(nu.size)
    for sel_f, sel_n in ((f > CH_CM, nu > CH_CM), (f <= CH_CM, nu <= CH_CM)):
        if sel_f.sum() != sel_n.sum():
            raise ValueError("different numbers of CH stretches")
        out[np.where(sel_n)[0][np.argsort(-nu[sel_n])]] = inten[np.where(sel_f)[0][np.argsort(-f[sel_f])]]
    return out


@dataclass
class CascadeSpectrum:
    nu_fund: np.ndarray
    photons: np.ndarray          # [mode, bin] per absorbed UV photon
    temperature_k: np.ndarray    # per bin
    poly: Polyad
    chi: np.ndarray
    omega: np.ndarray
    t_grid: np.ndarray
    fwhm_floor_cm: float = 1.0

    def spectrum_distribution(self, nu_cm, lo_cm=2600.0, hi_cm=3300.0, smooth_cm=1.0, step_cm=0.5):
        """As ``spectrum``, with the exact shift distributions (``shift_distribution``) instead of Gaussians,
        and the polyad's mixing at T = 0 (each eigenstate's shift vector the coefficient-weighted one)."""
        nu = np.asarray(nu_cm, dtype=float)
        out = np.zeros(nu.size)
        poly = self.poly
        e0, c0 = np.linalg.eigh(np.diag(poly.e0) + poly.coupling)
        off = self.chi - np.diag(np.diag(self.chi))
        srow = np.zeros((len(poly.states), self.omega.size))
        for n, st in enumerate(poly.states):
            srow[n] = off[st[1]] if st[0] == "1" else (off[st[1]] + off[st[1] if st[0] == "2" else st[2]])
        brows = np.where(poly.bright >= 0)[0]
        shares = (c0[brows, :] ** 2).T                    # [eigenstate, bright row]
        rows_eig = (c0**2).T @ srow                       # [eigenstate, mode]
        bright_index = {int(poly.bright[r]): n for n, r in enumerate(brows)}
        sel = [k for k in range(self.nu_fund.size) if lo_cm <= self.nu_fund[k] <= hi_cm]
        it = np.clip(np.searchsorted(self.t_grid, self.temperature_k), 0, self.t_grid.size - 1)
        for t in np.unique(it):
            bins = it == t
            T = self.t_grid[t]
            ph = self.photons[:, bins].sum(axis=1)
            w_eig = np.zeros(e0.size)
            for k, n in bright_index.items():
                w_eig += ph[k] * shares[:, n]
            lines = [(e0[a], rows_eig[a], w_eig[a]) for a in range(e0.size) if w_eig[a] > 0]
            lines += [(self.nu_fund[k], off[k], ph[k]) for k in sel if k not in bright_index and ph[k] > 0]
            if not lines:
                continue
            tot = max(w for _, _, w in lines)
            lines = [ln for ln in lines if ln[2] > 1e-3 * tot]
            grid, dens = shift_distribution(np.array([ln[1] for ln in lines]), self.omega, T, step_cm,
                                            smooth_cm=smooth_cm)
            for (e, _, w), d in zip(lines, dens):
                out += w * e * np.interp(nu - e, grid, d, left=0.0, right=0.0)
        return out

    def spectrum(self, nu_cm, lo_cm=None, hi_cm=None):
        """Emitted power per cm^-1 on ``nu_cm`` per absorbed photon; Gaussians with FWHM >= ``fwhm_floor_cm``."""
        nu = np.asarray(nu_cm, dtype=float)
        out = np.zeros(nu.size)
        it = np.clip(np.searchsorted(self.t_grid, self.temperature_k), 0, self.t_grid.size - 1)
        bright_rows = {k: n for n, k in enumerate(self.poly.bright[self.poly.bright >= 0])}
        sig_floor = self.fwhm_floor_cm / np.sqrt(8 * np.log(2))
        for t in np.unique(it):
            bins = it == t
            T = self.t_grid[t]
            e, shares, var = polyad_lines(self.poly, self.chi, self.omega, T)
            mean, mvar = mode_shifts(self.chi, self.omega, T)
            ph = self.photons[:, bins].sum(axis=1)
            centres, weights, sig = [], [], []
            for k in range(self.nu_fund.size):
                if ph[k] <= 0:
                    continue
                if k in bright_rows:
                    w = shares[:, bright_rows[k]]
                    centres.append(e)
                    weights.append(ph[k] * w * e)
                    sig.append(np.sqrt(var + sig_floor**2))
                else:
                    c = self.nu_fund[k] + mean[k]
                    centres.append([c])
                    weights.append([ph[k] * c])
                    sig.append([np.sqrt(mvar[k] + sig_floor**2)])
            c, w, s = (np.concatenate(v) for v in (centres, weights, sig))
            keep = w > 1e-9 * w.max()
            if lo_cm is not None:
                keep &= (c > lo_cm - 5 * s) & (c < (hi_cm if hi_cm is not None else np.inf) + 5 * s)
            c, w, s = c[keep], w[keep], s[keep]
            out += (w[None, :] * np.exp(-0.5 * ((nu[:, None] - c[None, :]) / s[None, :]) ** 2)
                    / (s[None, :] * np.sqrt(2 * np.pi))).sum(axis=1)
        return out


def run(ff, e_uv_ev, intensity_km_mol, de_cm=20.0, t_grid=None):
    """``CascadeSpectrum`` of force field ``ff`` after one ``e_uv_ev`` photon."""
    chi = vpt2.anharmonic_constants(ff)
    nu = vpt2.fundamentals(ff.omega, chi)
    poly = build_polyad(ff, chi, nu)
    a = _EINSTEIN_A_PER_KM_MOL * nu**2 * np.asarray(intensity_km_mol, dtype=float)
    e_uv = e_uv_ev * dos.CM_PER_EV
    cm = energy_me.CascadeModel(nu, a, e_uv + 5 * de_cm, de_cm, 2.0)
    start = np.zeros(cm.n)
    x = e_uv / de_cm
    lo = int(np.floor(x))
    start[lo], start[lo + 1] = 1 - (x - lo), x - lo
    X = emission.emission_by_energy(cm, start)
    from pycalima.pahchem.demyk2026 import temperature

    T = temperature(ff.omega, cm.energy_cm)
    grid = np.arange(25.0, 3000.0, 25.0) if t_grid is None else t_grid
    return CascadeSpectrum(nu, X, T, poly, chi, ff.omega, grid)
